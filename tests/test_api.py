"""Auth, rate limits, body size, validation, health/readiness, streaming, cache, startup, break-glass."""

from __future__ import annotations

import json
import re
from typing import Any

import httpx
import pytest
import respx

from app.config import ModelRegistry, Settings
from app.llm.startup import ModelValidationError, check_key_tier, validate_models
from app.main import create_app, run_startup_checks
from tests.conftest import (
    AUTH,
    BASE,
    CLAUDE,
    FREE,
    GEMINI,
    LLAMA,
    PAID_OSS,
    AppFactory,
    FakeTracer,
    chat,
    completion,
    error,
    make_settings,
    no_sleep,
    sent_bodies,
    sse_body,
)


# ----------------------------------------------------------------- auth & limits
@pytest.mark.parametrize(
    "headers",
    [{}, {"Authorization": "Bearer wrong-key"}, {"Authorization": "Basic abc"}, {"Authorization": "Bearer "}],
)
async def test_missing_or_invalid_key_401(client: httpx.AsyncClient, headers: dict[str, str]) -> None:
    resp = await client.post("/v1/chat/completions", json=chat("hi"), headers=headers)
    assert resp.status_code == 401
    assert resp.json()["error"]["code"] == "invalid_api_key"
    assert resp.headers["WWW-Authenticate"] == "Bearer"


async def test_no_configured_keys_fails_closed(app_factory: AppFactory) -> None:
    app = app_factory(gateway_api_keys="")
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://gw") as c:
        resp = await c.post("/v1/chat/completions", json=chat("hi"), headers=AUTH)
    assert resp.status_code == 401


async def test_oversized_body_413(app_factory: AppFactory, mock_router: respx.MockRouter) -> None:
    route = mock_router.post("/chat/completions")
    app = app_factory(max_body_bytes=2048)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://gw") as c:
        resp = await c.post("/v1/chat/completions", json=chat("x " * 5000), headers=AUTH)
        # Chunked upload without Content-Length is also bounded.

        async def gen():  # type: ignore[no-untyped-def]
            for _ in range(10):
                yield b"a" * 1000

        resp2 = await c.post("/v1/chat/completions", content=gen(), headers=AUTH)
    assert resp.status_code == 413
    assert resp.json()["error"]["code"] == "request_too_large"
    assert resp.headers["X-Request-ID"]
    assert resp2.status_code == 413
    assert route.call_count == 0


async def test_per_key_rate_limit(app_factory: AppFactory, mock_router: respx.MockRouter) -> None:
    mock_router.post("/chat/completions").mock(return_value=httpx.Response(200, json=completion(FREE)))
    app = app_factory(rate_limit_rpm_per_key=2)
    other = {"Authorization": "Bearer gw-test-key-0002"}
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://gw") as c:
        codes = [(await c.post("/v1/chat/completions", json=chat("hi"), headers=AUTH)).status_code for _ in range(3)]
        limited = await c.post("/v1/chat/completions", json=chat("hi"), headers=AUTH)
        other_key = await c.post("/v1/chat/completions", json=chat("hi"), headers=other)
    assert codes == [200, 200, 429]
    assert limited.status_code == 429 and int(limited.headers["Retry-After"]) >= 1
    assert other_key.status_code == 200


async def test_invalid_body_400(client: httpx.AsyncClient) -> None:
    r1 = await client.post("/v1/chat/completions", json={"model": "auto", "messages": []}, headers=AUTH)
    r2 = await client.post(
        "/v1/chat/completions", content=b"{not json", headers={**AUTH, "content-type": "application/json"}
    )
    assert r1.status_code == 400 and r1.json()["error"]["code"] == "invalid_request"
    assert r2.status_code == 400


async def test_request_id_propagated(client: httpx.AsyncClient, mock_router: respx.MockRouter) -> None:
    mock_router.post("/chat/completions").mock(return_value=httpx.Response(200, json=completion(FREE)))
    resp = await client.post("/v1/chat/completions", json=chat("hi"), headers={**AUTH, "X-Request-ID": "abc-123"})
    assert resp.headers["X-Request-ID"] == "abc-123"
    bad = await client.post("/v1/chat/completions", json=chat("hi"), headers={**AUTH, "X-Request-ID": "bad id\n"})
    assert bad.headers["X-Request-ID"] != "bad id\n"


# ----------------------------------------------------------------- health
async def test_healthz_and_readyz(client: httpx.AsyncClient) -> None:
    assert (await client.get("/healthz")).json() == {"status": "ok"}
    ready = await client.get("/readyz")
    assert ready.status_code == 200
    body = ready.json()
    assert body["status"] == "ready" and "free_tier" in body and "circuit_breakers" in body


async def test_readyz_not_ready_without_key(app_factory: AppFactory) -> None:
    app = app_factory(openrouter_api_key=None)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://gw") as c:
        resp = await c.get("/readyz")
    assert resp.status_code == 503


# ----------------------------------------------------------------- streaming
async def test_streaming_passthrough_with_headers(
    client: httpx.AsyncClient, mock_router: respx.MockRouter, tracer: FakeTracer
) -> None:
    route = mock_router.post("/chat/completions").mock(
        return_value=httpx.Response(
            200, text=sse_body(FREE, ["Hel", "lo"]), headers={"content-type": "text/event-stream"}
        )
    )
    resp = await client.post("/v1/chat/completions", json=chat("hi", stream=True), headers=AUTH)
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/event-stream")
    assert resp.headers["X-Routed-Model"] == FREE and resp.headers["X-Free-Tier"] == "true"
    events = [line[6:] for line in resp.text.splitlines() if line.startswith("data: ")]
    assert events[-1] == "[DONE]"
    text = "".join(
        c["choices"][0]["delta"].get("content", "") for c in map(json.loads, events[:-1]) if c.get("choices")
    )
    assert text == "Hello"
    body = sent_bodies(route)[0]
    assert body["stream"] is True and body["stream_options"] == {"include_usage": True}
    assert tracer.last.completion_tokens == 2 and tracer.last.cost_usd == 0.0


async def test_streaming_fallback_before_first_token(client: httpx.AsyncClient, mock_router: respx.MockRouter) -> None:
    def handler(req: httpx.Request) -> httpx.Response:
        model = json.loads(req.content)["model"]
        if model == FREE:
            return error(429)
        return httpx.Response(200, text=sse_body(model, ["ok"]), headers={"content-type": "text/event-stream"})

    mock_router.post("/chat/completions").mock(side_effect=handler)
    resp = await client.post("/v1/chat/completions", json=chat("hi", stream=True), headers=AUTH)
    assert resp.status_code == 200
    assert resp.headers["X-Routed-Model"] == PAID_OSS
    assert resp.headers["X-Fallback-Used"] == "true"


async def test_streaming_output_leak_terminates_stream(
    client: httpx.AsyncClient, mock_router: respx.MockRouter
) -> None:
    mock_router.post("/chat/completions").mock(
        return_value=httpx.Response(
            200,
            text=sse_body(FREE, ["key: sk-or-v1-testkey", "testkeytestkey"]),
            headers={"content-type": "text/event-stream"},
        )
    )
    resp = await client.post("/v1/chat/completions", json=chat("hi", stream=True), headers=AUTH)
    assert "output_rejected" in resp.text
    assert "testkeytestkeytestkey" not in resp.text


async def test_streaming_empty_upstream(client: httpx.AsyncClient, mock_router: respx.MockRouter) -> None:
    mock_router.post("/chat/completions").mock(
        return_value=httpx.Response(200, text="data: [DONE]\n\n", headers={"content-type": "text/event-stream"})
    )
    resp = await client.post("/v1/chat/completions", json=chat("hi", stream=True), headers=AUTH)
    assert resp.status_code == 502
    assert resp.json()["error"]["code"] == "upstream_empty_stream"


async def test_streaming_mid_stream_error_event(client: httpx.AsyncClient, mock_router: respx.MockRouter) -> None:
    chunk = {
        "id": "g",
        "object": "chat.completion.chunk",
        "created": 1,
        "model": FREE,
        "choices": [{"index": 0, "delta": {"content": "par"}, "finish_reason": None}],
    }
    broken = f'data: {json.dumps(chunk)}\n\ndata: {{"error": {{"message": "provider died"}}}}\n\n'
    mock_router.post("/chat/completions").mock(
        return_value=httpx.Response(200, text=broken, headers={"content-type": "text/event-stream"})
    )
    resp = await client.post("/v1/chat/completions", json=chat("hi", stream=True), headers=AUTH)
    assert resp.status_code == 200  # headers already sent: no fallback after first token
    assert "stream_interrupted" in resp.text


# ----------------------------------------------------------------- semantic cache
async def test_semantic_cache_hit_scoped_per_key(
    app_factory: AppFactory, mock_router: respx.MockRouter, tracer: FakeTracer
) -> None:
    app = app_factory(enable_semantic_cache=True)
    route = mock_router.post("/chat/completions").mock(return_value=httpx.Response(200, json=completion(FREE, "Paris")))
    q = chat("What is the capital of France?", temperature=0)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://gw") as c:
        first = await c.post("/v1/chat/completions", json=q, headers=AUTH)
        second = await c.post(
            "/v1/chat/completions", json=chat("what is the capital of France", temperature=0), headers=AUTH
        )
        other_key = await c.post("/v1/chat/completions", json=q, headers={"Authorization": "Bearer gw-test-key-0002"})
        warm = await c.post(
            "/v1/chat/completions", json=chat("What is the capital of France?", temperature=0.7), headers=AUTH
        )
        opt_out = await c.post("/v1/chat/completions", json=q, headers={**AUTH, "Cache-Control": "no-cache"})
    assert first.headers["X-Cache"] == "miss"
    assert second.headers["X-Cache"] == "hit" and second.json()["choices"][0]["message"]["content"] == "Paris"
    assert second.headers["X-Routed-Model"] == FREE
    assert other_key.headers["X-Cache"] == "miss"
    assert warm.headers["X-Cache"] == "bypass"
    assert opt_out.headers["X-Cache"] == "bypass"
    assert route.call_count == 4
    assert tracer.records[1].cache == "hit"


def test_semantic_cache_ttl_and_eviction() -> None:
    from app.cache.semantic import SemanticCache

    now = [0.0]
    cache = SemanticCache(threshold=0.9, ttl_s=10, max_entries=1, clock=lambda: now[0])
    cache.put("k", "ctx", "hello world", {"v": 1})
    assert cache.get("k", "ctx", "hello world") == {"v": 1}
    assert cache.get("k", "other-ctx", "hello world") is None
    assert cache.get("k", "ctx", "completely different question about taxes") is None
    now[0] = 11
    assert cache.get("k", "ctx", "hello world") is None
    cache.put("k", "ctx", "a", {"v": 1})
    cache.put("k", "ctx", "b", {"v": 2})
    assert cache.get("k", "ctx", "a") is None
    assert SemanticCache.eligible(None, False, False) is False
    assert SemanticCache.eligible(0.0, False, False) is True
    assert SemanticCache.eligible(0.0, True, False) is False


# ----------------------------------------------------------------- startup checks
def _models_listing(ids: list[str]) -> dict[str, Any]:
    return {
        "data": [
            {
                "id": i,
                "context_length": 999_999 if i == CLAUDE else 131072,
                "pricing": {"prompt": "0.000003", "completion": "0.000015"},
            }
            for i in ids
        ]
    }


async def test_startup_validation_passes_and_syncs(mock_router: respx.MockRouter) -> None:
    registry = ModelRegistry.load()
    mock_router.get("/models").mock(
        return_value=httpx.Response(200, json=_models_listing([FREE, PAID_OSS, LLAMA, GEMINI, CLAUDE, "x/y"]))
    )
    async with httpx.AsyncClient() as http:
        await validate_models(make_settings(), registry, http)
    assert registry.models[CLAUDE].context_window == 999_999
    assert registry.models[PAID_OSS].input_price_per_m == pytest.approx(3.0)
    assert registry.models[FREE].input_price_per_m == 0.0  # free models stay free


async def test_startup_validation_fails_fast_on_missing_id(mock_router: respx.MockRouter) -> None:
    mock_router.get("/models").mock(return_value=httpx.Response(200, json=_models_listing([FREE, PAID_OSS])))
    async with httpx.AsyncClient() as http:
        with pytest.raises(ModelValidationError, match=re.escape(CLAUDE)):
            await validate_models(make_settings(), ModelRegistry.load(), http)


async def test_key_tier_check(mock_router: respx.MockRouter, capsys: pytest.CaptureFixture[str]) -> None:
    from app.observability.logging import configure_logging

    configure_logging("INFO", [])
    route = mock_router.get("/key")
    async with httpx.AsyncClient() as http:
        route.mock(return_value=httpx.Response(200, json={"data": {"is_free_tier": True}}))
        assert (await check_key_tier(make_settings(), http))["is_free_tier"] is True
        route.mock(return_value=httpx.Response(200, json={"data": {"is_free_tier": False}}))
        await check_key_tier(make_settings(), http)
        route.mock(return_value=httpx.Response(200, json={"data": {"is_free_tier": True}}))
        await check_key_tier(make_settings(free_tier_daily_cap=1000), http)
        route.mock(
            return_value=httpx.Response(
                200, json={"data": {"is_free_tier": False, "free_model_daily_requests": {"limit": 1000}}}
            )
        )
        await check_key_tier(make_settings(), http)
    err = capsys.readouterr().err
    assert "raising FREE_TIER_DAILY_CAP to 1000" in err
    assert "differs from OpenRouter-reported" in err
    assert "sk-or-v1-testkeytestkeytestkey" not in err


async def test_lifespan_runs_startup_checks(mock_router: respx.MockRouter) -> None:
    mock_router.get("/models").mock(
        return_value=httpx.Response(200, json=_models_listing([FREE, PAID_OSS, LLAMA, GEMINI, CLAUDE]))
    )
    mock_router.get("/key").mock(return_value=error(500))
    tracer = FakeTracer()
    app = create_app(make_settings(validate_models_on_startup=True), tracer=tracer, sleep=no_sleep)
    async with app.router.lifespan_context(app):
        assert app.state.startup_ok is True
    assert tracer.flushed


async def test_startup_skipped_without_key() -> None:
    app = create_app(make_settings(openrouter_api_key=None, validate_models_on_startup=True), tracer=FakeTracer())
    await run_startup_checks(app)
    assert app.state.startup_ok is False


# ----------------------------------------------------------------- break-glass
async def test_break_glass_used_when_openrouter_exhausted() -> None:
    settings = make_settings(
        break_glass_enabled=True,
        anthropic_api_key="sk-ant-test-000000000",
        break_glass_anthropic_model="claude-direct-test",
        retry_max_attempts=1,
    )
    app = create_app(settings, tracer=FakeTracer(), sleep=no_sleep)
    with respx.mock(assert_all_mocked=True) as router:
        router.post(f"{BASE}/chat/completions").mock(return_value=error(503))
        direct = router.post("https://api.anthropic.com/v1/chat/completions").mock(
            return_value=httpx.Response(200, json=completion("claude-direct-test"))
        )
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://gw") as c:
            resp = await c.post("/v1/chat/completions", json=chat("hi"), headers={**AUTH, "X-Model-Override": "claude"})
    assert resp.status_code == 200
    assert resp.headers["X-Routed-Model"] == "anthropic/claude-direct-test"
    assert direct.call_count == 1


def test_break_glass_disabled_by_default() -> None:
    from app.llm.client import build_break_glass_clients

    s = make_settings(anthropic_api_key="sk-ant-x", break_glass_anthropic_model="m")
    assert build_break_glass_clients(s, httpx.AsyncClient()) == {}


# ----------------------------------------------------------------- config
def test_registry_validation_errors(tmp_path: Any) -> None:
    good = json.loads((ModelRegistry.load.__globals__["DEFAULT_REGISTRY_PATH"]).read_text())
    bad = json.loads(json.dumps(good))
    bad["tiers"]["SIMPLE"]["model"] = "nope/missing"
    p = tmp_path / "r.json"
    p.write_text(json.dumps(bad))
    with pytest.raises(ValueError, match="unknown model"):
        ModelRegistry.load(p)
    bad = json.loads(json.dumps(good))
    bad["models"][FREE]["input_price_per_m"] = 1.0
    p.write_text(json.dumps(bad))
    with pytest.raises(ValueError, match="zero prices"):
        ModelRegistry.load(p)
    bad = json.loads(json.dumps(good))
    bad["models"][CLAUDE]["fallbacks"] = ["ghost/model"]
    p.write_text(json.dumps(bad))
    with pytest.raises(ValueError, match="unknown fallback"):
        ModelRegistry.load(p)
    bad = json.loads(json.dumps(good))
    del bad["tiers"]["COMPLEX"]
    p.write_text(json.dumps(bad))
    with pytest.raises(ValueError, match="missing tiers"):
        ModelRegistry.load(p)


def test_settings_parsing() -> None:
    s = Settings(_env_file=None, gateway_api_keys=" a-key-000001 , ,b-key-000002 ", openrouter_api_key="")  # type: ignore[call-arg]
    assert s.gateway_keys == frozenset({"a-key-000001", "b-key-000002"})
    assert s.openrouter_api_key is None
    assert "a-key-000001" in s.secret_values()


def test_langfuse_host_alias(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LANGFUSE_HOST", "https://legacy.example")
    assert Settings(_env_file=None).langfuse_base_url == "https://legacy.example"  # type: ignore[call-arg]
    monkeypatch.setenv("LANGFUSE_BASE_URL", "https://new.example")
    assert Settings(_env_file=None).langfuse_base_url == "https://new.example"  # type: ignore[call-arg]


async def test_break_glass_failure_then_503() -> None:
    settings = make_settings(
        break_glass_enabled=True,
        gemini_api_key="g-key-00000000",
        break_glass_gemini_model="gemini-direct",
        retry_max_attempts=1,
    )
    app = create_app(settings, tracer=FakeTracer(), sleep=no_sleep)
    with respx.mock(assert_all_mocked=True) as router:
        router.post(f"{BASE}/chat/completions").mock(return_value=error(503))
        direct = router.post("https://generativelanguage.googleapis.com/v1beta/openai/chat/completions").mock(
            return_value=error(500)
        )
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://gw") as c:
            resp = await c.post("/v1/chat/completions", json=chat("hi"), headers={**AUTH, "X-Model-Override": "gemini"})
    assert direct.call_count == 1
    assert resp.status_code == 503


def test_base_url_constant() -> None:
    assert make_settings().openrouter_base_url == BASE


def test_env_example_parses_to_safe_defaults() -> None:
    s = Settings(_env_file=".env.example")  # type: ignore[call-arg]
    assert s.openrouter_api_key is None
    assert s.gateway_keys == frozenset()
    assert s.free_tier_daily_cap == 50 and s.free_tier_rpm_cap == 20
    assert s.enable_tracing is True and s.enable_semantic_cache is False
    assert s.break_glass_enabled is False


def test_inline_comment_values_are_ignored(tmp_path: Any) -> None:
    env = tmp_path / ".env"
    env.write_text(
        "GATEWAY_API_KEYS=   # comma-separated keys\nOPENROUTER_API_KEY=  # required\nFREE_TIER_DAILY_CAP=50\n"
    )
    s = Settings(_env_file=str(env))  # type: ignore[call-arg]
    assert s.gateway_keys == frozenset()  # a comment must never become a valid gateway key
    assert s.openrouter_api_key is None
