"""Tracing: fail-open behaviour, trace content, Langfuse exporter wiring, log redaction."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any

import httpx
import pytest
import respx
from loguru import logger

from app.observability.logging import configure_logging, redact
from app.observability.tracing import (
    LangfuseTracer,
    NoopTracer,
    SafeTracer,
    TraceRecord,
    build_tracer,
    hash_identifier,
)
from tests.conftest import AUTH, CLAUDE, FREE, GATEWAY_KEY, AppFactory, FakeTracer, chat, completion, make_settings


async def test_tracing_failure_does_not_break_requests(app_factory: AppFactory, mock_router: respx.MockRouter) -> None:
    app = app_factory(tracer_override=SafeTracer(FakeTracer(fail=True)))
    mock_router.post("/chat/completions").mock(return_value=httpx.Response(200, json=completion(FREE)))
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://gw") as c:
        resp = await c.post("/v1/chat/completions", json=chat("hi"), headers=AUTH)
    assert resp.status_code == 200


async def test_tracing_disabled_does_not_break_requests(app_factory: AppFactory, mock_router: respx.MockRouter) -> None:
    app = app_factory(tracer_override=NoopTracer())
    mock_router.post("/chat/completions").mock(return_value=httpx.Response(200, json=completion(FREE)))
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://gw") as c:
        resp = await c.post("/v1/chat/completions", json=chat("hi"), headers=AUTH)
    assert resp.status_code == 200


async def test_trace_contains_required_fields(
    client: httpx.AsyncClient, mock_router: respx.MockRouter, tracer: FakeTracer
) -> None:
    mock_router.post("/chat/completions").mock(
        return_value=httpx.Response(200, json=completion(FREE, prompt_tokens=12, completion_tokens=3))
    )
    resp = await client.post("/v1/chat/completions", json=chat("what's 2+2?", user="alice@example.com"), headers=AUTH)
    assert resp.status_code == 200
    t = tracer.last
    assert t.request_id == resp.headers["X-Request-ID"]
    assert t.tier == "SIMPLE"
    assert t.served_model == FREE and t.requested_model == FREE
    assert t.free_tier is True
    assert (t.prompt_tokens, t.completion_tokens, t.total_tokens) == (12, 3, 15)
    assert t.cost_usd == 0.0
    assert t.latency_ms > 0
    assert t.guardrail["flagged"] is False
    assert t.route_reasons and "est_tokens" in t.route_features
    assert t.attempts and t.attempts[0]["status"] == 200
    # Identity is hashed, raw content is not captured by default.
    assert t.key_hash and GATEWAY_KEY not in str(t.as_dict())
    assert t.user_hash and "alice@example.com" not in str(t.as_dict())
    assert t.raw_input is None and t.raw_output is None
    assert t.prompt_chars == len("what's 2+2?")


async def test_trace_raw_content_opt_in(
    app_factory: AppFactory, mock_router: respx.MockRouter, tracer: FakeTracer
) -> None:
    app = app_factory(trace_raw_content=True)
    mock_router.post("/chat/completions").mock(return_value=httpx.Response(200, json=completion(FREE, "4")))
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://gw") as c:
        await c.post("/v1/chat/completions", json=chat("2+2?"), headers=AUTH)
    assert tracer.last.raw_input[0]["content"] == "2+2?"
    assert tracer.last.raw_output == ["4"]


async def test_trace_recorded_on_error(
    client: httpx.AsyncClient, mock_router: respx.MockRouter, tracer: FakeTracer
) -> None:
    resp = await client.post("/v1/chat/completions", json=chat("hi"), headers={**AUTH, "X-Model-Override": "nope"})
    assert resp.status_code == 400
    assert tracer.last.status_code == 400 and tracer.last.error_code == "invalid_model_override"


def test_hash_identifier_is_stable_and_salted() -> None:
    assert hash_identifier("k", "s") == hash_identifier("k", "s")
    assert hash_identifier("k", "s") != hash_identifier("k", "t")
    assert len(hash_identifier("k", "")) == 16


def test_safe_tracer_swallows_all_errors() -> None:
    class Boom:
        def record(self, trace: TraceRecord) -> None:
            raise RuntimeError

        def flush(self) -> None:
            raise RuntimeError

        def shutdown(self) -> None:
            raise RuntimeError

    safe = SafeTracer(Boom())
    safe.record(TraceRecord(request_id="x"))
    safe.flush()
    safe.shutdown()


def test_build_tracer_variants(monkeypatch: pytest.MonkeyPatch) -> None:
    assert isinstance(build_tracer(make_settings(enable_tracing=False)), NoopTracer)
    assert isinstance(build_tracer(make_settings(enable_tracing=True)), NoopTracer)  # no keys

    def boom(settings: Any) -> None:
        raise ValueError("bad config")

    monkeypatch.setattr("app.observability.tracing.LangfuseTracer", boom)
    s = make_settings(enable_tracing=True, langfuse_public_key="pk-lf-x", langfuse_secret_key="sk-lf-x")
    assert isinstance(build_tracer(s), NoopTracer)


class _FakeObservation:
    def __init__(self, sink: list[dict[str, Any]], kwargs: dict[str, Any]) -> None:
        sink.append(kwargs)


class _FakeLangfuse:
    def __init__(self, **kwargs: Any) -> None:
        self.init_kwargs = kwargs
        self.observations: list[dict[str, Any]] = []
        self.flushed = self.closed = False

    @contextmanager
    def start_as_current_observation(self, **kwargs: Any):  # type: ignore[no-untyped-def]
        yield _FakeObservation(self.observations, kwargs)

    def flush(self) -> None:
        self.flushed = True

    def shutdown(self) -> None:
        self.closed = True


def test_langfuse_tracer_emits_one_trace_with_children(monkeypatch: pytest.MonkeyPatch) -> None:
    import langfuse

    propagated: list[dict[str, Any]] = []

    @contextmanager
    def fake_propagate(**kwargs: Any):  # type: ignore[no-untyped-def]
        propagated.append(kwargs)
        yield

    monkeypatch.setattr(langfuse, "Langfuse", _FakeLangfuse)
    monkeypatch.setattr(langfuse, "propagate_attributes", fake_propagate)
    settings = make_settings(
        enable_tracing=True,
        langfuse_public_key="pk-lf-test",
        langfuse_secret_key="sk-lf-test",
        langfuse_base_url="https://langfuse.example",
    )
    tracer = LangfuseTracer(settings)
    client: _FakeLangfuse = tracer._client  # type: ignore[assignment]
    assert client.init_kwargs["base_url"] == "https://langfuse.example"

    trace = TraceRecord(
        request_id="rid",
        key_hash="abc",
        tier="COMPLEX",
        served_model=CLAUDE,
        free_tier=False,
        fallback_used=True,
        prompt_tokens=5,
        completion_tokens=7,
        cost_usd=0.001,
        attempts=[
            {"model": CLAUDE, "outcome": "error", "status": 429},
            {"model": CLAUDE, "outcome": "ok", "status": 200, "served_model": CLAUDE},
        ],
    )
    tracer.record(trace)
    tracer.flush()
    tracer.shutdown()

    assert propagated[0]["user_id"] == "abc"
    assert "COMPLEX" in propagated[0]["tags"] and "fallback" in propagated[0]["tags"]
    names = [o["name"] for o in client.observations]
    assert names == ["gateway.request", "guardrail.input", "route", "upstream.attempt.0", "upstream.attempt.1"]
    ok_gen = client.observations[-1]
    assert ok_gen["as_type"] == "generation"
    assert ok_gen["usage_details"] == {"input": 5, "output": 7}
    assert ok_gen["cost_details"] == {"total": 0.001}
    assert client.observations[-2]["usage_details"] is None
    assert client.flushed and client.closed


def test_log_redaction(capsys: pytest.CaptureFixture[str]) -> None:
    assert redact("key=sk-or-v1-abcdefghijklmnopqrstuv", []) == "key=[REDACTED]"
    assert redact("Authorization: Bearer abc.def.ghi-123", []) == "Authorization: Bearer [REDACTED]"
    assert redact("my custom-secret-value here", ["custom-secret-value"]) == "my [REDACTED] here"
    configure_logging("INFO", ["gw-super-secret-1"])
    logger.info("presented key gw-super-secret-1")
    err = capsys.readouterr().err
    assert "gw-super-secret-1" not in err and "[REDACTED]" in err
