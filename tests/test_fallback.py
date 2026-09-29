"""Gateway-level fallback: 429, 5xx, timeouts, non-retryable 4xx, circuit breaker."""

from __future__ import annotations

import httpx
import pytest
import respx
from fastapi import FastAPI

from app.llm.circuit_breaker import BreakerState, CircuitBreakerRegistry
from app.llm.errors import parse_retry_after
from tests.conftest import (
    AUTH,
    CLAUDE,
    GEMINI,
    PAID_OSS,
    AppFactory,
    FakeTracer,
    chat,
    completion,
    error,
    sent_bodies,
)

ARCH = "Design a scalable distributed system architecture for an event-sourced order service."


def by_model(responses: dict[str, list[httpx.Response]]):  # type: ignore[no-untyped-def]
    """respx side effect returning queued responses per requested model."""
    import json

    def side_effect(request: httpx.Request) -> httpx.Response:
        model = json.loads(request.content)["model"]
        queue = responses[model]
        return queue.pop(0) if len(queue) > 1 else queue[0]

    return side_effect


async def test_primary_429_walks_to_next_model(
    client: httpx.AsyncClient, mock_router: respx.MockRouter, tracer: FakeTracer
) -> None:
    route = mock_router.post("/chat/completions").mock(
        side_effect=by_model({CLAUDE: [error(429)], GEMINI: [httpx.Response(200, json=completion(GEMINI))]})
    )
    resp = await client.post("/v1/chat/completions", json=chat(ARCH), headers=AUTH)

    assert resp.status_code == 200
    assert resp.headers["X-Fallback-Used"] == "true"
    assert resp.headers["X-Routed-Model"] == GEMINI
    models_sent = [b["model"] for b in sent_bodies(route)]
    # 3 attempts on the primary (default retry_max_attempts), then walk to Gemini.
    assert models_sent == [CLAUDE, CLAUDE, CLAUDE, GEMINI]
    # The walked hop still carries the remaining chain for OpenRouter-native fallback.
    assert sent_bodies(route)[-1]["models"] == [PAID_OSS]
    outcomes = [a["outcome"] for a in tracer.last.attempts]
    assert outcomes == ["error", "error", "error", "ok"]


async def test_5xx_sequence_retried_then_succeeds(client: httpx.AsyncClient, mock_router: respx.MockRouter) -> None:
    route = mock_router.post("/chat/completions").mock(
        side_effect=[error(502), error(503), httpx.Response(200, json=completion(CLAUDE))]
    )
    resp = await client.post("/v1/chat/completions", json=chat(ARCH), headers=AUTH)
    assert resp.status_code == 200
    assert resp.headers["X-Fallback-Used"] == "false"
    assert [b["model"] for b in sent_bodies(route)] == [CLAUDE, CLAUDE, CLAUDE]


async def test_timeout_walks_chain(client: httpx.AsyncClient, mock_router: respx.MockRouter) -> None:
    def side_effect(request: httpx.Request) -> httpx.Response:
        import json

        if json.loads(request.content)["model"] == CLAUDE:
            raise httpx.ReadTimeout("slow", request=request)
        return httpx.Response(200, json=completion(GEMINI))

    mock_router.post("/chat/completions").mock(side_effect=side_effect)
    resp = await client.post("/v1/chat/completions", json=chat(ARCH), headers=AUTH)
    assert resp.status_code == 200
    assert resp.headers["X-Routed-Model"] == GEMINI
    assert resp.headers["X-Fallback-Used"] == "true"


@pytest.mark.parametrize("status", [400, 401, 402, 403, 404, 422])
async def test_non_429_4xx_not_retried(client: httpx.AsyncClient, mock_router: respx.MockRouter, status: int) -> None:
    route = mock_router.post("/chat/completions").mock(return_value=error(status))
    resp = await client.post("/v1/chat/completions", json=chat(ARCH), headers=AUTH)
    assert route.call_count == 1  # no retry, no chain walk
    assert resp.status_code == (400 if status == 400 else 502)
    assert resp.json()["error"]["code"] == "upstream_error"


async def test_retry_after_is_honoured(app_factory: AppFactory, mock_router: respx.MockRouter) -> None:
    sleeps: list[float] = []

    async def record_sleep(seconds: float) -> None:
        sleeps.append(seconds)

    from app.main import create_app
    from tests.conftest import make_settings

    app = create_app(make_settings(retry_max_delay_s=5.0), tracer=FakeTracer(), sleep=record_sleep)
    mock_router.post("/chat/completions").mock(
        side_effect=[error(429, {"Retry-After": "2"}), httpx.Response(200, json=completion(CLAUDE))]
    )
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://gw") as c:
        resp = await c.post("/v1/chat/completions", json=chat(ARCH), headers=AUTH)
    assert resp.status_code == 200
    assert sleeps == [2.0]


async def test_retry_after_beyond_cap_walks_immediately(
    client: httpx.AsyncClient, mock_router: respx.MockRouter
) -> None:
    route = mock_router.post("/chat/completions").mock(
        side_effect=by_model(
            {CLAUDE: [error(429, {"Retry-After": "120"})], GEMINI: [httpx.Response(200, json=completion(GEMINI))]}
        )
    )
    resp = await client.post("/v1/chat/completions", json=chat(ARCH), headers=AUTH)
    assert resp.status_code == 200
    assert [b["model"] for b in sent_bodies(route)] == [CLAUDE, GEMINI]


async def test_all_models_fail_returns_503(
    client: httpx.AsyncClient, mock_router: respx.MockRouter, tracer: FakeTracer
) -> None:
    mock_router.post("/chat/completions").mock(return_value=error(503))
    resp = await client.post("/v1/chat/completions", json=chat(ARCH), headers=AUTH)
    assert resp.status_code == 503
    body = resp.json()["error"]
    assert body["code"] == "upstream_unavailable"
    assert body["request_id"] == resp.headers["X-Request-ID"]
    assert tracer.last.status_code == 503
    assert len(tracer.last.attempts) >= 3


async def test_circuit_breaker_opens_and_skips_model(app_factory: AppFactory, mock_router: respx.MockRouter) -> None:
    app: FastAPI = app_factory(circuit_failure_threshold=2, retry_max_attempts=1)
    route = mock_router.post("/chat/completions").mock(
        side_effect=by_model({CLAUDE: [error(500)], GEMINI: [httpx.Response(200, json=completion(GEMINI))]})
    )
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://gw") as c:
        for _ in range(2):
            await c.post("/v1/chat/completions", json=chat(ARCH), headers=AUTH)
        assert app.state.breakers.state(CLAUDE) is BreakerState.OPEN
        before = route.call_count
        resp = await c.post("/v1/chat/completions", json=chat(ARCH), headers=AUTH)
    assert resp.status_code == 200
    # Claude is skipped entirely while the breaker is open.
    assert [b["model"] for b in sent_bodies(route)[before:]] == [GEMINI]


def test_circuit_breaker_half_open_recovery() -> None:
    now = [0.0]
    cb = CircuitBreakerRegistry(failure_threshold=2, cooldown_s=30.0, clock=lambda: now[0])
    cb.record_failure("m")
    assert cb.allow("m")
    cb.record_failure("m")
    assert cb.state("m") is BreakerState.OPEN
    assert not cb.allow("m")
    now[0] = 31.0
    assert cb.allow("m")  # half-open trial
    assert cb.state("m") is BreakerState.HALF_OPEN
    assert not cb.allow("m")  # only one trial at a time
    cb.record_success("m")
    assert cb.state("m") is BreakerState.CLOSED
    assert cb.allow("m")


def test_circuit_breaker_half_open_failure_reopens() -> None:
    now = [0.0]
    cb = CircuitBreakerRegistry(failure_threshold=1, cooldown_s=10.0, clock=lambda: now[0])
    cb.record_failure("m")
    now[0] = 11.0
    assert cb.allow("m")
    cb.record_failure("m")
    assert cb.state("m") is BreakerState.OPEN
    assert cb.is_open("m")
    assert cb.snapshot() == {"m": "open"}


def test_parse_retry_after_formats() -> None:
    assert parse_retry_after("3") == 3.0
    assert parse_retry_after(None) is None
    assert parse_retry_after("garbage") is None
    assert parse_retry_after("Wed, 21 Oct 2015 07:28:00 GMT") == 0.0
