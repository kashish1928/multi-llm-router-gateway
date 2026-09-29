"""Free-tier handling: 429 walks to paid same-model, local caps skip the free model, cost is 0."""

from __future__ import annotations

import json
from datetime import UTC, datetime

import httpx
import respx

from app.llm.free_tier import FreeTierLimiter
from app.llm.pricing import estimate_cost_usd
from tests.conftest import AUTH, FREE, LLAMA, PAID_OSS, AppFactory, FakeTracer, chat, completion, error, sent_bodies


def _side_effect(free_response: httpx.Response):  # type: ignore[no-untyped-def]
    def handler(request: httpx.Request) -> httpx.Response:
        model = json.loads(request.content)["model"]
        if model == FREE:
            return free_response
        return httpx.Response(200, json=completion(model))

    return handler


async def test_free_model_429_uses_paid_same_model(
    client: httpx.AsyncClient, mock_router: respx.MockRouter, tracer: FakeTracer
) -> None:
    route = mock_router.post("/chat/completions").mock(side_effect=_side_effect(error(429)))
    resp = await client.post("/v1/chat/completions", json=chat("what's 2+2?"), headers=AUTH)

    assert resp.status_code == 200
    sent = [b["model"] for b in sent_bodies(route)]
    # Free-model 429 is NOT retried (failed attempts burn daily quota): walk immediately.
    assert sent == [FREE, PAID_OSS]
    assert sent_bodies(route)[1]["models"] == [LLAMA]
    assert resp.headers["X-Routed-Model"] == PAID_OSS
    assert resp.headers["X-Fallback-Used"] == "true"
    assert resp.headers["X-Free-Tier"] == "false"
    assert tracer.last.cost_usd is not None and tracer.last.cost_usd > 0


async def test_daily_cap_reached_skips_free_model_without_calling_it(
    app_factory: AppFactory, mock_router: respx.MockRouter, tracer: FakeTracer
) -> None:
    app = app_factory(free_tier_daily_cap=2)
    route = mock_router.post("/chat/completions").mock(
        side_effect=_side_effect(httpx.Response(200, json=completion(FREE)))
    )
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://gw") as c:
        for _ in range(2):
            r = await c.post("/v1/chat/completions", json=chat("hi"), headers=AUTH)
            assert r.headers["X-Free-Tier"] == "true"
        before = route.call_count
        resp = await c.post("/v1/chat/completions", json=chat("hi"), headers=AUTH)

    new_calls = sent_bodies(route)[before:]
    assert [b["model"] for b in new_calls] == [PAID_OSS]
    # The exhausted free model is also removed from the native fallback list.
    assert FREE not in new_calls[0].get("models", [])
    assert resp.headers["X-Free-Tier"] == "false"
    assert resp.headers["X-Fallback-Used"] == "true"
    assert any(a["outcome"] == "skipped_free_cap" for a in tracer.last.attempts)


async def test_rpm_cap_reached_skips_free_model(app_factory: AppFactory, mock_router: respx.MockRouter) -> None:
    app = app_factory(free_tier_rpm_cap=1)
    route = mock_router.post("/chat/completions").mock(
        side_effect=_side_effect(httpx.Response(200, json=completion(FREE)))
    )
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://gw") as c:
        await c.post("/v1/chat/completions", json=chat("hi"), headers=AUTH)
        before = route.call_count
        await c.post("/v1/chat/completions", json=chat("hello"), headers=AUTH)
    assert [b["model"] for b in sent_bodies(route)[before:]] == [PAID_OSS]


async def test_free_response_cost_zero_and_header(
    client: httpx.AsyncClient, mock_router: respx.MockRouter, tracer: FakeTracer
) -> None:
    mock_router.post("/chat/completions").mock(
        return_value=httpx.Response(200, json=completion(FREE, prompt_tokens=1000, completion_tokens=500))
    )
    resp = await client.post("/v1/chat/completions", json=chat("hello"), headers=AUTH)
    assert resp.headers["X-Free-Tier"] == "true"
    trace = tracer.last
    assert trace.free_tier is True
    assert trace.cost_usd == 0.0
    assert trace.prompt_tokens == 1000 and trace.completion_tokens == 500


async def test_failed_free_attempts_count_against_quota(app_factory: AppFactory, mock_router: respx.MockRouter) -> None:
    app = app_factory(free_tier_daily_cap=1)
    route = mock_router.post("/chat/completions").mock(side_effect=_side_effect(error(429)))
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://gw") as c:
        await c.post("/v1/chat/completions", json=chat("hi"), headers=AUTH)
        before = route.call_count
        await c.post("/v1/chat/completions", json=chat("hi"), headers=AUTH)
    assert [b["model"] for b in sent_bodies(route)[before:]] == [PAID_OSS]
    assert app.state.free_limiter.snapshot()["daily_used"] == 1


def test_limiter_resets_on_new_utc_day_and_minute() -> None:
    day1 = datetime(2026, 1, 1, 23, 59, 30, tzinfo=UTC).timestamp()
    now = [day1]
    lim = FreeTierLimiter(daily_cap=2, rpm_cap=1, clock=lambda: now[0])
    assert lim.try_acquire()
    assert not lim.try_acquire()  # rpm cap
    now[0] += 61
    assert lim.try_acquire()  # new minute, and new UTC day -> daily counter reset
    assert lim.snapshot()["day"] == "2026-01-02"
    assert lim.snapshot()["daily_used"] == 1
    now[0] += 61
    assert lim.try_acquire()
    now[0] += 61
    assert not lim.available()  # daily cap 2 reached
    lim.record_external()
    assert lim.snapshot()["daily_used"] == 3


def test_cost_estimates(app_factory: AppFactory) -> None:
    registry = app_factory().state.registry
    assert estimate_cost_usd(registry, FREE, 10_000, 10_000) == 0.0
    assert estimate_cost_usd(registry, "anthropic/claude-sonnet-5.5", 1_000_000, 0) == 3.0
    assert estimate_cost_usd(registry, "unknown/model", 1, 1) is None
