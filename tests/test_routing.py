"""Routing: labelled eval set (accuracy + confusion matrix), context-fit, overrides, cascade."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import httpx
import pytest
import respx

from app.config import ModelRegistry
from app.models import TIER_ORDER, ChatCompletionRequest, Tier
from app.routing.cascade import validity_problem
from app.routing.heuristic import HeuristicRouter
from tests.conftest import (
    AUTH,
    CLAUDE,
    FREE,
    GEMINI,
    AppFactory,
    FakeTracer,
    chat,
    completion,
    sent_bodies,
)

EVAL_PATH = Path(__file__).parent / "data" / "routing_eval.jsonl"


def load_eval() -> list[dict[str, Any]]:
    return [json.loads(line) for line in EVAL_PATH.read_text().splitlines() if line.strip()]


def to_request(row: dict[str, Any]) -> ChatCompletionRequest:
    text = row["prompt"] + row.get("filler", "") * row.get("filler_repeat", 0)
    extra = {"response_format": row["response_format"]} if "response_format" in row else {}
    return ChatCompletionRequest.model_validate(chat(text, **extra))


@pytest.fixture(scope="module")
def router() -> HeuristicRouter:
    return HeuristicRouter(ModelRegistry.load())


def test_eval_set_shape() -> None:
    rows = load_eval()
    assert len(rows) >= 30
    for tier in TIER_ORDER:
        assert sum(r["label"] == tier.value for r in rows) >= 10


def test_routing_eval_accuracy(router: HeuristicRouter, capsys: pytest.CaptureFixture[str]) -> None:
    rows = load_eval()
    matrix = {a: dict.fromkeys(TIER_ORDER, 0) for a in TIER_ORDER}
    misses: list[str] = []
    for row in rows:
        predicted = router.classify(to_request(row)).tier
        actual = Tier(row["label"])
        matrix[actual][predicted] += 1
        if predicted is not actual:
            misses.append(f"{row['id']}: expected {actual.value}, got {predicted.value}")
    correct = sum(matrix[t][t] for t in TIER_ORDER)
    accuracy = correct / len(rows)

    header = "actual \\ predicted".ljust(20) + "".join(t.value.rjust(14) for t in TIER_ORDER)
    lines = [header] + [
        t.value.ljust(20) + "".join(str(matrix[t][p]).rjust(14) for p in TIER_ORDER) for t in TIER_ORDER
    ]
    with capsys.disabled():
        print(f"\n\nRouting confusion matrix (n={len(rows)}, accuracy={accuracy * 100:.1f}%)")
        print("\n".join(lines))
        if misses:
            print("Misrouted: " + "; ".join(misses))
    assert accuracy >= 0.80


@pytest.mark.parametrize(
    ("row_id", "expected"),
    [("s01", Tier.SIMPLE), ("i01", Tier.INTERMEDIATE), ("c01", Tier.COMPLEX)],
)
def test_canonical_cases(router: HeuristicRouter, row_id: str, expected: Tier) -> None:
    row = next(r for r in load_eval() if r["id"] == row_id)
    decision = router.classify(to_request(row))
    assert decision.tier is expected
    assert decision.reasons
    assert "est_tokens" in decision.features


async def test_canonical_cases_reach_expected_models(client: httpx.AsyncClient, mock_router: respx.MockRouter) -> None:
    route = mock_router.post("/chat/completions").mock(
        side_effect=lambda req: httpx.Response(200, json=completion(json.loads(req.content)["model"]))
    )
    rows = {r["id"]: r for r in load_eval()}
    expected = {"s01": FREE, "i01": GEMINI, "c01": CLAUDE}
    for row_id, model in expected.items():
        payload = to_request(rows[row_id]).model_dump(exclude_none=True)
        resp = await client.post("/v1/chat/completions", json=payload, headers=AUTH)
        assert resp.status_code == 200
        assert resp.headers["X-Routed-Model"] == model
    assert [b["model"] for b in sent_bodies(route)] == list(expected.values())


# ----------------------------------------------------------------- context fit
def test_context_fit_escalates_past_simple_window(router: HeuristicRouter) -> None:
    registry = ModelRegistry.load()
    simple_window = registry.primary(Tier.SIMPLE).context_window
    # A trivially "simple" instruction whose payload exceeds the SIMPLE model's window.
    text = "Say hi. " + "lorem ipsum dolor " * (simple_window // 4 + 100)
    decision = router.classify(ChatCompletionRequest.model_validate(chat(text)))
    assert decision.tier is not Tier.SIMPLE
    assert registry.primary(decision.tier).context_window >= decision.features["est_tokens"]


def test_context_fit_considers_output_budget(router: HeuristicRouter) -> None:
    registry = ModelRegistry.load()
    window = registry.primary(Tier.SIMPLE).context_window
    req = ChatCompletionRequest.model_validate(chat("hi", max_tokens=window))
    assert router.classify(req).tier is not Tier.SIMPLE


async def test_oversized_prompt_never_sent_to_simple_model(
    client: httpx.AsyncClient, mock_router: respx.MockRouter
) -> None:
    route = mock_router.post("/chat/completions").mock(
        side_effect=lambda req: httpx.Response(200, json=completion(json.loads(req.content)["model"]))
    )
    registry = ModelRegistry.load()
    window = registry.primary(Tier.SIMPLE).context_window
    text = "hello " + "lorem ipsum dolor " * (window // 4 + 100)
    resp = await client.post("/v1/chat/completions", json=chat(text), headers=AUTH)
    assert resp.status_code == 200
    for body in sent_bodies(route):
        assert body["model"] != FREE
        assert FREE not in body.get("models", [])
    # The gpt-oss models (131k window) are also excluded from any native fallback list.
    assert all(registry.models[m].context_window >= window for m in [sent_bodies(route)[0]["model"]])


async def test_override_to_small_model_with_oversized_prompt_is_rejected(
    client: httpx.AsyncClient, mock_router: respx.MockRouter
) -> None:
    route = mock_router.post("/chat/completions")
    window = ModelRegistry.load().primary(Tier.SIMPLE).context_window
    text = "lorem ipsum dolor " * (window // 4 + 1000)
    resp = await client.post("/v1/chat/completions", json=chat(text), headers={**AUTH, "X-Model-Override": "gpt-oss"})
    assert resp.status_code == 413
    assert resp.json()["error"]["code"] == "context_length_exceeded"
    assert route.call_count == 0


# ----------------------------------------------------------------- overrides
@pytest.mark.parametrize(("alias", "model"), [("gpt-oss", FREE), ("gemini", GEMINI), ("claude", CLAUDE)])
async def test_valid_override_header_honoured(
    client: httpx.AsyncClient, mock_router: respx.MockRouter, tracer: FakeTracer, alias: str, model: str
) -> None:
    route = mock_router.post("/chat/completions").mock(return_value=httpx.Response(200, json=completion(model)))
    resp = await client.post("/v1/chat/completions", json=chat("hi"), headers={**AUTH, "X-Model-Override": alias})
    assert resp.status_code == 200
    assert sent_bodies(route)[0]["model"] == model
    assert tracer.last.override == alias
    assert f"override:{alias}" in tracer.last.route_reasons


async def test_valid_override_body_field(client: httpx.AsyncClient, mock_router: respx.MockRouter) -> None:
    route = mock_router.post("/chat/completions").mock(return_value=httpx.Response(200, json=completion(CLAUDE)))
    resp = await client.post("/v1/chat/completions", json=chat("hi", model_override="Claude"), headers=AUTH)
    assert resp.status_code == 200
    assert sent_bodies(route)[0]["model"] == CLAUDE


async def test_alias_in_model_field(client: httpx.AsyncClient, mock_router: respx.MockRouter) -> None:
    route = mock_router.post("/chat/completions").mock(return_value=httpx.Response(200, json=completion(GEMINI)))
    payload = {"model": "gemini", "messages": [{"role": "user", "content": "hi"}]}
    resp = await client.post("/v1/chat/completions", json=payload, headers=AUTH)
    assert resp.status_code == 200
    assert sent_bodies(route)[0]["model"] == GEMINI


async def test_invalid_override_alias_rejected(client: httpx.AsyncClient, mock_router: respx.MockRouter) -> None:
    route = mock_router.post("/chat/completions")
    resp = await client.post("/v1/chat/completions", json=chat("hi"), headers={**AUTH, "X-Model-Override": "gpt-5"})
    assert resp.status_code == 400
    assert resp.json()["error"]["code"] == "invalid_model_override"
    assert route.call_count == 0


@pytest.mark.parametrize("raw", ["anthropic/claude-opus-4", "openai/o3-pro", "google/gemini-3.1-pro-preview"])
async def test_arbitrary_model_strings_rejected(
    client: httpx.AsyncClient, mock_router: respx.MockRouter, raw: str
) -> None:
    route = mock_router.post("/chat/completions")
    r1 = await client.post("/v1/chat/completions", json=chat("hi", model_override=raw), headers=AUTH)
    r2 = await client.post(
        "/v1/chat/completions", json={"model": raw, "messages": [{"role": "user", "content": "hi"}]}, headers=AUTH
    )
    r3 = await client.post("/v1/chat/completions", json=chat("hi"), headers={**AUTH, "X-Model-Override": raw})
    assert r1.status_code == r2.status_code == r3.status_code == 400
    assert route.call_count == 0


# ----------------------------------------------------------------- cascade
def test_validity_checks() -> None:
    assert validity_problem("", False) == "empty"
    assert validity_problem("   ", False) == "empty"
    assert validity_problem(None, False) == "empty"
    assert validity_problem("I'm sorry, but I can't help with that.", False) == "refusal"
    assert validity_problem("{not json", True) == "malformed_json"
    assert validity_problem('```json\n{"a": 1}\n```', True) is None
    assert validity_problem('{"a": 1}', True) is None
    assert validity_problem("Sure, here you go.", False) is None


async def test_cascade_escalates_on_invalid_response(
    app_factory: AppFactory, mock_router: respx.MockRouter, tracer: FakeTracer
) -> None:
    app = app_factory(routing_mode="cascade")

    def handler(req: httpx.Request) -> httpx.Response:
        model = json.loads(req.content)["model"]
        content = "" if model == FREE else '{"answer": 42}'
        return httpx.Response(200, json=completion(model, content))

    route = mock_router.post("/chat/completions").mock(side_effect=handler)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://gw") as c:
        resp = await c.post(
            "/v1/chat/completions", json=chat("hi", response_format={"type": "json_object"}), headers=AUTH
        )
    assert resp.status_code == 200
    assert [b["model"] for b in sent_bodies(route)] == [FREE, GEMINI]
    assert resp.headers["X-Route-Tier"] == "INTERMEDIATE"
    assert [s["outcome"] for s in tracer.last.cascade_steps] == ["empty", "accepted"]


async def test_cascade_accepts_cheapest_valid(app_factory: AppFactory, mock_router: respx.MockRouter) -> None:
    app = app_factory()
    route = mock_router.post("/chat/completions").mock(return_value=httpx.Response(200, json=completion(FREE, "fine")))
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://gw") as c:
        resp = await c.post(
            "/v1/chat/completions",
            json=chat("Design a distributed system architecture", routing_mode="cascade"),
            headers=AUTH,
        )
    assert resp.status_code == 200
    assert [b["model"] for b in sent_bodies(route)] == [FREE]


async def test_cascade_off_by_default(client: httpx.AsyncClient, mock_router: respx.MockRouter) -> None:
    route = mock_router.post("/chat/completions").mock(return_value=httpx.Response(200, json=completion(FREE, "")))
    resp = await client.post("/v1/chat/completions", json=chat("hi"), headers=AUTH)
    assert resp.status_code == 200
    assert route.call_count == 1


async def test_cascade_escalates_past_upstream_failure(
    app_factory: AppFactory, mock_router: respx.MockRouter, tracer: FakeTracer
) -> None:
    app = app_factory(routing_mode="cascade", retry_max_attempts=1)

    def handler(req: httpx.Request) -> httpx.Response:
        model = json.loads(req.content)["model"]
        if model in {GEMINI, CLAUDE}:
            return httpx.Response(200, json=completion(model, "done"))
        return httpx.Response(503, json={"error": {"message": "down"}})

    mock_router.post("/chat/completions").mock(side_effect=handler)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://gw") as c:
        resp = await c.post("/v1/chat/completions", json=chat("hi"), headers=AUTH)
    assert resp.status_code == 200
    assert resp.headers["X-Route-Tier"] == "INTERMEDIATE"
    assert tracer.last.cascade_steps[0]["outcome"] == "upstream_error"


async def test_feature_log_written(app_factory: AppFactory, mock_router: respx.MockRouter, tmp_path: Path) -> None:
    app = app_factory()
    mock_router.post("/chat/completions").mock(return_value=httpx.Response(200, json=completion(FREE)))
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://gw") as c:
        await c.post("/v1/chat/completions", json=chat("secret-ish user text"), headers=AUTH)
    lines = (tmp_path / "features.jsonl").read_text().splitlines()
    record = json.loads(lines[-1])
    assert record["tier"] == "SIMPLE"
    assert "est_tokens" in record["features"]
    assert "secret-ish user text" not in lines[-1]  # features only, no raw prompt
