"""Dispatch: correct URL, model, extra_body.models chain, reasoning and headers per tier."""

from __future__ import annotations

import httpx
import respx

from tests.conftest import AUTH, CLAUDE, FREE, GEMINI, LLAMA, PAID_OSS, chat, completion, sent_bodies

ARCH_PROMPT = "Design a scalable distributed system architecture for a multi-region payments platform."
LARGE_PROMPT = "Check this document for inconsistencies:\n" + ("The quarterly revenue grew in the east region. " * 1200)


async def test_simple_tier_dispatch(client: httpx.AsyncClient, mock_router: respx.MockRouter) -> None:
    route = mock_router.post("/chat/completions").mock(return_value=httpx.Response(200, json=completion(FREE)))
    resp = await client.post("/v1/chat/completions", json=chat("Translate 'hello' to French"), headers=AUTH)

    assert resp.status_code == 200
    assert route.calls.last.request.url == "https://openrouter.ai/api/v1/chat/completions"
    body = sent_bodies(route)[0]
    assert body["model"] == FREE
    assert body["models"] == [PAID_OSS, LLAMA]
    assert body["reasoning"] == {"effort": "low"}
    assert resp.headers["X-Route-Tier"] == "SIMPLE"
    assert resp.headers["X-Routed-Model"] == FREE
    assert resp.headers["X-Fallback-Used"] == "false"
    assert resp.headers["X-Free-Tier"] == "true"
    assert resp.headers["X-Request-ID"]
    assert "X-Degraded" not in resp.headers
    assert resp.json()["model"] == FREE


async def test_intermediate_tier_dispatch(client: httpx.AsyncClient, mock_router: respx.MockRouter) -> None:
    route = mock_router.post("/chat/completions").mock(return_value=httpx.Response(200, json=completion(GEMINI)))
    resp = await client.post("/v1/chat/completions", json=chat(LARGE_PROMPT), headers=AUTH)

    assert resp.status_code == 200
    body = sent_bodies(route)[0]
    assert body["model"] == GEMINI
    assert body["models"] == [PAID_OSS]
    assert "reasoning" not in body
    assert resp.headers["X-Route-Tier"] == "INTERMEDIATE"
    assert resp.headers["X-Free-Tier"] == "false"


async def test_complex_tier_dispatch(client: httpx.AsyncClient, mock_router: respx.MockRouter) -> None:
    route = mock_router.post("/chat/completions").mock(return_value=httpx.Response(200, json=completion(CLAUDE)))
    resp = await client.post("/v1/chat/completions", json=chat(ARCH_PROMPT), headers=AUTH)

    assert resp.status_code == 200
    body = sent_bodies(route)[0]
    assert body["model"] == CLAUDE
    assert body["models"] == [GEMINI, PAID_OSS]
    assert resp.headers["X-Route-Tier"] == "COMPLEX"
    assert resp.headers["X-Routed-Model"] == CLAUDE


async def test_attribution_headers_and_auth_sent(client: httpx.AsyncClient, mock_router: respx.MockRouter) -> None:
    route = mock_router.post("/chat/completions").mock(return_value=httpx.Response(200, json=completion(FREE)))
    await client.post("/v1/chat/completions", json=chat("hi"), headers=AUTH)
    sent = route.calls.last.request
    assert sent.headers["X-Title"] == "multi-llm-router-gateway"
    assert sent.headers["Authorization"] == "Bearer sk-or-v1-testkeytestkeytestkey"


async def test_system_prompt_is_separate_message_and_generic_on_free_tier(
    client: httpx.AsyncClient, mock_router: respx.MockRouter
) -> None:
    route = mock_router.post("/chat/completions").mock(return_value=httpx.Response(200, json=completion(FREE)))
    await client.post("/v1/chat/completions", json=chat("hi there"), headers=AUTH)
    messages = sent_bodies(route)[0]["messages"]
    assert messages[0]["role"] == "system"
    assert "ACME internal" not in messages[0]["content"]  # gateway prompt withheld from free tier
    assert messages[1] == {"role": "user", "content": "hi there"}


async def test_paid_tier_gets_gateway_system_prompt_structurally(
    client: httpx.AsyncClient, mock_router: respx.MockRouter
) -> None:
    route = mock_router.post("/chat/completions").mock(return_value=httpx.Response(200, json=completion(CLAUDE)))
    await client.post("/v1/chat/completions", json=chat(ARCH_PROMPT), headers=AUTH)
    messages = sent_bodies(route)[0]["messages"]
    assert messages[0]["role"] == "system" and "ACME internal" in messages[0]["content"]
    assert ARCH_PROMPT not in messages[0]["content"]
    assert messages[-1] == {"role": "user", "content": ARCH_PROMPT}


async def test_client_cannot_inject_openrouter_models_param(
    client: httpx.AsyncClient, mock_router: respx.MockRouter
) -> None:
    route = mock_router.post("/chat/completions").mock(return_value=httpx.Response(200, json=completion(FREE)))
    payload = chat("hi", models=["openai/o3-pro"], provider={"order": ["x"]}, temperature=0.2)
    resp = await client.post("/v1/chat/completions", json=payload, headers=AUTH)
    assert resp.status_code == 200
    body = sent_bodies(route)[0]
    assert "openai/o3-pro" not in body["models"]
    assert "provider" not in body
    assert body["temperature"] == 0.2


async def test_complex_degraded_to_gpt_oss_sets_header(
    client: httpx.AsyncClient, mock_router: respx.MockRouter
) -> None:
    # OpenRouter's native fallback served the last model in the COMPLEX chain.
    mock_router.post("/chat/completions").mock(return_value=httpx.Response(200, json=completion(PAID_OSS)))
    resp = await client.post("/v1/chat/completions", json=chat(ARCH_PROMPT), headers=AUTH)
    assert resp.headers["X-Route-Tier"] == "COMPLEX"
    assert resp.headers["X-Routed-Model"] == PAID_OSS
    assert resp.headers["X-Fallback-Used"] == "true"
    assert resp.headers["X-Degraded"] == "true"
