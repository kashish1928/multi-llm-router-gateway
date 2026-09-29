"""OpenRouter (OpenAI-compatible) client construction."""

from __future__ import annotations

import httpx
import httpx2  # openai>=3 transport package; its Timeout type is what the SDK expects
from openai import AsyncOpenAI

from app.config import Settings


def build_http_client(settings: Settings) -> httpx.AsyncClient:
    # openai>=3 accepts a legacy `httpx.AsyncClient` as http_client; using it keeps
    # explicit connect/read timeouts under our control and lets respx mock traffic.
    timeout = httpx.Timeout(settings.read_timeout_s, connect=settings.connect_timeout_s)
    return httpx.AsyncClient(timeout=timeout)


def _sdk_timeout(settings: Settings) -> httpx2.Timeout:
    return httpx2.Timeout(settings.read_timeout_s, connect=settings.connect_timeout_s)


def attribution_headers(settings: Settings) -> dict[str, str]:
    headers: dict[str, str] = {}
    if settings.openrouter_http_referer:
        headers["HTTP-Referer"] = settings.openrouter_http_referer
    if settings.openrouter_app_title:
        headers["X-Title"] = settings.openrouter_app_title
    return headers


def build_openrouter_client(settings: Settings, http_client: httpx.AsyncClient) -> AsyncOpenAI:
    api_key = settings.openrouter_api_key.get_secret_value() if settings.openrouter_api_key else "missing"
    return AsyncOpenAI(
        api_key=api_key,
        base_url=settings.openrouter_base_url,
        # SDK retries are disabled (or at most 1) so they never stack with the
        # gateway's own retry + fallback walk.
        max_retries=settings.sdk_max_retries,
        timeout=_sdk_timeout(settings),
        default_headers=attribution_headers(settings),
        http_client=http_client,  # type: ignore[arg-type]
    )


def build_break_glass_clients(settings: Settings, http_client: httpx.AsyncClient) -> dict[str, tuple[AsyncOpenAI, str]]:
    """Optional direct-provider clients, used only when the OpenRouter chain is exhausted.

    Returns {provider_name: (client, model_id)}. Empty unless BREAK_GLASS_ENABLED and
    both an API key and a model ID are configured for the provider.
    """
    out: dict[str, tuple[AsyncOpenAI, str]] = {}
    if not settings.break_glass_enabled:
        return out
    candidates = [
        (
            "anthropic",
            settings.anthropic_api_key,
            settings.break_glass_anthropic_base_url,
            settings.break_glass_anthropic_model,
        ),
        ("gemini", settings.gemini_api_key, settings.break_glass_gemini_base_url, settings.break_glass_gemini_model),
    ]
    for name, key, base_url, model in candidates:
        if key is None or not model:
            continue
        client = AsyncOpenAI(
            api_key=key.get_secret_value(),
            base_url=base_url,
            max_retries=settings.sdk_max_retries,
            timeout=_sdk_timeout(settings),
            http_client=http_client,  # type: ignore[arg-type]
        )
        out[name] = (client, model)
    return out
