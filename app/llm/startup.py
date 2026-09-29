"""Startup checks against OpenRouter: model availability and key tier."""

from __future__ import annotations

from typing import Any

import httpx
from loguru import logger

from app.config import ModelRegistry, Settings


class ModelValidationError(RuntimeError):
    pass


def _auth_headers(settings: Settings) -> dict[str, str]:
    assert settings.openrouter_api_key is not None
    return {"Authorization": f"Bearer {settings.openrouter_api_key.get_secret_value()}"}


def _per_token_to_per_m(value: Any) -> float | None:
    try:
        return float(value) * 1_000_000
    except (TypeError, ValueError):
        return None


async def validate_models(settings: Settings, registry: ModelRegistry, http: httpx.AsyncClient) -> None:
    """GET /models and fail fast if any configured model ID is missing.

    When SYNC_REGISTRY_FROM_OPENROUTER is set, context windows and prices are refreshed
    from the live listing (`context_length`, `pricing.prompt`/`pricing.completion`,
    which OpenRouter reports in USD per token).
    """
    resp = await http.get(f"{settings.openrouter_base_url}/models", headers=_auth_headers(settings))
    resp.raise_for_status()
    listing = {m["id"]: m for m in resp.json().get("data", []) if isinstance(m, dict) and "id" in m}
    missing = sorted(registry.all_ids() - set(listing))
    if missing:
        raise ModelValidationError(
            "Configured model IDs not available on OpenRouter: "
            + ", ".join(missing)
            + ". Update app/model_registry.json (or MODEL_REGISTRY_PATH) with current IDs."
        )
    if not settings.sync_registry_from_openrouter:
        return
    for mid, spec in registry.models.items():
        live = listing[mid]
        if isinstance(live.get("context_length"), int) and live["context_length"] > 0:
            spec.context_window = live["context_length"]
        pricing = live.get("pricing") or {}
        if not spec.is_free:
            prompt = _per_token_to_per_m(pricing.get("prompt"))
            completion = _per_token_to_per_m(pricing.get("completion"))
            if prompt is not None:
                spec.input_price_per_m = prompt
            if completion is not None:
                spec.output_price_per_m = completion
    logger.info("model registry validated and synced ({} models)", len(registry.models))


async def check_key_tier(settings: Settings, http: httpx.AsyncClient) -> dict[str, Any]:
    """GET /key and log whether this key is free-tier so cap defaults match reality."""
    resp = await http.get(f"{settings.openrouter_base_url}/key", headers=_auth_headers(settings))
    resp.raise_for_status()
    data: dict[str, Any] = resp.json().get("data", {})
    is_free_tier = data.get("is_free_tier")
    logger.info("openrouter key is_free_tier={}", is_free_tier)
    free_reqs = data.get("free_model_daily_requests")
    if isinstance(free_reqs, dict) and isinstance(free_reqs.get("limit"), int):
        if free_reqs["limit"] != settings.free_tier_daily_cap:
            logger.warning(
                "FREE_TIER_DAILY_CAP={} differs from OpenRouter-reported free daily limit {}",
                settings.free_tier_daily_cap,
                free_reqs["limit"],
            )
    elif is_free_tier is False and settings.free_tier_daily_cap <= 50:
        logger.warning("key has purchased credits; consider raising FREE_TIER_DAILY_CAP to 1000")
    elif is_free_tier is True and settings.free_tier_daily_cap > 50:
        logger.warning("key is free-tier (50 free req/day) but FREE_TIER_DAILY_CAP={}", settings.free_tier_daily_cap)
    return data
