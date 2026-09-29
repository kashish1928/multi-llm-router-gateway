"""Cost estimation from registry prices (USD per 1M tokens)."""

from __future__ import annotations

from app.config import ModelRegistry


def estimate_cost_usd(
    registry: ModelRegistry, model_id: str, prompt_tokens: int, completion_tokens: int
) -> float | None:
    """Estimated cost, 0.0 for free models, None when the model is not in the registry."""
    spec = registry.get(model_id)
    if spec is None:
        return None
    if spec.is_free:
        return 0.0
    cost = (prompt_tokens * spec.input_price_per_m + completion_tokens * spec.output_price_per_m) / 1_000_000
    return round(cost, 8)
