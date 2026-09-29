"""Manual override resolution. Only allowlisted aliases are accepted."""

from __future__ import annotations

from app.config import ModelRegistry
from app.models import Tier

AUTO_MODEL_VALUES = frozenset({"", "auto", "router", "gateway/auto"})


class InvalidOverride(ValueError):
    pass


def resolve_override(
    registry: ModelRegistry,
    header_value: str | None,
    body_override: str | None,
    model_field: str | None,
) -> tuple[Tier | None, str | None]:
    """Return (tier, alias) for an override, or (None, None) for automatic routing.

    Precedence: X-Model-Override header > body `model_override` > body `model`.
    `model` accepts "auto" (route automatically) or an allowlisted alias; any other
    string (e.g. a raw provider model ID) is rejected so clients cannot bypass the
    allowlist.
    """
    for value in (header_value, body_override):
        if value is not None:
            alias = value.strip().lower()
            if alias not in registry.aliases:
                raise InvalidOverride(f"unknown model override alias; allowed: {sorted(registry.aliases)}")
            return registry.aliases[alias], alias
    model = (model_field or "").strip().lower()
    if model in AUTO_MODEL_VALUES:
        return None, None
    if model in registry.aliases:
        return registry.aliases[model], model
    raise InvalidOverride(f"unsupported model; use 'auto' or one of {sorted(registry.aliases)}")
