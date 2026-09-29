"""Gateway configuration (environment-driven) and the model registry."""

from __future__ import annotations

import json
from functools import cached_property
from pathlib import Path
from typing import Literal

from pydantic import AliasChoices, BaseModel, Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from app.models import TIER_ORDER, Tier

DEFAULT_REGISTRY_PATH = Path(__file__).with_name("model_registry.json")


class ModelSpec(BaseModel):
    id: str
    context_window: int = Field(gt=0)
    input_price_per_m: float = Field(ge=0.0)
    output_price_per_m: float = Field(ge=0.0)
    is_free: bool = False
    fallbacks: list[str] = Field(default_factory=list)
    reasoning_effort: str | None = None

    @model_validator(mode="after")
    def _free_models_cost_nothing(self) -> ModelSpec:
        if self.is_free and (self.input_price_per_m or self.output_price_per_m):
            raise ValueError(f"free model {self.id} must have zero prices")
        return self


class TierSpec(BaseModel):
    model: str
    degraded_models: list[str] = Field(default_factory=list)


class ModelRegistry(BaseModel):
    """Config-driven registry: routing logic never hard-codes model IDs."""

    models: dict[str, ModelSpec]
    tiers: dict[Tier, TierSpec]
    aliases: dict[str, Tier]

    @model_validator(mode="after")
    def _check_references(self) -> ModelRegistry:
        missing_tiers = [t for t in TIER_ORDER if t not in self.tiers]
        if missing_tiers:
            raise ValueError(f"registry missing tiers: {missing_tiers}")
        for tier, spec in self.tiers.items():
            if spec.model not in self.models:
                raise ValueError(f"tier {tier} references unknown model {spec.model}")
        for mid, m in self.models.items():
            if m.id != mid:
                raise ValueError(f"model key {mid} does not match id {m.id}")
            for fb in m.fallbacks:
                if fb not in self.models:
                    raise ValueError(f"model {mid} has unknown fallback {fb}")
        return self

    @classmethod
    def load(cls, path: Path | None = None) -> ModelRegistry:
        raw = json.loads((path or DEFAULT_REGISTRY_PATH).read_text())
        models = {mid: {"id": mid, **spec} for mid, spec in raw["models"].items()}
        return cls.model_validate({"models": models, "tiers": raw["tiers"], "aliases": raw["aliases"]})

    def primary(self, tier: Tier) -> ModelSpec:
        return self.models[self.tiers[tier].model]

    def chain(self, tier: Tier) -> list[ModelSpec]:
        """Primary model followed by its configured fallback chain."""
        primary = self.primary(tier)
        return [primary, *(self.models[f] for f in primary.fallbacks)]

    def all_ids(self) -> set[str]:
        return set(self.models)

    def get(self, model_id: str) -> ModelSpec | None:
        return self.models.get(model_id)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # --- OpenRouter ---
    openrouter_api_key: SecretStr | None = None
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    openrouter_http_referer: str | None = None
    openrouter_app_title: str | None = "multi-llm-router-gateway"
    connect_timeout_s: float = 5.0
    read_timeout_s: float = 60.0
    sdk_max_retries: int = Field(default=0, ge=0, le=1)
    validate_models_on_startup: bool = True
    sync_registry_from_openrouter: bool = True
    model_registry_path: Path | None = None

    # --- Free tier ---
    free_tier_daily_cap: int = Field(default=50, ge=0)
    free_tier_rpm_cap: int = Field(default=20, ge=0)
    free_tier_system_prompt: str = "You are a concise, helpful assistant."
    # When false (default), the gateway system prompt (which may contain internal
    # instructions) is NOT sent on the SIMPLE tier, because free providers may log
    # or train on inputs. Only the generic free_tier_system_prompt is used.
    free_tier_send_gateway_system_prompt: bool = False

    # --- Gateway-level resilience ---
    retry_max_attempts: int = Field(default=3, ge=1)
    retry_base_delay_s: float = 0.25
    retry_max_delay_s: float = 4.0
    retry_after_cap_s: float = 10.0
    native_fallback_enabled: bool = True
    circuit_failure_threshold: int = Field(default=5, ge=1)
    circuit_cooldown_s: float = 30.0

    # --- Break-glass direct providers (OpenAI-compatible endpoints) ---
    break_glass_enabled: bool = False
    anthropic_api_key: SecretStr | None = None
    break_glass_anthropic_base_url: str = "https://api.anthropic.com/v1/"
    break_glass_anthropic_model: str | None = None
    gemini_api_key: SecretStr | None = None
    break_glass_gemini_base_url: str = "https://generativelanguage.googleapis.com/v1beta/openai/"
    break_glass_gemini_model: str | None = None

    # --- Gateway API ---
    gateway_api_keys: SecretStr = SecretStr("")
    rate_limit_rpm_per_key: int = Field(default=60, ge=1)
    max_body_bytes: int = Field(default=1_000_000, ge=1024)
    gateway_system_prompt: str | None = None
    routing_mode: Literal["direct", "cascade"] = "direct"

    # --- Routing ---
    simple_max_tokens: int = 1500
    intermediate_token_threshold: int = 6000
    routing_features_log_path: Path | None = Path("logs/routing_features.jsonl")

    # --- Guardrails ---
    guardrail_max_base64_chars: int = 2048
    guardrail_classifier_enabled: bool = False
    guardrail_classifier_model: str = "meta-llama/Llama-Prompt-Guard-2-86M"
    guardrail_classifier_threshold: float = 0.9
    output_guard_enabled: bool = True

    # --- Observability ---
    enable_tracing: bool = True
    langfuse_public_key: str | None = None
    langfuse_secret_key: SecretStr | None = None
    langfuse_base_url: str | None = Field(
        default=None, validation_alias=AliasChoices("LANGFUSE_BASE_URL", "LANGFUSE_HOST")
    )
    trace_raw_content: bool = False
    trace_hash_salt: SecretStr = SecretStr("")
    log_level: str = "INFO"
    log_json: bool = False

    # --- Semantic cache ---
    enable_semantic_cache: bool = False
    semantic_cache_threshold: float = Field(default=0.95, gt=0.0, le=1.0)
    semantic_cache_ttl_s: float = 600.0
    semantic_cache_max_entries: int = 1000

    @model_validator(mode="before")
    @classmethod
    def _drop_comment_values(cls, data: object) -> object:
        """Treat values that start with '#' as unset.

        python-dotenv reads `KEY=   # note` as the literal value "# note"; without this
        a commented-out GATEWAY_API_KEYS line would become a valid gateway key.
        """
        if isinstance(data, dict):
            return {k: v for k, v in data.items() if not (isinstance(v, str) and v.strip().startswith("#"))}
        return data

    @field_validator("openrouter_api_key", "anthropic_api_key", "gemini_api_key", "langfuse_secret_key", mode="before")
    @classmethod
    def _blank_is_none(cls, v: object) -> object:
        return None if v == "" else v

    @cached_property
    def gateway_keys(self) -> frozenset[str]:
        raw = self.gateway_api_keys.get_secret_value()
        return frozenset(k.strip() for k in raw.split(",") if k.strip() and not k.strip().startswith("#"))

    @cached_property
    def registry(self) -> ModelRegistry:
        return ModelRegistry.load(self.model_registry_path)

    def secret_values(self) -> list[str]:
        """All configured secret strings, for log redaction and output leak checks."""
        secrets: list[str] = list(self.gateway_keys)
        for s in (self.openrouter_api_key, self.anthropic_api_key, self.gemini_api_key, self.langfuse_secret_key):
            if s is not None and s.get_secret_value():
                secrets.append(s.get_secret_value())
        return [s for s in secrets if len(s) >= 8]
