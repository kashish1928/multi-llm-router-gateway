"""Request/response schemas and core domain types."""

from __future__ import annotations

from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class Tier(StrEnum):
    SIMPLE = "SIMPLE"
    INTERMEDIATE = "INTERMEDIATE"
    COMPLEX = "COMPLEX"

    @property
    def rank(self) -> int:
        return _TIER_ORDER.index(self)

    def next_up(self) -> Tier | None:
        idx = self.rank + 1
        return _TIER_ORDER[idx] if idx < len(_TIER_ORDER) else None


_TIER_ORDER: list[Tier] = [Tier.SIMPLE, Tier.INTERMEDIATE, Tier.COMPLEX]
TIER_ORDER: tuple[Tier, ...] = tuple(_TIER_ORDER)


class ChatMessage(BaseModel):
    model_config = ConfigDict(extra="allow")

    role: Literal["system", "developer", "user", "assistant", "tool"]
    content: str | list[dict[str, Any]] | None = None
    name: str | None = None
    tool_call_id: str | None = None
    tool_calls: list[dict[str, Any]] | None = None

    def text(self) -> str:
        """Plain text of the message (text parts only for multi-part content)."""
        if self.content is None:
            return ""
        if isinstance(self.content, str):
            return self.content
        parts: list[str] = []
        for part in self.content:
            if part.get("type") == "text" and isinstance(part.get("text"), str):
                parts.append(part["text"])
        return "\n".join(parts)


# OpenAI-compatible parameters the gateway forwards upstream. Anything else a client
# sends (e.g. OpenRouter's `models`, `route`, `provider`) is dropped so clients cannot
# bypass the routing allowlist.
FORWARDED_PARAMS: tuple[str, ...] = (
    "temperature",
    "top_p",
    "max_tokens",
    "max_completion_tokens",
    "stop",
    "presence_penalty",
    "frequency_penalty",
    "seed",
    "response_format",
    "tools",
    "tool_choice",
    "parallel_tool_calls",
    "logit_bias",
    "logprobs",
    "top_logprobs",
    "n",
)


class ChatCompletionRequest(BaseModel):
    """OpenAI-compatible chat completion request with gateway extensions."""

    model_config = ConfigDict(extra="allow")

    model: str | None = "auto"
    messages: list[ChatMessage] = Field(min_length=1)
    stream: bool = False
    temperature: float | None = None
    top_p: float | None = None
    max_tokens: int | None = Field(default=None, ge=1)
    max_completion_tokens: int | None = Field(default=None, ge=1)
    stop: str | list[str] | None = None
    presence_penalty: float | None = None
    frequency_penalty: float | None = None
    seed: int | None = None
    response_format: dict[str, Any] | None = None
    tools: list[dict[str, Any]] | None = None
    tool_choice: str | dict[str, Any] | None = None
    parallel_tool_calls: bool | None = None
    logit_bias: dict[str, float] | None = None
    logprobs: bool | None = None
    top_logprobs: int | None = None
    n: int | None = Field(default=None, ge=1, le=8)

    # Gateway extensions
    model_override: str | None = None
    routing_mode: Literal["direct", "cascade"] | None = None
    cache: bool | None = None

    def forwarded_params(self) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for key in FORWARDED_PARAMS:
            value = getattr(self, key)
            if value is not None:
                out[key] = value
        return out

    def output_token_budget(self) -> int:
        return self.max_completion_tokens or self.max_tokens or 0

    def wants_json(self) -> bool:
        if not self.response_format:
            return False
        return self.response_format.get("type") in {"json_object", "json_schema"}


class RouteDecision(BaseModel):
    tier: Tier
    confidence: float = Field(ge=0.0, le=1.0)
    reasons: list[str] = Field(default_factory=list)
    features: dict[str, Any] = Field(default_factory=dict)
    override: str | None = None
