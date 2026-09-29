"""Per-request trace records and Langfuse (SDK v4) export. Tracing always fails open."""

from __future__ import annotations

import hashlib
import hmac
from dataclasses import dataclass, field
from typing import Any, Protocol

from loguru import logger

from app.config import Settings


def hash_identifier(value: str, salt: str) -> str:
    """Stable, non-reversible identifier for a key/user (HMAC-SHA256, 16 hex chars)."""
    return hmac.new(salt.encode(), value.encode(), hashlib.sha256).hexdigest()[:16]


@dataclass
class TraceRecord:
    request_id: str
    key_hash: str | None = None
    user_hash: str | None = None
    prompt_chars: int = 0
    prompt_token_estimate: int = 0
    message_count: int = 0
    stream: bool = False
    raw_input: Any = None  # only populated when TRACE_RAW_CONTENT=true
    raw_output: Any = None
    guardrail: dict[str, Any] = field(default_factory=dict)
    output_guard: dict[str, Any] = field(default_factory=dict)
    tier: str | None = None
    route_confidence: float | None = None
    route_reasons: list[str] = field(default_factory=list)
    route_features: dict[str, Any] = field(default_factory=dict)
    routing_mode: str = "direct"
    override: str | None = None
    cascade_steps: list[dict[str, Any]] = field(default_factory=list)
    cache: str = "disabled"  # disabled | bypass | miss | hit
    requested_model: str | None = None
    served_model: str | None = None
    fallback_used: bool = False
    degraded: bool = False
    free_tier: bool = False
    break_glass_provider: str | None = None
    attempts: list[dict[str, Any]] = field(default_factory=list)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cost_usd: float | None = None
    latency_ms: float = 0.0
    status_code: int = 200
    error_code: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


class Tracer(Protocol):
    def record(self, trace: TraceRecord) -> None: ...

    def flush(self) -> None: ...

    def shutdown(self) -> None: ...


class NoopTracer:
    def record(self, trace: TraceRecord) -> None:
        return None

    def flush(self) -> None:
        return None

    def shutdown(self) -> None:
        return None


class SafeTracer:
    """Wraps any tracer so exceptions never propagate into request handling."""

    def __init__(self, inner: Tracer) -> None:
        self.inner = inner

    def record(self, trace: TraceRecord) -> None:
        try:
            self.inner.record(trace)
        except Exception as exc:
            logger.warning("tracing failed (fail-open): {}", type(exc).__name__)

    def flush(self) -> None:
        try:
            self.inner.flush()
        except Exception as exc:
            logger.warning("tracing flush failed: {}", type(exc).__name__)

    def shutdown(self) -> None:
        try:
            self.inner.shutdown()
        except Exception as exc:
            logger.warning("tracing shutdown failed: {}", type(exc).__name__)


class LangfuseTracer:
    """Emits one Langfuse trace per request.

    Uses the v4 SDK API: `Langfuse(...)`, `propagate_attributes(...)` for trace-level
    attributes, and nested `start_as_current_observation(as_type=...)` observations.
    The record is emitted after the request completes, so child observation timings
    reflect emission time; true per-attempt latencies are in each observation's
    metadata (`latency_ms`).
    """

    def __init__(self, settings: Settings) -> None:
        from langfuse import Langfuse

        assert settings.langfuse_secret_key is not None
        self._client = Langfuse(
            public_key=settings.langfuse_public_key,
            secret_key=settings.langfuse_secret_key.get_secret_value(),
            base_url=settings.langfuse_base_url,
        )

    def record(self, trace: TraceRecord) -> None:
        from langfuse import propagate_attributes

        tags = [
            t
            for t in (
                trace.tier,
                "free-tier" if trace.free_tier else None,
                "fallback" if trace.fallback_used else None,
                trace.cache,
            )
            if t
        ]
        with (
            propagate_attributes(
                trace_name="chat.completions",
                user_id=trace.key_hash,
                tags=tags,
                metadata={
                    "request_id": trace.request_id,
                    "tier": trace.tier or "",
                    "served_model": trace.served_model or "",
                },
            ),
            self._client.start_as_current_observation(
                as_type="span",
                name="gateway.request",
                input=trace.raw_input
                if trace.raw_input is not None
                else {"prompt_chars": trace.prompt_chars, "prompt_token_estimate": trace.prompt_token_estimate},
                output=trace.raw_output,
                metadata={k: v for k, v in trace.as_dict().items() if k not in {"raw_input", "raw_output", "attempts"}},
                level="ERROR" if trace.status_code >= 500 else "WARNING" if trace.status_code >= 400 else "DEFAULT",
            ),
        ):
            with self._client.start_as_current_observation(
                as_type="guardrail", name="guardrail.input", metadata=trace.guardrail
            ):
                pass
            with self._client.start_as_current_observation(
                as_type="span",
                name="route",
                metadata={
                    "tier": trace.tier,
                    "confidence": trace.route_confidence,
                    "reasons": trace.route_reasons,
                    "features": trace.route_features,
                    "override": trace.override,
                    "routing_mode": trace.routing_mode,
                    "cascade_steps": trace.cascade_steps,
                    "cache": trace.cache,
                },
            ):
                pass
            for i, attempt in enumerate(trace.attempts):
                ok = attempt.get("outcome") == "ok"
                with self._client.start_as_current_observation(
                    as_type="generation",
                    name=f"upstream.attempt.{i}",
                    model=attempt.get("served_model") or attempt.get("model"),
                    metadata=attempt,
                    level="DEFAULT" if ok else "WARNING",
                    usage_details={"input": trace.prompt_tokens, "output": trace.completion_tokens} if ok else None,
                    cost_details={"total": trace.cost_usd} if ok and trace.cost_usd is not None else None,
                ):
                    pass

    def flush(self) -> None:
        self._client.flush()

    def shutdown(self) -> None:
        self._client.shutdown()


def build_tracer(settings: Settings) -> Tracer:
    if not settings.enable_tracing:
        return NoopTracer()
    if not (settings.langfuse_public_key and settings.langfuse_secret_key):
        logger.info("tracing enabled but Langfuse keys not configured; using no-op tracer")
        return NoopTracer()
    try:
        return SafeTracer(LangfuseTracer(settings))
    except Exception as exc:
        logger.warning("Langfuse init failed ({}); tracing disabled", type(exc).__name__)
        return NoopTracer()
