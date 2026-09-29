"""Tier dispatch: OpenRouter-native fallback + gateway-level retry/fallback walk.

Layer (a): each upstream call carries the remaining chain in `extra_body.models`, so
OpenRouter itself fails over between models within a single HTTP request.

Layer (b): if OpenRouter itself returns 429/5xx/times out, the gateway retries the
same hop with jittered exponential backoff (honouring Retry-After), then walks to the
next model in the chain. Non-429 4xx errors are never retried or walked. A per-model
circuit breaker skips models that keep failing, and free models are skipped up front
when the local free-tier counters are exhausted.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

from loguru import logger
from openai import AsyncOpenAI
from openai.types.chat import ChatCompletion, ChatCompletionChunk
from tenacity import AsyncRetrying, retry_if_exception, stop_after_attempt

from app.config import ModelRegistry, ModelSpec, Settings
from app.llm.circuit_breaker import CircuitBreakerRegistry
from app.llm.errors import EmptyStreamError, ErrorInfo, RetryAfterWait, classify
from app.llm.free_tier import FreeTierLimiter
from app.models import Tier

Sleep = Callable[[float], Awaitable[None]]

BREAK_GLASS_ORDER: dict[Tier, list[str]] = {
    Tier.SIMPLE: [],
    Tier.INTERMEDIATE: ["gemini", "anthropic"],
    Tier.COMPLEX: ["anthropic", "gemini"],
}


@dataclass
class Attempt:
    model: str
    native_fallbacks: list[str]
    outcome: str  # ok | error | skipped_circuit_open | skipped_free_cap | skipped_context
    status: int | None = None
    latency_ms: float = 0.0
    error_kind: str | None = None
    served_model: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "native_fallbacks": self.native_fallbacks,
            "outcome": self.outcome,
            "status": self.status,
            "latency_ms": round(self.latency_ms, 2),
            "error_kind": self.error_kind,
            "served_model": self.served_model,
        }


@dataclass
class DispatchResult:
    requested_model: str
    served_model: str
    attempts: list[Attempt]
    completion: ChatCompletion | None = None
    first_chunk: ChatCompletionChunk | None = None
    stream: AsyncIterator[ChatCompletionChunk] | None = None
    break_glass_provider: str | None = None

    @property
    def fallback_used(self) -> bool:
        return self.served_model != self.requested_model


class UpstreamError(Exception):
    def __init__(self, status: int, code: str, message: str, attempts: list[Attempt]) -> None:
        super().__init__(message)
        self.status = status
        self.code = code
        self.attempts = attempts


@dataclass
class DispatchRequest:
    tier: Tier
    messages: list[dict[str, Any]]
    params: dict[str, Any] = field(default_factory=dict)
    stream: bool = False
    prompt_tokens_estimate: int = 0
    output_token_budget: int = 0


class Dispatcher:
    def __init__(
        self,
        settings: Settings,
        registry: ModelRegistry,
        client: AsyncOpenAI,
        free_limiter: FreeTierLimiter,
        breakers: CircuitBreakerRegistry,
        sleep: Sleep = asyncio.sleep,
        break_glass: dict[str, tuple[AsyncOpenAI, str]] | None = None,
    ) -> None:
        self.settings = settings
        self.registry = registry
        self.client = client
        self.free_limiter = free_limiter
        self.breakers = breakers
        self._sleep = sleep
        self._wait = RetryAfterWait(settings.retry_base_delay_s, settings.retry_max_delay_s)
        self.break_glass = break_glass or {}

    # ------------------------------------------------------------------ helpers
    def _fits(self, spec: ModelSpec, req: DispatchRequest) -> bool:
        return spec.context_window >= req.prompt_tokens_estimate + req.output_token_budget

    def candidates(self, req: DispatchRequest) -> tuple[list[ModelSpec], list[Attempt]]:
        """Tier chain filtered by context fit (never send a request that exceeds a window)."""
        fitting: list[ModelSpec] = []
        skipped: list[Attempt] = []
        for spec in self.registry.chain(req.tier):
            if self._fits(spec, req):
                fitting.append(spec)
            else:
                skipped.append(Attempt(spec.id, [], "skipped_context"))
        return fitting, skipped

    def _native_list(self, remaining: list[ModelSpec]) -> list[str]:
        if not self.settings.native_fallback_enabled:
            return []
        out: list[str] = []
        free_ok = self.free_limiter.available()
        for spec in remaining:
            if self.breakers.is_open(spec.id):
                continue
            if spec.is_free and not free_ok:
                continue
            out.append(spec.id)
        return out

    def _build_kwargs(self, spec: ModelSpec, native: list[str], req: DispatchRequest) -> dict[str, Any]:
        extra_body: dict[str, Any] = {}
        if native:
            extra_body["models"] = native
        if spec.reasoning_effort:
            extra_body["reasoning"] = {"effort": spec.reasoning_effort}
        kwargs: dict[str, Any] = {"model": spec.id, "messages": req.messages, **req.params}
        if extra_body:
            kwargs["extra_body"] = extra_body
        if req.stream:
            kwargs["stream"] = True
            kwargs["stream_options"] = {"include_usage": True}
        return kwargs

    def _should_retry(self, spec: ModelSpec) -> Callable[[BaseException], bool]:
        def predicate(exc: BaseException) -> bool:
            info = classify(exc)
            if not info.retryable:
                return False
            # Free-model 429s are not retried: OpenRouter counts failed attempts
            # against the daily free quota, so walk to the paid fallback instead.
            if spec.is_free and info.kind == "rate_limited":
                return False
            # A Retry-After longer than the cap means "walk the chain now" rather than wait.
            return info.retry_after_s is None or info.retry_after_s <= self.settings.retry_after_cap_s

        return predicate

    # ------------------------------------------------------------------ calls
    async def _call_once(
        self, client: AsyncOpenAI, kwargs: dict[str, Any], stream: bool
    ) -> tuple[ChatCompletion | None, ChatCompletionChunk | None, AsyncIterator[ChatCompletionChunk] | None]:
        if not stream:
            completion = await client.chat.completions.create(**kwargs)
            return completion, None, None
        stream_obj = await client.chat.completions.create(**kwargs)
        iterator: AsyncIterator[ChatCompletionChunk] = stream_obj.__aiter__()
        # Fallback only applies before the first token: pull the first chunk inside
        # the retry scope so connect-time and first-byte failures are still retried.
        try:
            first = await iterator.__anext__()
        except StopAsyncIteration:
            # Must not escape as StopAsyncIteration: inside tenacity's `async for` it
            # would silently end the retry loop instead of surfacing as an error.
            raise EmptyStreamError from None
        return None, first, iterator

    async def _try_model(
        self, spec: ModelSpec, native: list[str], req: DispatchRequest, attempts: list[Attempt]
    ) -> DispatchResult | None:
        kwargs = self._build_kwargs(spec, native, req)
        last_error: ErrorInfo | None = None
        retrying = AsyncRetrying(
            stop=stop_after_attempt(self.settings.retry_max_attempts),
            wait=self._wait,
            retry=retry_if_exception(self._should_retry(spec)),
            sleep=self._sleep,
            reraise=True,
        )
        try:
            async for attempt_ctx in retrying:
                with attempt_ctx:
                    if spec.is_free and not self.free_limiter.try_acquire():
                        attempts.append(Attempt(spec.id, native, "skipped_free_cap"))
                        return None
                    if attempt_ctx.retry_state.attempt_number > 1 and not self.breakers.allow(spec.id):
                        attempts.append(Attempt(spec.id, native, "skipped_circuit_open"))
                        return None
                    started = time.perf_counter()
                    try:
                        completion, first, iterator = await self._call_once(self.client, kwargs, req.stream)
                    except Exception as exc:
                        info = classify(exc)
                        last_error = info
                        attempts.append(
                            Attempt(
                                spec.id,
                                native,
                                "error",
                                status=info.status,
                                latency_ms=(time.perf_counter() - started) * 1000,
                                error_kind=info.kind,
                            )
                        )
                        if info.retryable:
                            self.breakers.record_failure(spec.id)
                        logger.warning(
                            "upstream attempt failed model={} kind={} status={}", spec.id, info.kind, info.status
                        )
                        raise
                    served = (completion.model if completion else first.model if first else None) or spec.id
                    attempts.append(
                        Attempt(
                            spec.id,
                            native,
                            "ok",
                            status=200,
                            latency_ms=(time.perf_counter() - started) * 1000,
                            served_model=served,
                        )
                    )
                    self.breakers.record_success(spec.id)
                    served_spec = self.registry.get(served)
                    if served != spec.id and served_spec is not None and served_spec.is_free:
                        self.free_limiter.record_external()
                    return DispatchResult(
                        requested_model="",
                        served_model=served,
                        attempts=attempts,
                        completion=completion,
                        first_chunk=first,
                        stream=iterator,
                    )
        except Exception as exc:
            info = last_error or classify(exc)
            if info.kind == "empty_stream":
                raise UpstreamError(502, "upstream_empty_stream", info.message, attempts) from None
            if not info.retryable:
                status = 400 if info.status == 400 else 502
                raise UpstreamError(status, "upstream_error", info.message, attempts) from exc
            return None
        return None  # pragma: no cover - AsyncRetrying always yields at least once

    async def _break_glass(self, req: DispatchRequest, attempts: list[Attempt]) -> DispatchResult | None:
        for name in BREAK_GLASS_ORDER[req.tier]:
            if name not in self.break_glass:
                continue
            client, model = self.break_glass[name]
            kwargs: dict[str, Any] = {"model": model, "messages": req.messages, **req.params}
            if req.stream:
                kwargs["stream"] = True
            started = time.perf_counter()
            try:
                completion, first, iterator = await self._call_once(client, kwargs, req.stream)
            except Exception as exc:
                info = classify(exc)
                logger.warning("break-glass attempt failed provider={} kind={}", name, info.kind)
                attempts.append(
                    Attempt(
                        f"break_glass:{name}:{model}",
                        [],
                        "error",
                        info.status,
                        (time.perf_counter() - started) * 1000,
                        info.kind,
                    )
                )
                continue
            served = f"{name}/{model}"
            attempts.append(
                Attempt(
                    f"break_glass:{name}:{model}",
                    [],
                    "ok",
                    200,
                    (time.perf_counter() - started) * 1000,
                    served_model=served,
                )
            )
            logger.warning("break-glass provider used provider={} model={}", name, model)
            return DispatchResult("", served, attempts, completion, first, iterator, break_glass_provider=name)
        return None

    # ------------------------------------------------------------------ public
    async def dispatch(self, req: DispatchRequest) -> DispatchResult:
        requested = self.registry.primary(req.tier).id
        candidates, attempts = self.candidates(req)
        if not candidates:
            raise UpstreamError(
                413, "context_length_exceeded", "request exceeds every model's context window", attempts
            )

        for idx, spec in enumerate(candidates):
            if not self.breakers.allow(spec.id):
                attempts.append(Attempt(spec.id, [], "skipped_circuit_open"))
                continue
            if spec.is_free and not self.free_limiter.available():
                attempts.append(Attempt(spec.id, [], "skipped_free_cap"))
                continue
            native = self._native_list(candidates[idx + 1 :])
            result = await self._try_model(spec, native, req, attempts)
            if result is not None:
                result.requested_model = requested
                return result

        if self.break_glass:
            bg = await self._break_glass(req, attempts)
            if bg is not None:
                bg.requested_model = requested
                return bg

        raise UpstreamError(503, "upstream_unavailable", "all upstream models failed", attempts)
