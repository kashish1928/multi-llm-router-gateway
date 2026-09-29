"""Request orchestration: guardrails -> override/route -> cache -> dispatch -> output check -> trace."""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any

from loguru import logger
from openai.types.chat import ChatCompletion

from app.cache.semantic import SemanticCache
from app.config import ModelRegistry, Settings
from app.guardrails.base import GuardrailPipeline
from app.guardrails.output import OutputGuard
from app.llm.dispatcher import Dispatcher, DispatchRequest, DispatchResult, UpstreamError
from app.llm.pricing import estimate_cost_usd
from app.models import TIER_ORDER, ChatCompletionRequest, RouteDecision, Tier
from app.observability.tracing import Tracer, TraceRecord, hash_identifier
from app.routing.base import Router
from app.routing.cascade import validity_problem
from app.routing.features import FeatureLogger, estimate_tokens, request_text
from app.routing.overrides import InvalidOverride, resolve_override

STREAM_GUARD_WINDOW = 4096


class GatewayError(Exception):
    def __init__(self, status: int, code: str, message: str, headers: dict[str, str] | None = None) -> None:
        super().__init__(message)
        self.status = status
        self.code = code
        self.message = message
        self.headers = headers or {}


@dataclass
class RequestContext:
    request_id: str
    api_key: str
    override_header: str | None = None
    cache_opt_out: bool = False


@dataclass
class GatewayResponse:
    headers: dict[str, str]
    body: dict[str, Any] | None = None
    stream: AsyncIterator[bytes] | None = None
    extra: dict[str, Any] = field(default_factory=dict)


def _sse(payload: dict[str, Any] | str) -> bytes:
    data = payload if isinstance(payload, str) else json.dumps(payload, separators=(",", ":"))
    return f"data: {data}\n\n".encode()


class GatewayService:
    def __init__(
        self,
        settings: Settings,
        registry: ModelRegistry,
        router: Router,
        dispatcher: Dispatcher,
        guardrails: GuardrailPipeline,
        output_guard: OutputGuard,
        tracer: Tracer,
        feature_logger: FeatureLogger,
        cache: SemanticCache | None = None,
    ) -> None:
        self.settings = settings
        self.registry = registry
        self.router = router
        self.dispatcher = dispatcher
        self.guardrails = guardrails
        self.output_guard = output_guard
        self.tracer = tracer
        self.feature_logger = feature_logger
        self.cache = cache
        self._salt = settings.trace_hash_salt.get_secret_value()

    # ------------------------------------------------------------------ helpers
    def _system_prompt_for(self, tier: Tier) -> str | None:
        if tier is Tier.SIMPLE and not self.settings.free_tier_send_gateway_system_prompt:
            return self.settings.free_tier_system_prompt or None
        return self.settings.gateway_system_prompt

    def build_messages(self, req: ChatCompletionRequest, tier: Tier) -> list[dict[str, Any]]:
        """Gateway instructions go in their own system message; user text is never
        concatenated into a system prompt."""
        messages = [m.model_dump(exclude_none=True) for m in req.messages]
        system = self._system_prompt_for(tier)
        if system:
            messages.insert(0, {"role": "system", "content": system})
        return messages

    def _headers(self, trace: TraceRecord) -> dict[str, str]:
        headers = {
            "X-Route-Tier": trace.tier or "",
            "X-Routed-Model": trace.served_model or "",
            "X-Fallback-Used": str(trace.fallback_used).lower(),
            "X-Free-Tier": str(trace.free_tier).lower(),
        }
        if trace.degraded:
            headers["X-Degraded"] = "true"
        if trace.cache != "disabled":
            headers["X-Cache"] = trace.cache
        return headers

    def _apply_result(self, trace: TraceRecord, tier: Tier, result: DispatchResult) -> None:
        spec = self.registry.get(result.served_model)
        trace.tier = tier.value
        trace.requested_model = result.requested_model
        trace.served_model = result.served_model
        trace.fallback_used = result.fallback_used
        trace.free_tier = bool(spec and spec.is_free)
        trace.degraded = result.served_model in self.registry.tiers[tier].degraded_models
        trace.break_glass_provider = result.break_glass_provider
        trace.attempts = [a.as_dict() for a in result.attempts]
        logger.info(
            "routed request_id={} tier={} requested={} served={} fallback={} free={}",
            trace.request_id,
            tier.value,
            result.requested_model,
            result.served_model,
            trace.fallback_used,
            trace.free_tier,
        )

    def _apply_usage(self, trace: TraceRecord, prompt_tokens: int, completion_tokens: int) -> None:
        trace.prompt_tokens = prompt_tokens
        trace.completion_tokens = completion_tokens
        trace.total_tokens = prompt_tokens + completion_tokens
        if trace.served_model:
            trace.cost_usd = estimate_cost_usd(self.registry, trace.served_model, prompt_tokens, completion_tokens)

    def _cache_context_key(self, req: ChatCompletionRequest, alias: str | None) -> str:
        context = {
            "prior": [m.model_dump(exclude_none=True) for m in req.messages[:-1]],
            "params": req.forwarded_params(),
            "override": alias,
        }
        return hashlib.sha256(json.dumps(context, sort_keys=True, default=str).encode()).hexdigest()

    def _dispatch_request(self, req: ChatCompletionRequest, tier: Tier, est_tokens: int) -> DispatchRequest:
        messages = self.build_messages(req, tier)
        return DispatchRequest(
            tier=tier,
            messages=messages,
            params=req.forwarded_params(),
            stream=req.stream,
            prompt_tokens_estimate=estimate_tokens("\n".join(str(m.get("content") or "") for m in messages)),
            output_token_budget=req.output_token_budget(),
        )

    # ------------------------------------------------------------------ pre-dispatch
    def _prepare(
        self, req: ChatCompletionRequest, ctx: RequestContext, trace: TraceRecord
    ) -> tuple[RouteDecision, str | None]:
        full_text = request_text(req)
        trace.prompt_chars = len(full_text)
        trace.prompt_token_estimate = estimate_tokens(full_text)
        trace.message_count = len(req.messages)
        trace.stream = req.stream
        if self.settings.trace_raw_content:
            trace.raw_input = [m.model_dump(exclude_none=True) for m in req.messages]

        guard = self.guardrails.run([m.text() for m in req.messages if m.role in {"user", "tool"}])
        trace.guardrail = guard.as_dict()
        if guard.flagged:
            # The matched rule is logged/traced internally but never returned to the client.
            logger.warning("input rejected request_id={} check={} rule={}", ctx.request_id, guard.check, guard.rule)
            raise GatewayError(400, "input_rejected", "The request was rejected by the gateway input policy.")

        try:
            override_tier, alias = resolve_override(self.registry, ctx.override_header, req.model_override, req.model)
        except InvalidOverride as exc:
            raise GatewayError(400, "invalid_model_override", str(exc)) from None

        decision = self.router.classify(req)
        if override_tier is not None:
            decision = decision.model_copy(
                update={
                    "tier": override_tier,
                    "override": alias,
                    "confidence": 1.0,
                    "reasons": [*decision.reasons, f"override:{alias}"],
                }
            )
        trace.override = alias
        trace.tier = decision.tier.value
        trace.route_confidence = decision.confidence
        trace.route_reasons = decision.reasons
        trace.route_features = decision.features
        self.feature_logger.log(
            {
                "ts": time.time(),
                "request_id": ctx.request_id,
                "key_hash": trace.key_hash,
                "tier": decision.tier.value,
                "confidence": decision.confidence,
                "reasons": decision.reasons,
                "override": alias,
                "features": decision.features,
            }
        )
        return decision, alias

    # ------------------------------------------------------------------ public
    async def handle(self, req: ChatCompletionRequest, ctx: RequestContext) -> GatewayResponse:
        started = time.perf_counter()
        trace = TraceRecord(request_id=ctx.request_id, key_hash=hash_identifier(ctx.api_key, self._salt))
        if req.model_extra and isinstance(req.model_extra.get("user"), str):
            trace.user_hash = hash_identifier(req.model_extra["user"], self._salt)
        streaming = False
        try:
            decision, alias = self._prepare(req, ctx, trace)
            mode = req.routing_mode or self.settings.routing_mode
            if req.stream or alias is not None:
                mode = "direct"  # cascade is non-streaming and never overrides an explicit alias
            trace.routing_mode = mode

            cache_key: tuple[str, str, str] | None = None
            if self.cache is not None:
                opt_out = ctx.cache_opt_out or req.cache is False
                if SemanticCache.eligible(req.temperature, req.stream, opt_out):
                    last_user = next((m.text() for m in reversed(req.messages) if m.role == "user"), "")
                    cache_key = (trace.key_hash or "", self._cache_context_key(req, alias), last_user)
                    hit = self.cache.get(*cache_key)
                    if hit is not None:
                        trace.cache = "hit"
                        for k in ("tier", "served_model", "requested_model", "fallback_used", "free_tier", "degraded"):
                            setattr(trace, k, hit["meta"][k])
                        return GatewayResponse(headers=self._headers(trace), body=hit["body"])
                    trace.cache = "miss"
                else:
                    trace.cache = "bypass"

            if req.stream:
                streaming = True
                return await self._handle_stream(req, decision, trace, started)

            if mode == "cascade":
                tier, result = await self._cascade(req, decision, trace)
            else:
                tier = decision.tier
                result = await self.dispatcher.dispatch(
                    self._dispatch_request(req, tier, decision.features["est_tokens"])
                )
            self._apply_result(trace, tier, result)
            completion = result.completion
            assert completion is not None
            body = self._finalize_completion(completion, trace)
            if cache_key is not None and self.cache is not None:
                meta = {
                    k: getattr(trace, k)
                    for k in ("tier", "served_model", "requested_model", "fallback_used", "free_tier", "degraded")
                }
                self.cache.put(*cache_key, {"body": body, "meta": meta})
            return GatewayResponse(headers=self._headers(trace), body=body)
        except UpstreamError as exc:
            trace.attempts = [a.as_dict() for a in exc.attempts]
            trace.status_code = exc.status
            trace.error_code = exc.code
            streaming = False
            raise GatewayError(exc.status, exc.code, str(exc)) from None
        except GatewayError as exc:
            trace.status_code = exc.status
            trace.error_code = exc.code
            streaming = False
            raise
        finally:
            if not streaming:
                trace.latency_ms = (time.perf_counter() - started) * 1000
                self.tracer.record(trace)

    def _finalize_completion(self, completion: ChatCompletion, trace: TraceRecord) -> dict[str, Any]:
        body = completion.model_dump(mode="json", exclude_none=True)
        if self.settings.output_guard_enabled:
            for choice in completion.choices:
                verdict = self.output_guard.check(choice.message.content)
                if verdict.leaked:
                    trace.output_guard = {"leaked": True, "rule": verdict.rule}
                    logger.warning("output rejected request_id={} rule={}", trace.request_id, verdict.rule)
                    raise GatewayError(
                        502, "output_rejected", "The upstream response was withheld by the output policy."
                    )
            trace.output_guard = {"leaked": False}
        usage = completion.usage
        self._apply_usage(trace, usage.prompt_tokens if usage else 0, usage.completion_tokens if usage else 0)
        if self.settings.trace_raw_content:
            trace.raw_output = [c.message.content for c in completion.choices]
        return body

    async def _cascade(
        self, req: ChatCompletionRequest, decision: RouteDecision, trace: TraceRecord
    ) -> tuple[Tier, DispatchResult]:
        """Cheapest fitting tier first; escalate while the answer fails a validity check."""
        est = decision.features["est_tokens"] + req.output_token_budget()
        tiers = [t for t in TIER_ORDER if self.registry.primary(t).context_window >= est] or [decision.tier]
        last: tuple[Tier, DispatchResult] | None = None
        last_error: UpstreamError | None = None
        for tier in tiers:
            try:
                result = await self.dispatcher.dispatch(
                    self._dispatch_request(req, tier, decision.features["est_tokens"])
                )
            except UpstreamError as exc:
                if exc.status < 500:
                    raise
                trace.cascade_steps.append({"tier": tier.value, "outcome": "upstream_error", "code": exc.code})
                last_error = exc
                continue
            content = None
            has_tool_calls = False
            if result.completion is not None and result.completion.choices:
                message = result.completion.choices[0].message
                content = message.content
                has_tool_calls = bool(message.tool_calls)
            problem = None if has_tool_calls else validity_problem(content, req.wants_json())
            trace.cascade_steps.append(
                {"tier": tier.value, "served_model": result.served_model, "outcome": problem or "accepted"}
            )
            last = (tier, result)
            if problem is None:
                return last
        if last is not None:
            return last
        assert last_error is not None
        raise last_error

    async def _handle_stream(
        self, req: ChatCompletionRequest, decision: RouteDecision, trace: TraceRecord, started: float
    ) -> GatewayResponse:
        result = await self.dispatcher.dispatch(
            self._dispatch_request(req, decision.tier, decision.features["est_tokens"])
        )
        self._apply_result(trace, decision.tier, result)
        assert result.first_chunk is not None and result.stream is not None
        first_chunk = result.first_chunk
        upstream = result.stream

        async def body() -> AsyncIterator[bytes]:
            buffer = ""
            prompt_tokens = completion_tokens = 0
            try:
                chunk = first_chunk
                while True:
                    for choice in chunk.choices:
                        if choice.delta and choice.delta.content:
                            buffer = (buffer + choice.delta.content)[-STREAM_GUARD_WINDOW:]
                    if chunk.usage:
                        prompt_tokens = chunk.usage.prompt_tokens
                        completion_tokens = chunk.usage.completion_tokens
                    if self.settings.output_guard_enabled:
                        verdict = self.output_guard.check(buffer)
                        if verdict.leaked:
                            trace.output_guard = {"leaked": True, "rule": verdict.rule}
                            trace.status_code, trace.error_code = 502, "output_rejected"
                            logger.warning(
                                "stream output rejected request_id={} rule={}", trace.request_id, verdict.rule
                            )
                            yield _sse(
                                {"error": {"code": "output_rejected", "message": "stream withheld by output policy"}}
                            )
                            return
                    yield _sse(chunk.model_dump(mode="json", exclude_none=True))
                    try:
                        chunk = await upstream.__anext__()
                    except StopAsyncIteration:
                        break
                yield _sse("[DONE]")
            except Exception as exc:
                # Mid-stream failure: no fallback after the first token (documented).
                logger.warning("stream interrupted request_id={} error={}", trace.request_id, type(exc).__name__)
                trace.status_code, trace.error_code = 502, "stream_interrupted"
                yield _sse({"error": {"code": "stream_interrupted", "message": "upstream stream interrupted"}})
            finally:
                if trace.output_guard == {}:
                    trace.output_guard = {"leaked": False}
                self._apply_usage(trace, prompt_tokens, completion_tokens)
                trace.latency_ms = (time.perf_counter() - started) * 1000
                self.tracer.record(trace)

        return GatewayResponse(headers=self._headers(trace), stream=body())
