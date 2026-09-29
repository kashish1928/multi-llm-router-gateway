"""FastAPI application factory and lifespan."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from loguru import logger

from app.api.middleware import GatewayMiddleware, error_body
from app.api.routes import router as api_router
from app.api.security import KeyAuthenticator, KeyRateLimiter
from app.cache.semantic import SemanticCache
from app.config import Settings
from app.guardrails import InputCheck, build_input_pipeline
from app.guardrails.output import OutputGuard
from app.llm.circuit_breaker import CircuitBreakerRegistry
from app.llm.client import build_break_glass_clients, build_http_client, build_openrouter_client
from app.llm.dispatcher import Dispatcher, Sleep
from app.llm.free_tier import FreeTierLimiter
from app.llm.startup import check_key_tier, validate_models
from app.observability.logging import configure_logging
from app.observability.tracing import Tracer, build_tracer
from app.routing.base import Router
from app.routing.features import FeatureLogger
from app.routing.heuristic import HeuristicRouter
from app.service import GatewayError, GatewayService


async def run_startup_checks(app: FastAPI) -> None:
    settings: Settings = app.state.settings
    if settings.openrouter_api_key is None:
        logger.warning("OPENROUTER_API_KEY not set; gateway will report not-ready")
        return
    if not settings.validate_models_on_startup:
        return
    http: httpx.AsyncClient = app.state.http_client
    await validate_models(settings, app.state.registry, http)  # fail fast on missing IDs
    try:
        await check_key_tier(settings, http)
    except httpx.HTTPError as exc:
        logger.warning("could not read OpenRouter key info: {}", type(exc).__name__)
    app.state.startup_ok = True


def create_app(
    settings: Settings | None = None,
    *,
    tracer: Tracer | None = None,
    sleep: Sleep | None = None,
    router: Router | None = None,
    classifier: InputCheck | None = None,
) -> FastAPI:
    settings = settings or Settings()
    configure_logging(settings.log_level, settings.secret_values(), settings.log_json)
    registry = settings.registry
    http_client = build_http_client(settings)
    client = build_openrouter_client(settings, http_client)
    free_limiter = FreeTierLimiter(settings.free_tier_daily_cap, settings.free_tier_rpm_cap)
    breakers = CircuitBreakerRegistry(settings.circuit_failure_threshold, settings.circuit_cooldown_s)
    dispatcher = Dispatcher(
        settings,
        registry,
        client,
        free_limiter,
        breakers,
        sleep=sleep or asyncio.sleep,
        break_glass=build_break_glass_clients(settings, http_client),
    )
    if classifier is None and settings.guardrail_classifier_enabled:  # pragma: no cover - needs HF weights
        from app.guardrails.classifier import load_prompt_guard

        classifier = load_prompt_guard(settings.guardrail_classifier_model, settings.guardrail_classifier_threshold)
    protected_prompts = [p for p in (settings.gateway_system_prompt,) if p]
    tracer = tracer or build_tracer(settings)
    service = GatewayService(
        settings=settings,
        registry=registry,
        router=router or HeuristicRouter(registry, settings.simple_max_tokens, settings.intermediate_token_threshold),
        dispatcher=dispatcher,
        guardrails=build_input_pipeline(settings.guardrail_max_base64_chars, classifier),
        output_guard=OutputGuard(protected_prompts, settings.secret_values()),
        tracer=tracer,
        feature_logger=FeatureLogger(settings.routing_features_log_path),
        cache=SemanticCache(
            settings.semantic_cache_threshold, settings.semantic_cache_ttl_s, settings.semantic_cache_max_entries
        )
        if settings.enable_semantic_cache
        else None,
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        await run_startup_checks(app)
        try:
            yield
        finally:
            tracer.flush()
            tracer.shutdown()
            await http_client.aclose()

    app = FastAPI(title="Multi-LLM Router Gateway", version="0.1.0", lifespan=lifespan)
    app.state.settings = settings
    app.state.registry = registry
    app.state.http_client = http_client
    app.state.free_limiter = free_limiter
    app.state.breakers = breakers
    app.state.dispatcher = dispatcher
    app.state.service = service
    app.state.tracer = tracer
    app.state.authenticator = KeyAuthenticator(settings.gateway_keys)
    app.state.rate_limiter = KeyRateLimiter(settings.rate_limit_rpm_per_key)
    app.state.startup_ok = False

    @app.exception_handler(GatewayError)
    async def gateway_error_handler(request: Request, exc: GatewayError) -> JSONResponse:
        rid = str(getattr(request.state, "request_id", ""))
        err_type = "server_error" if exc.status >= 500 else "invalid_request_error"
        return JSONResponse(
            error_body(exc.code, exc.message, rid, err_type), status_code=exc.status, headers=exc.headers
        )

    @app.exception_handler(RequestValidationError)
    async def validation_error_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
        rid = str(getattr(request.state, "request_id", ""))
        fields = sorted({".".join(str(p) for p in e.get("loc", ())[1:]) for e in exc.errors()})
        return JSONResponse(
            error_body("invalid_request", f"Invalid request body: {', '.join(fields) or 'malformed JSON'}", rid),
            status_code=400,
        )

    app.include_router(api_router)
    app.add_middleware(GatewayMiddleware, max_body_bytes=settings.max_body_bytes)
    return app


def get_app() -> FastAPI:
    """uvicorn factory entrypoint: `uvicorn app.main:get_app --factory`."""
    return create_app()
