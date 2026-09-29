"""HTTP endpoints: OpenAI-compatible chat completions + health/readiness."""

from __future__ import annotations

from typing import Annotated, Any

from fastapi import APIRouter, Header, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

from app.models import ChatCompletionRequest
from app.service import GatewayError, RequestContext

router = APIRouter()


def _request_id(request: Request) -> str:
    return str(getattr(request.state, "request_id", ""))


@router.get("/healthz")
async def healthz() -> dict[str, str]:
    return {"status": "ok"}


@router.get("/readyz")
async def readyz(request: Request) -> Response:
    state = request.app.state
    settings = state.settings
    checks: dict[str, Any] = {
        "openrouter_key_configured": settings.openrouter_api_key is not None,
        "startup_checks_passed": state.startup_ok or not settings.validate_models_on_startup,
    }
    ready = all(checks.values())
    body = {
        "status": "ready" if ready else "not_ready",
        "checks": checks,
        "circuit_breakers": state.breakers.snapshot(),
        "free_tier": state.free_limiter.snapshot(),
    }
    return JSONResponse(body, status_code=200 if ready else 503)


@router.post("/v1/chat/completions", response_model=None)
async def chat_completions(
    request: Request,
    payload: ChatCompletionRequest,
    authorization: Annotated[str | None, Header()] = None,
    x_model_override: Annotated[str | None, Header()] = None,
    cache_control: Annotated[str | None, Header()] = None,
) -> Response:
    state = request.app.state
    api_key = state.authenticator.authenticate(authorization)
    state.rate_limiter.check(api_key)
    ctx = RequestContext(
        request_id=_request_id(request),
        api_key=api_key,
        override_header=x_model_override,
        cache_opt_out=bool(cache_control and "no-cache" in cache_control.lower()),
    )
    result = await state.service.handle(payload, ctx)
    if result.stream is not None:
        return StreamingResponse(
            result.stream,
            media_type="text/event-stream",
            headers={**result.headers, "Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )
    if result.body is None:  # pragma: no cover - defensive
        raise GatewayError(502, "empty_response", "Upstream returned no body.")
    return JSONResponse(result.body, headers=result.headers)
