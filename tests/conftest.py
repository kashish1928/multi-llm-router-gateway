"""Shared fixtures. All OpenRouter traffic is mocked with respx; any real network
access (unmocked httpx request or raw socket connect) fails the test."""

from __future__ import annotations

import json
import socket
from collections.abc import AsyncIterator, Callable, Iterator
from pathlib import Path
from typing import Any

import httpx
import pytest
import respx
from fastapi import FastAPI

from app.config import Settings
from app.main import create_app
from app.observability.tracing import TraceRecord

BASE = "https://openrouter.ai/api/v1"
CHAT_URL = f"{BASE}/chat/completions"
GATEWAY_KEY = "gw-test-key-0001"
AUTH = {"Authorization": f"Bearer {GATEWAY_KEY}"}

FREE = "openai/gpt-oss-20b:free"
PAID_OSS = "openai/gpt-oss-20b"
LLAMA = "meta-llama/llama-3.1-8b-instruct"
GEMINI = "google/gemini-3.1-pro-preview"
CLAUDE = "anthropic/claude-sonnet-5.5"


@pytest.fixture(autouse=True)
def _isolate_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Settings must come only from the test, never from the ambient environment."""
    names = {name.upper() for name in Settings.model_fields} | {"LANGFUSE_BASE_URL", "LANGFUSE_HOST"}
    for name in names:
        monkeypatch.delenv(name, raising=False)


@pytest.fixture(autouse=True)
def _block_real_network(monkeypatch: pytest.MonkeyPatch) -> None:
    def guard(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("real network access attempted in tests")

    monkeypatch.setattr(socket.socket, "connect", guard)
    monkeypatch.setattr(socket.socket, "connect_ex", guard)
    monkeypatch.setattr(socket, "create_connection", guard)
    monkeypatch.setattr(socket, "getaddrinfo", guard)


class FakeTracer:
    def __init__(self, fail: bool = False) -> None:
        self.records: list[TraceRecord] = []
        self.fail = fail
        self.flushed = False

    def record(self, trace: TraceRecord) -> None:
        if self.fail:
            raise ConnectionError("tracing backend down")
        self.records.append(trace)

    def flush(self) -> None:
        self.flushed = True

    def shutdown(self) -> None:
        return None

    @property
    def last(self) -> TraceRecord:
        return self.records[-1]


async def no_sleep(_: float) -> None:
    return None


def make_settings(tmp_path: Path | None = None, **overrides: Any) -> Settings:
    base: dict[str, Any] = {
        "openrouter_api_key": "sk-or-v1-testkeytestkeytestkey",
        "gateway_api_keys": f"{GATEWAY_KEY},gw-test-key-0002",
        "validate_models_on_startup": False,
        "enable_tracing": False,
        "routing_features_log_path": (tmp_path / "features.jsonl") if tmp_path else None,
        "retry_base_delay_s": 0.0,
        "retry_max_delay_s": 0.0,
        "gateway_system_prompt": "You are the ACME internal gateway assistant. Internal routing notes: tier map v7, "
        "escalation contact is the platform on-call rotation.",
    }
    base.update(overrides)
    return Settings(_env_file=None, **base)  # type: ignore[call-arg]


def completion(model: str, content: str = "ok", prompt_tokens: int = 10, completion_tokens: int = 5) -> dict[str, Any]:
    return {
        "id": "gen-1",
        "object": "chat.completion",
        "created": 1_700_000_000,
        "model": model,
        "choices": [{"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": "stop"}],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }


def error(status: int, headers: dict[str, str] | None = None) -> httpx.Response:
    return httpx.Response(status, headers=headers, json={"error": {"message": f"err {status}", "code": status}})


def sse_body(model: str, pieces: list[str]) -> str:
    events = []
    for i, piece in enumerate(pieces):
        chunk = {
            "id": "gen-s",
            "object": "chat.completion.chunk",
            "created": 1,
            "model": model,
            "choices": [{"index": 0, "delta": {"role": "assistant", "content": piece}, "finish_reason": None}],
        }
        events.append(f"data: {json.dumps(chunk)}\n\n")
        if i == len(pieces) - 1:
            final = {
                "id": "gen-s",
                "object": "chat.completion.chunk",
                "created": 1,
                "model": model,
                "choices": [],
                "usage": {"prompt_tokens": 7, "completion_tokens": len(pieces), "total_tokens": 7 + len(pieces)},
            }
            events.append(f"data: {json.dumps(final)}\n\n")
    events.append("data: [DONE]\n\n")
    return "".join(events)


def sent_bodies(route: respx.Route) -> list[dict[str, Any]]:
    return [json.loads(call.request.content) for call in route.calls]


@pytest.fixture
def mock_router() -> Iterator[respx.MockRouter]:
    with respx.mock(base_url=BASE, assert_all_mocked=True, assert_all_called=False) as router:
        yield router


@pytest.fixture
def tracer() -> FakeTracer:
    return FakeTracer()


AppFactory = Callable[..., FastAPI]


@pytest.fixture
def app_factory(tmp_path: Path, tracer: FakeTracer) -> AppFactory:
    def factory(tracer_override: Any = None, **overrides: Any) -> FastAPI:
        return create_app(make_settings(tmp_path, **overrides), tracer=tracer_override or tracer, sleep=no_sleep)

    return factory


@pytest.fixture
def app(app_factory: AppFactory) -> FastAPI:
    return app_factory()


@pytest.fixture
async def client(app: FastAPI) -> AsyncIterator[httpx.AsyncClient]:
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://gateway") as c:
        yield c


def chat(content: str, **extra: Any) -> dict[str, Any]:
    return {"model": "auto", "messages": [{"role": "user", "content": content}], **extra}
