"""Upstream error classification shared by the dispatcher and the API layer."""

from __future__ import annotations

import random
from dataclasses import dataclass
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime

import openai
from tenacity import RetryCallState

RETRYABLE_STATUS = frozenset({408, 409, 429, 500, 502, 503, 504})


@dataclass(frozen=True)
class ErrorInfo:
    status: int | None
    kind: str  # rate_limited | server_error | timeout | connection | client_error | unknown
    retryable: bool
    retry_after_s: float | None
    message: str


class EmptyStreamError(Exception):
    """Upstream closed a streaming response before sending any chunk."""


def parse_retry_after(value: str | None) -> float | None:
    if not value:
        return None
    try:
        return max(0.0, float(value))
    except ValueError:
        pass
    try:
        when = parsedate_to_datetime(value)
    except (TypeError, ValueError):
        return None
    if when.tzinfo is None:
        when = when.replace(tzinfo=UTC)
    return max(0.0, (when - datetime.now(UTC)).total_seconds())


def classify(exc: BaseException) -> ErrorInfo:
    if isinstance(exc, EmptyStreamError):
        return ErrorInfo(None, "empty_stream", False, None, "upstream returned an empty stream")
    if isinstance(exc, openai.APITimeoutError):
        return ErrorInfo(None, "timeout", True, None, "upstream timeout")
    if isinstance(exc, openai.APIConnectionError):
        return ErrorInfo(None, "connection", True, None, "upstream connection error")
    if isinstance(exc, openai.APIStatusError):
        status = exc.status_code
        headers = exc.response.headers
        retry_after = parse_retry_after(headers.get("retry-after-ms"))
        retry_after = retry_after / 1000.0 if retry_after is not None else parse_retry_after(headers.get("retry-after"))
        if status == 429:
            return ErrorInfo(status, "rate_limited", True, retry_after, "upstream rate limited")
        if status >= 500 or status in RETRYABLE_STATUS:
            return ErrorInfo(status, "server_error", True, retry_after, f"upstream status {status}")
        # Non-429 4xx: never retried, never walked (the request itself is the problem).
        return ErrorInfo(status, "client_error", False, None, f"upstream status {status}")
    return ErrorInfo(None, "unknown", False, None, type(exc).__name__)


class RetryAfterWait:
    """tenacity wait strategy: honour Retry-After, else jittered exponential backoff."""

    def __init__(self, base_s: float, max_s: float, rng: random.Random | None = None) -> None:
        self.base_s = base_s
        self.max_s = max_s
        self._rng = rng or random.Random()  # noqa: S311 - jitter, not cryptography

    def __call__(self, retry_state: RetryCallState) -> float:
        exc = retry_state.outcome.exception() if retry_state.outcome else None
        if exc is not None:
            info = classify(exc)
            if info.retry_after_s is not None:
                return min(info.retry_after_s, self.max_s)
        exp = self.base_s * (2 ** (retry_state.attempt_number - 1))
        # "Full jitter" (AWS architecture blog): uniform(0, min(cap, base * 2^n)).
        return self._rng.uniform(0, min(self.max_s, exp))
