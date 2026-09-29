"""Gateway bearer-key authentication and per-key rate limiting."""

from __future__ import annotations

import hmac
import math
import threading
import time
from collections import deque
from collections.abc import Callable

from app.service import GatewayError


class KeyAuthenticator:
    def __init__(self, keys: frozenset[str]) -> None:
        self._keys = [k.encode() for k in keys]

    def authenticate(self, authorization: str | None) -> str:
        """Return the presented key if valid; raise 401 otherwise. Fails closed when no keys are configured."""
        unauthorized = GatewayError(
            401, "invalid_api_key", "Missing or invalid gateway API key.", {"WWW-Authenticate": "Bearer"}
        )
        if not authorization:
            raise unauthorized
        scheme, _, token = authorization.partition(" ")
        token = token.strip()
        if scheme.lower() != "bearer" or not token:
            raise unauthorized
        presented = token.encode()
        # Compare against every key (no early exit) to keep timing independent of position.
        matched = False
        for key in self._keys:
            matched |= hmac.compare_digest(presented, key)
        if not matched:
            raise unauthorized
        return token


class KeyRateLimiter:
    """Sliding one-minute window per API key (in-process)."""

    def __init__(self, rpm: int, clock: Callable[[], float] = time.monotonic) -> None:
        self.rpm = rpm
        self._clock = clock
        self._lock = threading.Lock()
        self._hits: dict[str, deque[float]] = {}

    def check(self, key_id: str) -> None:
        now = self._clock()
        with self._lock:
            hits = self._hits.setdefault(key_id, deque())
            while hits and hits[0] <= now - 60.0:
                hits.popleft()
            if len(hits) >= self.rpm:
                retry_after = max(1, math.ceil(60.0 - (now - hits[0])))
                raise GatewayError(
                    429,
                    "rate_limit_exceeded",
                    "Gateway rate limit exceeded for this API key.",
                    {"Retry-After": str(retry_after)},
                )
            hits.append(now)
