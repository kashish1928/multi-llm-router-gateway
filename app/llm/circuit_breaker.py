"""Minimal per-model circuit breaker (closed -> open -> half-open -> closed)."""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum


class BreakerState(StrEnum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


@dataclass
class _Breaker:
    state: BreakerState = BreakerState.CLOSED
    failures: int = 0
    opened_at: float = 0.0
    trial_in_flight: bool = False


class CircuitBreakerRegistry:
    def __init__(
        self,
        failure_threshold: int,
        cooldown_s: float,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.failure_threshold = failure_threshold
        self.cooldown_s = cooldown_s
        self._clock = clock
        self._lock = threading.Lock()
        self._breakers: dict[str, _Breaker] = {}

    def _get(self, model: str) -> _Breaker:
        return self._breakers.setdefault(model, _Breaker())

    def allow(self, model: str) -> bool:
        """Whether a request to `model` may proceed. Moves open -> half-open after cooldown."""
        with self._lock:
            b = self._get(model)
            if b.state is BreakerState.CLOSED:
                return True
            if b.state is BreakerState.OPEN:
                if self._clock() - b.opened_at >= self.cooldown_s:
                    b.state = BreakerState.HALF_OPEN
                    b.trial_in_flight = True
                    return True
                return False
            # HALF_OPEN: allow a single trial request at a time.
            if b.trial_in_flight:
                return False
            b.trial_in_flight = True
            return True

    def is_open(self, model: str) -> bool:
        """Non-mutating check used when building native fallback lists."""
        with self._lock:
            b = self._get(model)
            return b.state is BreakerState.OPEN and self._clock() - b.opened_at < self.cooldown_s

    def record_success(self, model: str) -> None:
        with self._lock:
            b = self._get(model)
            b.state = BreakerState.CLOSED
            b.failures = 0
            b.trial_in_flight = False

    def record_failure(self, model: str) -> None:
        with self._lock:
            b = self._get(model)
            b.failures += 1
            b.trial_in_flight = False
            if b.state is BreakerState.HALF_OPEN or b.failures >= self.failure_threshold:
                b.state = BreakerState.OPEN
                b.opened_at = self._clock()

    def state(self, model: str) -> BreakerState:
        with self._lock:
            return self._get(model).state

    def snapshot(self) -> dict[str, str]:
        with self._lock:
            return {m: b.state.value for m, b in self._breakers.items()}
