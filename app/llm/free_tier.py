"""In-process free-tier quota tracking (per UTC day and per rolling minute).

OpenRouter counts failed free-model attempts against the daily quota, so the gateway
reserves a slot *before* each free-model call and proactively skips the free model
once either local counter is exhausted.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from collections.abc import Callable
from datetime import UTC, date, datetime


class FreeTierLimiter:
    def __init__(
        self,
        daily_cap: int,
        rpm_cap: int,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self.daily_cap = daily_cap
        self.rpm_cap = rpm_cap
        self._clock = clock
        self._lock = threading.Lock()
        self._day: date = self._today()
        self._daily_count = 0
        self._minute: deque[float] = deque()

    def _today(self) -> date:
        return datetime.fromtimestamp(self._clock(), tz=UTC).date()

    def _roll(self) -> None:
        today = self._today()
        if today != self._day:
            self._day = today
            self._daily_count = 0
        cutoff = self._clock() - 60.0
        while self._minute and self._minute[0] <= cutoff:
            self._minute.popleft()

    def available(self) -> bool:
        with self._lock:
            self._roll()
            return self._daily_count < self.daily_cap and len(self._minute) < self.rpm_cap

    def try_acquire(self) -> bool:
        """Reserve one free-model request. Returns False if a cap is exhausted."""
        with self._lock:
            self._roll()
            if self._daily_count >= self.daily_cap or len(self._minute) >= self.rpm_cap:
                return False
            self._daily_count += 1
            self._minute.append(self._clock())
            return True

    def record_external(self) -> None:
        """Count a free-model request we did not reserve (served via native fallback)."""
        with self._lock:
            self._roll()
            self._daily_count += 1
            self._minute.append(self._clock())

    def snapshot(self) -> dict[str, int | str]:
        with self._lock:
            self._roll()
            return {
                "day": self._day.isoformat(),
                "daily_used": self._daily_count,
                "daily_cap": self.daily_cap,
                "minute_used": len(self._minute),
                "rpm_cap": self.rpm_cap,
            }
