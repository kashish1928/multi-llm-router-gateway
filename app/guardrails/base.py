"""Pluggable input-guardrail pipeline (cheapest checks first)."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Protocol

from app.guardrails.normalize import NormalizedText, normalize


@dataclass(frozen=True)
class CheckResult:
    check: str
    rule: str
    score: float = 1.0


@dataclass
class GuardrailResult:
    flagged: bool
    check: str | None = None
    rule: str | None = None  # internal only: never returned to clients
    score: float = 0.0
    checks_run: list[str] = field(default_factory=list)
    latency_ms: float = 0.0

    def as_dict(self) -> dict[str, object]:
        return {
            "flagged": self.flagged,
            "check": self.check,
            "rule": self.rule,
            "score": self.score,
            "checks_run": self.checks_run,
            "latency_ms": round(self.latency_ms, 3),
        }


class InputCheck(Protocol):
    name: str

    def check(self, text: NormalizedText) -> CheckResult | None: ...


class GuardrailPipeline:
    def __init__(self, checks: list[InputCheck]) -> None:
        self.checks = checks

    def run(self, texts: list[str]) -> GuardrailResult:
        started = time.perf_counter()
        normalized = [normalize(t) for t in texts if t]
        ran: list[str] = []
        for check in self.checks:
            ran.append(check.name)
            for text in normalized:
                hit = check.check(text)
                if hit is not None:
                    return GuardrailResult(
                        True, hit.check, hit.rule, hit.score, ran, (time.perf_counter() - started) * 1000
                    )
        return GuardrailResult(False, checks_run=ran, latency_ms=(time.perf_counter() - started) * 1000)
