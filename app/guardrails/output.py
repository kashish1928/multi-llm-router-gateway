"""Lightweight output check: gateway system-prompt or secret leakage."""

from __future__ import annotations

from dataclasses import dataclass

from app.guardrails.normalize import strip_invisible

LEAK_WINDOW_CHARS = 60


@dataclass(frozen=True)
class OutputCheckResult:
    leaked: bool
    rule: str | None = None


class OutputGuard:
    def __init__(self, system_prompts: list[str], secrets: list[str]) -> None:
        self._secrets = [s for s in secrets if s]
        # Pre-compute overlapping fragments of each protected system prompt. A response
        # containing any fragment verbatim (after normalization) is treated as a leak.
        self._fragments: set[str] = set()
        for prompt in system_prompts:
            norm = " ".join(strip_invisible(prompt).casefold().split())
            if len(norm) <= LEAK_WINDOW_CHARS:
                if len(norm) >= 30:
                    self._fragments.add(norm)
                continue
            step = LEAK_WINDOW_CHARS // 2
            for i in range(0, len(norm) - LEAK_WINDOW_CHARS + 1, step):
                self._fragments.add(norm[i : i + LEAK_WINDOW_CHARS])

    def check(self, content: str | None) -> OutputCheckResult:
        if not content:
            return OutputCheckResult(False)
        for secret in self._secrets:
            if secret in content:
                return OutputCheckResult(True, "secret_leak")
        if self._fragments:
            norm = " ".join(strip_invisible(content).casefold().split())
            for frag in self._fragments:
                if frag in norm:
                    return OutputCheckResult(True, "system_prompt_leak")
        return OutputCheckResult(False)
