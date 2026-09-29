"""Cascade-mode validity check: cheap signals that a cheaper tier's answer is unusable."""

from __future__ import annotations

import json
import re

REFUSAL_RE = re.compile(
    r"^\s*(?:i'?m sorry,? but |sorry,? )?(?:i can(?:not|'t)|i am unable to|i'm unable to|i won't be able to)"
    r"\s+(?:help|assist|comply|provide|do that|answer)",
    re.IGNORECASE,
)


def validity_problem(content: str | None, wants_json: bool) -> str | None:
    """Return a reason string if the response should be escalated, else None."""
    if content is None or not content.strip():
        return "empty"
    if REFUSAL_RE.search(content[:300]):
        return "refusal"
    if wants_json:
        text = content.strip()
        if text.startswith("```"):
            text = text.strip("`")
            text = text[4:] if text.lower().startswith("json") else text
        try:
            json.loads(text)
        except ValueError:
            return "malformed_json"
    return None
