"""Heuristic prompt-injection checks (regex, run on normalized text)."""

from __future__ import annotations

import base64
import binascii
import re

from app.guardrails.base import CheckResult
from app.guardrails.normalize import NormalizedText, normalize

_TARGETS = r"(?:instructions?|prompts?|rules?|directions?|directives?|guidelines?|guardrails?|system messages?|context)"

INJECTION_RULES: dict[str, re.Pattern[str]] = {
    # "ignore all previous instructions", "disregard the above rules" ...
    # The object must be an instruction-like noun, so "ignore the previous error" passes.
    "instruction_override": re.compile(
        r"\b(?:ignore|disregard|forget|override|bypass|skip)\b[\w\s,'-]{0,30}?"
        r"\b(?:all|any|every|the|your|my|these|those|previous|prior|above|earlier|preceding|original|initial|system)\s+"
        r"(?:(?:previous|prior|above|earlier|preceding|original|initial|system|safety|existing)\s+)*" + _TARGETS + r"\b"
    ),
    "new_instructions": re.compile(r"(?:^|\n)\s*(?:new|updated|real) (?:system )?instructions?\s*:"),
    "role_hijack": re.compile(
        r"\byou are (?:now|no longer)\b.{0,40}\b(?:dan|jailbr\w*|unrestricted|unfiltered|developer mode|evil|"
        r"without (?:any )?(?:rules|restrictions|limits))\b"
        r"|\b(?:enable|enter|activate)\s+(?:dan|developer|god|jailbreak)\s+mode\b"
        r"|\bdo anything now\b"
    ),
    "system_prompt_extraction": re.compile(
        r"\b(?:reveal|show|print|repeat|output|display|leak|dump|disclose|tell me|give me|what (?:is|are|was|were))\b"
        r".{0,40}?\b(?:your|the)\s+(?:(?:full|exact|entire|original|initial|hidden|secret|system)\s+)*"
        r"(?:system (?:prompt|message|instructions?)|initial (?:prompt|instructions?)|hidden (?:prompt|instructions?)|"
        r"developer (?:prompt|message)|instructions you were given)\b"
    ),
    "verbatim_above": re.compile(
        r"\b(?:repeat|print|output)\b.{0,30}\b(?:everything|all text|the text|all) (?:above|before) (?:this|here)\b"
    ),
    "chat_template_tokens": re.compile(r"<\|im_start\|>\s*system|<\|system\|>|\[/?inst\]|</?system>|<<sys>>"),
}

_BASE64_RE = re.compile(r"[A-Za-z0-9+/]{40,}={0,2}")


class HeuristicInjectionCheck:
    name = "heuristic"

    def check(self, text: NormalizedText) -> CheckResult | None:
        return match_injection(text.folded, self.name)


def match_injection(folded: str, check_name: str) -> CheckResult | None:
    for rule, pattern in INJECTION_RULES.items():
        if pattern.search(folded):
            return CheckResult(check_name, rule)
    return None


def _decode_printable(blob: str) -> str | None:
    padded = blob + "=" * (-len(blob) % 4)
    try:
        raw = base64.b64decode(padded, validate=True)
    except (binascii.Error, ValueError):
        return None
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        return None
    printable = sum(ch.isprintable() or ch.isspace() for ch in text)
    return text if text and printable / len(text) >= 0.95 else None


class EncodedPayloadCheck:
    """Flags base64 blobs that decode to injection text, and oversize opaque blobs.

    Runs on the case-preserved normalized text, since casefolding corrupts base64.
    """

    name = "encoded_payload"

    def __init__(self, max_blob_chars: int = 2048) -> None:
        self.max_blob_chars = max_blob_chars

    def check(self, text: NormalizedText) -> CheckResult | None:
        for match in _BASE64_RE.finditer(text.cased):
            blob = match.group()
            if len(blob) >= self.max_blob_chars:
                return CheckResult(self.name, "oversize_base64_blob")
            decoded = _decode_printable(blob)
            if decoded is not None and match_injection(normalize(decoded).folded, self.name) is not None:
                return CheckResult(self.name, "base64_encoded_injection")
        return None
