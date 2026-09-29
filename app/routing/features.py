"""Feature extraction for routing, plus a JSONL feature log for training a learned router."""

from __future__ import annotations

import json
import re
import threading
from pathlib import Path
from typing import Any

from loguru import logger

from app.models import ChatCompletionRequest

# Token estimate: ~4 characters per token for English text with OpenAI-style BPE
# tokenizers. Cheap, dependency-free, and deliberately conservative for routing only.
CHARS_PER_TOKEN = 4

CODE_BLOCK_RE = re.compile(r"```")
# Inline structured payloads: a JSON object/array literal, or >=3 consecutive CSV-like lines.
JSON_PAYLOAD_RE = re.compile(r"[\[{]\s*\{?\s*\"[^\"]+\"\s*:")
CSV_PAYLOAD_RE = re.compile(r"(?:^[^,\n]+(?:,[^,\n]+){2,}\n){2,}^[^,\n]+(?:,[^,\n]+){2,}$", re.MULTILINE)
FILE_REF_RE = re.compile(
    r"(?<![\w/.-])(?:[\w.-]+/)*[\w-]+\.(?:py|js|ts|tsx|jsx|go|rs|java|kt|rb|php|cs|cpp|cc|c|h|hpp|swift|scala|sql|yaml|yml|toml|json|sh)\b"
)
STACK_TRACE_RES = (
    re.compile(r"Traceback \(most recent call last\)"),
    re.compile(r"^\s+at [\w$.<>]+\(.*:\d+(?::\d+)?\)", re.MULTILINE),  # JVM / JS
    re.compile(r"^\s+at .+ \(.+:\d+:\d+\)", re.MULTILINE),  # Node
    re.compile(r"panic: .+\n+goroutine \d+", re.MULTILINE),  # Go
)

COMPLEX_PATTERNS: dict[str, re.Pattern[str]] = {
    "architecture": re.compile(r"\barchitect(?:ure|ing)?\b"),
    "system_design": re.compile(
        r"\b(?:system design|design (?:a|an|the)\s+(?:\w+\s+){0,3}"
        r"(?:system|service|platform|pipeline|architecture|backend|database schema|api|protocol))\b"
    ),
    "distributed": re.compile(r"\b(?:distributed|microservices?|horizontally scal\w*|fault[- ]toleran\w*|consensus)\b"),
    "refactor": re.compile(r"\brefactor\w*\b"),
    "debug": re.compile(r"\b(?:debug\w*|root cause|troubleshoot\w*)\b"),
    "algorithm": re.compile(r"\b(?:algorithm\w*|dynamic programming|time complexity|big-?o|o\(n)"),
    "prove": re.compile(r"\b(?:prove|proof|theorem|lemma|invariant)\b"),
    "optimize": re.compile(r"\boptimi[sz]\w*\b"),
    "implement": re.compile(r"\bimplement\w*\b"),
    "concurrency": re.compile(r"\b(?:race condition|deadlock|concurren\w+|lock-free)\b"),
    "data_structure": re.compile(
        r"\b(?:queue|b-?tree|trie|red-black tree|skip list|lru cache|bloom filter|consistent hashing|"
        r"parser|compiler|interpreter|allocator|memory ordering)\b"
    ),
}

INTERMEDIATE_PATTERNS: dict[str, re.Pattern[str]] = {
    "data_processing": re.compile(
        r"\b(?:csv|json|dataset|dataframe|table|spreadsheet|parse|transform|aggregate|normali[sz]e|etl|sql query)\b"
    ),
    "analysis": re.compile(r"\b(?:analy[sz]e|analysis|compare|contrast|evaluate|assess|review|audit|inconsisten\w*)\b"),
    "long_form": re.compile(r"\b(?:summari[sz]e|report|essay|explain (?:why|how)|step[- ]by[- ]step|outline)\b"),
    "reasoning": re.compile(r"\b(?:trade-?offs?|pros and cons|why does|root cause|reason about|migrat\w*)\b"),
}


def estimate_tokens(text: str) -> int:
    return (len(text) + CHARS_PER_TOKEN - 1) // CHARS_PER_TOKEN


def request_text(request: ChatCompletionRequest) -> str:
    return "\n".join(m.text() for m in request.messages)


def user_text(request: ChatCompletionRequest) -> str:
    return "\n".join(m.text() for m in request.messages if m.role in {"user", "tool"})


INSTRUCTION_WINDOW_CHARS = 2000


def instruction_text(last_user: str) -> str:
    """The part of the last user turn that most likely holds the instruction.

    Long prompts are usually "<instruction> + <pasted document>" or the reverse, so
    intent keywords are matched only against the head and tail. This stops words
    inside a pasted document (e.g. an architecture doc being proofread) from
    dominating the routing decision.
    """
    if len(last_user) <= 2 * INSTRUCTION_WINDOW_CHARS:
        return last_user
    return last_user[:INSTRUCTION_WINDOW_CHARS] + "\n" + last_user[-INSTRUCTION_WINDOW_CHARS:]


def extract_features(request: ChatCompletionRequest) -> dict[str, Any]:
    full = request_text(request)
    user = user_text(request)
    last_user = next((m.text() for m in reversed(request.messages) if m.role == "user"), "")
    lowered = instruction_text(last_user).lower()
    return {
        "est_tokens": estimate_tokens(full),
        "last_user_tokens": estimate_tokens(last_user),
        "message_count": len(request.messages),
        "code_blocks": len(CODE_BLOCK_RE.findall(user)) // 2,
        "distinct_files": len(set(FILE_REF_RE.findall(user))),
        "structured_data": bool(JSON_PAYLOAD_RE.search(user) or CSV_PAYLOAD_RE.search(user)),
        "stack_traces": sum(len(r.findall(user)) for r in STACK_TRACE_RES),
        "complex_hits": sorted(k for k, r in COMPLEX_PATTERNS.items() if r.search(lowered)),
        "intermediate_hits": sorted(k for k, r in INTERMEDIATE_PATTERNS.items() if r.search(lowered)),
        "wants_json": request.wants_json(),
        "has_tools": bool(request.tools),
        "output_budget": request.output_token_budget(),
    }


class FeatureLogger:
    """Append-only JSONL log of routing features and decisions (no raw prompt text)."""

    def __init__(self, path: Path | None) -> None:
        self.path = path
        self._lock = threading.Lock()
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, record: dict[str, Any]) -> None:
        if self.path is None:
            return
        try:
            line = json.dumps(record, default=str, sort_keys=True)
            with self._lock, self.path.open("a", encoding="utf-8") as fh:
                fh.write(line + "\n")
        except OSError as exc:  # never fail a request because of feature logging
            logger.warning("feature log write failed: {}", exc)
