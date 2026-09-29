"""Optional semantic response cache (ENABLE_SEMANTIC_CACHE, off by default).

Entries are scoped per gateway API key (hashed), TTL-bound, and matched by cosine
similarity of prompt embeddings above a configurable threshold. The default embedder
is a local hashed character-trigram vectorizer: no network calls, deterministic, and
good enough for near-duplicate detection. Plug in a real embedding model by
implementing the `Embedder` protocol.
"""

from __future__ import annotations

import hashlib
import math
import re
import threading
import time
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol

_PUNCT_RE = re.compile(r"[^\w\s]")


class Embedder(Protocol):
    def embed(self, text: str) -> list[float]: ...


class HashingEmbedder:
    def __init__(self, dims: int = 512) -> None:
        self.dims = dims

    def embed(self, text: str) -> list[float]:
        vec = [0.0] * self.dims
        # Punctuation is dropped so trivial variations ("France?" vs "france") still match.
        norm_text = " ".join(_PUNCT_RE.sub(" ", text.casefold()).split())
        padded = f"  {norm_text}  "
        for i in range(len(padded) - 2):
            h = int.from_bytes(hashlib.blake2b(padded[i : i + 3].encode(), digest_size=4).digest(), "big")
            vec[h % self.dims] += 1.0 if (h >> 31) & 1 else -1.0
        n = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / n for v in vec]


def cosine(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b, strict=True))


@dataclass
class _Entry:
    embedding: list[float]
    context_key: str
    response: dict[str, Any]
    expires_at: float


class SemanticCache:
    def __init__(
        self,
        threshold: float,
        ttl_s: float,
        max_entries: int,
        embedder: Embedder | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.threshold = threshold
        self.ttl_s = ttl_s
        self.max_entries = max_entries
        self.embedder = embedder or HashingEmbedder()
        self._clock = clock
        self._lock = threading.Lock()
        self._scopes: dict[str, OrderedDict[int, _Entry]] = {}
        self._seq = 0

    @staticmethod
    def eligible(temperature: float | None, stream: bool, opt_out: bool) -> bool:
        # Provider default temperature is ~1.0, so an unset temperature is treated as > 0.
        return not opt_out and not stream and temperature is not None and temperature <= 0.0

    def get(self, scope: str, context_key: str, prompt: str) -> dict[str, Any] | None:
        emb = self.embedder.embed(prompt)
        now = self._clock()
        with self._lock:
            entries = self._scopes.get(scope)
            if not entries:
                return None
            best: tuple[float, _Entry] | None = None
            for key in list(entries):
                entry = entries[key]
                if entry.expires_at <= now:
                    del entries[key]
                    continue
                if entry.context_key != context_key:
                    continue
                sim = cosine(emb, entry.embedding)
                if sim >= self.threshold and (best is None or sim > best[0]):
                    best = (sim, entry)
            return dict(best[1].response) if best else None

    def put(self, scope: str, context_key: str, prompt: str, response: dict[str, Any]) -> None:
        emb = self.embedder.embed(prompt)
        with self._lock:
            entries = self._scopes.setdefault(scope, OrderedDict())
            self._seq += 1
            entries[self._seq] = _Entry(emb, context_key, response, self._clock() + self.ttl_s)
            while len(entries) > self.max_entries:
                entries.popitem(last=False)
