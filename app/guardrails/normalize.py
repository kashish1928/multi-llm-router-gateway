"""Text normalization applied before every guardrail check."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

# Zero-width, bidi-control, and other invisible code points commonly used to split
# trigger words (e.g. "ig\u200bnore") so naive substring filters miss them.
_INVISIBLE_RE = re.compile(
    "["
    "\u00ad"  # soft hyphen
    "\u034f"  # combining grapheme joiner
    "\u061c"  # arabic letter mark
    "\u115f\u1160\u3164\uffa0"  # hangul fillers
    "\u17b4\u17b5"
    "\u180b-\u180f"
    "\u200b-\u200f"  # zero-width space/joiners, LRM/RLM
    "\u202a-\u202e"  # bidi embeddings/overrides
    "\u2060-\u206f"  # word joiner, invisible operators, bidi isolates
    "\ufe00-\ufe0f"  # variation selectors
    "\ufeff"  # BOM / zero-width no-break space
    "\U000e0000-\U000e007f"  # tag characters
    "]"
)
_WS_RE = re.compile(r"[ \t\f\v]+")


@dataclass(frozen=True)
class NormalizedText:
    cased: str  # NFKC + invisible chars stripped + whitespace collapsed (case preserved)
    folded: str  # `cased`, casefolded: what pattern checks match against


def strip_invisible(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = _INVISIBLE_RE.sub("", text)
    text = "".join(ch for ch in text if unicodedata.category(ch) != "Cf")
    return _WS_RE.sub(" ", text)


def normalize(text: str) -> NormalizedText:
    cased = strip_invisible(text)
    return NormalizedText(cased=cased, folded=cased.casefold())
