"""Optional ML prompt-injection classifier (e.g. Llama Prompt Guard 2). Disabled by default.

Llama Prompt Guard 2 (meta-llama/Llama-Prompt-Guard-2-86M / -22M on Hugging Face) is
license-gated: you must accept the Llama license on Hugging Face and authenticate
(HF_TOKEN) before the weights can be downloaded. `transformers` + `torch` are not
runtime dependencies of this gateway; install them only if you enable this check.

The model has a 512-token context window, so long prompts are scored in overlapping
windows and the maximum malicious probability is used.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

from app.guardrails.base import CheckResult
from app.guardrails.normalize import NormalizedText

Scorer = Callable[[str], float]  # returns P(malicious) in [0, 1]
Tokenizer = Callable[[str], Sequence[str]]
Detokenizer = Callable[[Sequence[str]], str]


def _whitespace_tokenize(text: str) -> list[str]:
    return text.split()


def _whitespace_detokenize(tokens: Sequence[str]) -> str:
    return " ".join(tokens)


def overlapping_windows(tokens: Sequence[str], window: int, stride: int) -> list[Sequence[str]]:
    if window <= 0 or stride <= 0:
        raise ValueError("window and stride must be positive")
    if len(tokens) <= window:
        return [tokens]
    chunks: list[Sequence[str]] = []
    start = 0
    while True:
        chunks.append(tokens[start : start + window])
        if start + window >= len(tokens):
            break
        start += stride
    return chunks


class ChunkedClassifierCheck:
    name = "ml_classifier"

    def __init__(
        self,
        scorer: Scorer,
        threshold: float = 0.9,
        window: int = 512,
        stride: int = 384,
        tokenize: Tokenizer = _whitespace_tokenize,
        detokenize: Detokenizer = _whitespace_detokenize,
    ) -> None:
        self.scorer = scorer
        self.threshold = threshold
        self.window = window
        self.stride = stride
        self.tokenize = tokenize
        self.detokenize = detokenize

    def max_score(self, text: str) -> float:
        tokens = self.tokenize(text)
        if not tokens:
            return 0.0
        return max(
            self.scorer(self.detokenize(chunk)) for chunk in overlapping_windows(tokens, self.window, self.stride)
        )

    def check(self, text: NormalizedText) -> CheckResult | None:
        score = self.max_score(text.cased)
        if score >= self.threshold:
            return CheckResult(self.name, "classifier_score", score)
        return None


def load_prompt_guard(model_name: str, threshold: float) -> ChunkedClassifierCheck:  # pragma: no cover - needs HF
    """Build a ChunkedClassifierCheck backed by a Hugging Face sequence classifier."""
    try:
        import torch  # type: ignore[import-not-found]
        from transformers import AutoModelForSequenceClassification, AutoTokenizer  # type: ignore[import-not-found]
    except ImportError as exc:
        raise RuntimeError(
            "GUARDRAIL_CLASSIFIER_ENABLED=true requires `transformers` and `torch` "
            "(and accepted Llama license + HF_TOKEN for Prompt Guard 2)."
        ) from exc

    tok: Any = AutoTokenizer.from_pretrained(model_name)
    model: Any = AutoModelForSequenceClassification.from_pretrained(model_name)
    model.eval()

    def scorer(text: str) -> float:
        inputs = tok(text, return_tensors="pt", truncation=True, max_length=512)
        with torch.no_grad():
            logits = model(**inputs).logits
        probs = torch.softmax(logits, dim=-1)[0]
        return float(probs[-1])  # last label = malicious for Prompt Guard 2

    # Window over model tokens (minus special tokens) so each chunk fits in 512.
    def tokenize(text: str) -> list[str]:
        return list(tok.tokenize(text))

    def detokenize(tokens: Sequence[str]) -> str:
        return str(tok.convert_tokens_to_string(list(tokens)))

    return ChunkedClassifierCheck(scorer, threshold, window=500, stride=375, tokenize=tokenize, detokenize=detokenize)
