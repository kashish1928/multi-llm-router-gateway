"""HeuristicRouter: cheap, explainable three-tier classification."""

from __future__ import annotations

from typing import Any

from app.config import ModelRegistry
from app.models import TIER_ORDER, ChatCompletionRequest, RouteDecision, Tier
from app.routing.features import extract_features

STRONG_COMPLEX = {
    "architecture": 3,
    "system_design": 3,
    "distributed": 3,
    "algorithm": 2,
    "prove": 2,
    "concurrency": 2,
    "data_structure": 1,
}
WEAK_COMPLEX = {"refactor", "debug", "optimize", "implement"}
COMPLEX_THRESHOLD = 3
INTERMEDIATE_THRESHOLD = 2


class HeuristicRouter:
    def __init__(
        self,
        registry: ModelRegistry,
        simple_max_tokens: int = 1500,
        intermediate_token_threshold: int = 6000,
    ) -> None:
        self.registry = registry
        self.simple_max_tokens = simple_max_tokens
        self.intermediate_token_threshold = intermediate_token_threshold

    # ------------------------------------------------------------------ scoring
    def _complex_score(self, f: dict[str, Any], reasons: list[str]) -> int:
        score = 0
        hits: list[str] = f["complex_hits"]
        for hit in hits:
            if hit in STRONG_COMPLEX:
                score += STRONG_COMPLEX[hit]
                reasons.append(f"complex_keyword:{hit}")
        has_code_context = f["code_blocks"] > 0 or f["distinct_files"] >= 2 or f["stack_traces"] > 0
        for hit in hits:
            if hit in WEAK_COMPLEX:
                score += 2 if has_code_context else 1
                reasons.append(f"code_intent:{hit}")
        if f["distinct_files"] >= 2:
            score += 2
            reasons.append(f"multi_file:{f['distinct_files']}")
        if f["stack_traces"] > 0 and f["distinct_files"] >= 2:
            score += 1
            reasons.append("multi_file_stack_trace")
        return score

    def _intermediate_score(self, f: dict[str, Any], reasons: list[str]) -> int:
        score = 0
        for hit in f["intermediate_hits"]:
            score += 1
            reasons.append(f"intermediate_keyword:{hit}")
        if f["code_blocks"] > 0:
            score += 1
            reasons.append("code_block")
        if f["structured_data"]:
            score += 1
            reasons.append("structured_data")
        if f["stack_traces"] > 0:
            score += 1
            reasons.append("stack_trace")
        if f["est_tokens"] > self.simple_max_tokens:
            score += 2
            reasons.append(f"tokens>{self.simple_max_tokens}")
        if f["wants_json"] and f["est_tokens"] > 300:
            score += 1
            reasons.append("json_output_nontrivial")
        return score

    # ------------------------------------------------------------------ context fit
    def _fits(self, tier: Tier, needed: int) -> bool:
        return self.registry.primary(tier).context_window >= needed

    def _fit_tier(self, tier: Tier, needed: int, reasons: list[str]) -> Tier:
        if self._fits(tier, needed):
            return tier
        upward = [t for t in TIER_ORDER if t.rank > tier.rank and self._fits(t, needed)]
        any_fit = [t for t in TIER_ORDER if self._fits(t, needed)]
        choice = (upward or any_fit or [max(TIER_ORDER, key=lambda t: self.registry.primary(t).context_window)])[0]
        reasons.append(f"context_fit:{tier.value}->{choice.value}")
        return choice

    # ------------------------------------------------------------------ public
    def classify(self, request: ChatCompletionRequest) -> RouteDecision:
        features = extract_features(request)
        reasons: list[str] = []
        c_score = self._complex_score(features, reasons)
        i_score = self._intermediate_score(features, reasons)

        if c_score >= COMPLEX_THRESHOLD:
            tier = Tier.COMPLEX
            confidence = min(0.95, 0.6 + 0.07 * (c_score - COMPLEX_THRESHOLD + 1))
        elif features["est_tokens"] > self.intermediate_token_threshold:
            tier = Tier.INTERMEDIATE
            reasons.append(f"large_context>{self.intermediate_token_threshold}")
            confidence = 0.85
        elif i_score + c_score >= INTERMEDIATE_THRESHOLD:
            tier = Tier.INTERMEDIATE
            confidence = min(0.9, 0.55 + 0.08 * (i_score + c_score - INTERMEDIATE_THRESHOLD + 1))
        else:
            tier = Tier.SIMPLE
            reasons.append("short_utility")
            confidence = 0.8 if i_score + c_score == 0 else 0.6

        needed = features["est_tokens"] + features["output_budget"]
        tier = self._fit_tier(tier, needed, reasons)
        features["complex_score"] = c_score
        features["intermediate_score"] = i_score
        return RouteDecision(tier=tier, confidence=round(confidence, 3), reasons=reasons, features=features)
