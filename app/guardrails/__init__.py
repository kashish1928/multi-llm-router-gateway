from app.guardrails.base import GuardrailPipeline, GuardrailResult, InputCheck
from app.guardrails.heuristics import EncodedPayloadCheck, HeuristicInjectionCheck

__all__ = [
    "EncodedPayloadCheck",
    "GuardrailPipeline",
    "GuardrailResult",
    "HeuristicInjectionCheck",
    "InputCheck",
    "build_input_pipeline",
]


def build_input_pipeline(
    max_base64_chars: int = 2048,
    classifier: InputCheck | None = None,
) -> GuardrailPipeline:
    """Cheapest checks first: regex heuristics, encoded payloads, then optional ML classifier."""
    checks: list[InputCheck] = [HeuristicInjectionCheck(), EncodedPayloadCheck(max_base64_chars)]
    if classifier is not None:
        checks.append(classifier)
    return GuardrailPipeline(checks)
