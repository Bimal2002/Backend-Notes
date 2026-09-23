from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List


FORBIDDEN_PHRASES = [
    "you have",
    "you definitely",
    "diagnosis",
    "take ",
    "dose",
    "mg",
    "antibiotic",
    "paracetamol",
    "acetaminophen",
    "ibuprofen",
]


@dataclass(frozen=True)
class GuardrailResult:
    ok: bool
    issues: List[str]


def enforce_explanation_guardrails(text: str) -> GuardrailResult:
    lowered = (text or "").lower()
    issues: List[str] = []
    for phrase in FORBIDDEN_PHRASES:
        if phrase in lowered:
            issues.append(f"forbidden_phrase:{phrase.strip()}")

    return GuardrailResult(ok=(len(issues) == 0), issues=issues)


def format_probabilities(probs: Dict[str, float]) -> str:
    items = sorted(probs.items(), key=lambda kv: kv[1], reverse=True)
    return "\n".join([f"- {label}: {score:.2f}" for label, score in items])
