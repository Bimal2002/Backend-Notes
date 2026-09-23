from __future__ import annotations

from typing import Dict, Optional

import requests

from core.guardrails import enforce_explanation_guardrails, format_probabilities
from core.schema import DISCLAIMER


def _fallback_explanation(probs: Dict[str, float]) -> str:
    ordered = sorted(probs.items(), key=lambda kv: kv[1], reverse=True)
    top_label, top_score = ordered[0]

    lines = [
        "Based on the information you provided, these patterns are consistent with several fever-related possibilities.",
        f"The highest-scoring category here is '{top_label}' (probability {top_score:.2f}).",
        "This is not a diagnosis, and different conditions can share similar patterns.",
        "If symptoms are severe, worsening, or you are worried, seek care from a qualified clinician.",
        "",
        "Probability scores:",
        format_probabilities(probs),
        "",
        f"Disclaimer: {DISCLAIMER}",
    ]
    return "\n".join(lines)


def explain_with_ollama(
    probs: Dict[str, float],
    *,
    model: str = "phi3",
    base_url: str = "http://localhost:11434",
    timeout_s: int = 15,
) -> str:
    prompt = (
        "You are an educational medical decision-support explainer.\n"
        "You MUST NOT diagnose or prescribe.\n"
        "You MUST use cautious language like 'consistent with' and mention uncertainty.\n"
        "You MUST NOT recommend medications, dosing, or treatment plans.\n\n"
        "Explain these probability scores in plain language for a layperson:\n"
        f"{format_probabilities(probs)}\n\n"
        f"End with this disclaimer verbatim:\n{DISCLAIMER}\n"
    )

    try:
        r = requests.post(
            f"{base_url}/api/generate",
            json={"model": model, "prompt": prompt, "stream": False},
            timeout=timeout_s,
        )
        r.raise_for_status()
        text = (r.json().get("response") or "").strip()
        if not text:
            return _fallback_explanation(probs)

        gr = enforce_explanation_guardrails(text)
        if not gr.ok:
            return _fallback_explanation(probs)

        if DISCLAIMER not in text:
            text = text + "\n\n" + f"Disclaimer: {DISCLAIMER}"
        return text
    except Exception:
        return _fallback_explanation(probs)


def explain(probs: Dict[str, float], *, use_ollama: bool = True) -> str:
    if use_ollama:
        return explain_with_ollama(probs)
    return _fallback_explanation(probs)
