from __future__ import annotations

import re
from typing import Optional

from core.schema import FeatureVector


_TEMP_RE = re.compile(r"(?P<temp>\d{2,3}(?:\.\d)?)\s*(?:f|fahrenheit|°f)", re.IGNORECASE)
_DAYS_RE = re.compile(r"(?:fever\s*(?:for)?\s*)?(?P<days>\d{1,2})\s*(?:day|days)", re.IGNORECASE)


def _extract_temperature_f(text: str) -> Optional[float]:
    m = _TEMP_RE.search(text)
    if not m:
        return None
    try:
        value = float(m.group("temp"))
    except ValueError:
        return None
    if value < 80 or value > 115:
        return None
    return value


def _extract_fever_days(text: str) -> Optional[int]:
    m = _DAYS_RE.search(text)
    if not m:
        return None
    try:
        value = int(m.group("days"))
    except ValueError:
        return None
    if value < 0 or value > 60:
        return None
    return value


def extract_features_from_text(text: str) -> FeatureVector:
    if not text:
        return FeatureVector()

    lowered = text.lower()

    headache = 1 if "headache" in lowered else None
    rash = 1 if "rash" in lowered else None
    chills = 1 if "chills" in lowered or "shivering" in lowered else None

    temperature_f = _extract_temperature_f(text)
    fever_days = _extract_fever_days(text)

    # Conservative: do not infer lab values from vague words.
    # Example: "platelets are low" => leave as None.

    return FeatureVector(
        temperature_f=temperature_f,
        fever_days=fever_days,
        headache=headache,
        rash=rash,
        chills=chills,
    )
