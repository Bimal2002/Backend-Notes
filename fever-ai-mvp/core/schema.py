from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict, Any


DISCLAIMER = (
    "Educational use only. This system does not provide medical diagnosis or treatment. "
    "If you are concerned or symptoms are severe, seek care from a qualified clinician or emergency services."
)


@dataclass(frozen=True)
class FeatureVector:
    temperature_f: Optional[float] = None
    fever_days: Optional[int] = None
    platelets: Optional[int] = None
    wbc: Optional[int] = None
    crp: Optional[float] = None
    headache: Optional[int] = None
    rash: Optional[int] = None
    chills: Optional[int] = None

    def to_model_dict(self) -> Dict[str, Any]:
        return {
            "temperature_f": self.temperature_f,
            "fever_days": self.fever_days,
            "platelets": self.platelets,
            "wbc": self.wbc,
            "crp": self.crp,
            "headache": self.headache,
            "rash": self.rash,
            "chills": self.chills,
        }


MODEL_FEATURE_COLUMNS = [
    "temperature_f",
    "fever_days",
    "platelets",
    "wbc",
    "crp",
    "headache",
    "rash",
    "chills",
]


LABELS = [
    "viral_fever",
    "dengue_like",
    "bacterial",
    "unknown",
]
