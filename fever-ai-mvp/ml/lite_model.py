from __future__ import annotations

import csv
import json
import math
from dataclasses import asdict
from typing import Dict, Optional, Tuple

from core.schema import FeatureVector, LABELS, MODEL_FEATURE_COLUMNS


def _softmax(logits: Dict[str, float]) -> Dict[str, float]:
    m = max(logits.values())
    exps = {k: math.exp(v - m) for k, v in logits.items()}
    s = sum(exps.values())
    return {k: (v / s) for k, v in exps.items()}


def predict_probabilities_lite(features: FeatureVector) -> Dict[str, float]:
    """Pure-Python fallback model.

    This is intentionally conservative and heuristic-based so the project can run
    on Python versions where scientific wheels (numpy/sklearn) may be unavailable.

    It is NOT a diagnosis and is not medically validated.
    """

    logits = {
        "viral_fever": 0.0,
        "dengue_like": 0.0,
        "bacterial": 0.0,
        "unknown": -0.2,
    }

    # Fever duration
    if features.fever_days is not None:
        if features.fever_days >= 5:
            logits["dengue_like"] += 0.7
            logits["bacterial"] += 0.3
        elif features.fever_days <= 2:
            logits["viral_fever"] += 0.4

    # Temperature
    if features.temperature_f is not None:
        if features.temperature_f >= 102.0:
            logits["dengue_like"] += 0.3
            logits["bacterial"] += 0.3
        elif features.temperature_f < 100.4:
            logits["unknown"] += 0.2

    # Platelets
    if features.platelets is not None:
        if features.platelets < 100_000:
            logits["dengue_like"] += 1.1
        elif features.platelets < 150_000:
            logits["dengue_like"] += 0.6

    # WBC
    if features.wbc is not None:
        if features.wbc > 12_000:
            logits["bacterial"] += 0.8
        elif features.wbc < 4_000:
            logits["dengue_like"] += 0.4
            logits["viral_fever"] += 0.2

    # CRP
    if features.crp is not None:
        if features.crp >= 20:
            logits["bacterial"] += 1.0
        elif features.crp >= 10:
            logits["bacterial"] += 0.5

    # Symptoms (binary)
    if features.headache == 1:
        logits["dengue_like"] += 0.2
        logits["viral_fever"] += 0.1
    if features.rash == 1:
        logits["dengue_like"] += 0.2
    if features.chills == 1:
        logits["bacterial"] += 0.2

    # Uncertainty baseline
    logits["unknown"] += 0.3

    return _softmax(logits)


# --- Optional trainable lite model (pure Python) ---


def _safe_float(x) -> Optional[float]:
    if x is None:
        return None
    try:
        return float(x)
    except Exception:
        return None


def _mean_var(values: list[float]) -> Tuple[float, float]:
    if not values:
        return 0.0, 1.0
    m = sum(values) / len(values)
    var = sum((v - m) ** 2 for v in values) / max(1, (len(values) - 1))
    return m, max(var, 1e-6)


def train_from_rows(rows: list[dict]) -> dict:
    by_label: Dict[str, list[dict]] = {l: [] for l in LABELS}
    for r in rows:
        label = str(r.get("label") or "unknown")
        if label not in by_label:
            label = "unknown"
        by_label[label].append(r)

    total = sum(len(v) for v in by_label.values())
    priors = {l: (len(by_label[l]) / total if total else 1.0 / len(LABELS)) for l in LABELS}

    stats: Dict[str, dict] = {}
    for label in LABELS:
        stats[label] = {}
        rows_l = by_label[label]
        for col in MODEL_FEATURE_COLUMNS:
            numeric_vals: list[float] = []
            bin_vals: list[float] = []

            for r in rows_l:
                v = _safe_float(r.get(col))
                if v is None:
                    continue
                if col in ("headache", "rash", "chills"):
                    bin_vals.append(1.0 if v >= 0.5 else 0.0)
                else:
                    numeric_vals.append(v)

            if col in ("headache", "rash", "chills"):
                p = (sum(bin_vals) + 1.0) / (len(bin_vals) + 2.0) if bin_vals else 0.5
                stats[label][col] = {"type": "bernoulli", "p": p}
            else:
                m, var = _mean_var(numeric_vals)
                stats[label][col] = {"type": "gaussian", "mean": m, "var": var}

    return {
        "type": "lite_nb_v1",
        "labels": LABELS,
        "priors": priors,
        "stats": stats,
    }


def train_from_csv(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    return train_from_rows(rows)


def save_model(model: dict, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(model, f, indent=2)


def load_model(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _log_gaussian(x: float, mean: float, var: float) -> float:
    return -0.5 * (math.log(2.0 * math.pi * var) + ((x - mean) ** 2) / var)


def predict_proba(model: dict, features: FeatureVector) -> Dict[str, float]:
    # If the model isn't the expected shape, fall back to heuristic.
    if not isinstance(model, dict) or model.get("type") != "lite_nb_v1":
        return predict_probabilities_lite(features)

    fv = asdict(features)
    priors = model.get("priors", {})
    stats = model.get("stats", {})
    labels = model.get("labels", LABELS)

    logps: Dict[str, float] = {}
    for label in labels:
        lp = math.log(max(float(priors.get(label, 1e-9)), 1e-12))
        st = stats.get(label, {})
        for col in MODEL_FEATURE_COLUMNS:
            v = _safe_float(fv.get(col))
            if v is None:
                continue
            entry = st.get(col)
            if not entry:
                continue

            if entry.get("type") == "bernoulli":
                p = float(entry.get("p", 0.5))
                x = 1.0 if v >= 0.5 else 0.0
                lp += math.log(p if x >= 0.5 else (1.0 - p))
            else:
                mean = float(entry.get("mean", 0.0))
                var = float(entry.get("var", 1.0))
                lp += _log_gaussian(float(v), mean, var)
        logps[str(label)] = lp

    return _softmax(logps)
