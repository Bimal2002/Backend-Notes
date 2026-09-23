from __future__ import annotations

import os
from typing import Dict, Any, Tuple

from core.schema import FeatureVector, MODEL_FEATURE_COLUMNS
from ml.lite_model import predict_probabilities_lite


def load_model(path: str) -> Tuple[str, Any]:
    """Load a model from disk.

    Returns: (kind, model)
      - ('full', sklearn_pipeline)
      - ('lite', json_dict_model)
      - ('heuristic', None)
    """
    if not path or not os.path.exists(path):
        return "heuristic", None

    if path.lower().endswith(".json"):
        from ml.lite_model import load_model as load_lite

        return "lite", load_lite(path)

    # Otherwise, attempt joblib/sklearn pipeline.
    try:
        import joblib

        return "full", joblib.load(path)
    except Exception:
        return "heuristic", None


def predict_probabilities(kind: str, model: Any, features: FeatureVector) -> Dict[str, float]:
    if kind == "lite":
        from ml.lite_model import predict_proba

        return predict_proba(model, features)

    if kind == "heuristic" or model is None:
        return predict_probabilities_lite(features)

    # Full sklearn pipeline path (requires pandas)
    try:
        import pandas as pd
    except Exception as e:
        raise RuntimeError(
            "Full model is present but pandas is not installed. "
            "Use lite mode on Python 3.13+ or install full deps on Python 3.11/3.12."
        ) from e

    row = features.to_model_dict()
    X = pd.DataFrame([row], columns=MODEL_FEATURE_COLUMNS)
    probs = model.predict_proba(X)[0]
    labels = list(model.classes_)
    return {labels[i]: float(probs[i]) for i in range(len(labels))}
