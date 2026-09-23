from __future__ import annotations

import os
import sys
from dataclasses import asdict
from typing import Optional

# Ensure project root is importable when uvicorn is launched from other directories.
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

try:
    from fastapi import FastAPI
    from pydantic import BaseModel
except Exception as e:  # pragma: no cover
    raise SystemExit(
        "API mode requires FastAPI + Pydantic (recommended on Python 3.11/3.12).\n"
        "If you are on Python 3.13+ (e.g., 3.14), use Streamlit lite mode instead:\n"
        "  pip install -r requirements-lite.txt\n"
        "  python -m ml.train_lite_model --data data\\training_data.csv --out ml\\model.json\n"
        "  streamlit run ui\\app.py\n"
        f"\nImport error: {e}"
    )

from core.schema import DISCLAIMER, FeatureVector
from llm.explain import explain
from ml.inference import load_model, predict_probabilities
from nlp.symptom_extractor import extract_features_from_text


DEFAULT_FULL_MODEL = os.path.join(os.path.dirname(__file__), "..", "ml", "model.joblib")
DEFAULT_LITE_MODEL = os.path.join(os.path.dirname(__file__), "..", "ml", "model.json")

MODEL_PATH = os.environ.get("FEVER_MODEL_PATH") or (
    DEFAULT_FULL_MODEL if os.path.exists(DEFAULT_FULL_MODEL) else DEFAULT_LITE_MODEL
)

app = FastAPI(title="Fever AI MVP", version="0.1.0")


class ChatRequest(BaseModel):
    text: str
    use_ollama: bool = True


class ChatResponse(BaseModel):
    features: dict
    probabilities: dict
    explanation: str
    disclaimer: str


def _merge_features(a: FeatureVector, b: FeatureVector) -> FeatureVector:
    da = asdict(a)
    db = asdict(b)
    merged = {}
    for k in da.keys():
        merged[k] = da[k] if da[k] is not None else db.get(k)
    return FeatureVector(**merged)


@app.get("/health")
def health():
    return {"ok": True}


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    fv_text = extract_features_from_text(req.text)

    if not os.path.exists(MODEL_PATH):
        probabilities = {"unknown": 1.0}
        explanation = (
            "Model not trained yet. Train the model locally, then retry.\n\n"
            f"Disclaimer: {DISCLAIMER}"
        )
        return ChatResponse(
            features=fv_text.to_model_dict(),
            probabilities=probabilities,
            explanation=explanation,
            disclaimer=DISCLAIMER,
        )

    kind, model = load_model(MODEL_PATH)
    probs = predict_probabilities(kind, model, fv_text)
    return ChatResponse(
        features=fv_text.to_model_dict(),
        probabilities=probs,
        explanation=explain(probs, use_ollama=req.use_ollama),
        disclaimer=DISCLAIMER,
    )
