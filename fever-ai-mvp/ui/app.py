from __future__ import annotations

import os
import sys
from dataclasses import asdict

import streamlit as st

# Streamlit runs this file as a script; ensure project root is importable.
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from core.schema import DISCLAIMER, FeatureVector
from llm.explain import explain
from ml.inference import load_model, predict_probabilities
from nlp.symptom_extractor import extract_features_from_text
from ocr.report_parser import extract_features_from_ocr_text, ocr_image_to_text


DEFAULT_FULL_MODEL = os.path.join(os.path.dirname(__file__), "..", "ml", "model.joblib")
DEFAULT_LITE_MODEL = os.path.join(os.path.dirname(__file__), "..", "ml", "model.json")

MODEL_PATH = os.environ.get("FEVER_MODEL_PATH") or (
    DEFAULT_FULL_MODEL if os.path.exists(DEFAULT_FULL_MODEL) else DEFAULT_LITE_MODEL
)


def _merge_features(a: FeatureVector, b: FeatureVector) -> FeatureVector:
    da = asdict(a)
    db = asdict(b)
    merged = {}
    for k in da.keys():
        merged[k] = da[k] if da[k] is not None else db.get(k)
    return FeatureVector(**merged)


st.set_page_config(page_title="Fever AI MVP", layout="centered")

st.title("Fever AI MVP (Educational Decision Support)")
st.caption("Not a diagnosis. No medication advice.")

with st.expander("Safety disclaimer", expanded=True):
    st.write(DISCLAIMER)

if "messages" not in st.session_state:
    st.session_state.messages = []

if "use_ollama" not in st.session_state:
    st.session_state.use_ollama = True

st.sidebar.header("Options")
st.session_state.use_ollama = st.sidebar.toggle("Use Ollama for explanation (optional)", value=st.session_state.use_ollama)

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

st.write("")

uploaded = st.file_uploader("Optional: upload a lab/report image (CBC/CRP)", type=["png", "jpg", "jpeg"], accept_multiple_files=False)

user_text = st.chat_input("Describe symptoms and any known values (e.g., 'fever for 5 days, 102F, headache')")

if user_text:
    st.session_state.messages.append({"role": "user", "content": user_text})

    # Extract structured features from text.
    fv_text = extract_features_from_text(user_text)

    # Optional OCR path.
    fv_ocr = FeatureVector()
    ocr_debug = None
    if uploaded is not None:
        img_bytes = uploaded.getvalue()
        ocr_res = ocr_image_to_text(img_bytes)
        if ocr_res.used_ocr:
            fv_ocr = extract_features_from_ocr_text(ocr_res.text)
            ocr_debug = (ocr_res.text[:1200] + "...") if len(ocr_res.text) > 1200 else ocr_res.text
        else:
            ocr_debug = f"OCR skipped/unavailable: {ocr_res.error}"

    features = _merge_features(fv_text, fv_ocr)

    with st.chat_message("assistant"):
        if not os.path.exists(MODEL_PATH):
            st.error(
                "Model not found. Train one of these:\n"
                "- Full: python -m ml.generate_synthetic_data ; python -m ml.train_model\n"
                "- Lite: python -m ml.generate_synthetic_data ; python -m ml.train_lite_model --out ml\\model.json"
            )
        else:
            kind, model = load_model(MODEL_PATH)
            probs = predict_probabilities(kind, model, features)

            st.subheader("Structured features (no guessing)")
            st.json(features.to_model_dict())

            st.subheader("Probability scores")
            st.json({k: round(v, 4) for k, v in probs.items()})

            st.subheader("Explanation")
            st.markdown(explain(probs, use_ollama=st.session_state.use_ollama))

            if uploaded is not None:
                with st.expander("OCR debug (optional)", expanded=False):
                    st.text(ocr_debug or "")

    st.session_state.messages.append({"role": "assistant", "content": "(Results shown above.)"})
