# Fever AI MVP (Educational Decision Support)

This project is a **research/educational medical decision-support demo**, **not** a diagnostic tool.

It:
- Extracts structured fever-related features from **text** and optionally from a **report image** (OCR)
- Runs a small ML classifier trained from scratch on **synthetic data**
- Returns **probability scores** + a conservative explanation (optionally via a local SLM like Ollama)

## Safety (non-negotiable)
- No diagnosis.
- No medication recommendations.
- Always includes uncertainty + a disclaimer.

## Quickstart (Windows PowerShell)

### Prerequisite: install Python

You need **Python 3.11 or 3.12** installed (recommended). On Windows, many scientific packages may not ship wheels for the newest Python immediately.

- Recommended: install from https://www.python.org/downloads/ (check **Add Python to PATH**)
- Alternative: Microsoft Store Python works too, but ensure `py` / `python` resolves correctly.

1) Create and activate a venv

```powershell
cd c:\ML\fever-ai-mvp
# Prefer Python 3.12 (or 3.11) if available
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2) Install deps

```powershell
pip install -r requirements.txt
```

### Python 3.14 (lite mode)

If you are on Python 3.14 and `pip install -r requirements.txt` fails, use lite mode (UI-only). This avoids FastAPI/Pydantic, which may not have wheels yet on the newest Python.

```powershell
pip install -r requirements-lite.txt
python -m ml.generate_synthetic_data --out data\training_data.csv --rows 800
python -m ml.train_lite_model --data data\training_data.csv --out ml\model.json
streamlit run ui\app.py
```

Note: the FastAPI server in [api/main.py](api/main.py) is intended for Python 3.11/3.12 full installs.

3) Generate synthetic data + train model

```powershell
# Use the venv's Python (works even if you didn't use the `py` launcher)
python -m ml.generate_synthetic_data --out data\training_data.csv --rows 800
python -m ml.train_model --data data\training_data.csv --out ml\model.joblib
```

4) Run Streamlit UI

```powershell
streamlit run ui\app.py
```

## OCR (optional)
This uses `pytesseract` and requires the **Tesseract OCR** binary installed on your machine.
If Tesseract isn't installed, image OCR will be skipped gracefully.

### Install Tesseract (Windows)

Option A (recommended): install from the official Windows builds (UB Mannheim):
- Download and install: https://github.com/UB-Mannheim/tesseract/wiki

Option B: install via package manager (if you use one):
- `winget install -e --id UB-Mannheim.TesseractOCR`

After installing:
- Close and reopen PowerShell (so PATH refreshes)
- Verify: `tesseract --version`

If `tesseract --version` works but the app still says it can’t find Tesseract, set an explicit path:

```powershell
$env:FEVER_TESSERACT_CMD = "C:\\Program Files\\Tesseract-OCR\\tesseract.exe"
streamlit run ui\app.py
```

## Troubleshooting

### `numpy` tries to build from source (mentions `meson-python` / `ninja`)

That usually means you are on a Python version without compatible wheels (often Python 3.13) or an older pip.

Fix options (recommended order):

1) Use Python 3.11 or 3.12 for this MVP (best wheel availability), then recreate the venv.

2) Upgrade pip tooling inside your venv:

```powershell
python -m pip install --upgrade pip setuptools wheel
```

Then retry:

```powershell
pip install -r requirements.txt
```

## Ollama (optional)
If you have Ollama running locally (default `http://localhost:11434`), the UI can use it to generate **explanations only**.
The ML model always produces the probabilities.

## What this is (and is not)
- ✅ Educational decision support demonstration
- ❌ Not a clinical product
- ❌ Not medically validated

