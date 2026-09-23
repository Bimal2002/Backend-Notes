from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Optional

from core.schema import FeatureVector


@dataclass(frozen=True)
class OcrResult:
    text: str
    used_ocr: bool
    error: Optional[str] = None


_PLATELETS_RE = re.compile(r"(?:platelet(?:\s+count)?|plt)\s*[:\-]?\s*(?P<val>[\d,]{2,9})", re.IGNORECASE)
_WBC_RE = re.compile(r"(?:wbc|white\s+blood\s+cell(?:\s+count)?)\s*[:\-]?\s*(?P<val>[\d,]{2,9})", re.IGNORECASE)
_CRP_RE = re.compile(r"(?:crp)\s*[:\-]?\s*(?P<val>\d{1,3}(?:\.\d{1,2})?)", re.IGNORECASE)


def _to_int(s: str) -> Optional[int]:
    try:
        return int(s.replace(",", "").strip())
    except Exception:
        return None


def _to_float(s: str) -> Optional[float]:
    try:
        return float(s.strip())
    except Exception:
        return None


def extract_features_from_ocr_text(text: str) -> FeatureVector:
    if not text:
        return FeatureVector()

    platelets = None
    wbc = None
    crp = None

    m = _PLATELETS_RE.search(text)
    if m:
        platelets = _to_int(m.group("val"))
        if platelets is not None and (platelets < 1000 or platelets > 2000000):
            platelets = None

    m = _WBC_RE.search(text)
    if m:
        wbc = _to_int(m.group("val"))
        if wbc is not None and (wbc < 100 or wbc > 200000):
            wbc = None

    m = _CRP_RE.search(text)
    if m:
        crp = _to_float(m.group("val"))
        if crp is not None and (crp < 0 or crp > 500):
            crp = None

    return FeatureVector(platelets=platelets, wbc=wbc, crp=crp)


def ocr_image_to_text(image_bytes: bytes) -> OcrResult:
    # OCR is optional; if Tesseract isn't installed, we return a graceful error.
    try:
        import pytesseract
        from PIL import Image
        import io

        # Allow overriding the tesseract path (useful on Windows).
        tesseract_cmd = os.environ.get("FEVER_TESSERACT_CMD") or os.environ.get("TESSERACT_CMD")
        if tesseract_cmd:
            pytesseract.pytesseract.tesseract_cmd = tesseract_cmd

        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        # If OpenCV is available, do light preprocessing; otherwise OCR raw image.
        try:
            import cv2
            import numpy as np

            arr = np.array(img)
            gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
            gray = cv2.medianBlur(gray, 3)
            _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            text = pytesseract.image_to_string(thresh)
        except Exception:
            text = pytesseract.image_to_string(img)

        return OcrResult(text=text, used_ocr=True)
    except Exception as e:
        msg = str(e)
        if "tesseract is not installed" in msg.lower() or "not in your path" in msg.lower():
            msg = (
                "Tesseract is not installed or not on PATH. Install it, then restart the app. "
                "On Windows you can also set FEVER_TESSERACT_CMD to the full path of tesseract.exe. "
                f"Original error: {e}"
            )
        return OcrResult(text="", used_ocr=False, error=msg)
