"""
Page 3 — Accessibility Mode.

Facial-only fallback view: large text, high contrast, minimal motion — for
users where EEG is unavailable, too noisy, or simply not the right channel
for them, and for anyone who benefits from a simpler, higher-contrast
interface generally. Styling approach mirrors the existing Accessibility
Mode in mi-bci-pipeline (larger text, higher contrast, reduced motion —
see that repo's README screenshots for the pattern this follows).
"""

import sys
from pathlib import Path

import numpy as np
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from emotion_analyzer.inference.predictor import EmotionPredictor

st.set_page_config(page_title="Accessibility Mode", page_icon="♿", layout="centered")

st.markdown(
    """
    <style>
    .big-result { font-size: 48px; font-weight: 700; text-align: center; padding: 24px; border-radius: 12px; }
    .big-label { font-size: 22px; text-align: center; color: #444; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown("<h1 style='text-align:center;'>♿ Accessibility Mode</h1>", unsafe_allow_html=True)
st.markdown(
    "<p class='big-label'>Facial-emotion-only view — large text, high contrast, no EEG required.</p>",
    unsafe_allow_html=True,
)


@st.cache_resource
def get_facial_predictor():
    return EmotionPredictor()


COLOR_BY_EMOTION = {
    "happy": "#2E7D32", "surprise": "#F9A825", "angry": "#C62828",
    "fear": "#6A1B9A", "disgust": "#4E342E", "sad": "#1565C0", "neutral": "#546E7A",
}

img_file = st.camera_input("Capture your expression", label_visibility="visible")

if img_file is not None:
    import cv2

    file_bytes = np.frombuffer(img_file.getvalue(), dtype=np.uint8)
    bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    predictor = get_facial_predictor()
    predictions = predictor.predict_image(bgr)

    if not predictions:
        st.markdown(
            "<div class='big-result' style='background:#455A64; color:white;'>No face detected</div>",
            unsafe_allow_html=True,
        )
    else:
        result = predictions[0]
        color = COLOR_BY_EMOTION.get(result["label"], "#455A64")
        st.markdown(
            f"<div class='big-result' style='background:{color}; color:white;'>"
            f"{result['label'].upper()}</div>",
            unsafe_allow_html=True,
        )
        st.markdown(
            f"<p class='big-label'>{result['confidence'] * 100:.0f}% confidence</p>",
            unsafe_allow_html=True,
        )
