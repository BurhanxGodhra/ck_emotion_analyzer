"""
FastAPI inference server.

POST /predict — takes a base64-encoded image (JSON body), returns one
prediction per detected face. This matches what webapp/app.js sends — no
more multipart-vs-JSON mismatch like the original prototype had.

Run from repo root:
    uvicorn emotion_analyzer.api.main:app --reload --port 8000
"""

from __future__ import annotations

import base64

import cv2
import numpy as np
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from emotion_analyzer.api.schemas import FacePrediction, PredictRequest, PredictResponse
from emotion_analyzer.inference.predictor import EmotionPredictor

app = FastAPI(title="Emotion Analyzer API")

# Wide-open CORS for local dev only — tighten allow_origins before any public deploy.
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)

predictor = EmotionPredictor()


def _decode_base64_image(data_url: str) -> np.ndarray:
    if "," in data_url:
        data_url = data_url.split(",", 1)[1]
    img_bytes = base64.b64decode(data_url)
    arr = np.frombuffer(img_bytes, dtype=np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest) -> PredictResponse:
    bgr_image = _decode_base64_image(req.image)
    predictions = predictor.predict_image(bgr_image)
    return PredictResponse(faces=[FacePrediction(**p) for p in predictions])


@app.get("/health")
def health():
    return {"status": "ok"}
