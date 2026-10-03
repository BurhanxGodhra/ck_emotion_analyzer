"""
Tests for the inference pipeline: preprocessing math, label round-tripping,
API request/response schemas, and defensive failure behavior.

Tests that need a real trained model or the YuNet face-detector file are
marked skipif and skip cleanly on a fresh clone that hasn't trained/
downloaded those yet, rather than failing — but DO run and actually
exercise the real pipeline once those files exist (e.g. in CI after a
training step, or locally after setup).
"""

import json

import numpy as np
import pytest

from emotion_analyzer.inference.labels import load_labels
from emotion_analyzer.inference.predictor import (
    EmotionPredictor,
    MODEL_PATH,
    FACE_DETECTOR_MODEL_PATH,
)
from emotion_analyzer.models.architectures import IMG_SIZE


class TestLoadLabels:
    def test_round_trip(self, tmp_path):
        labels_file = tmp_path / "labels.json"
        original = {0: "angry", 1: "happy", 2: "sad"}
        with open(labels_file, "w") as f:
            json.dump(original, f)  # json.dump always writes keys as strings

        loaded = load_labels(labels_file)
        assert loaded == {0: "angry", 1: "happy", 2: "sad"}
        assert all(isinstance(k, int) for k in loaded), (
            "keys must be converted back to int — json always round-trips "
            "them as strings, and predictor.py indexes this dict with an "
            "int from np.argmax()"
        )


class TestPreprocessFace:
    """_preprocess_face() doesn't touch self.model or self.labels, so it can
    be tested directly without a trained model — bypass __init__ with
    __new__() to avoid loading anything heavy for a pure preprocessing test."""

    @pytest.fixture
    def bare_predictor(self):
        return EmotionPredictor.__new__(EmotionPredictor)

    def test_output_shape_matches_model_input_size(self, bare_predictor):
        dummy_bgr_face = np.random.randint(0, 255, (100, 80, 3), dtype=np.uint8)
        result = bare_predictor._preprocess_face(dummy_bgr_face)
        assert result.shape == (IMG_SIZE[0], IMG_SIZE[1], 3)

    def test_output_is_normalized_to_mobilenet_range(self, bare_predictor):
        # mobilenet_v2.preprocess_input maps [0, 255] -> [-1, 1]
        dummy_bgr_face = np.random.randint(0, 255, (100, 80, 3), dtype=np.uint8)
        result = bare_predictor._preprocess_face(dummy_bgr_face)
        assert result.min() >= -1.0 - 1e-5
        assert result.max() <= 1.0 + 1e-5

    def test_handles_non_square_input(self, bare_predictor):
        # Face crops are rarely square — this must not crash or distort
        # catastrophically on an extreme aspect ratio.
        dummy_bgr_face = np.random.randint(0, 255, (200, 40, 3), dtype=np.uint8)
        result = bare_predictor._preprocess_face(dummy_bgr_face)
        assert result.shape == (IMG_SIZE[0], IMG_SIZE[1], 3)


class TestEmotionPredictorFailsClearly:
    """A missing model file should raise a clear, actionable error — not a
    confusing low-level TensorFlow exception. This check was previously
    missing (EEGEmotionClassifier already had the equivalent check; this
    was an inconsistency caught while writing this test)."""

    def test_missing_model_raises_file_not_found(self, tmp_path):
        nonexistent_path = tmp_path / "does_not_exist.keras"
        with pytest.raises(FileNotFoundError, match="Train it first"):
            EmotionPredictor(model_path=nonexistent_path)


class TestAPISchemas:
    def test_predict_request_requires_image_field(self):
        from emotion_analyzer.api.schemas import PredictRequest
        with pytest.raises(Exception):  # pydantic.ValidationError
            PredictRequest()  # missing required 'image' field

        req = PredictRequest(image="data:image/jpeg;base64,abc123")
        assert req.image == "data:image/jpeg;base64,abc123"

    def test_face_prediction_schema_shape(self):
        from emotion_analyzer.api.schemas import FacePrediction, BoundingBox
        pred = FacePrediction(
            label="happy",
            confidence=0.95,
            all_probs={"happy": 0.95, "sad": 0.05},
            box=BoundingBox(x=10, y=10, w=50, h=50),
        )
        assert pred.label == "happy"
        assert pred.box.w == 50


@pytest.mark.skipif(
    not MODEL_PATH.exists(),
    reason="Requires a trained model (python -m emotion_analyzer.models.train)",
)
@pytest.mark.skipif(
    not FACE_DETECTOR_MODEL_PATH.exists(),
    reason="Requires the YuNet face-detector ONNX file — see README setup",
)
class TestEndToEndPrediction:
    """Only runs once the real model + face detector actually exist — on a
    fresh clone with neither trained/downloaded yet, these skip cleanly
    rather than failing, but DO exercise the real pipeline once available."""

    @pytest.fixture(scope="class")
    def predictor(self):
        return EmotionPredictor()

    def test_predict_image_on_blank_frame_finds_no_face(self, predictor):
        # A blank frame should detect zero faces, not crash or hallucinate one.
        blank_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        results = predictor.predict_image(blank_frame)
        assert results == []

    def test_predict_face_returns_one_of_the_known_labels(self, predictor):
        dummy_face = np.random.randint(0, 255, (150, 150, 3), dtype=np.uint8)
        result = predictor.predict_face(dummy_face)
        assert result["label"] in predictor.labels.values()
        assert 0.0 <= result["confidence"] <= 1.0
        assert abs(sum(result["all_probs"].values()) - 1.0) < 1e-4
