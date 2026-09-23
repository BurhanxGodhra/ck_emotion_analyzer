"""
Loads the trained model + label map and runs face detection + emotion
prediction on a raw image (e.g. a webcam frame).

Face detection uses OpenCV's YuNet (cv2.FaceDetectorYN) — OpenCV's own
currently-maintained face detector, which takes an ONNX model directly and
handles all box decoding + NMS internally. This replaces both the legacy
Haar cascade API (removed in this opencv-python build) and MediaPipe (whose
Tasks API crashes on macOS with a GPU/Metal service error unrelated to this
code). YuNet is CPU-only and has no such platform issues.

Requires the model file — see download command in the project README:
    mkdir -p models/opencv_dnn
    curl -L -o models/opencv_dnn/face_detection_yunet.onnx \
        https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf

from emotion_analyzer.inference.labels import load_labels
from emotion_analyzer.models.architectures import IMG_SIZE

MODEL_PATH = Path("models/facial_emotion/emotion_model.keras")
LABELS_PATH = Path("models/facial_emotion/labels.json")
FACE_DETECTOR_MODEL_PATH = Path("models/opencv_dnn/face_detection_yunet.onnx")
FACE_CONFIDENCE_THRESHOLD = 0.6


class EmotionPredictor:
    def __init__(
        self,
        model_path: Path = MODEL_PATH,
        labels_path: Path = LABELS_PATH,
        face_detector_model_path: Path = FACE_DETECTOR_MODEL_PATH,
    ):
        self.model = tf.keras.models.load_model(model_path)
        self.labels = load_labels(labels_path)

        if not face_detector_model_path.exists():
            raise FileNotFoundError(
                f"YuNet model not found at {face_detector_model_path}. "
                "Download it first — see the module docstring for the curl command."
            )

        # input_size is set per-frame in detect_faces() via setInputSize,
        # since webcam frame dimensions aren't known until we see one.
        self._detector = cv2.FaceDetectorYN.create(
            str(face_detector_model_path),
            "",
            (320, 320),
            score_threshold=FACE_CONFIDENCE_THRESHOLD,
        )

    def detect_faces(self, bgr_image: np.ndarray) -> list[tuple[int, int, int, int]]:
        h, w = bgr_image.shape[:2]
        self._detector.setInputSize((w, h))
        try:
            _, faces = self._detector.detect(bgr_image)
        except cv2.error:
            # YuNet's newer graph backend occasionally throws a shape-mismatch
            # assertion on a frame-size change mid-stream (seen intermittently
            # in practice). Skip this single frame rather than 500 the request —
            # the next frame a second later works fine.
            return []

        boxes = []
        if faces is not None:
            for face in faces:
                x, y, box_w, box_h = face[:4].astype(int)

                # Pad the box ~15% on each side: YuNet's box is tighter than
                # the crop margins in FER2013/RAF-DB's training images, and
                # feeding the classifier a tighter-than-trained-on crop hurts
                # accuracy even when the model itself is fine.
                pad_x = int(box_w * 0.15)
                pad_y = int(box_h * 0.15)
                x -= pad_x
                y -= pad_y
                box_w += 2 * pad_x
                box_h += 2 * pad_y

                x, y = max(x, 0), max(y, 0)
                box_w = min(box_w, w - x)
                box_h = min(box_h, h - y)
                if box_w > 0 and box_h > 0:
                    boxes.append((int(x), int(y), int(box_w), int(box_h)))
        return boxes

    def _preprocess_face(self, bgr_face: np.ndarray) -> np.ndarray:
        rgb = cv2.cvtColor(bgr_face, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, IMG_SIZE)
        return tf.keras.applications.mobilenet_v2.preprocess_input(
            resized.astype(np.float32)
        )

    def predict_face(self, bgr_face: np.ndarray) -> dict:
        arr = self._preprocess_face(bgr_face)
        batch = np.expand_dims(arr, axis=0)
        probs = self.model.predict(batch, verbose=0)[0]
        top_idx = int(np.argmax(probs))
        return {
            "label": self.labels[top_idx],
            "confidence": float(probs[top_idx]),
            "all_probs": {self.labels[i]: float(p) for i, p in enumerate(probs)},
        }

    def predict_image(self, bgr_image: np.ndarray) -> list[dict]:
        """One prediction per detected face, including its box (for drawing)."""
        results = []
        for (x, y, w, h) in self.detect_faces(bgr_image):
            face_crop = bgr_image[y : y + h, x : x + w]
            if face_crop.size == 0:
                continue
            pred = self.predict_face(face_crop)
            pred["box"] = {"x": x, "y": y, "w": w, "h": h}
            results.append(pred)
        return results
