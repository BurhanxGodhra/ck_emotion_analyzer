"""
Loads the trained EEGNet ONNX model and runs quadrant (valence/arousal)
classification on an EEG window.

Uses onnxruntime for inference — mirrors mi-bci-pipeline's own choice
(benchmarked there at ~0.29ms CPU inference vs ~1.6ms for PyTorch CPU,
see that repo's README) rather than loading the PyTorch model directly.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import onnxruntime as ort

MODEL_PATH = Path("models/eeg/eegnet_dreamer.onnx")
LABELS_PATH = Path("models/eeg/labels.json")


class EEGEmotionClassifier:
    def __init__(self, model_path: Path = MODEL_PATH, labels_path: Path = LABELS_PATH):
        if not model_path.exists():
            raise FileNotFoundError(
                f"EEG model not found at {model_path}. Train it first: "
                "python -m bci_suite.eeg.train"
            )
        self.session = ort.InferenceSession(str(model_path))
        with open(labels_path) as f:
            raw = json.load(f)
        self.labels = {int(k): v for k, v in raw.items()}

    def predict_window(self, eeg_window: np.ndarray) -> dict:
        """
        eeg_window: (n_channels, n_samples) float32, e.g. (14, 512) for DREAMER.
        Returns the predicted quadrant label + per-class probabilities.
        """
        x = eeg_window.astype(np.float32)[np.newaxis, np.newaxis, :, :]  # -> (1, 1, C, T)
        logits = self.session.run(None, {"eeg_input": x})[0][0]
        probs = np.exp(logits - logits.max())
        probs /= probs.sum()
        top_idx = int(np.argmax(probs))
        return {
            "quadrant": self.labels[top_idx],
            "confidence": float(probs[top_idx]),
            "all_probs": {self.labels[i]: float(p) for i, p in enumerate(probs)},
        }
