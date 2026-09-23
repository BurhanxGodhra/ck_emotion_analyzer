"""
Fine-tunes the generic EEGNet (trained on DREAMER, ~48.6% cross-subject) on
a small set of live (facial-label, EEG window) calibration pairs collected
from one specific person, producing a personalized model for them.

This is genuinely a fine-tune, not a from-scratch train: with realistically
few calibration examples (tens to low hundreds), training from scratch would
just overfit noise. Starting from the generic model's learned features and
adjusting at a low learning rate is the standard few-shot personalization
approach.

Also provides the personalized-model registry: list/save/load models under
models/eeg/personalized/, each with a small metadata sidecar (owner-given
name, example count, creation time) so the prediction pages can list and
pick between them.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from bci_suite.eeg.eegnet import EEGNet
from bci_suite.eeg.loaders import EEG_CHANNELS, WINDOW_SAMPLES
from shared.valence_arousal import IDX_TO_QUADRANT, QUADRANT_TO_IDX

GENERIC_CHECKPOINT = Path("models/eeg/eegnet_dreamer.pt")
PERSONALIZED_DIR = Path("models/eeg/personalized")
PERSONALIZED_DIR.mkdir(parents=True, exist_ok=True)

FINETUNE_EPOCHS = 15
FINETUNE_LR = 1e-4  # much lower than the 1e-3 used for the original training —
                     # we're adjusting a working model slightly, not learning from scratch


def finetune_from_calibration(
    windows: list[np.ndarray],   # each (n_channels, window_samples)
    quadrant_labels: list[str],  # e.g. "HVHA" — same vocabulary as the generic model
    model_name: str,
) -> dict:
    """Fine-tunes the generic model on the given pairs and saves the result
    under models/eeg/personalized/<model_name>.{pt,onnx} + metadata.json.
    Returns a summary dict (final training accuracy, example count, path)."""

    if not GENERIC_CHECKPOINT.exists():
        raise FileNotFoundError(
            f"Generic model checkpoint not found at {GENERIC_CHECKPOINT}. "
            "Train it first: python -m bci_suite.eeg.train"
        )

    checkpoint = torch.load(GENERIC_CHECKPOINT, map_location="cpu")
    model = EEGNet(n_classes=4, n_channels=len(EEG_CHANNELS), n_timepoints=WINDOW_SAMPLES)
    model.load_state_dict(checkpoint["model_state_dict"])

    X = np.stack(windows).astype(np.float32)[:, np.newaxis, :, :]  # (N, 1, C, T)
    y = np.array([QUADRANT_TO_IDX[q] for q in quadrant_labels], dtype=np.int64)

    X_t = torch.from_numpy(X)
    y_t = torch.from_numpy(y)

    optimizer = torch.optim.Adam(model.parameters(), lr=FINETUNE_LR)
    criterion = nn.CrossEntropyLoss()

    model.train()
    final_acc = 0.0
    for epoch in range(FINETUNE_EPOCHS):
        optimizer.zero_grad()
        logits = model(X_t)
        loss = criterion(logits, y_t)
        loss.backward()
        optimizer.step()
        final_acc = (logits.argmax(1) == y_t).float().mean().item()

    model.eval()

    # Export — same pattern as train.py's fallback path
    onnx_path = PERSONALIZED_DIR / f"{model_name}.onnx"
    dummy_input = torch.randn(1, 1, len(EEG_CHANNELS), WINDOW_SAMPLES)
    try:
        torch.onnx.export(
            model, dummy_input, str(onnx_path),
            input_names=["eeg_input"], output_names=["class_logits"],
            dynamic_axes={"eeg_input": {0: "batch_size"}, "class_logits": {0: "batch_size"}},
            opset_version=18, do_constant_folding=True,
        )
    except ModuleNotFoundError:
        torch.onnx.export(
            model, dummy_input, str(onnx_path),
            input_names=["eeg_input"], output_names=["class_logits"],
            dynamic_axes={"eeg_input": {0: "batch_size"}, "class_logits": {0: "batch_size"}},
            opset_version=17, do_constant_folding=True, dynamo=False,
        )

    labels_path = PERSONALIZED_DIR / f"{model_name}_labels.json"
    with open(labels_path, "w") as f:
        json.dump(IDX_TO_QUADRANT, f, indent=2)

    metadata = {
        "name": model_name,
        "n_examples": len(windows),
        "final_train_acc": final_acc,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "base_model": str(GENERIC_CHECKPOINT),
    }
    metadata_path = PERSONALIZED_DIR / f"{model_name}_metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    return metadata


def list_personalized_models() -> list[dict]:
    """Returns metadata for every saved personalized model."""
    results = []
    for meta_path in sorted(PERSONALIZED_DIR.glob("*_metadata.json")):
        with open(meta_path) as f:
            results.append(json.load(f))
    return results
