"""
Trains EEGNet on DREAMER for 4-class valence/arousal quadrant classification
(HVHA/HVLA/LVHA/LVLA — see shared/valence_arousal.py), then exports to ONNX
for inference (mirroring mi-bci-pipeline's PyTorch -> ONNX Runtime pattern,
see its bci_utils.ensure_onnx_model for the source of this approach).

NOT YET RUN against real data — see loaders.py's docstring. Run
loaders.py's own sanity check first; only run this after that looks correct.

Usage (from repo root, with the venv active):
    python -m bci_suite.eeg.train
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import GroupShuffleSplit
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader, TensorDataset

from bci_suite.eeg.eegnet import EEGNet
from bci_suite.eeg.loaders import WINDOW_SAMPLES, EEG_CHANNELS, load_dreamer_windows
from shared.valence_arousal import IDX_TO_QUADRANT

MODEL_BASE_DIR = Path("models/eeg")
MODEL_BASE_DIR.mkdir(parents=True, exist_ok=True)

BATCH_SIZE = 64
EPOCHS = 30
LEARNING_RATE = 1e-3
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def main():
    print("Loading DREAMER windows...")
    X, y, groups = load_dreamer_windows()
    print(f"Loaded {X.shape[0]} windows, {len(EEG_CHANNELS)} channels, {WINDOW_SAMPLES} samples each")

    # Split by PARTICIPANT, not by window — windows from the same trial are
    # highly correlated (same person, same stimulus), so a plain random
    # split lets the model partly "recognize" a trial it's already seen
    # slices of rather than generalizing to a genuinely new person. This is
    # also why the previous run's val_acc bounced around so erratically.
    test_split = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_val_idx, test_idx = next(test_split.split(X, y, groups=groups))

    val_split = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=42)
    train_idx, val_idx = next(
        val_split.split(X[train_val_idx], y[train_val_idx], groups=groups[train_val_idx])
    )
    train_idx, val_idx = train_val_idx[train_idx], train_val_idx[val_idx]

    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]
    X_test, y_test = X[test_idx], y[test_idx]

    n_train_p = len(np.unique(groups[train_idx]))
    n_val_p = len(np.unique(groups[val_idx]))
    n_test_p = len(np.unique(groups[test_idx]))
    print(f"Participants — train: {n_train_p}, val: {n_val_p}, test: {n_test_p} "
          f"(should not overlap)")

    class_weights = compute_class_weight(
        class_weight="balanced", classes=np.arange(4), y=y_train
    )
    class_weights_t = torch.tensor(class_weights, dtype=torch.float32).to(DEVICE)
    print("Class weights:", dict(zip([IDX_TO_QUADRANT[i] for i in range(4)], class_weights)))

    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train)),
        batch_size=BATCH_SIZE, shuffle=True,
    )
    val_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val)),
        batch_size=BATCH_SIZE,
    )
    test_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test)),
        batch_size=BATCH_SIZE,
    )

    model = EEGNet(
        n_classes=4, n_channels=len(EEG_CHANNELS), n_timepoints=WINDOW_SAMPLES
    ).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss(weight=class_weights_t)

    best_val_acc = -1.0
    best_state = None

    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * xb.size(0)
            train_correct += (logits.argmax(1) == yb).sum().item()
            train_total += xb.size(0)

        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                logits = model(xb)
                val_correct += (logits.argmax(1) == yb).sum().item()
                val_total += xb.size(0)

        train_acc = train_correct / train_total
        val_acc = val_correct / val_total
        print(
            f"Epoch {epoch}/{EPOCHS} — train_loss: {train_loss / train_total:.4f} "
            f"train_acc: {train_acc:.4f} val_acc: {val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)

    model.eval()
    test_correct, test_total = 0, 0
    with torch.no_grad():
        for xb, yb in test_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            logits = model(xb)
            test_correct += (logits.argmax(1) == yb).sum().item()
            test_total += xb.size(0)
    test_acc = test_correct / test_total
    print(f"\nFinal test accuracy: {test_acc:.4f}  (chance level for 4 classes: 0.25)")

    # Save PyTorch checkpoint + export ONNX (mirrors mi-bci-pipeline's ensure_onnx_model pattern)
    checkpoint_path = MODEL_BASE_DIR / "eegnet_dreamer.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "n_channels": len(EEG_CHANNELS),
            "n_timepoints": WINDOW_SAMPLES,
            "label_map": IDX_TO_QUADRANT,
        },
        checkpoint_path,
    )

    onnx_path = MODEL_BASE_DIR / "eegnet_dreamer.onnx"
    model_cpu = model.to("cpu")
    model_cpu.eval()
    dummy_input = torch.randn(1, 1, len(EEG_CHANNELS), WINDOW_SAMPLES)
    try:
        torch.onnx.export(
            model_cpu, dummy_input, str(onnx_path),
            input_names=["eeg_input"], output_names=["class_logits"],
            dynamic_axes={"eeg_input": {0: "batch_size"}, "class_logits": {0: "batch_size"}},
            opset_version=18, do_constant_folding=True,
        )
    except ModuleNotFoundError as e:
        # PyTorch's newer opset-18 exporter needs the optional `onnxscript`
        # package; fall back to the older, dependency-free exporter path.
        print(f"New ONNX exporter unavailable ({e}); falling back to dynamo=False")
        torch.onnx.export(
            model_cpu, dummy_input, str(onnx_path),
            input_names=["eeg_input"], output_names=["class_logits"],
            dynamic_axes={"eeg_input": {0: "batch_size"}, "class_logits": {0: "batch_size"}},
            opset_version=17, do_constant_folding=True, dynamo=False,
        )

    with open(MODEL_BASE_DIR / "labels.json", "w") as f:
        json.dump(IDX_TO_QUADRANT, f, indent=2)

    print(f"Saved checkpoint + ONNX model to {MODEL_BASE_DIR}/")


if __name__ == "__main__":
    main()
