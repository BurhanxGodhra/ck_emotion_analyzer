"""
Trains the facial emotion recognition model in two phases:
  Phase 1: MobileNetV2 backbone frozen, train the classifier head only.
  Phase 2: unfreeze the top of the backbone, fine-tune at a low learning rate.

Each run is saved to its own timestamped folder under models/facial_emotion/,
and is only promoted to the "production" path (emotion_model.keras,
labels.json — what the API actually loads) if its test accuracy beats the
previously recorded best. A run that regresses can never silently overwrite
a better model again.

Usage (from repo root, with the venv active):
    python -m emotion_analyzer.models.train
"""

from __future__ import annotations

import json
import shutil
import time
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

from emotion_analyzer.data.loaders import build_manifest
from emotion_analyzer.models.architectures import IMG_SIZE, build_mobilenet_emotion_model

MODEL_BASE_DIR = Path("models/facial_emotion")
RUN_DIR = MODEL_BASE_DIR / f"run_{time.strftime('%Y%m%d_%H%M%S')}"
RUN_DIR.mkdir(parents=True, exist_ok=True)

PROD_MODEL_PATH = MODEL_BASE_DIR / "emotion_model.keras"
PROD_LABELS_PATH = MODEL_BASE_DIR / "labels.json"
BEST_ACC_MARKER = MODEL_BASE_DIR / "best_test_accuracy.txt"

BATCH_SIZE = 32
PHASE1_EPOCHS = 12
PHASE2_EPOCHS = 15
FINE_TUNE_AT = 60

# Augmentation runs on raw 0-255-scale images, BEFORE preprocess_input's
# normalization to [-1, 1] — RandomBrightness/RandomContrast assume standard
# 0-255 input by default, and applying them after normalization corrupts
# every image (this is what broke the previous run). Never move this after
# the preprocess_input step.
_augment = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.08),
    tf.keras.layers.RandomZoom(0.1),
    tf.keras.layers.RandomContrast(0.15),
    tf.keras.layers.RandomBrightness(0.15),  # default value_range=(0,255) — matches this stage
])


def _decode_and_resize(filepath, label):
    img = tf.io.read_file(filepath)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, IMG_SIZE)  # still 0-255 scale here, resize doesn't rescale values
    return img, label


def _normalize(img, label):
    img = tf.keras.applications.mobilenet_v2.preprocess_input(img)  # -> [-1, 1]
    return img, label


def _make_dataset(filepaths, labels, training: bool) -> tf.data.Dataset:
    ds = tf.data.Dataset.from_tensor_slices((filepaths, labels))
    ds = ds.map(_decode_and_resize, num_parallel_calls=tf.data.AUTOTUNE)
    if training:
        ds = ds.shuffle(2048)
        ds = ds.map(
            lambda x, y: (_augment(x, training=True), y),
            num_parallel_calls=tf.data.AUTOTUNE,
        )
    ds = ds.map(_normalize, num_parallel_calls=tf.data.AUTOTUNE)  # applied to ALL splits, after augmentation
    return ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)


def main():
    train_df = build_manifest("train")
    test_df = build_manifest("test")

    train_df = train_df[train_df["label"] != "contempt"].reset_index(drop=True)

    classes = sorted(train_df["label"].unique())
    label_to_idx = {c: i for i, c in enumerate(classes)}
    print(f"Classes ({len(classes)}): {classes}")

    train_df["label_idx"] = train_df["label"].map(label_to_idx)
    test_df["label_idx"] = test_df["label"].map(label_to_idx)

    train_split, val_split = train_test_split(
        train_df, test_size=0.1, stratify=train_df["label_idx"], random_state=42
    )

    class_weights_arr = compute_class_weight(
        class_weight="balanced",
        classes=np.arange(len(classes)),
        y=train_split["label_idx"].values,
    )
    class_weight = dict(enumerate(class_weights_arr))
    print("Class weights:", class_weight)

    train_ds = _make_dataset(
        train_split["filepath"].values, train_split["label_idx"].values, training=True
    )
    val_ds = _make_dataset(
        val_split["filepath"].values, val_split["label_idx"].values, training=False
    )
    test_ds = _make_dataset(
        test_df["filepath"].values, test_df["label_idx"].values, training=False
    )

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            str(RUN_DIR / "best.keras"), save_best_only=True, monitor="val_accuracy"
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy", patience=4, restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=2, min_lr=1e-7
        ),
    ]

    print("\n=== Phase 1: training head (backbone frozen) ===")
    model = build_mobilenet_emotion_model(num_classes=len(classes), fine_tune_at=None)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.fit(
        train_ds, validation_data=val_ds, epochs=PHASE1_EPOCHS,
        class_weight=class_weight, callbacks=callbacks,
    )

    print("\n=== Phase 2: fine-tuning top of backbone ===")
    model = build_mobilenet_emotion_model(num_classes=len(classes), fine_tune_at=FINE_TUNE_AT)
    model.load_weights(str(RUN_DIR / "best.keras"))
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-5),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.fit(
        train_ds, validation_data=val_ds, epochs=PHASE2_EPOCHS,
        class_weight=class_weight, callbacks=callbacks,
    )

    print("\n=== Final evaluation on held-out test set ===")
    test_loss, test_acc = model.evaluate(test_ds)
    print(f"Test accuracy: {test_acc:.4f}")

    model.save(RUN_DIR / "emotion_model.keras")
    with open(RUN_DIR / "labels.json", "w") as f:
        json.dump({v: k for k, v in label_to_idx.items()}, f, indent=2)
    print(f"Saved this run to {RUN_DIR}/")

    previous_best = -1.0
    if BEST_ACC_MARKER.exists():
        previous_best = float(BEST_ACC_MARKER.read_text().strip())

    if test_acc > previous_best:
        shutil.copy(RUN_DIR / "emotion_model.keras", PROD_MODEL_PATH)
        shutil.copy(RUN_DIR / "labels.json", PROD_LABELS_PATH)
        BEST_ACC_MARKER.write_text(str(test_acc))
        print(
            f"New best ({test_acc:.4f} > previous {previous_best:.4f}) — "
            f"promoted to production model at {PROD_MODEL_PATH}"
        )
    else:
        print(
            f"This run ({test_acc:.4f}) did not beat the previous best "
            f"({previous_best:.4f}) — production model left unchanged."
        )


if __name__ == "__main__":
    main()
