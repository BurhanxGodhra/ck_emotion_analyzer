"""
Model architecture: MobileNetV2 transfer learning for facial emotion classification.

build_mobilenet_emotion_model(num_classes, fine_tune_at=None):
  - fine_tune_at=None  -> backbone fully frozen (phase 1: train the head only)
  - fine_tune_at=N     -> unfreeze backbone layers from index N onward (phase 2: fine-tune)

BatchNorm layers are always run in inference mode (training=False), even during
fine-tuning — this is standard practice: it stops fine-tuning from corrupting
the pretrained batch-norm statistics, which otherwise destabilizes training
badly on a dataset much smaller than ImageNet.
"""

from __future__ import annotations

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2

IMG_SIZE = (224, 224)  # MobileNetV2's standard pretrained input size


def build_mobilenet_emotion_model(
    num_classes: int, fine_tune_at: int | None = None
) -> tf.keras.Model:
    base = MobileNetV2(input_shape=IMG_SIZE + (3,), include_top=False, weights="imagenet")

    if fine_tune_at is None:
        base.trainable = False
    else:
        base.trainable = True
        for layer in base.layers[:fine_tune_at]:
            layer.trainable = False

    inputs = layers.Input(shape=IMG_SIZE + (3,))
    x = base(inputs, training=False)  # always inference-mode BatchNorm, see note above
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    return models.Model(inputs, outputs)
