"""
Tests for data-side preprocessing logic: label normalization, class
taxonomy consistency, and the model architecture's basic shape contracts.

These run without needing any trained model or dataset files present —
pure logic tests, meant to work on a completely fresh clone.
"""

import numpy as np
import pytest

from emotion_analyzer.config import (
    FER2013_CLASSES,
    AFFECTNET_CLASSES,
    RAFDB_LABEL_MAP,
    CANONICAL_CLASSES,
)
from emotion_analyzer.data.loaders import _canon
from emotion_analyzer.models.architectures import build_mobilenet_emotion_model, IMG_SIZE


class TestLabelNormalization:
    """_canon() reconciles dataset-specific synonyms (e.g. AffectNet's
    'anger' vs FER2013's 'angry') into one canonical taxonomy. Getting this
    wrong silently creates duplicate classes (see docs/decisions.md — this
    exact class of bug is why AffectNet's casing inconsistency mattered)."""

    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("anger", "angry"),
            ("ANGER", "angry"),
            ("Anger", "angry"),
            ("happiness", "happy"),
            ("sadness", "sad"),
            ("angry", "angry"),  # already-canonical labels pass through unchanged
            ("happy", "happy"),
            ("neutral", "neutral"),
            ("  anger  ", "angry"),  # whitespace should not create a new class
        ],
    )
    def test_canon_maps_synonyms(self, raw, expected):
        assert _canon(raw) == expected


class TestClassTaxonomyConsistency:
    """Every class any dataset loader can emit, after _canon() normalization,
    must be a member of CANONICAL_CLASSES — if not, build_manifest() would
    silently produce a label the model's output layer has no slot for."""

    def test_fer2013_classes_are_canonical(self):
        for cls in FER2013_CLASSES:
            assert _canon(cls) in CANONICAL_CLASSES, (
                f"FER2013 class '{cls}' (normalized: '{_canon(cls)}') is not "
                f"in CANONICAL_CLASSES"
            )

    def test_affectnet_classes_are_canonical(self):
        for cls in AFFECTNET_CLASSES:
            assert _canon(cls) in CANONICAL_CLASSES, (
                f"AffectNet class '{cls}' (normalized: '{_canon(cls)}') is not "
                f"in CANONICAL_CLASSES"
            )

    def test_rafdb_label_map_values_are_canonical(self):
        for numeric_code, cls in RAFDB_LABEL_MAP.items():
            assert _canon(cls) in CANONICAL_CLASSES, (
                f"RAF-DB code {numeric_code} ('{cls}', normalized: "
                f"'{_canon(cls)}') is not in CANONICAL_CLASSES"
            )

    def test_canonical_classes_has_no_duplicates_after_normalization(self):
        normalized = [_canon(c) for c in CANONICAL_CLASSES]
        assert len(normalized) == len(set(normalized)), (
            "CANONICAL_CLASSES contains two entries that normalize to the "
            "same label — this is exactly the kind of silent duplicate-class "
            "bug _canon() exists to prevent."
        )


class TestModelArchitecture:
    """Shape/contract tests for the model builder — catches a broken input/
    output shape before it wastes hours of training time discovering it."""

    def test_output_shape_matches_num_classes(self):
        num_classes = 7
        model = build_mobilenet_emotion_model(num_classes=num_classes, fine_tune_at=None)
        dummy_batch = np.zeros((2, IMG_SIZE[0], IMG_SIZE[1], 3), dtype=np.float32)
        output = model.predict(dummy_batch, verbose=0)
        assert output.shape == (2, num_classes)

    def test_output_is_a_valid_probability_distribution(self):
        model = build_mobilenet_emotion_model(num_classes=7, fine_tune_at=None)
        dummy_batch = np.zeros((1, IMG_SIZE[0], IMG_SIZE[1], 3), dtype=np.float32)
        output = model.predict(dummy_batch, verbose=0)[0]
        assert np.isclose(output.sum(), 1.0, atol=1e-5), "softmax output should sum to 1"
        assert (output >= 0).all() and (output <= 1).all()

    def test_frozen_backbone_has_no_trainable_backbone_layers(self):
        model = build_mobilenet_emotion_model(num_classes=7, fine_tune_at=None)
        backbone = model.layers[1]  # the MobileNetV2 sub-model
        assert backbone.trainable is False

    def test_fine_tune_at_unfreezes_only_requested_layers(self):
        fine_tune_at = 60
        model = build_mobilenet_emotion_model(num_classes=7, fine_tune_at=fine_tune_at)
        backbone = model.layers[1]
        assert backbone.trainable is True
        # layers before fine_tune_at should individually still be frozen
        for layer in backbone.layers[:fine_tune_at]:
            assert layer.trainable is False
