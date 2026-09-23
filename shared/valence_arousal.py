"""
Shared valence/arousal vocabulary used by BOTH the facial model and the EEG
model, so their outputs land on the same circumplex space for the fusion page.

The facial model outputs one of 7 discrete emotions (see
emotion_analyzer/config.py CANONICAL_CLASSES, minus "contempt" — that class
was dropped from the trained model, see models/train.py). This maps each to
an approximate (valence, arousal) point on Russell's circumplex model
(Russell, 1980), both axes in [-1, 1].

These placements are commonly-cited approximations from the affective-
computing literature, not measured from either of our actual datasets — a
reasonable default for visualizing fusion, not a precise scientific mapping.
"""

from __future__ import annotations

FACIAL_EMOTION_TO_VA: dict[str, tuple[float, float]] = {
    "happy":    (0.80,  0.40),
    "surprise": (0.20,  0.75),
    "angry":    (-0.60,  0.65),
    "fear":     (-0.65,  0.70),
    "disgust":  (-0.70,  0.20),
    "sad":      (-0.65, -0.35),
    "neutral":  (0.0,    0.0),
}


def facial_label_to_va(label: str) -> tuple[float, float]:
    return FACIAL_EMOTION_TO_VA.get(label, (0.0, 0.0))


def va_to_quadrant(valence: float, arousal: float) -> str:
    """Maps a (valence, arousal) point to one of the 4 standard quadrants —
    this is also exactly the label space the EEG classifier is trained on
    (see eeg/loaders.py), so both channels output directly-comparable labels."""
    if valence >= 0 and arousal >= 0:
        return "HVHA"
    if valence >= 0 and arousal < 0:
        return "HVLA"
    if valence < 0 and arousal >= 0:
        return "LVHA"
    return "LVLA"


QUADRANT_LABELS: dict[str, str] = {
    "HVHA": "Excited / Happy",
    "HVLA": "Calm / Content",
    "LVHA": "Angry / Afraid",
    "LVLA": "Sad / Bored",
}

QUADRANT_TO_IDX = {"HVHA": 0, "HVLA": 1, "LVHA": 2, "LVLA": 3}
IDX_TO_QUADRANT = {v: k for k, v in QUADRANT_TO_IDX.items()}
