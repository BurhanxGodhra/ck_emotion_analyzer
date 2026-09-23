"""Loads the index -> emotion-name mapping saved by training."""

import json
from pathlib import Path


def load_labels(labels_path: Path) -> dict[int, str]:
    with open(labels_path) as f:
        raw = json.load(f)
    return {int(k): v for k, v in raw.items()}  # json keys are always strings
