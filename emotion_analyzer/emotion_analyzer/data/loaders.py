"""
Unified loaders for FER2013, AffectNet, and RAF-DB.

Each loader returns a list of (absolute_filepath: str, label: str) tuples,
with labels already normalized to the CANONICAL_CLASSES taxonomy in config.py.
A combined `build_manifest()` merges all three into one DataFrame you can
split/sample/filter however training needs.

Run directly for a fast sanity check:
    python -m emotion_analyzer.data.loaders
"""

from __future__ import annotations

import csv
from pathlib import Path

import pandas as pd

from emotion_analyzer.config import (
    FER2013_ROOT,
    AFFECTNET_ROOT,
    AFFECTNET_LABELS_CSV,
    AFFECTNET_MIN_CONFIDENCE,
    RAFDB_ROOT,
    RAFDB_LABEL_MAP,
)

# Reconcile dataset-specific synonyms to one canonical taxonomy.
_LABEL_ALIASES = {
    "anger": "angry",
    "happiness": "happy",
    "sadness": "sad",
}


def _canon(label: str) -> str:
    label = label.strip().lower()
    return _LABEL_ALIASES.get(label, label)


def load_fer2013(split: str = "train") -> list[tuple[str, str]]:
    """FER2013: clean ImageFolder layout, train/<class>/*.jpg or test/<class>/*.jpg."""
    root = FER2013_ROOT / split
    records = []
    for class_dir in sorted(root.iterdir()):
        if not class_dir.is_dir():
            continue
        label = _canon(class_dir.name)
        for img_path in class_dir.glob("*.jpg"):
            records.append((str(img_path), label))
    return records


def load_affectnet(min_confidence: float = AFFECTNET_MIN_CONFIDENCE) -> list[tuple[str, str]]:
    """
    AffectNet mirror: labels.csv is ground truth, NOT the folder the image sits in.
    Columns: (index), pth, label, relFCs
    pth is relative, e.g. "anger/image0000006.jpg" — resolved against both
    Train/ and Test/ since the CSV doesn't indicate split.
    """
    records = []
    skipped_low_conf = 0
    skipped_missing = 0

    with open(AFFECTNET_LABELS_CSV, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                confidence = float(row["relFCs"])
            except (KeyError, ValueError):
                confidence = 1.0  # if missing, don't silently drop the row

            if confidence < min_confidence:
                skipped_low_conf += 1
                continue

            rel_path = row["pth"]
            label = _canon(row["label"])

            resolved = None
            for split_dir in ("Train", "Test"):
                candidate = AFFECTNET_ROOT / split_dir / rel_path
                if candidate.exists():
                    resolved = candidate
                    break
                # folder casing is inconsistent (Test/Contempt vs Train/contempt) —
                # retry with the label-cased folder name matching rel_path's own case variants
                alt = AFFECTNET_ROOT / split_dir / rel_path.capitalize()
                if alt.exists():
                    resolved = alt
                    break

            if resolved is None:
                skipped_missing += 1
                continue

            records.append((str(resolved), label))

    print(
        f"[affectnet] loaded {len(records)} | "
        f"skipped {skipped_low_conf} low-confidence, {skipped_missing} missing-file"
    )
    return records


def load_rafdb(split: str = "train") -> list[tuple[str, str]]:
    """RAF-DB (was mislabeled 'ferplus'): DATASET/<split>/<1-7>/*_aligned.jpg."""
    root = RAFDB_ROOT / "DATASET" / split
    records = []
    for num_dir in sorted(root.iterdir(), key=lambda p: p.name):
        if not num_dir.is_dir():
            continue
        try:
            label = RAFDB_LABEL_MAP[int(num_dir.name)]
        except (ValueError, KeyError):
            print(f"[rafdb] skipping unrecognized folder: {num_dir}")
            continue
        for img_path in num_dir.glob("*.jpg"):
            records.append((str(img_path), label))
    return records


def build_manifest(split: str = "train") -> pd.DataFrame:
    """Merge all three sources into one DataFrame: filepath, label, source."""
    rows = []
    for filepath, label in load_fer2013(split):
        rows.append({"filepath": filepath, "label": label, "source": "fer2013"})
    for filepath, label in load_rafdb(split):
        rows.append({"filepath": filepath, "label": label, "source": "rafdb"})

    # AffectNet has no clean split signal of its own — include it only once,
    # on the "train" call, and skip on "test" to avoid double-loading it into both.
    if split == "train":
        for filepath, label in load_affectnet():
            rows.append({"filepath": filepath, "label": label, "source": "affectnet"})

    df = pd.DataFrame(rows)
    return df


if __name__ == "__main__":
    for split in ("train", "test"):
        df = build_manifest(split)
        print(f"\n=== {split} ===")
        print(f"total: {len(df)}")
        print(df.groupby(["source", "label"]).size())
