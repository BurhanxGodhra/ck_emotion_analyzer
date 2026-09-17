"""
Central config for dataset paths and label taxonomies.
Update DATA_ROOT to wherever data/raw/ lives in your local clone.
"""

from pathlib import Path

DATA_ROOT = Path("data/raw")

# ---------------------------------------------------------------------------
# FER2013 — confirmed clean ImageFolder layout: train/<class>/*.jpg, test/<class>/*.jpg
# ---------------------------------------------------------------------------
FER2013_ROOT = DATA_ROOT / "fer2013"
FER2013_CLASSES = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]

# ---------------------------------------------------------------------------
# AffectNet (unofficial Kaggle mirror) — Train/<class>/*.png, Test/<class>/*.png
# NOTE 1: casing is inconsistent between splits (Test/Contempt vs Train/contempt,
# Test/Anger vs Train/anger). Always normalize with .lower() when reading
# folder names as labels — do not trust case as given.
# NOTE 2 (important): the folder an image sits in does NOT reliably match its
# true label. labels.csv is the ground truth — columns are:
#   pth      -> relative path, e.g. "anger/image0000006.jpg" (folder of origin)
#   label    -> the actual label to train on (can disagree with the folder!)
#   relFCs   -> confidence score of the label, ~0-1. Filter on this (e.g. >0.7)
#               to drop low-confidence auto-labeled samples.
# Always load labels from the CSV, never from folder position.
# ---------------------------------------------------------------------------
AFFECTNET_ROOT = DATA_ROOT / "affectnet"
AFFECTNET_LABELS_CSV = AFFECTNET_ROOT / "labels.csv"
AFFECTNET_CLASSES = [
    "happy", "sad", "fear", "surprise", "neutral", "disgust",
    "anger",      # note: FER2013 calls this "angry", AffectNet calls it "anger" — reconcile in loader
    "contempt",   # AffectNet-only class, not present in FER2013
]
AFFECTNET_MIN_CONFIDENCE = 0.7  # relFCs threshold; tune later once we see label quality

# ---------------------------------------------------------------------------
# RAF-DB — mislabeled as "ferplus" in the downloaded folder; it is NOT Microsoft's
# FER+ (vote-distribution relabeling of FER2013). Confirmed by class-size
# fingerprint: folder totals (12,271) and per-class counts match RAF-DB's
# published basic-emotion training set exactly. Layout:
#   DATASET/train/<1-7>/*_aligned.jpg, DATASET/test/<1-7>/*_aligned.jpg
#   train_labels.csv / test_labels.csv duplicate this as image,label (int 1-7)
# RECOMMENDATION: rename/move this folder from data/raw/ferplus/ to
# data/raw/rafdb/ before writing the loader, so paths match what it is.
# ---------------------------------------------------------------------------
RAFDB_ROOT = DATA_ROOT / "rafdb"  # move ferplus/ contents here
RAFDB_LABEL_MAP = {
    1: "surprise",
    2: "fear",
    3: "disgust",
    4: "happy",
    5: "sad",
    6: "angry",
    7: "neutral",
}

# The genuine Microsoft FER+ (vote-count relabeling of FER2013 itself) is not
# yet in hand — separate from the above. Leave unset until/if it's downloaded.
FERPLUS_ROOT = None

# Canonical class list every part of the system should agree on (union of all datasets,
# FER2013 naming convention preferred where two datasets use a synonym)
CANONICAL_CLASSES = [
    "angry", "disgust", "fear", "happy", "sad", "surprise", "neutral", "contempt",
]
