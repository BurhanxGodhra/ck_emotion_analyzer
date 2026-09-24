"""
Fetches datasets into data/raw/. FER2013 and FER+ can be pulled automatically
via the Kaggle API (requires a configured ~/.kaggle/kaggle.json).
AffectNet, RAF-DB, and DREAMER are all research-only / request-gated: this
script prints the official request-form links instead of downloading them,
since they cannot legally be mirrored or auto-fetched.

NOTE: DEAP was the originally planned EEG dataset for this project but its
official host (Queen Mary University) was confirmed discontinued (a
scicrunch.org resource listing marked it "no longer in service," documented
mid-December 2025). DREAMER replaced it — see docs/decisions.md D-008.
This script was not updated to reflect that switch until an external audit
caught the mismatch (EXTERNAL_REVIEW.md F-006); if you see any other
references to DEAP elsewhere in this repo that should say DREAMER, that's
the same stale-reference class of issue, please flag it.
"""

RESTRICTED_DATASETS = {
    "AffectNet": "http://mohammadmahoor.com/pages/databases/affectnet/",
    "RAF-DB": "http://www.whdeng.cn/RAF/model1.html",
    "DREAMER": "https://zenodo.org/records/546113",
}


def fetch_fer2013():
    import subprocess
    subprocess.run(
        ["kaggle", "datasets", "download", "-d", "msambare/fer2013",
         "-p", "data/raw/fer2013", "--unzip"],
        check=True,
    )


def fetch_ferplus():
    import subprocess
    subprocess.run(
        ["git", "clone", "https://github.com/microsoft/FERPlus.git",
         "data/raw/ferplus"],
        check=True,
    )


def print_restricted_instructions():
    print("\nThe following datasets require manual request approval:")
    for name, url in RESTRICTED_DATASETS.items():
        print(f"  - {name}: {url}")
    print(
        "\nOnce approved, place the extracted contents into the matching "
        "data/raw/<dataset>/ folder:\n"
        "  - AffectNet -> data/raw/affectnet/ (Train/, Test/, labels.csv)\n"
        "  - RAF-DB     -> data/raw/rafdb/ (DATASET/train/<1-7>/, DATASET/test/<1-7>/)\n"
        "  - DREAMER    -> data/raw/dreamer/DREAMER.mat\n"
    )
    print(
        "NOTE: an unofficial AffectNet mirror also exists on Kaggle "
        "(mstjebashazida/affectnet) if the official request is slow — see "
        "docs/decisions.md for the caveats on using an unofficial mirror "
        "(unverified provenance, same underlying license terms apply "
        "regardless of source)."
    )


if __name__ == "__main__":
    fetch_fer2013()
    fetch_ferplus()
    print_restricted_instructions()
