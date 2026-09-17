"""
Fetches datasets into data/raw/. FER2013 and FER+ can be pulled automatically
via the Kaggle API (requires a configured ~/.kaggle/kaggle.json).
AffectNet, RAF-DB, and DEAP are research-only / no-redistribution licenses:
this script prints the official request-form links instead of downloading
them, since they cannot legally be mirrored or auto-fetched.
"""

RESTRICTED_DATASETS = {
    "AffectNet": "http://mohammadmahoor.com/pages/databases/affectnet/",
    "RAF-DB": "http://www.whdeng.cn/RAF/model1.html",
    "DEAP": "https://www.eecs.qmul.ac.uk/mmv/datasets/deap/download.html",
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
    print("Once approved, place the extracted contents into the matching "
          "data/raw/<dataset>/ folder.\n")


if __name__ == "__main__":
    fetch_fer2013()
    fetch_ferplus()
    print_restricted_instructions()
