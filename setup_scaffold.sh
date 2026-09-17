#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# ck_emotion_analyzer — project scaffold
# Run this from an EMPTY local clone of the repo:
#   git clone https://github.com/BurhanxGodhra/ck_emotion_analyzer.git
#   cd ck_emotion_analyzer
#   bash setup_scaffold.sh
# ---------------------------------------------------------------------------
set -e

echo "Scaffolding ck_emotion_analyzer hybrid project..."

# ---- top-level ----
mkdir -p requirements data/raw/{fer2013,ferplus,affectnet,rafdb,deap} data/processed
mkdir -p models/facial_emotion models/eeg
mkdir -p scripts shared

# ---- SYSTEM 1: emotion_analyzer (standalone public product) ----
mkdir -p emotion_analyzer/emotion_analyzer/{data,models,inference,api}
mkdir -p emotion_analyzer/notebooks
mkdir -p emotion_analyzer/webapp
mkdir -p emotion_analyzer/tests

touch emotion_analyzer/emotion_analyzer/__init__.py
touch emotion_analyzer/emotion_analyzer/config.py
touch emotion_analyzer/emotion_analyzer/data/__init__.py
touch emotion_analyzer/emotion_analyzer/data/loaders.py
touch emotion_analyzer/emotion_analyzer/data/preprocessing.py
touch emotion_analyzer/emotion_analyzer/data/augmentation.py
touch emotion_analyzer/emotion_analyzer/models/__init__.py
touch emotion_analyzer/emotion_analyzer/models/architectures.py
touch emotion_analyzer/emotion_analyzer/models/train.py
touch emotion_analyzer/emotion_analyzer/inference/__init__.py
touch emotion_analyzer/emotion_analyzer/inference/predictor.py
touch emotion_analyzer/emotion_analyzer/inference/labels.py
touch emotion_analyzer/emotion_analyzer/api/__init__.py
touch emotion_analyzer/emotion_analyzer/api/main.py
touch emotion_analyzer/emotion_analyzer/api/schemas.py
touch emotion_analyzer/tests/__init__.py
touch emotion_analyzer/tests/test_preprocessing.py
touch emotion_analyzer/tests/test_inference.py

cat > emotion_analyzer/pyproject.toml << 'EOF'
[project]
name = "emotion_analyzer"
version = "0.1.0"
description = "Standalone real-time facial emotion recognition system"
requires-python = ">=3.10"

[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"
EOF

cat > emotion_analyzer/webapp/index.html << 'EOF'
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <title>Emotion Analyzer</title>
  <link rel="stylesheet" href="style.css" />
</head>
<body>
  <h1>Facial Emotion Analyzer</h1>
  <video id="video" width="480" height="360" autoplay muted></video>
  <canvas id="canvas" width="480" height="360" style="display:none;"></canvas>
  <div id="emotion-output">Waiting for camera...</div>
  <script src="app.js"></script>
</body>
</html>
EOF

touch emotion_analyzer/webapp/style.css
touch emotion_analyzer/webapp/app.js
touch emotion_analyzer/notebooks/training_experiments.ipynb

# ---- SYSTEM 2: bci_suite (Streamlit multi-page app) ----
mkdir -p bci_suite/pages
mkdir -p bci_suite/bci_suite/{eeg,fusion,utils}
mkdir -p bci_suite/assets

touch bci_suite/Home.py
touch bci_suite/pages/1_Calibration_Ground_Truth.py
touch bci_suite/pages/2_Multimodal_Fusion_Demo.py
touch bci_suite/pages/3_Accessibility_Mode.py
touch bci_suite/bci_suite/__init__.py
touch bci_suite/bci_suite/eeg/__init__.py
touch bci_suite/bci_suite/eeg/loaders.py
touch bci_suite/bci_suite/eeg/preprocessing.py
touch bci_suite/bci_suite/eeg/classifier.py
touch bci_suite/bci_suite/fusion/__init__.py
touch bci_suite/bci_suite/fusion/fusion_model.py
touch bci_suite/bci_suite/utils/__init__.py
touch bci_suite/bci_suite/utils/sync.py

# ---- shared ----
touch shared/__init__.py
touch shared/valence_arousal.py
touch shared/io_utils.py

# ---- requirements ----
cat > requirements/analyzer.txt << 'EOF'
tensorflow>=2.16
opencv-python
mediapipe
fastapi
uvicorn[standard]
numpy
pillow
EOF

cat > requirements/bci_suite.txt << 'EOF'
streamlit
mne
scikit-learn
numpy
pandas
plotly
EOF

cat > requirements/dev.txt << 'EOF'
pytest
black
ruff
EOF

# ---- scripts ----
cat > scripts/setup_env.sh << 'EOF'
#!/usr/bin/env bash
set -e
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements/analyzer.txt
pip install -r requirements/bci_suite.txt
pip install -r requirements/dev.txt
pip install -e emotion_analyzer
echo "Environment ready. Activate with: source .venv/bin/activate"
EOF
chmod +x scripts/setup_env.sh

cat > scripts/download_datasets.py << 'EOF'
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
EOF

# ---- gitignore ----
cat > .gitignore << 'EOF'
# datasets & model artifacts — never commit
data/
models/
*.keras
*.h5
*.pth

# python
.venv/
__pycache__/
*.pyc
.pytest_cache/

# notebooks
.ipynb_checkpoints/

# streamlit
.streamlit/secrets.toml

# os
.DS_Store
EOF

echo "Done. Directory tree:"
find . -not -path '*/.git*' | sort
