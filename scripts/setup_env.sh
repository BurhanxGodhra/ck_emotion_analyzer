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
