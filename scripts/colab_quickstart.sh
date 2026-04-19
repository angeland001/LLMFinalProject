#!/usr/bin/env bash
set -euo pipefail

if [[ ! -f "requirements.txt" ]]; then
  echo "Run this script from the repository root (LLMFinalProject)."
  exit 1
fi

echo "Installing Python dependencies..."
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

echo "Downloading/creating RSNA-style metadata at data/rsna/metadata.csv ..."
python scripts/download_dataset.py

echo
echo "Colab bootstrap complete."
echo "Run an experiment with:"
echo "python scripts/run_experiment.py --config configs/experiments/zero_shot.yaml --num-workers 2"
