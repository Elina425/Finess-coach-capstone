#!/usr/bin/env bash
# Step 5: BiLSTM on Riccio NPZs (--kaggle-angles-dir = RICCIO_OUTPUT_DIR by default).
# Default model: cnn_attn + classification-only. Override with extra args after this script.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
source "$(dirname "${BASH_SOURCE[0]}")/riccio_env.sh"

exec ./venv/bin/python train_exercise_bilstm.py \
  --preset riccio --standardize --eval-test \
  --architecture cnn_attn --classification-only \
  --kaggle-angles-dir "$RICCIO_OUTPUT_DIR" \
  --kaggle-stem "$RICCIO_STEM" \
  --output-dir "$EXERCISE_BILSTM_OUT" \
  "$@"
