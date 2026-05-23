#!/usr/bin/env bash
# train_paper_classification.py — xLSTM[7:1] on ViTPose-S 256-D windows (same NPZ as PosePulse vit_backbone export).
#
# From repo root:
#   ./scripts/train_paper_classification_xlstm_vit256.sh
#
# Overrides:
#   KAGGLE_DIR=results/riccio_vit256_features OUT_DIR=results/my_run ./scripts/train_paper_classification_xlstm_vit256.sh

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

KAGGLE_DIR="${KAGGLE_DIR:-results/riccio_vit256_features}"
KAGGLE_STEM="${KAGGLE_STEM:-riccio_realtime_exercise_recognition}"
OUT_DIR="${OUT_DIR:-results/paper_xlstm_vit256}"
PYTHON="${PYTHON:-./venv/bin/python}"

exec "$PYTHON" train_paper_classification.py \
  --kaggle-angles-dir "$KAGGLE_DIR" \
  --kaggle-stem "$KAGGLE_STEM" \
  --exclude-classes "${EXCLUDE_CLASSES:-hammer curl}" \
  --feature-dim 256 \
  --seq-len 30 \
  --stride 15 \
  --num-classes 4 \
  --xlstm-hidden 256 \
  --xlstm-num-heads 4 \
  --xlstm-block-pattern mmmmmmms \
  --xlstm-conv-kernel-size 4 \
  --xlstm-projection-factor 1.333 \
  --dropout 0.15 \
  --epochs 50 \
  --batch-size 64 \
  --lr 3e-4 \
  --min-lr 3e-6 \
  --weight-decay 1e-4 \
  --grad-clip 1.0 \
  --label-smoothing 0.1 \
  --models xlstm \
  --output-dir "$OUT_DIR"
#
# Optional (causal-friendly readouts — append before output-dir line when tuning):
#   --xlstm-pool last \
#   --xlstm-linear-classifier \
#   --xlstm-input-dropout 0.05 \
