#!/usr/bin/env bash
# Train xLSTM on the same Kaggle/Riccio NPZ protocol as BiLSTM --preset posepulse:
# mixed (T,42), train-only standardization, window label = first frame, exclude hammer curl,
# AdamW, hidden 256, Beck xLSTM[7:1] stack (pattern mmmmmmmms: 7 mLSTM + 1 sLSTM), CE-only (reg_weight=0).
#
# Usage:
#   export KAGGLE_DIR=/path/to/folder/with/riccio_*_npz
#   ./scripts/train_xlstm_posepulse_parity.sh
#
# Optional: add --linear-classifier for a single Linear head after pool+dropout.

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

: "${KAGGLE_DIR:?Set KAGGLE_DIR to the directory containing {stem}_biomechanics.npz and labels}"

STEM="${KAGGLE_STEM:-riccio_realtime_exercise_recognition}"
OUT="${OUTPUT_DIR:-results/xlstm_posepulse_parity}"

PYTHON="${PYTHON:-./venv/bin/python}"
exec "$PYTHON" train_xlstm_keypoints.py \
  --preset posepulse \
  --kaggle-keypoints-dir "$KAGGLE_DIR" \
  --kaggle-stem "$STEM" \
  --output-dir "$OUT" \
  --epochs "${EPOCHS:-80}" \
  --eval-test \
  "$@"
