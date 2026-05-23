#!/usr/bin/env bash
# xLSTM[7:1] classifier on the same 30×256 windows as the paper (block pattern mmmmmmmms: 7×m + 1×s).
# Input: vit_backbone NPZ (frame_features). Classification CE only (reg_weight=0).
#
# Note: Paper also mentions DoRA on all linear projections; this trainer updates full xLSTM weights
# (no DoRA hooks in fitness_coach.models.xlstm_model yet). For strict paper parity on adaptation,
# extend the model or freeze+adapter in a follow-up.
#
# From repo root:
#   KAGGLE_DIR=results/riccio_vit256_features ./scripts/run_paper_posepulse_xlstm_vit256.sh
#
# Env: KAGGLE_STEM, OUT_DIR, EXCLUDE_CLASSES (default hammer curl), EPOCHS override, EXTRA_TRAIN

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

KAGGLE_DIR="${KAGGLE_DIR:?Set KAGGLE_DIR to folder with vit_backbone NPZ}"
KAGGLE_STEM="${KAGGLE_STEM:-riccio_realtime_exercise_recognition}"
OUT_DIR="${OUT_DIR:-results/xlstm_paper_vit256}"
PYTHON="${PYTHON:-./venv/bin/python}"

EPOCH_FLAGS=()
if [[ -n "${EPOCHS+x}" && -n "${EPOCHS}" ]]; then
  EPOCH_FLAGS=(--epochs "${EPOCHS}")
fi

# shellcheck disable=SC2086
exec "$PYTHON" train_xlstm_keypoints.py \
  --preset paper_posepulse_vit \
  --kaggle-keypoints-dir "$KAGGLE_DIR" \
  --kaggle-stem "$KAGGLE_STEM" \
  --window 30 \
  --stride 15 \
  --exclude-classes "${EXCLUDE_CLASSES:-hammer curl}" \
  --eval-test \
  --output-dir "$OUT_DIR" \
  --linear-classifier \
  "${EPOCH_FLAGS[@]}" \
  ${EXTRA_TRAIN:-}
