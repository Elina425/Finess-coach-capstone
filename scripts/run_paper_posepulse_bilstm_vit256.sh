#!/usr/bin/env bash
# BiLSTM–CNN (paper §Model): 1× BiLSTM 4/dir → (1,30,8) Conv2D tower, AdamW, cosine LR, CE+label smoothing,
# DoRA rank 8 on the final classifier Linear. Input: (30, D) windows from vit_backbone NPZ (D≈256 ViTPose-S export).
#
# From repo root:
#   KAGGLE_DIR=results/riccio_vit256_features ./scripts/run_paper_posepulse_bilstm_vit256.sh
#
# Env:
#   KAGGLE_DIR       directory with {stem}_biomechanics.npz and {stem}_labels.npz
#   KAGGLE_STEM      default: riccio_realtime_exercise_recognition
#   OUT_DIR          default: results/exercise_bilstm_paper_vit256
#   EXCLUDE_CLASSES  default: hammer curl  (use EXCLUDE_CLASSES="" for 5-class)
#   EPOCHS           overrides preset 50 if set
#   EXTRA_TRAIN      e.g. --cpu

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

KAGGLE_DIR="${KAGGLE_DIR:?Set KAGGLE_DIR to folder with vit_backbone *_biomechanics.npz}"
KAGGLE_STEM="${KAGGLE_STEM:-riccio_realtime_exercise_recognition}"
OUT_DIR="${OUT_DIR:-results/exercise_bilstm_paper_vit256}"
PYTHON="${PYTHON:-./venv/bin/python}"

EPOCH_FLAGS=()
if [[ -n "${EPOCHS+x}" && -n "${EPOCHS}" ]]; then
  EPOCH_FLAGS=(--epochs "${EPOCHS}")
fi

# shellcheck disable=SC2086
exec "$PYTHON" train_exercise_bilstm.py \
  --preset paper_posepulse_vit \
  --kaggle-angles-dir "$KAGGLE_DIR" \
  --kaggle-stem "$KAGGLE_STEM" \
  --window 30 \
  --stride 15 \
  --exclude-classes "${EXCLUDE_CLASSES:-hammer curl}" \
  --eval-test \
  --output-dir "$OUT_DIR" \
  "${EPOCH_FLAGS[@]}" \
  ${EXTRA_TRAIN:-}
