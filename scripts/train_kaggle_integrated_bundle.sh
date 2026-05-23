#!/usr/bin/env bash
# Train BiLSTM + xLSTM on results/kaggle_exercise_recognition_integrated/*.npz
# (stem kaggle_exercise_recognition — angles + keypoints + labels in one folder).
#
# Usage:
#   ./scripts/train_kaggle_integrated_bundle.sh [epochs]
#
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"
PY="${ROOT}/venv/bin/python"
DIR="${INTEGRATED_DIR:-${ROOT}/results/kaggle_exercise_recognition_integrated}"
STEM="${INTEGRATED_STEM:-kaggle_exercise_recognition}"
EPOCHS="${1:-30}"

echo "=== BiLSTM (angles, preset riccio) ==="
"$PY" "${ROOT}/fitness_coach/training/train_exercise_bilstm.py" \
  --preset riccio --standardize --eval-test \
  --kaggle-angles-dir "$DIR" \
  --kaggle-stem "$STEM" \
  --feature-mode angles \
  --epochs "$EPOCHS" \
  --output-dir "${ROOT}/results/bilstm_${STEM}_integrated"

echo ""
echo "=== xLSTM (angles, preset riccio) ==="
"$PY" "${ROOT}/train_xlstm_keypoints.py" \
  --preset riccio \
  --feature-mode angles \
  --kaggle-keypoints-dir "$DIR" \
  --kaggle-stem "$STEM" \
  --standardize \
  --epochs "$EPOCHS" \
  --eval-test \
  --output-dir "${ROOT}/results/xlstm_${STEM}_integrated_angles"

echo ""
echo "Optional PosePulse-style mixed (T,42), first-frame window labels:"
echo "  ${PY} ${ROOT}/fitness_coach/training/train_exercise_bilstm.py --preset posepulse --exclude-classes \"\" --standardize --eval-test \\"
echo "    --kaggle-angles-dir \"$DIR\" --kaggle-stem \"$STEM\" --feature-mode mixed --keypoints-dir \"$DIR\""
echo "  ${PY} ${ROOT}/train_xlstm_keypoints.py --preset posepulse --exclude-classes \"\" --feature-mode mixed \\"
echo "    --kaggle-keypoints-dir \"$DIR\" --kaggle-stem \"$STEM\" --eval-test"
