#!/usr/bin/env bash
# Train BiLSTM and xLSTM on Riccio exports where pose used YOLO26 (angles / mixed from keypoints).
#
# Requires under YOLO26_DIR:
#   {STEM}_biomechanics.npz   (key `angles` when representation=angles)
#   {STEM}_labels.npz         (pose, video_id)
# For --feature-mode mixed also:
#   {STEM}_keypoints.npz      (from riccio_kaggle_video_pipeline without --skip-keypoints)
#
# Usage:
#   chmod +x scripts/train_yolo26_riccio_bilstm_xlstm.sh
#   ./scripts/train_yolo26_riccio_bilstm_xlstm.sh
#   YOLO26_DIR=/path/to/dir STEM=my_stem ./scripts/train_yolo26_riccio_bilstm_xlstm.sh
#
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export PYTHONPATH="$ROOT"
PY="${ROOT}/venv/bin/python"

YOLO26_DIR="${YOLO26_DIR:-$ROOT/results/riccio_realtime_exercise_recognition_yolo26}"
STEM="${STEM:-riccio_realtime_exercise_recognition}"
FEATURE="${FEATURE:-angles}"   # angles | mixed

echo "PYTHONPATH=$ROOT"
echo "Data dir: $YOLO26_DIR"
echo "NPZ stem: $STEM"
echo "Feature mode: $FEATURE"
echo ""

if [[ "$FEATURE" != "angles" && "$FEATURE" != "mixed" ]]; then
  echo "FEATURE must be angles or mixed (got: $FEATURE)" >&2
  exit 1
fi

# --- BiLSTM (ExerciseBiLSTM CNN head in train_exercise_bilstm.py)
echo "=== BiLSTM (train_exercise_bilstm.py) ==="
if [[ "$FEATURE" == "angles" ]]; then
  "$PY" "$ROOT/fitness_coach/training/train_exercise_bilstm.py" \
    --preset riccio --standardize --eval-test \
    --kaggle-angles-dir "$YOLO26_DIR" --kaggle-stem "$STEM" \
    --feature-mode angles \
    --window-label first \
    --exclude-classes "hammer curl" \
    --epochs 50 \
    --output-dir "$ROOT/results/bilstm_yolo26_${STEM}_angles"
else
  "$PY" "$ROOT/fitness_coach/training/train_exercise_bilstm.py" \
    --preset posepulse --standardize --eval-test \
    --kaggle-angles-dir "$YOLO26_DIR" --kaggle-stem "$STEM" \
    --feature-mode mixed \
    --window-label first \
    --exclude-classes "hammer curl" \
    --epochs 50 \
    --output-dir "$ROOT/results/bilstm_yolo26_${STEM}_mixed"
fi

echo ""
echo "=== xLSTM (train_xlstm_keypoints.py) ==="
if [[ "$FEATURE" == "angles" ]]; then
  "$PY" "$ROOT/train_xlstm_keypoints.py" \
    --kaggle-keypoints-dir "$YOLO26_DIR" \
    --kaggle-stem "$STEM" \
    --feature-mode angles \
    --preset riccio \
    --window-label first \
    --exclude-classes "hammer curl" \
    --reg-weight 0 \
    --epochs 50 \
    --output-dir "$ROOT/results/xlstm_yolo26_${STEM}_angles"
else
  "$PY" "$ROOT/train_xlstm_keypoints.py" \
    --kaggle-keypoints-dir "$YOLO26_DIR" \
    --kaggle-stem "$STEM" \
    --feature-mode mixed \
    --preset posepulse \
    --window-label first \
    --exclude-classes "hammer curl" \
    --reg-weight 0 \
    --epochs 50 \
    --output-dir "$ROOT/results/xlstm_yolo26_${STEM}_mixed"
fi

echo ""
echo "Done."
