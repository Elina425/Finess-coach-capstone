#!/usr/bin/env bash
# Train exercise BiLSTM from outputs of fitness_coach/pipelines/riccio_kaggle_video_pipeline.py
# (merged *_biomechanics.npz + *_labels.npz in one directory).
#
# Usage:
#   ./scripts/train_from_riccio_export.sh <mode> <kaggle_angles_dir> <stem> [extra args to train_exercise_bilstm.py]
#
# Modes:
#   angles         — Row 1 → angles in NPZ keys `angles`  (representation=angles export)
#   mixed          — needs same stem *_keypoints.npz; use --keypoints-dir (defaults to kaggle dir)
#   vit            — ViTPose-style frame_features  (--preset paper_posepulse_vit)
#   resnet         — ResNet frame_features (--preset paper_posepulse_resnet)
#
# Examples:
#   ./scripts/train_from_riccio_export.sh angles ./results/riccio_angles_out riccio_realtime_exercise_recognition
#   ./scripts/train_from_riccio_export.sh resnet ./results/riccio_resnet50_features riccio_realtime_exercise_recognition
#
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PY="${ROOT}/venv/bin/python"
TRAIN="${ROOT}/fitness_coach/training/train_exercise_bilstm.py"

MODE="${1:?mode: angles|mixed|vit|resnet}"
DIR="${2:?path to folder containing {stem}_biomechanics.npz}"
STEM="${3:?stem prefix, same as riccio --output-stem}"
shift 3 || true

case "$MODE" in
  angles)
    exec "$PY" "$TRAIN" --preset riccio --standardize --eval-test \
      --kaggle-angles-dir "$DIR" --kaggle-stem "$STEM" \
      --feature-mode angles --epochs 30 \
      "$@"
    ;;
  mixed)
    exec "$PY" "$TRAIN" --preset posepulse --standardize --eval-test \
      --kaggle-angles-dir "$DIR" --kaggle-stem "$STEM" \
      --feature-mode mixed --keypoints-dir "$DIR" \
      --epochs 30 \
      "$@"
    ;;
  vit)
    exec "$PY" "$TRAIN" --preset paper_posepulse_vit --standardize --eval-test \
      --kaggle-angles-dir "$DIR" --kaggle-stem "$STEM" \
      --feature-mode vit_backbone --epochs 50 \
      "$@"
    ;;
  resnet)
    exec "$PY" "$TRAIN" --preset paper_posepulse_resnet --standardize --eval-test \
      --kaggle-angles-dir "$DIR" --kaggle-stem "$STEM" \
      --feature-mode resnet_backbone --epochs 50 \
      "$@"
    ;;
  *)
    echo "Unknown mode: $MODE (use angles|mixed|vit|resnet)" >&2
    exit 1
    ;;
esac
