#!/usr/bin/env bash
# Build Kaggle/Riccio NPZs with ViTPose-S-style 256-D frame_features (YOLO crop → backbone).
# Prerequisites: Ultralytics YOLO26-Pose, extracted Riccio-style video tree under DATASET_ROOT.
#
# From repo root:
#   DATASET_ROOT=/path/to/riccio_like_tree ./scripts/run_paper_posepulse_export_vit256_npz.sh
#
# Outputs (default OUT_DIR/stem_*):
#   *_biomechanics.npz  (frame_features + labels merged by pipeline)
#   *_labels.npz

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

PYTHON="${PYTHON:-./venv/bin/python}"
DATASET_ROOT="${DATASET_ROOT:?Set DATASET_ROOT to folder with per-exercise video subdirs}"
OUT_DIR="${OUT_DIR:-results/riccio_vit256_features}"
STEM="${NPZ_STEM:-riccio_realtime_exercise_recognition}"

mkdir -p "$OUT_DIR"

echo "==> NPZ export: vit_backbone → $OUT_DIR (stem=$STEM)"
"$PYTHON" fitness_coach/pipelines/riccio_kaggle_video_pipeline.py \
  --dataset-root "$DATASET_ROOT" \
  --output-dir "$OUT_DIR" \
  --output-stem "$STEM" \
  --representation vit_backbone \
  --vit-encoder paper \
  --pose-backend yolo26 \
  --yolo-pose-model "${YOLO_POSE_MODEL:-yolo26n-pose.pt}" \
  "${EXTRA_EXPORT:-}"

echo "Done. Train with:"
echo "  KAGGLE_DIR=$OUT_DIR ./scripts/run_paper_posepulse_bilstm_vit256.sh"
