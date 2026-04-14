#!/usr/bin/env bash
# Rich-style without DWT and without bone proportion: Laplacian + Savitzky–Golay only.
# Output: results/run_rich_no_dwt_no_bone/ (override with RICCIO_ABLATION_OUT).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
source "$(dirname "${BASH_SOURCE[0]}")/riccio_env.sh"

OUT="${RICCIO_ABLATION_OUT:-$ROOT/results/run_rich_no_dwt_no_bone}"
STEM="${RICCIO_STEM:-riccio_realtime_exercise_recognition}"

_extra=(--output-dir "$OUT" --output-stem "$STEM" --laplacian-spatial --savgol)
if [[ -n "${EXERCISE_RECOGNITION_ROOT:-}" ]]; then
  _extra+=(--dataset-root "$EXERCISE_RECOGNITION_ROOT")
fi
if [[ -z "${NO_SKIP_KP:-}" ]]; then
  _extra+=(--skip-keypoints)
fi

exec ./venv/bin/python riccio_kaggle_video_pipeline.py "${_extra[@]}" "$@"
