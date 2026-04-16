#!/usr/bin/env bash
# Steps 3 + 4: Riccio videos → NPZs (biomechanics + labels + optional keypoints).
# Defaults: output-dir / stem from riccio_env.sh; dataset from EXERCISE_RECOGNITION_ROOT or kagglehub.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
source "$(dirname "${BASH_SOURCE[0]}")/riccio_env.sh"

if [[ -z "${EXERCISE_RECOGNITION_ROOT:-}" ]]; then
  echo "step_03_04: EXERCISE_RECOGNITION_ROOT is unset. Use --download or --dataset-root PATH (or set EXERCISE_RECOGNITION_ROOT before running)." >&2
fi

_extra=(
  --output-dir "$RICCIO_OUTPUT_DIR"
  --output-stem "$RICCIO_STEM"
)
if [[ -n "${EXERCISE_RECOGNITION_ROOT:-}" ]]; then
  _extra+=(--dataset-root "$EXERCISE_RECOGNITION_ROOT")
fi
# Default-on full stack from riccio_env.sh (override: RICCIO_USE_RICH_PREPROCESS=0)
if [[ "${RICCIO_USE_RICH_PREPROCESS:-1}" != "0" ]]; then
  _extra+=(--rich-preprocess)
fi
_mpv="${RICCIO_MAX_VIDEOS:-0}"
if [[ -n "$_mpv" ]] && [[ "$_mpv" =~ ^[0-9]+$ ]] && [[ "$_mpv" -gt 0 ]]; then
  _extra+=(--max-videos "$_mpv")
fi
exec ./venv/bin/python riccio_kaggle_video_pipeline.py "${_extra[@]}" "$@"
