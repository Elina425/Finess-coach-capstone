#!/usr/bin/env bash
# Shared Riccio (Kaggle) paths for pipeline/step_*.sh — source this file, do not run alone.
# Override any variable before sourcing, or export EXERCISE_RECOGNITION_ROOT to your dataset root.

_RICCIO_PIPELINE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RICCIO_REPO_ROOT="$(cd "$_RICCIO_PIPELINE_DIR/.." && pwd)"

# NPZ output from step 3–4 (also input to step 5)
export RICCIO_OUTPUT_DIR="${RICCIO_OUTPUT_DIR:-$RICCIO_REPO_ROOT/results/riccio_realtime_exercise_recognition}"
export RICCIO_STEM="${RICCIO_STEM:-riccio_realtime_exercise_recognition}"

# Step 5 checkpoint / metrics directory
export EXERCISE_BILSTM_OUT="${EXERCISE_BILSTM_OUT:-$RICCIO_REPO_ROOT/results/exercise_bilstm}"

# MediaPipe / step 3–4: --workers 0 uses min(RICCIO_MP_MAX_WORKERS, cpu_count); override pool size:
# export RICCIO_MP_WORKERS=4
export RICCIO_MP_MAX_WORKERS="${RICCIO_MP_MAX_WORKERS:-8}"

# Video dataset root: user-set or newest kagglehub version that looks like Riccio layout
if [[ -z "${EXERCISE_RECOGNITION_ROOT:-}" ]]; then
  _vbase="${KAGGLEHUB_CACHE:-$HOME/.cache/kagglehub}/datasets/riccardoriccio/real-time-exercise-recognition-dataset/versions"
  if [[ -d "$_vbase" ]]; then
    _resolved=""
    while IFS= read -r _d; do
      [[ -z "$_d" || ! -d "$_d" ]] && continue
      if [[ -d "$_d/similar_dataset" || -d "$_d/final_kaggle_with_additional_video" ]]; then
        _resolved="$_d"
      fi
    done < <(find "$_vbase" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | LC_ALL=C sort -V)
    if [[ -n "$_resolved" ]]; then
      export EXERCISE_RECOGNITION_ROOT="$_resolved"
    fi
  fi
else
  export EXERCISE_RECOGNITION_ROOT
fi
