#!/usr/bin/env bash
# Step 2: pose backend comparison on Riccio videos (uses EXERCISE_RECOGNITION_ROOT or kagglehub cache).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
source "$(dirname "${BASH_SOURCE[0]}")/riccio_env.sh"

if [[ -z "${EXERCISE_RECOGNITION_ROOT:-}" ]]; then
  echo "step_02: EXERCISE_RECOGNITION_ROOT is unset (no Riccio folder found under kagglehub cache). Comparison uses run_full_comparison defaults (env EXERCISE_RECOGNITION_ROOT or ./data). Set export EXERCISE_RECOGNITION_ROOT=... or pass --dataset-root ..." >&2
fi

_extra=()
if [[ -n "${EXERCISE_RECOGNITION_ROOT:-}" ]]; then
  _extra+=(--dataset-root "$EXERCISE_RECOGNITION_ROOT")
fi
exec ./venv/bin/python run_full_comparison.py "${_extra[@]}" "$@"
