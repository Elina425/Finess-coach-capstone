#!/usr/bin/env bash
# Download EgoExo-Fitness raw_annotations (and small files) only — fits small disks / Cloud Shell.
# Requires: HF_TOKEN or huggingface-cli login; dataset terms accepted on the Hub.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
PY="${ROOT}/venv/bin/python"
DEST="${EGOEXO_LOCAL_DIR:-data/EgoExo-Fitness}"

if [[ -z "${HF_TOKEN:-}" ]] && ! huggingface-cli whoami &>/dev/null; then
  echo "Set HF_TOKEN or run: huggingface-cli login" >&2
  exit 1
fi

exec "$PY" download_egoexo_fitness_dataset.py \
  --annotations-only \
  --local-dir "$DEST" \
  --min-free-gb "${MIN_FREE_GB:-0}"
