#!/usr/bin/env bash
# Export an existing exercise_bilstm_best.pt to ONNX for Netron (https://netron.app).
# Does not run training.
#
# From repo root:
#   ./scripts/export_bilstm_netron.sh
#
# Env overrides:
#   CKPT   checkpoint path (default: results/exercise_bilstm_posepulse_final/exercise_bilstm_best.pt)
#   ONNX   output .onnx path (default: same dir as CKPT, name exercise_bilstm_netron.onnx)

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

CKPT="${CKPT:-${ROOT}/results/exercise_bilstm_posepulse_final/exercise_bilstm_best.pt}"
if [[ -z "${ONNX:-}" ]]; then
  d="$(dirname "${CKPT}")"
  ONNX="${d}/exercise_bilstm_netron.onnx"
fi

PYTHON="${PYTHON:-python3}"

if [[ ! -f "${CKPT}" ]]; then
  echo "ERROR: checkpoint not found: ${CKPT}" >&2
  echo "Set CKPT=path/to/exercise_bilstm_best.pt" >&2
  exit 1
fi

${PYTHON} -c "import onnx" 2>/dev/null || {
  echo "Installing onnx..." >&2
  ${PYTHON} -m pip install -q onnx
}

echo "==> ONNX export"
echo "    checkpoint: ${CKPT}"
echo "    output:     ${ONNX}"
${PYTHON} export_bilstm_onnx_for_netron.py \
  --checkpoint "${CKPT}" \
  --output "${ONNX}"

echo ""
echo "Open in Netron: https://netron.app  →  Open file → ${ONNX}"
