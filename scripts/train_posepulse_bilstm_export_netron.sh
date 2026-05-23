#!/usr/bin/env bash
# Final PosePulse BiLSTM→CNN training (4 classes: no hammer curl) + ONNX for Netron.
#
# - Uses --preset posepulse (mixed 42-D, standardize, window label first, Adam, DoRA heads).
# - Drops hammer curl explicitly via --exclude-classes (matches trainer default; kept for clarity).
#
# From repo root:
#   ./scripts/train_posepulse_bilstm_export_netron.sh
#
# Env overrides:
#   KAGGLE_DIR   NPZ folder (default: results/riccio_realtime_exercise_recognition)
#   KAGGLE_STEM  file prefix (default: riccio_realtime_exercise_recognition)
#   OUT_DIR      checkpoints + ONNX (default: results/exercise_bilstm_posepulse_final)
#   EPOCHS       (default: 40)
#   EXCLUDE_CLASSES  comma-separated coarse names to omit (default: hammer curl)
#   EXTRA_TRAIN  more flags, e.g. EXTRA_TRAIN="--cpu"  or  EXTRA_TRAIN="--no-dora-head"
#
# Train all Kaggle classes including hammer curl:
#   EXCLUDE_CLASSES="" ./scripts/train_posepulse_bilstm_export_netron.sh

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

KAGGLE_DIR="${KAGGLE_DIR:-results/riccio_realtime_exercise_recognition}"
KAGGLE_STEM="${KAGGLE_STEM:-riccio_realtime_exercise_recognition}"
OUT_DIR="${OUT_DIR:-results/exercise_bilstm_posepulse_final}"
EPOCHS="${EPOCHS:-40}"
# Default exclude hammer curl; to keep all classes: EXCLUDE_CLASSES="" ./scripts/...
if [[ -z "${EXCLUDE_CLASSES+x}" ]]; then
  EXCLUDE_CLASSES="hammer curl"
fi

PYTHON="${PYTHON:-python3}"
CKPT="${OUT_DIR}/exercise_bilstm_best.pt"
ONNX_OUT="${OUT_DIR}/exercise_bilstm_netron.onnx"

echo "==> Training (posepulse, excluding: ${EXCLUDE_CLASSES:-<none>}) → ${OUT_DIR}"
# shellcheck disable=SC2086
${PYTHON} train_exercise_bilstm.py \
  --preset posepulse \
  --eval-test \
  --kaggle-angles-dir "${KAGGLE_DIR}" \
  --kaggle-stem "${KAGGLE_STEM}" \
  --exclude-classes "${EXCLUDE_CLASSES}" \
  --epochs "${EPOCHS}" \
  --output-dir "${OUT_DIR}" \
  ${EXTRA_TRAIN:-}

if [[ ! -f "${CKPT}" ]]; then
  echo "ERROR: checkpoint not found: ${CKPT}" >&2
  exit 1
fi

echo "==> Exporting ONNX for Netron → ${ONNX_OUT}"
${PYTHON} -c "import onnx" 2>/dev/null || {
  echo "Installing onnx (required for export)..." >&2
  ${PYTHON} -m pip install -q onnx
}

${PYTHON} export_bilstm_onnx_for_netron.py \
  --checkpoint "${CKPT}" \
  --output "${ONNX_OUT}"

echo ""
echo "Netron:"
echo "  Open https://netron.app and upload:"
echo "    ${ROOT}/${ONNX_OUT}"
echo "  Or from repo:  open -a Netron \"${ONNX_OUT}\"   (if Netron desktop is installed)"
