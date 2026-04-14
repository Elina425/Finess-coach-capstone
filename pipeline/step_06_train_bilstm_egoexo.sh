#!/usr/bin/env bash
# OPTIONAL — not part of the Riccio pipeline. EgoExo-Fitness: frame folders → angles → BiLSTM.
# Step 6: EgoExo-Fitness — MediaPipe angles from released frame folders, then BiLSTM
# (classification + quality regression; CSV quality is raw scale from build_egoexo_fitness_index --quality-scale raw).
# Text: train_exercise_bilstm enables --text-supervision by default; use --no-text-supervision to disable.
#
# ST-GCN: needs {stem}_keypoints.npz. Set EGOEXO_SAVE_KEYPOINTS=1 so batch_compute_angles_for_index.py
# writes them next to *_biomechanics.npz, then e.g.:
#   $PY train_exercise_stgcn.py --index-csv "$SPLIT" --keypoints-dir "$ANGLES" --output-dir results/stgcn_egoexo
#
# Prereqs:
#   - Hugging Face EgoExo-Fitness snapshot under data/EgoExo-Fitness (frames + annotations)
#   - results/egoexo_fitness_index.csv from build_egoexo_fitness_index.py (or use smoke CSV)
#
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
PY="${ROOT}/venv/bin/python"

INDEX="${INDEX_CSV:-results/egoexo_fitness_index.csv}"
SPLIT="${INDEX%.csv}_split.csv"
ANGLES="${EGOEXO_ANGLES_DIR:-results/egoexo_exercise_angles}"
OUT="${EGOEXO_BILSTM_DIR:-results/exercise_bilstm_egoexo}"
VIEW="${EGOEXO_VIEW:-ego_l}"
DATA_ROOT="${EGOEXO_DATA_ROOT:-data/EgoExo-Fitness}"

if [[ ! -d "$DATA_ROOT/frames_open" ]]; then
  echo "" >&2
  echo "ERROR: No frame folders under $DATA_ROOT/frames_open" >&2
  echo "  An annotations-only HF download has raw_annotations/ but not frames (tens of GB)." >&2
  echo "  Download frames, e.g. full snapshot (needs disk space + HF token):" >&2
  echo "    $PY download_egoexo_fitness_dataset.py --local-dir $DATA_ROOT" >&2
  echo "  Or a smaller dev subset (example — adjust pattern to one record id):" >&2
  echo "    $PY download_egoexo_fitness_dataset.py --local-dir $DATA_ROOT \\" >&2
  echo "      --allow-patterns 'frames_open/ThEnUZ/**' --allow-patterns 'raw_annotations/**'" >&2
  echo "" >&2
  exit 1
fi

echo "== Split index: $INDEX -> $SPLIT"
"$PY" split_exercise_index.py \
  --input "$INDEX" \
  --output "$SPLIT" \
  --test-ratio 0.15 --val-ratio 0.15 --seed 42

echo "== Angles (frame folders; view=$VIEW; data=$DATA_ROOT)"
SAVE_KP=()
if [[ "${EGOEXO_SAVE_KEYPOINTS:-0}" == "1" ]]; then
  SAVE_KP=(--save-keypoints)
  echo " (also writing *_keypoints.npz for ST-GCN)"
fi
"$PY" batch_compute_angles_for_index.py \
  --index-csv "$SPLIT" \
  --dataset-root "$DATA_ROOT" \
  --output-dir "$ANGLES" \
  --egoexo-view "$VIEW" \
  --max-videos "${MAX_VIDEOS:-0}" \
  --max-frames "${MAX_FRAMES:-0}" \
  "${SAVE_KP[@]}"

echo "== Train BiLSTM + quality head (+ coaching text)"
# Default --feature-mode angles (8-D per frame): only *_biomechanics.npz. Ablation vs raw coords:
#   EGOEXO_SAVE_KEYPOINTS=1 … batch step, then --feature-mode coords --keypoints-dir "$ANGLES"
# (or mixed). Add --eval-test if the split CSV has both val and test rows.
TRAIN_EXTRA=()
[[ "${EVAL_TEST:-0}" == "1" ]] && TRAIN_EXTRA+=(--eval-test)
"$PY" train_exercise_bilstm.py \
  --index-csv "$SPLIT" \
  --angles-dir "$ANGLES" \
  --standardize \
  --output-dir "$OUT" \
  --epochs "${EPOCHS:-30}" \
  "${TRAIN_EXTRA[@]}"

echo "Done. Checkpoint: $OUT/exercise_bilstm_best.pt"
