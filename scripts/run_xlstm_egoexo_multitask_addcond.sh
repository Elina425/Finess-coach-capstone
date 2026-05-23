#!/usr/bin/env bash
# EgoExo multitask training (addcond recipe): run from repo root with venv activated.
#
# Fixed: each `--flag \` must end its line; never put `\ --other-flag` on the same line.
#
# Four-level unit ratings (0.25, 0.5, 0.75, 1.0): add after --quality-head-mode classification:
#   --quality-encoding unit \
#   --comment-quality-buckets 4 \
#   --quality-bucket-edges 0.375,0.625,0.875 \

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-python3}"

nohup "$PYTHON" train_xlstm_egoexo_multitask.py \
  --index-csv notebooks/data/egoexo_fitness_full/egoexo_fitness_index.csv \
  --clip-features-root notebooks/data/egoexo_fitness_full/features_open/visual \
  --clip-view all \
  --clip-max-frames 300 --clip-subsample-stride 3 \
  --window 60 --stride 30 --standardize \
  --hidden 256 --layers 8 --num-heads 4 \
  --block-pattern mmmmmmmm \
  --conv-kernel-size 4 --projection-factor 1.333 \
  --dropout 0.15 \
  --use-attention-pool \
  --teacher-force-quality \
  --mtl-method dwa --dwa-window 25 --dwa-temp 2.0 \
  --filter-null-comments \
  --quality-head-mode classification \
  --quality-encoding unit \
  --comment-quality-buckets 4 \
  --quality-bucket-edges 0.375,0.625,0.875 \
  --freeze-backbone --unfreeze-last-n 6 \
  --epochs 25 --batch-size 32 \
  --lr 3e-4 --min-lr-ratio 0.01 --warmup-frac 0.1 \
  --weight-decay 1e-4 --grad-clip 1.0 --optimizer adamw \
  --error-weight 0.0 \
  --balanced-class-weights \
  --balanced-quality-weights \
  --eval-test \
  --output-dir results/xlstm_egoexo_multitask_addcond \
  > results/xlstm_egoexo_multitask_addcond.log 2>&1 &

echo $! > results/xlstm_egoexo_multitask_addcond.pid
disown
echo "PID: $(cat results/xlstm_egoexo_multitask_addcond.pid)"
