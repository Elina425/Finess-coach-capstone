# Preprocessing benchmark — ST-GCN (Riccio keypoints)

Generated: 2026-04-13 18:40 UTC

## Training configuration

| Setting | Value |
|--------|--------|
| Model | ExerciseSTGCN (COCO-17, xy) |
| Standardize | yes |
| Eval test | yes |
| Epochs | 70 |
| Kaggle split seed | 42 |
| NPZ stem | `riccio_realtime_exercise_recognition` |
| Default keypoints dir | `results/run_default` |
| Rich keypoints dir | `results/run_rich_no_dwt` |
| Extra train args | `--balanced-class-weights` |

## Test-set metrics (held-out video split)

| Metric | Default preprocess | Rich preprocess | Δ (rich − default) |
|--------|-------------------:|----------------:|-------------------:|
| Accuracy | 0.4271 | 0.9701 | +0.5430 |
| F1 macro | 0.2674 | 0.9190 | +0.6516 |
| F1 weighted | 0.2930 | 0.9706 | +0.6776 |
| Recall macro | 0.3871 | 0.9297 | +0.5426 |
| Precision macro | 0.2215 | 0.9103 | +0.6888 |

### Per-class F1

| Class | Default F1 | Rich F1 | Δ |
|-------|-----------:|--------:|--:|
| barbell biceps curl | 0.4463 | 0.9499 | +0.5036 |
| hammer curl | 0.0000 | 0.6885 | +0.6885 |
| push-up | 0.8909 | 0.9845 | +0.0936 |
| shoulder press | 0.0000 | 0.9929 | +0.9929 |
| squat | 0.0000 | 0.9794 | +0.9794 |

## Artifacts

- Default checkpoint: `/Users/emelkonyan/Finess-coach-capstone-1/results/preprocess_stgcn_benchmark/default/exercise_stgcn_best.pt`
- Rich checkpoint: `/Users/emelkonyan/Finess-coach-capstone-1/results/preprocess_stgcn_benchmark/rich/exercise_stgcn_best.pt`
- Default test metrics: `/Users/emelkonyan/Finess-coach-capstone-1/results/preprocess_stgcn_benchmark/default/test_classification_metrics.json`
- Rich test metrics: `/Users/emelkonyan/Finess-coach-capstone-1/results/preprocess_stgcn_benchmark/rich/test_classification_metrics.json`

## Note

ST-GCN uses **2D keypoint** windows, not angle features. Preprocessing affects `*_keypoints.npz` only if the video pipeline was run **without** `--skip-keypoints`.
