# Preprocessing benchmark — BiLSTM (Riccio)

Generated: 2026-04-13 08:01 UTC

## Training configuration

| Setting | Value |
|--------|--------|
| Preset | riccio |
| Architecture | cnn_attn |
| Classification only | yes |
| Standardize | yes |
| Eval test | yes |
| Epochs | 65 |
| Kaggle split seed | 42 |
| NPZ stem | `riccio_realtime_exercise_recognition` |
| Default angles dir | `results/run_default` |
| Rich angles dir | `results/run_rich_no_dwt` |
| Extra train args | `--balanced-class-weights` |

## Test-set metrics (held-out video split)

| Metric | Default preprocess | Rich preprocess | Δ (rich − default) |
|--------|-------------------:|----------------:|-------------------:|
| Accuracy | 0.8662 | 0.8507 | -0.0156 |
| F1 macro | 0.7468 | 0.7387 | -0.0081 |
| F1 weighted | 0.8645 | 0.8480 | -0.0165 |
| Recall macro | 0.7367 | 0.7277 | -0.0090 |
| Precision macro | 0.7912 | 0.7615 | -0.0296 |

### Per-class F1

| Class | Default F1 | Rich F1 | Δ |
|-------|-----------:|--------:|--:|
| barbell biceps curl | 0.8831 | 0.8559 | -0.0272 |
| hammer curl | 0.2456 | 0.2727 | +0.0271 |
| push-up | 0.7972 | 0.7947 | -0.0025 |
| shoulder press | 0.9281 | 0.9041 | -0.0240 |
| squat | 0.8798 | 0.8659 | -0.0139 |

## Artifacts

- Default checkpoint: `/Users/emelkonyan/Finess-coach-capstone-1/results/preprocess_train_benchmark_no_dwt_balanced/default/exercise_bilstm_best.pt`
- Rich checkpoint: `/Users/emelkonyan/Finess-coach-capstone-1/results/preprocess_train_benchmark_no_dwt_balanced/rich/exercise_bilstm_best.pt`
- Default test metrics JSON: `/Users/emelkonyan/Finess-coach-capstone-1/results/preprocess_train_benchmark_no_dwt_balanced/default/test_classification_metrics.json`
- Rich test metrics JSON: `/Users/emelkonyan/Finess-coach-capstone-1/results/preprocess_train_benchmark_no_dwt_balanced/rich/test_classification_metrics.json`

## Interpretation

Higher test **accuracy** / **F1 macro** on the **same split seed** supports the claim that the corresponding preprocessing is better **for this classifier**, subject to training noise. If differences are small, repeat with a different `--kaggle-seed` or longer `--epochs`.
