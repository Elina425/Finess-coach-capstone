# Preprocessing benchmark — BiLSTM (Riccio)

Generated: 2026-04-12 20:58 UTC

## Training configuration

| Setting | Value |
|--------|--------|
| Preset | riccio |
| Architecture | cnn_attn |
| Classification only | yes |
| Standardize | yes |
| Eval test | yes |
| Epochs | 73 |
| Kaggle split seed | 42 |
| NPZ stem | `riccio_realtime_exercise_recognition` |
| Default angles dir | `results/run_default` |
| Rich angles dir | `results/run_rich_no_dwt` |

## Test-set metrics (held-out video split)

| Metric | Default preprocess | Rich preprocess | Δ (rich − default) |
|--------|-------------------:|----------------:|-------------------:|
| Accuracy | 0.8575 | 0.8749 | +0.0174 |
| F1 macro | 0.7066 | 0.7497 | +0.0431 |
| F1 weighted | 0.8533 | 0.8723 | +0.0190 |
| Recall macro | 0.7074 | 0.7416 | +0.0342 |
| Precision macro | 0.7297 | 0.7702 | +0.0405 |

### Per-class F1

| Class | Default F1 | Rich F1 | Δ |
|-------|-----------:|--------:|--:|
| barbell biceps curl | 0.8667 | 0.8957 | +0.0291 |
| hammer curl | 0.0769 | 0.2222 | +0.1453 |
| push-up | 0.7972 | 0.8205 | +0.0233 |
| shoulder press | 0.9198 | 0.9206 | +0.0009 |
| squat | 0.8725 | 0.8894 | +0.0169 |

## Artifacts

- Default checkpoint: `/Users/emelkonyan/Finess-coach-capstone-1/results/preprocess_train_benchmark_no_dwt/default/exercise_bilstm_best.pt`
- Rich checkpoint: `/Users/emelkonyan/Finess-coach-capstone-1/results/preprocess_train_benchmark_no_dwt/rich/exercise_bilstm_best.pt`
- Default test metrics JSON: `/Users/emelkonyan/Finess-coach-capstone-1/results/preprocess_train_benchmark_no_dwt/default/test_classification_metrics.json`
- Rich test metrics JSON: `/Users/emelkonyan/Finess-coach-capstone-1/results/preprocess_train_benchmark_no_dwt/rich/test_classification_metrics.json`

## Interpretation

Higher test **accuracy** / **F1 macro** on the **same split seed** supports the claim that the corresponding preprocessing is better **for this classifier**, subject to training noise. If differences are small, repeat with a different `--kaggle-seed` or longer `--epochs`.
