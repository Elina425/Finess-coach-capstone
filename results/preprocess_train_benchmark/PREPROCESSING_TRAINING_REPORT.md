# Preprocessing benchmark — BiLSTM (Riccio)

Generated: 2026-04-12 19:29 UTC

## Training configuration

| Setting | Value |
|--------|--------|
| Preset | riccio |
| Architecture | cnn_attn |
| Classification only | yes |
| Standardize | yes |
| Eval test | yes |
| Epochs | 30 |
| Kaggle split seed | 42 |
| NPZ stem | `riccio_realtime_exercise_recognition` |
| Default angles dir | `results/run_default` |
| Rich angles dir | `results/run_rich` |

## Test-set metrics (held-out video split)

| Metric | Default preprocess | Rich preprocess | Δ (rich − default) |
|--------|-------------------:|----------------:|-------------------:|
| Accuracy | 0.8575 | 0.7627 | -0.0948 |
| F1 macro | 0.6993 | 0.6159 | -0.0834 |
| F1 weighted | 0.8529 | 0.7592 | -0.0937 |
| Recall macro | 0.7022 | 0.6165 | -0.0857 |
| Precision macro | 0.7162 | 0.6196 | -0.0966 |

### Per-class F1

| Class | Default F1 | Rich F1 | Δ |
|-------|-----------:|--------:|--:|
| barbell biceps curl | 0.8656 | 0.7569 | -0.1087 |
| hammer curl | 0.0400 | 0.0000 | -0.0400 |
| push-up | 0.7933 | 0.7165 | -0.0768 |
| shoulder press | 0.9321 | 0.8492 | -0.0830 |
| squat | 0.8654 | 0.7570 | -0.1084 |

## Artifacts

- Default checkpoint: `/Users/emelkonyan/Finess-coach-capstone-1/results/preprocess_train_benchmark/default/exercise_bilstm_best.pt`
- Rich checkpoint: `/Users/emelkonyan/Finess-coach-capstone-1/results/preprocess_train_benchmark/rich/exercise_bilstm_best.pt`
- Default test metrics JSON: `/Users/emelkonyan/Finess-coach-capstone-1/results/preprocess_train_benchmark/default/test_classification_metrics.json`
- Rich test metrics JSON: `/Users/emelkonyan/Finess-coach-capstone-1/results/preprocess_train_benchmark/rich/test_classification_metrics.json`

## Interpretation

Higher test **accuracy** / **F1 macro** on the **same split seed** supports the claim that the corresponding preprocessing is better **for this classifier**, subject to training noise. If differences are small, repeat with a different `--kaggle-seed` or longer `--epochs`.
