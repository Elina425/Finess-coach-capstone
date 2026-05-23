# xLSTM Training: Quick Summary

## For Your Capstone Project

You need to train the xLSTM model on **Riccio** and **EgoExo-Fitness** datasets using **Google Colab** (no local GPU required).

## Files Created

| File | Purpose |
|------|---------|
| `notebooks/colab_xlstm_simple.ipynb` | **Main Colab notebook** - use this |
| `notebooks/colab_xlstm_egoexo.ipynb` | Alternative Colab notebook with more features |
| `COLAB_SETUP_GUIDE.md` | Detailed Colab setup instructions |
| `prepare_colab_data.py` | Dataset download script for Colab |
| `fitness_coach/preprocessing/pose_extractor.py` | MediaPipe pose extraction |
| `XLSTM_TRAINING_GUIDE.md` | Complete training reference |

## Quick Start (5 minutes)

1. **Open Colab:**
   ```
   https://colab.research.google.com/github/elinelkonyan/fitness-coach-capstone-1/blob/main/notebooks/colab_xlstm_simple.ipynb
   ```

2. **Add HF Token to secrets:**
   - Click 🔑 in left sidebar
   - Add secret: `HF_TOKEN = hf_xxxxx...`
   - Get token from: https://huggingface.co/settings/tokens

3. **Connect to GPU:**
   - Runtime → Change runtime type → GPU → T4

4. **Run all cells** (Runtime → Run all)

## Architecture

```
Video → MediaPipe Pose → Chebyshev Interpolation → xLSTM → Classification + Quality
                                                      ↓
                                    5 exercises       Quality score (0-5)
```

## Training Configuration

### Smoke Test (first run)
```python
SMOKE_TEST = True   # 5 epochs, ~5-10 minutes
```

### Full Training (production model)
```python
SMOKE_TEST = False  # 100 epochs, ~45-90 minutes
```

## Expected Results

| Stage | Accuracy | Time |
|-------|----------|------|
| Smoke test | ~0.60-0.70 | 5-10 min |
| Full training | ~0.80-0.90 | 45-90 min |

## After Training

Model saved to Google Drive:
```
MyDrive/fitness-coach/xlstm_model/
├── xlstm_best.pt          # Model checkpoint
├── class_map.json         # Class mapping
├── test_results.json      # Test metrics
└── training_history.json  # Training curves
```

## Datasets

### EgoExo-Fitness (Primary)
- **HF Repo:** `ego-exo/egoexo-fitness`
- **Features:** Multiple exercises with quality scores
- **Quality:** 0-5 scale annotations

### Riccio (Optional/Supplementary)
- Use if available on HuggingFace
- Otherwise focus on EgoExo-Fitness only

## Key Hyperparameters

```python
{
    "input_size": 13,      # Joint angles
    "hidden_size": 128,
    "num_layers": 2,
    "dropout": 0.3,
    "bidirectional": True,
    "target_frames": 60,   # Sequence length
    "batch_size": 32,
    "lr": 0.0005,
    "interpolation": "chebyshev",
}
```

## Troubleshooting

| Issue | Solution |
|-------|----------|
| CUDA OOM | Reduce `batch_size` to 16 or 8 |
| HF download failed | Check token validity, re-login |
| No videos found | EgoExo may store videos separately |
| Training too slow | Verify GPU is connected |

## Next Steps After Training

1. Download model from Google Drive
2. Run inference on test videos
3. Add Gemma-X feedback generation
4. Export for deployment

## References

- **xLSTM Paper:** Beck et al. (2024) - https://arxiv.org/abs/2405.04517
- **EgoExo-Fitness:** https://huggingface.co/datasets/ego-exo/egoexo-fitness
- **Nyquist-Shannon:** Justifies resampling to 60 frames
- **Chebyshev Interpolation:** Trefethen (2000) - minimizes Runge oscillation
