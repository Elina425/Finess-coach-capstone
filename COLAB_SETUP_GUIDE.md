# Google Colab Training Guide for xLSTM Exercise Recognition

## Overview

This guide shows you how to train the xLSTM exercise recognition model on Google Colab using the EgoExo-Fitness dataset from HuggingFace.

## Why Colab?

- **Free GPU access** (T4, 16GB VRAM)
- **No local disk space needed** - everything runs in the cloud
- **Pre-configured environment** - no dependency issues
- **Easy integration** with HuggingFace and Google Drive

## Prerequisites

1. **Google account** (for Colab)
2. **HuggingFace account** with access to EgoExo-Fitness dataset
3. **HF Token** (get from https://huggingface.co/settings/tokens)

## Step-by-Step Instructions

### Step 1: Get HuggingFace Token

1. Go to https://huggingface.co/settings/tokens
2. Click "Create new token"
3. Select "Read" permission (write not needed)
4. Copy the token (starts with `hf_...`)

### Step 2: Open Colab Notebook

Choose one of these notebooks:

| Notebook | Description | Link |
|----------|-------------|------|
| **Simple** | Minimal setup, quick start | `notebooks/colab_xlstm_simple.ipynb` |
| **Full** | Complete pipeline with pose extraction | `notebooks/colab_xlstm_egoexo.ipynb` |

**To open in Colab:**
1. Go to https://colab.research.google.com
2. Click "GitHub" tab
3. Enter: `elinelkonyan/fitness-coach-capstone-1`
4. Select the notebook

Or directly:
```
https://colab.research.google.com/github/elinelkonyan/fitness-coach-capstone-1/blob/main/notebooks/colab_xlstm_simple.ipynb
```

### Step 3: Configure Colab

1. **Add HF Token to secrets:**
   - Click the 🔑 key icon in the left sidebar
   - Click "Create secret"
   - Name: `HF_TOKEN`
   - Value: `hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx`
   - Click "OK"

2. **Connect to GPU:**
   - Click "Runtime" → "Change runtime type"
   - Hardware accelerator: **GPU**
   - GPU type: **T4** (free tier)
   - Click "Save"

### Step 4: Run the Notebook

Execute cells in order (Runtime → Run all):

1. **Install dependencies** (~3-5 minutes)
2. **Clone repository** (~30 seconds)
3. **Login to HuggingFace** (~10 seconds)
4. **Download EgoExo-Fitness** (~10-30 minutes, depends on connection)
5. **View dataset info** (~5 seconds)
6. **Run training** (~15-60 minutes, depends on epochs)
7. **View results** (~5 seconds)
8. **Save to Google Drive** (~30 seconds)

### Step 5: Download Results

After training completes:

1. Model checkpoint saved to your Google Drive:
   ```
   MyDrive/fitness-coach/xlstm_model/xlstm_best.pt
   ```

2. Download the model files to your local machine:
   - `xlstm_best.pt` - Model checkpoint
   - `class_map.json` - Exercise class mapping
   - `test_results.json` - Test metrics

## Training Configuration

### Smoke Test (Recommended First)
```python
SMOKE_TEST = True
EPOCHS = 5
BATCH_SIZE = 16
```
- **Time:** ~5-10 minutes
- **Purpose:** Verify everything works
- **Expected accuracy:** ~0.60-0.70

### Full Training
```python
SMOKE_TEST = False
EPOCHS = 100
BATCH_SIZE = 32
```
- **Time:** ~45-90 minutes
- **Purpose:** Production model
- **Expected accuracy:** ~0.80-0.90

### Hyperparameters

| Parameter | Recommended | Description |
|-----------|-------------|-------------|
| `epochs` | 100 | Training iterations |
| `batch_size` | 32 | Batch size (reduce if OOM) |
| `lr` | 0.0005 | Learning rate |
| `hidden_size` | 128 | xLSTM hidden dimension |
| `num_layers` | 2 | Number of xLSTM layers |
| `target_frames` | 60 | Sequence length |

## Troubleshooting

### "CUDA out of memory"
```python
# Reduce batch size
BATCH_SIZE = 8  # or 16

# Or reduce model size
--hidden-size 64
--num-layers 1
```

### "HuggingFace download failed"
- Check that your HF token is valid
- Verify you have access to the egoexo-fitness dataset
- Try re-running the login cell

### "No videos found"
- EgoExo-Fitness may store videos separately from metadata
- Check the dataset structure on HuggingFace
- You may need to download video files separately

### Training is too slow
- Verify you're using GPU: `torch.cuda.is_available()` should return `True`
- Reduce `target_frames` to 30
- Use smoke test first to verify setup

## Cost Estimate

| Resource | Free Tier | Colab Pro |
|----------|-----------|-----------|
| GPU | T4 (16GB) | V100/A100 |
| Runtime limit | 12 hours | 24 hours |
| Price | $0 | ~$10/month |

For capstone project, **free tier is sufficient**.

## After Training

### Run Inference Locally

```bash
# Download model from Drive to local machine
# Then run:

python inference_xlstm_complete.py \
    --video path/to/test_video.mp4 \
    --model-path xlstm_best.pt \
    --output-dir results/inference \
    --use-gemma
```

### Export for Deployment

```python
import torch
from fitness_coach.models.xlstm_model import xLSTMExerciseClassifier

# Load model
model = xLSTMExerciseClassifier(
    input_size=13,
    hidden_size=128,
    num_layers=2,
    num_classes=5,
    dropout=0.3,
    bidirectional=True
)
model.load_state_dict(torch.load("xlstm_best.pt"))
model.eval()

# Save as TorchScript (for deployment)
scripted_model = torch.jit.script(model)
scripted_model.save("xlstm_deploy.pt")
```

## Dataset Information

### EgoExo-Fitness
- **Source:** https://huggingface.co/datasets/ego-exo/egoexo-fitness
- **Exercises:** Multiple fitness exercises
- **Labels:** Exercise class + quality score (0-5)
- **Split:** Train/Val/Test provided

### Riccio (Optional)
- If available on HuggingFace, add to the download step
- Otherwise, use EgoExo-Fitness only

## Support

If you encounter issues:

1. Check the Colab cell outputs for error messages
2. Verify HF token has "Read" permission
3. Ensure GPU is connected (Runtime → Change runtime type)
4. Try smoke test first before full training

## References

- [xLSTM Paper](https://arxiv.org/abs/2405.04517)
- [EgoExo-Fitness Dataset](https://huggingface.co/datasets/ego-exo/egoexo-fitness)
- [Colab Documentation](https://colab.research.google.com/docs)
