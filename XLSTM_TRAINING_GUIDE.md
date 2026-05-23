# xLSTM Exercise Recognition Training Guide

## Overview

This guide explains how to train the xLSTM-based exercise recognition model following the capstone architecture guidelines.

> **Note:** For training on **Google Colab** with EgoExo-Fitness dataset (recommended if you don't have a local GPU), see [`COLAB_SETUP_GUIDE.md`](COLAB_SETUP_GUIDE.md).

## Architecture

```
Video → Frame Sampling → Pose Extraction → Interpolation → xLSTM → Heads → Feedback
                                                                    ↓
                              ┌─────────────────────┬─────────────────────┐
                              ↓                     ↓                     ↓
                       Classification         Quality Score         Gemma-X Feedback
                       (5 exercises)            (0-5 scale)         (natural language)
```

### Key Components

1. **Pose Extractor** (MediaPipe): Extracts 33 landmarks → 13 joint angles
2. **Interpolation** (Chebyshev/Spline): Resamples to fixed length (60-90 frames)
3. **xLSTM Encoder**: Bidirectional temporal modeling with enhanced gating
4. **Multi-task Heads**: Classification + Quality regression
5. **Gemma-X**: Natural language feedback generation

## Quick Start

### Option A: Train on Google Colab (Recommended for most users)

If you don't have a local GPU or sufficient disk space:

1. Open the Colab notebook: `notebooks/colab_xlstm_simple.ipynb`
2. Add your HuggingFace token to Colab secrets (`HF_TOKEN`)
3. Connect to a GPU runtime (Runtime → Change runtime type → GPU)
4. Run all cells sequentially

See [`COLAB_SETUP_GUIDE.md`](COLAB_SETUP_GUIDE.md) for detailed instructions.

### Option B: Local Training

### Step 1: Extract Pose Features (if not already done)

```bash
# Extract pose features from Riccio dataset
python -m fitness_coach.preprocessing.pose_extractor \
    --video path/to/video.mp4 \
    --output-dir results/riccio_features \
    --target-frames 60

# Or batch process multiple videos
python -c "
from fitness_coach.preprocessing.pose_extractor import batch_extract_videos
import csv

# Load video paths from CSV
videos = []
with open('results/riccio_index.csv') as f:
    reader = csv.DictReader(f)
    for row in reader:
        if row.get('video_path'):
            videos.append(row['video_path'])

# Batch extract
batch_extract_videos(videos[:100], 'results/riccio_features')  # First 100 for testing
"
```

### Step 2: Run Training Pipeline

```bash
# Smoke test (fast validation)
python run_xlstm_training.py --smoke-test

# Full training with Riccio data
python run_xlstm_training.py --data-source riccio --epochs 100

# Full training with EgoExo data (has quality labels)
python run_xlstm_training.py --data-source egoexo --feature-type hybrid
```

### Step 3: Run Inference

```bash
python inference_xlstm_complete.py \
    --video path/to/test_video.mp4 \
    --model-path results/xlstm_stage3_optimized/xlstm_best.pt \
    --output-dir results/inference \
    --use-gemma
```

## Training Stages

The pipeline runs 4 stages automatically:

| Stage | Epochs | Batch Size | LR | Target Frames | Purpose |
|-------|--------|------------|-----|---------------|---------|
| 1. Validation | 15 | 32 | 0.001 | 60 | Verify setup works |
| 2. Baseline | 50 | 64 | 0.001 | 60 | Establish performance |
| 3. Optimized | 100 | 64 | 0.0005 | 60 | Best convergence |
| 4. Deep | 150 | 32 | 0.0003 | 90 | Final submission |

## Data Sources

### Riccio Dataset
- **Format**: NPZ files with keypoints and biomechanics
- **Features**: Pose angles (13 dimensions)
- **Labels**: 5 exercise classes
- **Quality**: Limited quality annotations

### EgoExo-Fitness Dataset
- **Format**: CSV with video paths and quality scores
- **Features**: Can use pose-only or hybrid (pose + DINOv3)
- **Labels**: Multiple exercise classes
- **Quality**: Detailed 0-5 quality scores

## Feature Types

### Pose-Only (Recommended for baseline)
- 13 joint angles
- Interpretable and debuggable
- Fast to train

```
Features: [shoulder_L, shoulder_R, elbow_L, elbow_R, hip_L, hip_R, 
           knee_L, knee_R, ankle_L, ankle_R, back, neck, wrist]
Shape: (60, 13)
```

### Hybrid Features (Recommended for final)
- Pose angles (13) + DINOv3 visual embeddings (512 or 768)
- Stronger performance on noisy data
- Requires GPU for visual feature extraction

```
Features: [pose_angles (13) | dinov3_embedding (512)]
Shape: (60, 525) or (60, 781)
```

## Interpolation Methods

### Chebyshev (Default)
- Minimizes Runge oscillation at boundaries
- Mathematically principled for bounded motion
- Best for smooth exercise motions

### Spline
- Piecewise polynomial curves
- Better for noisy data
- More stable numerically

### Linear (Baseline)
- Fast, simple
- Good for debugging
- May miss smooth motion patterns

## Model Configuration

### Recommended Hyperparameters

```python
{
    "input_size": 13,          # Pose angles
    "hidden_size": 128,        # xLSTM hidden dimension
    "num_layers": 2,           # Stack depth
    "dropout": 0.3,            # Regularization
    "bidirectional": True,     # Full context
    "target_frames": 60,       # Sequence length
    "batch_size": 64,          # Training batch size
    "lr": 0.0005,              # Learning rate
    "class_weight": 1.0,       # Classification loss weight
    "quality_weight": 0.5,     # Quality loss weight
}
```

### For Deep Model (Stage 4)

```python
{
    "hidden_size": 256,
    "num_layers": 4,
    "target_frames": 90,
    "lr": 0.0003,
    "batch_size": 32,
}
```

## Nyquist-Shannon Justification

When resampling motion sequences:

1. **Original sampling**: Video at 30 FPS captures motion up to 15 Hz (Nyquist limit)
2. **Exercise motion frequency**: Typical fitness exercises operate at 0.5-3 Hz
3. **Our resampling**: 60 frames over ~30 seconds = 2 FPS effective
4. **Nyquist check**: 2 FPS can represent motion up to 1 Hz safely

For faster motions, increase `target_frames` to 90 or 120.

## Expected Results

| Model | Test Accuracy | Quality RMSE | Training Time |
|-------|---------------|--------------|---------------|
| Stage 1 (15 ep) | ~0.75 | ~1.2 | 5 min |
| Stage 2 (50 ep) | ~0.82 | ~0.9 | 15 min |
| Stage 3 (100 ep) | ~0.85 | ~0.8 | 30 min |
| Stage 4 (150 ep) | ~0.87 | ~0.7 | 60 min |

*Results vary based on dataset and hardware*

## Troubleshooting

### "CUDA out of memory"
- Reduce `--batch-size` to 32 or 16
- Reduce `--target-frames` to 30
- Use `--hidden-size 64` instead of 128

### "Poor convergence"
- Lower learning rate to 0.0001
- Increase `--epochs` to 150
- Try `--interpolation spline` instead of chebyshev

### "Overfitting"
- Increase `--dropout` to 0.4 or 0.5
- Add `--weight-decay 1e-4`
- Use data augmentation (temporal jittering)

## File Structure

```
fitness_coach/
├── models/
│   └── xlstm_model.py         # xLSTM architecture
├── datasets/
│   └── advanced_video_dataset.py  # Data loading
├── preprocessing/
│   ├── interpolation.py       # Chebyshev/spline interpolation
│   └── pose_extractor.py      # MediaPipe pose extraction
├── inference/
│   └── gemma_feedback.py      # Gemma-X feedback generation
└── training/
    └── train_xlstm.py         # Training loop

run_xlstm_training.py          # Main pipeline runner
inference_xlstm_complete.py    # End-to-end inference
```

## Next Steps

1. Run smoke test to verify setup
2. Extract pose features for your dataset
3. Run full training pipeline
4. Evaluate on test videos
5. Integrate Gemma-X for feedback
6. Export model for deployment

## References

- xLSTM Paper: Beck et al. (2024) - "xLSTM: Extended Long Short-Term Memory"
- MediaPipe Pose: https://developers.google.com/mediapipe/solutions/vision/pose_landmarker
- Nyquist-Shannon: https://en.wikipedia.org/wiki/Nyquist%E2%80%93Shannon_sampling_theorem
- Chebyshev Interpolation: Trefethen (2000) - "Spectral Methods in MATLAB"
