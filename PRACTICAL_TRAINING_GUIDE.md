# Practical Training Guide: Fitness Coach Capstone

## Quick Start

### Prerequisites
```bash
cd /Users/emelkonyan/Finess-coach-capstone-1
source venv/bin/activate
```

### Verify Data Availability
```bash
# Check Riccio dataset (primary training data)
ls -lh results/riccio_realtime_exercise_recognition/
# Expected files:
#   - riccio_realtime_exercise_recognition_biomechanics.npz
#   - riccio_realtime_exercise_recognition_keypoints.npz
#   - riccio_realtime_exercise_recognition_labels.npz

# Check EgoExo dataset (enhancement data)
head results/egoexo_fitness_index.csv
```

---

## Training Stages

### Stage 1: Quick Validation (15 epochs)
**Purpose:** Verify pipeline works end-to-end  
**Time:** ~5-10 minutes  
**Expected Result:** test_acc ≈ 0.80-0.85

```bash
./venv/bin/python train_exercise_bilstm.py \
  --preset riccio \
  --standardize \
  --eval-test \
  --epochs 15 \
  --batch-size 54 \
  --lr 0.001 \
  --kaggle-angles-dir results/riccio_realtime_exercise_recognition \
  --kaggle-stem riccio_realtime_exercise_recognition
```

**Output:** `results/exercise_bilstm/`

---

### Stage 2: Baseline Model (50 epochs)
**Purpose:** Establish performance baseline  
**Time:** ~20-30 minutes  
**Expected Result:** test_acc ≈ 0.83-0.86

```bash
mkdir -p results/exercise_bilstm_baseline
./venv/bin/python train_exercise_bilstm.py \
  --preset riccio \
  --standardize \
  --eval-test \
  --epochs 50 \
  --batch-size 64 \
  --lr 0.001 \
  --kaggle-angles-dir results/riccio_realtime_exercise_recognition \
  --kaggle-stem riccio_realtime_exercise_recognition \
  --output-dir results/exercise_bilstm_baseline
```

**Output:** `results/exercise_bilstm_baseline/`

---

### Stage 3: Hyperparameter Tuning (100 epochs, lower LR)
**Purpose:** Stabilize training, improve convergence  
**Time:** ~40-60 minutes  
**Expected Result:** test_acc ≈ 0.84-0.87

```bash
mkdir -p results/exercise_bilstm_tuned
./venv/bin/python train_exercise_bilstm.py \
  --preset riccio \
  --standardize \
  --eval-test \
  --epochs 100 \
  --batch-size 64 \
  --lr 0.0005 \
  --kaggle-angles-dir results/riccio_realtime_exercise_recognition \
  --kaggle-stem riccio_realtime_exercise_recognition \
  --output-dir results/exercise_bilstm_tuned
```

**Output:** `results/exercise_bilstm_tuned/`

---

### Stage 4: GCN Model (Graph Convolutional Network)
**Purpose:** Capture skeletal topology, validate different architecture  
**Time:** ~30-40 minutes  
**Expected Result:** test_acc ≈ 0.80-0.86

```bash
mkdir -p results/gcn_pose_supervised
./venv/bin/python train_gcn_supervised.py \
  --preset riccio \
  --epochs 50 \
  --batch-size 32 \
  --lr 0.001 \
  --kaggle-angles-dir results/riccio_realtime_exercise_recognition \
  --kaggle-stem riccio_realtime_exercise_recognition \
  --output-dir results/gcn_pose_supervised
```

**Output:** `results/gcn_pose_supervised/`

---

### Stage 5: Ensemble Evaluation
**Purpose:** Compare models, select best for production  

```bash
./venv/bin/python model_comparison.py \
  --models-dir results/ \
  --models bilstm_baseline,bilstm_tuned,gcn_pose_supervised
```

---

## Results Analysis

### Inspect Metrics
```bash
# View classification metrics
cat results/exercise_bilstm_baseline/test_classification_metrics.json | jq .

# View confusion matrix
python -c "
import json
with open('results/exercise_bilstm_baseline/test_classification_metrics.json') as f:
    data = json.load(f)
    print('Confusion Matrix:')
    print(data.get('confusion_matrix', 'N/A'))
"
```

### Generate Visualizations
```bash
python visualize_confusion_roc.py \
  --metrics-file results/exercise_bilstm_baseline/test_classification_metrics.json \
  --output-dir results/exercise_bilstm_baseline/plots
```

### Compare Across Stages
```bash
python -c "
import json
import os

results = {}
for stage in ['exercise_bilstm', 'exercise_bilstm_baseline', 'exercise_bilstm_tuned']:
    metrics_file = f'results/{stage}/test_classification_metrics.json'
    if os.path.exists(metrics_file):
        with open(metrics_file) as f:
            data = json.load(f)
            results[stage] = {
                'test_accuracy': data.get('test_accuracy'),
                'f1_macro': data.get('f1_macro'),
                'quality_rmse': data.get('test_quality_rmse')
            }

for stage, metrics in results.items():
    print(f'\n{stage}:')
    for key, val in metrics.items():
        print(f'  {key}: {val:.4f}')
"
```

---

## Full Multi-Stage Pipeline

Run all stages sequentially:

```bash
cat > run_full_training.py << 'EOF'
#!/usr/bin/env python3
import subprocess
import sys
from pathlib import Path
import json

workspace = Path.cwd()

def run_stage(name, output_dir, cmd_args):
    """Run a training stage."""
    print(f"\n{'='*70}")
    print(f"STAGE: {name}")
    print(f"{'='*70}\n")
    
    output_dir = workspace / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    
    cmd = ["./venv/bin/python", "train_exercise_bilstm.py"] + cmd_args
    result = subprocess.run(cmd, cwd=str(workspace))
    
    if result.returncode != 0:
        print(f"\n✗ {name} failed!")
        return False
    
    # Read and display metrics
    metrics_file = output_dir / "test_classification_metrics.json"
    if metrics_file.exists():
        with open(metrics_file) as f:
            metrics = json.load(f)
            print(f"\n✅ {name} Results:")
            print(f"   Accuracy: {metrics.get('test_accuracy', 'N/A'):.4f}")
            print(f"   F1 (macro): {metrics.get('f1_macro', 'N/A'):.4f}")
            print(f"   Quality RMSE: {metrics.get('test_quality_rmse', 'N/A'):.4f}")
    
    return True

# Configuration
riccio_dir = "results/riccio_realtime_exercise_recognition"
riccio_stem = "riccio_realtime_exercise_recognition"

stages = [
    ("Baseline (50 epochs)", "results/exercise_bilstm_baseline", [
        "--preset", "riccio",
        "--standardize",
        "--eval-test",
        "--epochs", "50",
        "--batch-size", "64",
        "--lr", "0.001",
        "--kaggle-angles-dir", riccio_dir,
        "--kaggle-stem", riccio_stem,
        "--output-dir", "results/exercise_bilstm_baseline",
    ]),
    ("Tuned (100 epochs, 0.0005 LR)", "results/exercise_bilstm_tuned", [
        "--preset", "riccio",
        "--standardize",
        "--eval-test",
        "--epochs", "100",
        "--batch-size", "64",
        "--lr", "0.0005",
        "--kaggle-angles-dir", riccio_dir,
        "--kaggle-stem", riccio_stem,
        "--output-dir", "results/exercise_bilstm_tuned",
    ]),
]

print("\n" + "="*70)
print("FULL MULTI-STAGE TRAINING PIPELINE")
print("="*70)

for name, output_dir, cmd_args in stages:
    if not run_stage(name, output_dir, cmd_args):
        sys.exit(1)

print("\n" + "="*70)
print("✅ ALL STAGES COMPLETE")
print("="*70)
print("\n📊 Final Results Summary:")
print(f"  Baseline: results/exercise_bilstm_baseline/test_classification_metrics.json")
print(f"  Tuned: results/exercise_bilstm_tuned/test_classification_metrics.json")

EOF

chmod +x run_full_training.py
./venv/bin/python run_full_training.py
```

---

## Key Metrics to Track

| Metric | Target | Interpretation |
|--------|--------|-----------------|
| **Test Accuracy** | 0.84+ | Overall classification correctness |
| **F1 (macro)** | 0.75+ | Per-class balance (important for imbalanced data) |
| **Quality RMSE** | <0.05 | How well model predicts form quality |
| **Per-class F1** | >0.70 | Individual exercise detection rates |

---

## Troubleshooting

### Out of Memory
```bash
# Reduce batch size
--batch-size 32

# Reduce sequence length (if applicable)
--max-seq-length 64
```

### Poor Validation Accuracy
```bash
# Try higher learning rate
--lr 0.01

# Or lower learning rate with more epochs
--lr 0.0001 --epochs 200
```

### Model Not Converging
```bash
# Add dropout (if supported)
--dropout 0.3

# Use different optimizer preset
--preset riccio-dropout
```

---

## Next Steps

1. **Run Stage 1** (validation) → verify setup works
2. **Run Stage 2** (baseline) → establish performance baseline
3. **Run Stage 3** (tuned) → optimize hyperparameters
4. **Compare Results** → select best model
5. **Deploy** → use best model for inference

---

## Production Deployment

Once satisfied with results:

```bash
# Copy best model
cp results/exercise_bilstm_baseline/exercise_bilstm_best.pt \
   models/exercise_bilstm_production.pt

# Use for inference
python inference_exercise_bilstm.py \
  --model-path models/exercise_bilstm_production.pt \
  --video-path /path/to/video.mp4 \
  --output-json results/predictions.json
```

---

## References

- Model architectures: `fitness_coach/models/`
- Training code: `train_exercise_bilstm.py`, `train_exercise_stgcn.py`
- Inference: `inference_exercise_bilstm.py`
- Dataset utilities: `fitness_coach/datasets/`

