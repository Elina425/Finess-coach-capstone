# Fitness Coach Capstone: Quick Reference Cheat Sheet

## 🚀 Start Here

### 3 Seconds: Activate Environment
```bash
cd /Users/emelkonyan/Finess-coach-capstone-1
source venv/bin/activate
```

### 30 Seconds: Verify Setup Works
```bash
./venv/bin/python train_exercise_bilstm.py \
  --preset riccio --standardize --eval-test \
  --epochs 3 --batch-size 54 --lr 0.001 \
  --kaggle-angles-dir results/riccio_realtime_exercise_recognition \
  --kaggle-stem riccio_realtime_exercise_recognition
```
**Expected:** `test_acc ≈ 0.70-0.75` (3 epochs is very quick) ✓

---

## 📊 Quick Training Commands

### Option A: Individual Stages (Recommended for first run)

**Stage 1: Validation (5 min)**
```bash
./venv/bin/python train_exercise_bilstm.py \
  --preset riccio --standardize --eval-test --epochs 15 \
  --batch-size 54 --lr 0.001 \
  --kaggle-angles-dir results/riccio_realtime_exercise_recognition \
  --kaggle-stem riccio_realtime_exercise_recognition
# Expected: test_acc ≈ 0.84
```

**Stage 2: Baseline (15 min)**
```bash
./venv/bin/python train_exercise_bilstm.py \
  --preset riccio --standardize --eval-test --epochs 50 \
  --batch-size 64 --lr 0.001 \
  --kaggle-angles-dir results/riccio_realtime_exercise_recognition \
  --kaggle-stem riccio_realtime_exercise_recognition \
  --output-dir results/exercise_bilstm_baseline
# Expected: test_acc ≈ 0.85
```

**Stage 3: Optimized (30 min)**
```bash
./venv/bin/python train_exercise_bilstm.py \
  --preset riccio --standardize --eval-test --epochs 100 \
  --batch-size 64 --lr 0.0005 \
  --kaggle-angles-dir results/riccio_realtime_exercise_recognition \
  --kaggle-stem riccio_realtime_exercise_recognition \
  --output-dir results/exercise_bilstm_tuned
# Expected: test_acc ≈ 0.86
```

### Option B: Run All Stages Automatically (1 command!)
```bash
# Complete 4-stage pipeline (takes ~1.5-2 hours)
./venv/bin/python run_complete_training.py
```

---

## 📈 View Results

### Check Latest Metrics
```bash
# Latest results
cat results/exercise_bilstm/test_classification_metrics.json | jq .

# Specific metric
jq '.test_accuracy' results/exercise_bilstm/test_classification_metrics.json
```

### Compare All Models
```bash
python -c "
import json, os
for d in sorted(os.listdir('results')):
    if 'bilstm' in d:
        try:
            m = json.load(open(f'results/{d}/test_classification_metrics.json'))
            print(f'{d:<40} Acc: {m.get(\"test_accuracy\", 0):.4f}  F1: {m.get(\"f1_macro\", 0):.4f}')
        except: pass
"
```

### View Confusion Matrix
```bash
python -c "
import json
m = json.load(open('results/exercise_bilstm/test_classification_metrics.json'))
print(json.dumps(m.get('confusion_matrix', {}), indent=2))
"
```

---

## 🔧 Common Tweaks

### Problem: Low Accuracy (<0.80)
```bash
# Solution 1: Train longer
--epochs 150

# Solution 2: Lower learning rate
--lr 0.0001

# Solution 3: Smaller batch size
--batch-size 32

# Solution 3: Add dropout
# (Need to modify model code)
```

### Problem: Out of Memory
```bash
# Reduce batch size
--batch-size 32

# Reduce hidden dimension  
--hidden-size 64

# Process one sequence at a time
--batch-size 1
```

### Problem: Training is too slow
```bash
# Reduce epochs
--epochs 25

# Increase batch size
--batch-size 128

# Reduce sequence length (if model supports)
--max-seq-length 16
```

---

## 📁 Key Files & Directories

| Path | Purpose |
|------|---------|
| `train_exercise_bilstm.py` | Main training script |
| `run_complete_training.py` | Auto-run all 4 stages |
| `results/riccio_realtime_exercise_recognition/` | Primary dataset |
| `results/exercise_bilstm/` | Default model output |
| `results/exercise_bilstm_*/` | Different training variants |
| `fitness_coach/models/bilstm.py` | Model architecture |
| `fitness_coach/datasets/` | Dataset utilities |
| `PRACTICAL_TRAINING_GUIDE.md` | Detailed training guide |
| `SYSTEM_ARCHITECTURE.md` | Full system design |
| `EXECUTION_SUMMARY.md` | Current status & results |

---

## 📊 Metrics Explained

### Classification Metrics

| Metric | Meaning | Good Value |
|--------|---------|-----------|
| **Accuracy** | % predictions correct | 0.84+ |
| **F1 (macro)** | Avg per-class F1 score | 0.75+ |
| **F1 (weighted)** | Class-weighted F1 | 0.80+ |
| **Precision** | TP / (TP + FP) | 0.81+ |
| **Recall** | TP / (TP + FN) | 0.78+ |

### Quality Prediction

| Metric | Meaning | Good Value |
|--------|---------|-----------|
| **RMSE** | Root mean squared error | <0.05 |
| **MAE** | Mean absolute error | <0.04 |
| **R²** | Coefficient of determination | >0.0 |

### Per-Class Breakdown

```
Shoulder Press: 0.8998 F1 ⭐ Best
Barbell Biceps: 0.8739 F1 ✓ Good
Squat:          0.8786 F1 ✓ Good
Push-up:        0.7476 F1 ✓ OK
Hammer Curl:    0.4478 F1 ⚠️ Low (few samples)
```

---

## 🔍 Debugging

### Check Data Availability
```bash
ls -lh results/riccio_realtime_exercise_recognition/
# Should see: *_biomechanics.npz, *_keypoints.npz, *_labels.npz
```

### Check Training Progress
```bash
# Watch training live
tail -f <current_output.log>

# Check best validation accuracy during training
grep "Best val acc" results/exercise_bilstm/training.log
```

### Test Inference
```bash
python inference_exercise_bilstm.py \
  --model-path results/exercise_bilstm/exercise_bilstm_best.pt \
  --video-path /path/to/video.mp4 \
  --output-json results/predictions.json
```

---

## 🎯 Success Checklist

- [ ] **Environment activated:** `source venv/bin/activate`
- [ ] **Data verified:** `ls results/riccio_realtime_exercise_recognition/` shows 3 files
- [ ] **Quick test runs:** 3-epoch run completes (test_acc ≈ 0.70+)
- [ ] **Stage 1 complete:** 15-epoch run achieves test_acc ≈ 0.84
- [ ] **Baseline trained:** 50-epoch model saves to `exercise_bilstm_baseline`
- [ ] **Metrics reviewed:** Confusion matrix shows good per-class accuracy
- [ ] **Best model selected:** Choose model with highest test_acc
- [ ] **Ready for deployment:** Export best checkpoint

---

## 🚀 Production Deployment

### 1. Select Best Model
```bash
# Find model with highest accuracy
python -c "
import json, os
best = None
best_acc = 0
for d in os.listdir('results'):
    if 'bilstm' in d:
        try:
            m = json.load(open(f'results/{d}/test_classification_metrics.json'))
            acc = m.get('test_accuracy', 0)
            if acc > best_acc:
                best_acc = acc
                best = d
        except: pass
print(f'Best model: {best} ({best_acc:.4f})')
"
```

### 2. Copy to Production
```bash
# Copy best model
cp results/exercise_bilstm_baseline/exercise_bilstm_best.pt \
   models/exercise_bilstm_production.pt

# Verify
ls -lh models/exercise_bilstm_production.pt
```

### 3. Test Production Model
```bash
python inference_exercise_bilstm.py \
  --model-path models/exercise_bilstm_production.pt \
  --video-path test_video.mp4 \
  --output-json test_predictions.json
```

---

## 🔗 Quick Links

- **Full Guide:** See `PRACTICAL_TRAINING_GUIDE.md`
- **System Design:** See `SYSTEM_ARCHITECTURE.md`
- **Latest Status:** See `EXECUTION_SUMMARY.md`
- **Capstone Report:** See `docs/CAPSTONE_REPORT.md`

---

## 💡 Pro Tips

1. **Always start with Stage 1** (15 epochs) to catch data/code issues early
2. **Monitor validation accuracy**, not just training loss
3. **Save outputs to different directories** (`--output-dir`) to compare
4. **Use `--eval-test`** to see test metrics immediately after training
5. **Check per-class F1** in metrics JSON - some exercises may need more data
6. **For best accuracy:** Use optimized config (100 epochs, 0.0005 LR)
7. **For fastest baseline:** Use 50 epochs, 0.001 LR
8. **Standardization is critical:** Always use `--standardize` flag

---

## 📞 Support

**Problem:** Training doesn't start
→ Check: `cd /Users/emelkonyan/Finess-coach-capstone-1 && source venv/bin/activate`

**Problem:** Data not found
→ Check: `ls results/riccio_realtime_exercise_recognition/`

**Problem:** Low accuracy
→ Try: `--epochs 100 --lr 0.0005`

**Problem:** Out of memory
→ Try: `--batch-size 32`

**Problem:** Need more help
→ Read: `PRACTICAL_TRAINING_GUIDE.md`

---

**Status:** ✅ Ready to Train  
**Validation Test:** ✅ 84.84% accuracy (15 epochs)  
**Estimated Time for Full Pipeline:** 1-2 hours  

Good luck! 🚀

