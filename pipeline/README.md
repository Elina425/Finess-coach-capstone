# Riccio dataset — step-by-step pipeline (steps 2 → 5)

This is the **default capstone path**: [Riccio / real-time exercise recognition](https://www.kaggle.com/datasets/riccardoriccio/real-time-exercise-recognition-dataset) **video folders** → MediaPipe + preprocessing → `*_biomechanics.npz` / `*_labels.npz` → **BiLSTM** with two heads:

1. **Classification** — which exercise (coarse class from folder / phase label).  
2. **Regression** — continuous **quality** in **[0, 1]** (Riccio NPZs use a **placeholder** target unless you supply real scores elsewhere).

Train / val / test are split **by source video** when `*_labels.npz` contains **`video_id`** (this pipeline writes it), with **~70% / 15% / 15%** of videos in train / val / test by default (`--kaggle-val-ratio 0.15`, `--kaggle-test-ratio 0.15`). Training windows are **not** the same size as the test set; **most windows are in train**, by design.

---

## 0. One-time setup

From the repository root:

```bash
python -m venv venv
./venv/bin/pip install -r requirements.txt
./venv/bin/pip install -e .
```

GPU for step 5 is optional (`train_exercise_bilstm.py` uses CUDA if available; pass `--cpu` to force CPU).

### Riccio env

Steps **2–5** (`step_02_pose_comparison.sh` … `step_05_train_bilstm_riccio.sh`) `source` [`pipeline/riccio_env.sh`](riccio_env.sh). You can override before running:

| Variable | Default |
|----------|---------|
| `EXERCISE_RECOGNITION_ROOT` | Newest version under `$KAGGLEHUB_CACHE/.../riccardoriccio/real-time-exercise-recognition-dataset/versions/` that contains `similar_dataset` or `final_kaggle_with_additional_video`, else unset |
| `RICCIO_OUTPUT_DIR` | `<repo>/results/riccio_realtime_exercise_recognition` |
| `RICCIO_STEM` | `riccio_realtime_exercise_recognition` |
| `EXERCISE_BILSTM_OUT` | `<repo>/results/exercise_bilstm` |
| `KAGGLEHUB_CACHE` | `$HOME/.cache/kagglehub` when unset |
| `RICCIO_MP_MAX_WORKERS` | Cap when `--workers 0` (default `8`) |
| `RICCIO_MP_WORKERS` | If set to a positive integer, used when `--workers 0` instead of auto CPU count |

Extra CLI args appended to the shell scripts are forwarded last, so e.g. `--dataset-root /other` overrides earlier defaults where argparse allows.

**Full Riccio path (minimal):** after setup and dataset available (see below), from repo root:

```bash
./pipeline/step_02_pose_comparison.sh --num-videos 5 --output-dir results/02_pose_comparison
./pipeline/step_03_04_riccio_videos_to_angles.sh --workers 6 --skip-keypoints
./pipeline/step_05_train_bilstm_riccio.sh
```

---

## 1. Get the Riccio video dataset

Pick **one**:

- **A. Auto-download** (needs `pip install kagglehub`): pass **`--download`** to the step 3–4 command below, or run `download_riccio_kaggle_dataset.py` and then point `--dataset-root` at the extracted folder.  
- **B. Existing folder**: set **`EXERCISE_RECOGNITION_ROOT`** to the directory that contains subsets such as `similar_dataset/`, `final_kaggle_with_additional_video/`, … (see `riccio_kaggle_video_pipeline.py`).  
- **C. Explicit path**: `--dataset-root ~/.cache/kagglehub/datasets/riccardoriccio/real-time-exercise-recognition-dataset/versions/<N>` (or your copy).

---

## 2. Step 2 — pose backend comparison (optional)

Compare detectors (speed / overlays) before locking MediaPipe for steps 3–4:

```bash
./pipeline/step_02_pose_comparison.sh --num-videos 50 --output-dir results/02_pose_comparison
```

Uses `run_full_comparison.py` with `EXERCISE_RECOGNITION_ROOT` (or `--dataset-root`) for `.mp4` discovery.

**Speed:** pose models load **once** per run (not once per video). The heavy **FPS benchmark** (step 4) only uses the **first 3** sampled clips by default (`--benchmark-videos 3`). For many PNGs but no summary timing, add `--skip-benchmark`. To benchmark every sampled clip (slow), use `--benchmark-videos 0`. Lower `--benchmark-max-frames` (e.g. `40`) for a shorter per-clip timing pass.

---

## 3. Steps 3 + 4 — videos → keypoints (optional) + joint angles + labels

Single entry point: **`riccio_kaggle_video_pipeline.py`** (wrapped by the shell script).

### What the current run includes (default)

Pipeline order is implemented in `apply_keypoint_preprocessing_pipeline` (`fitness_coach/core/pose_estimation_core.py`). **MediaPipe** runs first on each video; then keypoints (unless `--raw-keypoints`):

| Step | Default? | Meaning |
|------|----------|--------|
| **Spatial imputation** | Yes | Low-confidence joints filled (COCO **neighbor** mean; **not** Laplacian unless `--laplacian-spatial`) |
| **Skeleton–torso normalization** | Yes | Hip-centered, torso-length scale (2D scale / distance) |
| **Bone proportion** | No | `--bone-proportion` — BioPose-style limb ratios after norm |
| **Temporal imputation** | Yes | Linear interpolation along time for flickering joints |
| **FPS sync** | Yes | Resample to `--target-fps` (default 30) unless `--no-fps-sync` |
| **Savitzky–Golay / Kalman** | No | `--savgol` or `--kalman` (mutually exclusive) |
| **DWT** | No | `--dwt` (PyWavelets) |

**Default technique list** (no extra flags): `normalization`, `imputation`, `fps_sync` — **no** Laplacian, bone, DWT, or Savitzky–Golay.

**`--rich-preprocess`** is a shortcut for: `--laplacian-spatial --bone-proportion --dwt --savgol`.

**Ablations (e.g. classification hurt by DWT):** use the wrappers (same env as step 3–4):

- `./pipeline/regenerate_rich_no_dwt.sh` — Laplacian + bone + Savitzky–Golay, **no DWT** → `results/run_rich_no_dwt/`
- `./pipeline/regenerate_rich_no_dwt_no_bone.sh` — Laplacian + Savitzky–Golay only → `results/run_rich_no_dwt_no_bone/`

Both default to `--skip-keypoints` for speed; set `NO_SKIP_KP=1` to also write keypoints. Forward extra args (e.g. `--workers 0`). Then re-run `benchmark_preprocessing_training.py` with `--rich-angles-dir results/run_rich_no_dwt`.

**Outputs** (under `--output-dir` / `RICCIO_OUTPUT_DIR`, stem `RICCIO_STEM`):

| File | Contents |
|------|----------|
| `{stem}_biomechanics.npz` | `angles` `(T, 8)` |
| `{stem}_labels.npz` | `pose` (class), `video_id` (per frame) |
| `{stem}_keypoints.npz` | `keypoints` `(T, 17, 2)` if not `--skip-keypoints` |
| `{stem}_pipeline_summary.json` | Metadata: preprocessing list, `target_fps`, paths, etc. |

**Comparing “baseline” vs “rich” runs:** use two output directories (or stems) and the same `--max-videos` / dataset so rows align. **Do not use `--skip-keypoints`** if you want joint-level metrics. Then compare NPZs:

```bash
./venv/bin/python compare_preprocessing_npz.py \
  --baseline-kp results/run_base/riccio_realtime_exercise_recognition_keypoints.npz \
  --alternate-kp results/run_rich/riccio_realtime_exercise_recognition_keypoints.npz \
  --baseline-ang results/run_base/riccio_realtime_exercise_recognition_biomechanics.npz \
  --alternate-ang results/run_rich/riccio_realtime_exercise_recognition_biomechanics.npz \
  --json-out results/preprocess_compare.json
```

That script reports MAE / RMSE / NME (RMSE normalized by median shoulder width), Pearson correlation per joint, and angle MAE/RMSE. **For PCK / MPJPE / OKS vs a reference pose model on raw video**, use `evaluate_pose_metrics.py` (pseudo–ground truth from ViTPose, not preprocessing A vs B).

**Task-level benchmark (default vs rich NPZs):** one command trains the same Riccio BiLSTM twice (`cnn_attn`, classification-only, same `--kaggle-seed`) and writes:

- `preprocessing_train_comparison.json` — test accuracy, F1 macro / weighted, recall, precision, deltas  
- `PREPROCESSING_TRAINING_REPORT.md` — Markdown table + per-class F1 for reports

```bash
./venv/bin/python benchmark_preprocessing_training.py \
  --default-angles-dir results/run_default \
  --rich-angles-dir results/run_rich \
  --output-root results/preprocess_train_benchmark \
  --epochs 30 --kaggle-seed 42
```

Optional extra training flags are forwarded to both runs (after the wrapper’s own options), e.g. `--cpu`, `--epochs 60`, or **`--balanced-class-weights`** (inverse-frequency cross-entropy — often raises **macro F1** when rare classes like hammer curl matter).

Other ways to push accuracy after dropping DWT: slightly **lower `--lr`** (e.g. `3e-4`), **more epochs**, try **`--window 40 --stride 15`**, or run **`tune_exercise_bilstm.py`** on the rich-no-DWT NPZ folder.

**ST-GCN benchmark (same split seed, keypoint NPZs required):** each folder must contain `{stem}_keypoints.npz` and `{stem}_labels.npz` (omit `--skip-keypoints` when building NPZs). Writes `preprocessing_stgcn_comparison.json` and `PREPROCESSING_STGCN_REPORT.md`.

```bash
./venv/bin/python benchmark_preprocessing_stgcn.py \
  --default-keypoints-dir results/run_default \
  --rich-keypoints-dir results/run_rich_no_dwt \
  --output-root results/preprocess_stgcn_benchmark \
  --epochs 30 --kaggle-seed 42 \
  --balanced-class-weights
```

(`--balanced-class-weights` is forwarded to both ST-GCN runs, like the BiLSTM benchmark.)

```bash
./pipeline/step_03_04_riccio_videos_to_angles.sh --skip-keypoints
```

(Output dir / stem / dataset root come from `riccio_env.sh`. **Worker count:** the Python entrypoint defaults to **`--workers 0`**, which auto-selects parallel processes from your CPU count, capped by `RICCIO_MP_MAX_WORKERS` (default 8, set in `riccio_env.sh`). Use `--workers 1` for a single process or `--workers N` to fix the pool. **GitHub Copilot does not run MediaPipe for you** and has no “free GPU” for this pipeline; **Google Colab’s GPU** speeds **PyTorch training**, not this MediaPipe step, which stays **CPU**-bound—on Colab use `--workers 0` or match `os.cpu_count()` (often 2).)

**What this does (conceptually):** capstone **§3** normalization / imputation / FPS sync, then **§4** eight joint angles per frame; writes:

| Output | Role |
|--------|------|
| `{stem}_biomechanics.npz` | Angles `(T, 8)` for BiLSTM |
| `{stem}_labels.npz` | Per-frame pose / class + **`video_id`** for split-by-video |
| `{stem}_keypoints.npz` | Optional; omit with `--skip-keypoints` to save disk (needed for ST-GCN / mixed BiLSTM) |

**Useful flags** (see `./venv/bin/python riccio_kaggle_video_pipeline.py --help`):

- `--download` — fetch dataset via kagglehub if no local layout found.  
- `--max-videos N` / `--max-frames M` — smoke tests.  
- `--dataset-root PATH` — override auto-discovery.  
- `--workers N` — parallel videos (CPU).  
- Richer preprocessing: e.g. `--laplacian-spatial`, `--bone-proportion`, `--dwt` (as in the module docstring).

---

## 4. Step 5 — train BiLSTM (**classification** on Riccio)

The wrapper passes **`--preset riccio`**, **`--standardize`**, **`--eval-test`**, and by default:

- **`--architecture cnn_attn`** — dilated **1D CNN** front-end (multi-scale temporal receptive field) → **BiLSTM** → **multi-head self-attention** (residual + LayerNorm) → mean pool → linear heads. Improves temporal modelling vs mean-pooled BiLSTM alone; report **macro F1** on the test split to compare against a baseline.  
- **`--classification-only`** — sets **`--reg-weight 0`** (Riccio quality targets are placeholders).

**Baseline (plain BiLSTM + mean pool):**  
`./pipeline/step_05_train_bilstm_riccio.sh … --architecture plain`  
(omit `--classification-only` if you still want the auxiliary regression head).

```bash
./pipeline/step_05_train_bilstm_riccio.sh
```

**Training behaviour:**

- **Checkpoint** when validation **accuracy** improves (quality RMSE is still logged if `reg_weight > 0`).  
- **`--eval-test`**: reloads best weights and evaluates the **held-out test** split; writes e.g.  
  `results/exercise_bilstm/test_classification_metrics.json` (includes **F1 macro**) and  
  `results/exercise_bilstm/test_classification_probs.npz`.

**Common tweaks** (append to the same command):

| Flag | Default | Note |
|------|---------|------|
| `--epochs` | 30 | Increase (e.g. `80`); watch **val acc** for overfitting. |
| `--architecture` | `cnn_attn` (via script) | `plain` = original BiLSTM. |
| `--cnn-hidden` | 64 | CNN trunk width (`cnn_attn` only). |
| `--attn-heads` | 4 | Adjusted automatically if `2×hidden` is not divisible by 4. |
| `--kaggle-val-ratio` / `--kaggle-test-ratio` | 0.15 | Keep fixed when comparing runs. |
| `--kaggle-seed` | 42 | Split reproducibility. |
| `--cpu` | off | Force CPU. |

**Hyperparameter search (Table 3–style ranges):** same Kaggle paths, plus e.g.

```bash
./venv/bin/python tune_exercise_bilstm.py --search-table3 --n-trials 25 \\
  --kaggle-angles-dir results/riccio_realtime_exercise_recognition \\
  --kaggle-stem riccio_realtime_exercise_recognition \\
  --architecture cnn_attn --classification-only \\
  --output-dir results/exercise_bilstm_tune_t3
```

Search space: **units** 50–150, **dropout** 0.2–0.5, **lr** uniform 1e-4–1e-3, **batch** 32–64, **epochs** 50–100 per trial (final retrain uses best trial’s epochs). For the original Riccio Table 4–local search, omit `--search-table3` and use `--narrow-space` / `--tune-epochs` as before.

---

## 5. After training — evaluation and inference

- **Rigorous window metrics** (optional, if you use an index CSV elsewhere): `evaluate_exercise_models.py`.  
- **Confusion matrix / ROC** (from test probs): `visualize_confusion_roc.py` — use the same checkpoint and Kaggle args as training when recomputing from `--checkpoint`.  
- **New video:** `inference_exercise_bilstm.py --checkpoint results/exercise_bilstm/exercise_bilstm_best.pt --video path/to.mp4`.

---

## 6. Optional — EgoExo-Fitness (real human quality labels)

Not required for the Riccio path. Needs **frame folders** (large download) or your own angles; see `step_06_train_bilstm_egoexo.sh` and `build_egoexo_fitness_index.py`.

**Hugging Face auth (gated dataset):** On [Lymann/EgoExo-Fitness](https://huggingface.co/datasets/Lymann/EgoExo-Fitness), accept the dataset terms. Then authenticate **without putting tokens in the repo**: e.g. `huggingface-cli login`, or `export HF_TOKEN="hf_…"` in your shell for one session, or a `.env` file (already gitignored) loaded by your shell. Full snapshot:

```bash
export HF_TOKEN="hf_…your_read_token…"
./venv/bin/python download_egoexo_fitness_dataset.py --local-dir data/EgoExo-Fitness
```

Labels-only (small): add `--annotations-only`. **Never commit tokens** or paste them into tracked files.

**Annotations-only on a small disk:** `./pipeline/download_egoexo_annotations_only.sh` (or `download_egoexo_fitness_dataset.py --annotations-only`). Then rebuild the index and train **`train_exercise_annotation_bilstm.py`** (BiLSTM on interpretable checks + text; no pose).

### Second head (quality regression) — exact commands

The quality head is trained by **`train_exercise_bilstm.py`** whenever you **do not** pass **`--classification-only`**. Default **`--reg-weight`** is `0.5` (MSE on the CSV `quality` column).

**One-shot (split → angles → train)** from the repo root, after you have `data/EgoExo-Fitness` with frames and `results/egoexo_fitness_index.csv`:

```bash
./pipeline/step_06_train_bilstm_egoexo.sh
```

**Equivalent explicit training step only** (after `split_exercise_index.py` and `batch_compute_angles_for_index.py` have populated the paths):

```bash
./venv/bin/python train_exercise_bilstm.py \
  --index-csv results/egoexo_fitness_index_split.csv \
  --angles-dir results/egoexo_exercise_angles \
  --standardize \
  --output-dir results/exercise_bilstm_egoexo \
  --epochs 30
```

**Text supervision** (comment + action guidance from the index) is **on by default**; add **`--no-text-supervision`** to turn it off. **`--text-supervision`** is optional and redundant with the default.

**Build the index first** (if you have not already):

```bash
./venv/bin/python build_egoexo_fitness_index.py \
  --annotations-json data/EgoExo-Fitness/raw_annotations/interpretable_action_judgement.json \
  --dataset-root data/EgoExo-Fitness \
  --format interpretable \
  --output results/egoexo_fitness_index.csv
```

Then either run `step_06_train_bilstm_egoexo.sh` or repeat split / angles / `train_exercise_bilstm.py` as above.

---

## 7. Google Colab (GPU for step 5)

- **Riccio / Kaggle NPZs:** [`notebooks/colab_gpu_training.ipynb`](../notebooks/colab_gpu_training.ipynb) — point `KAGGLE_DIR` at NPZs from steps 3–4, run `train_exercise_bilstm.py`.
- **EgoExo end-to-end (rubric 1–8 outline):** [`notebooks/colab_egoexo_capstone_pipeline.ipynb`](../notebooks/colab_egoexo_capstone_pipeline.ipynb) — Drive + `HF_TOKEN`, annotations download, index/split, annotation BiLSTM, optional pose track when frames exist on Drive.

---

## Reference

- Implementation map: [`docs/CAPSTONE_PIPELINE.md`](../docs/CAPSTONE_PIPELINE.md).  
- Shell scripts in this folder are thin wrappers; logic lives under `fitness_coach/` and root shims (`riccio_kaggle_video_pipeline.py`, `train_exercise_bilstm.py`).
