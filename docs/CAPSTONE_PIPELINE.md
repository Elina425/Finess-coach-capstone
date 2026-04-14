# Capstone pipeline: code and data map

This document maps your thesis-style steps (§2–§9) to **folders**, **annotation files**, and **Python entry points** in this repository.

**Default capstone workflow:** Riccio / Kaggle exercise videos → `pipeline/step_03_04_riccio_videos_to_angles.sh` → NPZs under e.g. `results/riccio_realtime_exercise_recognition/` → `train_exercise_bilstm.py --kaggle-angles-dir …`. The QEVD-style tree below is **optional legacy** layout only.

---

## Legacy data layout (`data/` or historical `qevd-fit-coach-data/`)

| Resource | Location | Role |
|----------|----------|------|
| Long-range videos | `videos/long_range/*.mp4` | Full-session captures (`0000.mp4` …) |
| Short clips | `videos/short_clips/*.mp4` | Segment clips aligned with labels (IDs may be 8-digit) |
| Global metadata | `metadata.csv` | Per-video `video_id`, `file_path`, `exercise_type`, `difficulty`, etc. |
| Fine-grained labels | `annotations/labels/fine_grained_labels.json` | Per-clip labels, splits, text labels for exercise + technique |
| Short-clip feedback | `annotations/labels/feedbacks_short_clips.json` | Text feedback keyed by `video_path` |
| Long-range feedback | `annotations/labels/feedbacks_long_range.json` | Feedback for long videos |
| Short-clip metadata | `annotations/metadata/short_clips/short_clips_metadata.json`, `clip_feedback_mapping.json` | Indexing and feedback linkage |

**Training indices (CSV)** — produced under `results/`:

- `exercise_training_index.csv` — from `build_exercise_training_index.py` (short clips + labels).
- `exercise_training_index_split.csv` — after `split_exercise_index.py` (**train / val / test** by `video_stem`).
- `exercise_training_index_long_range.csv` — from `setup_local_training_data.py` (local long-range + pseudo classes optional).
- `exercise_training_index_long_range_split.csv` — stratified split for long-range experiments.

**Processed features** — typical locations:

- `results/processed_keypoints_mediapipe/{stem}_keypoints.npz` — COCO-17 `(T,17,2)` + metadata.
- `results/processed_keypoints_mediapipe/{stem}_biomechanics.npz` — joint angles `(T,8)`.
- `results/exercise_angles/` — copies or batch outputs for BiLSTM (`{stem}_biomechanics.npz`).

**Important:** Short-clip stems (`00000000` …) must match NPZ file names. Long-range stems (`0000` …) use a different naming pattern — **do not mix index and NPZ sources** without regenerating features for every stem.

---

## §2 Pose estimation and model comparison

| Step | Implementation |
|------|------------------|
| MediaPipe BlazePose / COCO-17 | `pose_estimation_core.py` (`MediaPipeDetector`, `VideoProcessor`) |
| YOLO / other backends | `pose_estimation_core.py` (`RTMPose-X`, `ViTPose`, `DETRPose`), `model_comparison.py`, `run_full_comparison.py` |
| Benchmark speed & throughput | `model_comparison.py`, `run_full_comparison.py` |
| Documentation | This file (`docs/CAPSTONE_PIPELINE.md`) §2 |

**Typical command:** random clips + skeleton grids + aggregated FPS/confidence benchmark:

`./venv/bin/python run_full_comparison.py --num-videos 3 --seed 42 --output-dir results/skeleton_comparison_random`

**Video discovery:** (1) all `*.mp4` under `--dataset-root/videos/` (QEVD layout), or (2) if that is empty, all `*.mp4` recursively under `--dataset-root` (e.g. Riccio Kaggle cache — no need for a `videos/` subfolder). If you still see 0 files, the folder has no `.mp4` files or the path is wrong.

(`--no-random` uses the first N videos in sorted path order; OpenPose is skipped if not installed.)

---

## §3 Keypoint preprocessing (before feature extraction)

Do **not** train on raw detector coordinates until this step is done. Joint angles and mixed
angle+coord features assume a consistent preprocessing order.

### 3.1 Normalise coordinates (camera distance / subject size)

MediaPipe outputs are scaled to image `[0, 1]`; that removes resolution but not **depth** (person
appearing larger or smaller). **Skeleton-based normalization** (`PosePreprocessor.skeleton_based_normalize`):
center on hips, scale by torso length so 2D pose is comparable across cameras and distances.

**Entry point:** `apply_keypoint_preprocessing_pipeline` in `pose_estimation_core.py` (same steps as
`DatasetPreprocessor` in `qevd_dataset_integration.py`).

**Optional — bone proportion scaling:** [BioPose](https://arxiv.org/abs/2501.07800) (Koleini et al.) uses a biomechanical skeleton with bone scale and OpenSim-style subject scaling. For 2D COCO-17, `PosePreprocessor.bone_proportion_normalize` applies the same *idea*: limb lengths are set to anthropometric ratios × torso length while preserving each bone’s 2D direction. Add technique `"bone_proportion"` after `"normalization"`, or pass `--bone-proportion` to `keypoint_preprocessing_pipeline.py`.

### 3.2 Impute missing / occluded joints

- **Spatial:** low-confidence joints (`confidence < 0.3`) get values from COCO skeleton neighbors
  (`PosePreprocessor.impute_missing_joints`), **or** with technique `laplacian_spatial`, from **harmonic relaxation** on the skeleton graph: build **A** from COCO edges, **L = D − A**, and iterate missing nodes toward the degree-normalized neighbor average (Paoletti et al., [arXiv:2204.10312](https://arxiv.org/abs/2204.10312) skeletal Laplacian / smoothness). CLI: `--laplacian-spatial` on `keypoint_preprocessing_pipeline.py`.
- **Temporal:** per-joint linear interpolation along time where confidence is low
  (`PosePreprocessor.temporal_impute_sequence`).

### 3.3 Synchronise frame rates

Native video FPS varies. Sequences are **resampled in time** to a common `target_fps` (default 30 Hz)
via `PosePreprocessor.resample_to_standard_fps`, using FPS measured from the video file. NPZ files
store `source_fps` and `target_fps` for traceability.

### 3.4 Then extract features (→ §4)

After saving `*_keypoints.npz` — with whatever combination of §3.1–3.3 you used (torso normalization,
optional bone proportion, optional graph-Laplacian spatial imputation, temporal imputation, FPS sync,
optional DWT) — run **biomechanical feature extraction** (`biomechanical_features.py` /
`--biomechanics` on the keypoint script). For **mixed** models, `ExerciseAngleWindowDataset` detects
pipeline-normalized NPZ (via `techniques_json` in the archive) and **does not** apply a second
pelvis/shoulder normalization to coordinates.

### 3.5 Optional: batch KMeans + KNN imputation (OPSTL-style)

[Chen et al., arXiv:2309.12029](https://arxiv.org/html/2309.12029v2) (OPSTL, [code](https://github.com/cyfml/OPSTL)) imputes **occluded skeleton coordinates** using **KMeans** on sequence features, then **KNN** within each cluster with a **missing-value–aware distance** and **inverse-distance weighting**. The paper uses **self-supervised embeddings** for clustering; this repo’s optional step uses **time-resampled flattened 2D coordinates** as a practical stand-in unless you plug in SSL features.

**Requires ≥2 sequences** (e.g. a folder of `*_keypoints.npz`). Run **after** §3.1–3.3 if you want dataset-level completion:

`./venv/bin/python batch_knn_impute_keypoints.py --input-dir results/processed_keypoints_mediapipe --output-dir results/processed_keypoints_knn`

Outputs `*_keypoints_knn.npz` and `knn_imputation_meta.json`. Point training at the KNN folder if you adopt this step end-to-end.

| Concern | Implementation |
|--------|----------------|
| CLI wrapper | `keypoint_preprocessing_pipeline.py` (`--from-fine-grained-labels` + `--max-videos` for short_clips aligned with `fine_grained_labels.json`; `--short-clips-only`; `--all-videos`; `--biomechanics` for tasks 3+4) |
| Core pipeline function | `pose_estimation_core.apply_keypoint_preprocessing_pipeline` |
| DWT (optional) | `PosePreprocessor.dwt_normalize` |
| Batch KNN (optional, OPSTL Sec. II-D) | `knn_skeleton_imputation.py`, `batch_knn_impute_keypoints.py` |

---

## §4 Biomechanical features from preprocessed keypoints

**Task.** From **already preprocessed** COCO-17 sequences (§3), compute **biomechanical features** for downstream windows and models. The **primary** representation is **joint angles** (e.g. knee, hip, elbow, shoulder), not raw image coordinates.

**Why angles instead of raw coordinates?** After §3, keypoints are hip-centered and torso-scaled (and optionally bone-length–regularized or graph-smoothed), which stabilizes comparisons across people and cameras. **Planar joint angles** add **invariance to translation** and **uniform scaling** in the image plane: the same pose geometry yields similar angle values whether the subject stands nearer or farther from the camera or shifts laterally in frame—so angles track **limb configuration** more consistently than raw pixel coordinates for monocular coaching (remaining limits: out-of-plane motion and perspective still affect 2D angles).

| Step | Implementation |
|------|------------------|
| Eight planar angles (knee, hip, elbow, shoulder, …) | `biomechanical_features.py` — `compute_sequence_angles`, `ANGLE_FEATURE_NAMES` |
| Mixed angles + normalised coords (ablation / Riccio-style) | `biomechanical_features.py` — `compute_mixed_sequence_features`, `compute_coords_only_sequence_features` |
| Batch from video index | `batch_compute_angles_for_index.py` |
| NPZ I/O | `process_npz_file` / pipeline in `biomechanical_features.py` |

**Kaggle (pre-extracted MediaPipe):** [Physical Exercise Recognition](https://www.kaggle.com/datasets/muhannadtuameh/exercise-recognition) ships `landmarks.csv` (33× x,y,z) + `labels.csv`. **`kaggle_exercise_recognition_pipeline.py`** integrates **§3 + §4** like QEVD: `apply_keypoint_preprocessing_pipeline` (normalization, imputation, optional Laplacian / bone proportion, FPS sync, optional DWT) and `biomechanical_features.process_npz_file` (eight joint angles + `*_biomechanics_summary.json`). Flags mirror `keypoint_preprocessing_pipeline.py` (`--laplacian-spatial`, `--bone-proportion`, `--no-fps-sync`, `--dwt`, `--source-fps` / `--target-fps`). Auto-finds `~/.cache/kagglehub/.../exercise-recognition` or `EXERCISE_RECOGNITION_ROOT` if `--dataset-root` is omitted.

**Notebook (PyTorch on Kaggle):** `notebooks/exercise_pose_classification_pytorch.ipynb` — loads the same CSVs, COCO-17 windows, LSTM classifier; works under `/kaggle/input/…` or local `results/kaggle_exercise_recognition/`. Style aligned with [exercise pose classification with PyTorch](https://www.kaggle.com/code/teowaihong/exercise-pose-classification-with-pytorch).

---

## §5–§6 BiLSTM: exercise classification + quality regression

| Step | Implementation |
|------|------------------|
| Index from labels + heuristics | `build_exercise_training_index.py`, `exercise_label_utils.py` |
| Local long-range index | `setup_local_training_data.py` |
| Train/val/test split | `split_exercise_index.py` (QEVD CSV); **Riccio NPZs:** split inside `build_kaggle_angle_datasets` |
| Dataset (30-frame windows) | `exercise_bilstm_dataset.py` |
| Model (BiLSTM + cls + reg heads) | `exercise_bilstm_model.py` |
| Training | `train_exercise_bilstm.py` (`--feature-mode angles|coords|mixed`, `--standardize`, `--preset riccio`) |
| Inference | `inference_exercise_bilstm.py` |

**Riccio / `--kaggle-angles-dir`:** when `*_labels.npz` includes **`video_id`** (from `riccio_kaggle_video_pipeline.py`), angle windows are built **per video only** (no cross-clip windows), and **train / val / test are assigned by video** with **stratification on class**—same spirit as stem-level splitting for QEVD, and avoids leakage from shuffling windows that share the same recording. NPZs without `video_id`, or a single-video timeline, keep stratified **window** splits.

**Quality target:** CSV column `quality` ∈ [0,1] from `heuristic_quality_score` + optional feedback text.

---

## §7 Rigorous evaluation and ablations

| Step | Implementation |
|------|------------------|
| Accuracy, macro F1, per-class F1, MAE, R² | `evaluate_exercise_models.py` |
| Ablation: angles vs coords vs mixed | Train three BiLSTM checkpoints with `--feature-mode`; compare with `--checkpoint` |
| GCN baseline (supervised, same task) | `train_gcn_supervised.py`, `GCNSequenceExerciseNet` in `gcn_pose_model.py` |
| Contrastive GCN (pose similarity) | `train_gcn_pose_similarity.py`, `inference_gcn_pose_similarity.py` |
| JSON export | `evaluate_exercise_models.py --json-out` |

**Plots from metrics:** `export_evaluation_plots.py` (reads evaluation JSON → figures in `docs/figures/`).

---

## §8 Personalisation (planned architecture)

| Step | Status | Notes |
|------|--------|--------|
| Per-user adapter after each session | **Not implemented** | Recommended design: freeze `exercise_bilstm_best.pt`, train small **LoRA / linear adapter** on user’s saved windows, or fine-tune last layer only with low LR + EWC |
| Session data store | **Not implemented** | Store anonymised `(window_features, user_id, timestamp, label)` |
| Where to hook in | — | After `ExerciseBiLSTM` trunk, before heads; or separate calibration network on quality head |

---

## §9 Application integration (planned / partial)

| Feature | In repo |
|---------|---------|
| Live skeleton overlay | `extract_skeleton_visualizations.py`, OpenCV paths in `pose_estimation_core` |
| Real-time classification | `inference_exercise_bilstm.py` (batch over windows; wrap in loop for webcam) |
| Quality score display | Regression head output from same inference script |
| Session progress DB | **Not implemented** — suggest SQLite + Streamlit/React per prior README patterns |

---

## Suggested end-to-end command order

1. `build_exercise_training_index.py` **or** `setup_local_training_data.py`
2. `split_exercise_index.py` → `*_split.csv`
3. Ensure `*_keypoints.npz` / `*_biomechanics.npz` for **every stem** in the index (batch scripts as needed).
4. `train_exercise_bilstm.py` …  
5. `train_gcn_supervised.py` … (optional)  
6. `evaluate_exercise_models.py --split test --json-out …`  
7. `export_evaluation_plots.py` …  

---

## Key references (methods cited in code)

- BiLSTM + angles/coords: Riccio (arXiv:2411.11548) — see `train_exercise_bilstm.py` docstring.  
- GCN pose similarity: Zeng et al. (arXiv:2511.01194) — `gcn_pose_*.py`.  
- Topology / stride ideas (discussion): SasMamba (arXiv:2511.08872) — design notes only.
