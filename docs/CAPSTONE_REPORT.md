# AI-Powered Fitness Coaching from Monocular Video:  
# Pose Estimation, Biomechanical Features, and Sequence Models for Exercise Recognition and Quality Assessment

**Capstone project report (draft)**  
**Author:** [Your name]  
**Institution:** [Your institution]  
**Date:** March 2026  

---

## Abstract

We present an end-to-end pipeline for analysing fitness videos: multiple pose estimators produce COCO-style 2D keypoints; we normalise and impute missing data; derive joint-angle features invariant to camera scale; and train a bidirectional LSTM (**BiLSTM**) with dual heads for **exercise classification** and continuous **movement quality** regression in $[0,1]$. A stratified **train / validation / test** split by video stem avoids window leakage between sets. We compare **angle-only**, **coordinate-only**, and **mixed** inputs (following Riccio et al., arXiv:2411.11548) and report accuracy, per-class F1, MAE, and $R^2$. A secondary **graph convolutional network (GCN)** baseline (Zeng et al., arXiv:2511.01194) models skeleton topology for supervised classification and contrastive pose similarity. Primary experiments follow the **Riccio et al.** (arXiv:2411.11548) setup on **Kaggle-style exercise video** data processed through this repository’s angle pipeline (`*_biomechanics.npz`, stratified splits by video). We discuss limitations (pseudo-labels on small local subsets, short-clip NPZ coverage) and outline **personalisation** and a **real-time coaching application** as future work.

**Keywords:** human pose estimation; fitness; BiLSTM; joint angles; action quality; graph convolution.

---

## 1. Introduction

Digital fitness coaching benefits from automatic recognition of which exercise a user performs and how well they perform it. Monocular RGB video is widely available but introduces **depth ambiguity**, **occlusion**, and **viewpoint variation**. Representing the body as a **skeleton** reduces sensitivity to clothing and background; **joint angles** further reduce dependence on distance to the camera compared with raw pixel coordinates.

This work implements: (1) pose extraction with **MediaPipe BlazePose** and optional comparison to **YOLOv11**-class detectors; (2) **preprocessing** (normalisation, missing-joint handling, FPS metadata); (3) **biomechanical features**—eight planar angles from COCO-17 keypoints; (4) a **BiLSTM** on 30-frame windows with **cross-entropy** and **MSE** losses; (5) **evaluation** and **ablation** across feature types; (6) optional **GCN** encoders; (7) a roadmap for **user adaptation** and **deployment**.

---

## 2. Pose estimation and model selection

### 2.1 Models

We integrate:

- **MediaPipe BlazePose** — real-time, mobile-friendly, 33 landmarks mapped to **COCO-17** for compatibility with downstream code (`pose_estimation_core.py`).
- **YOLOv11** (Ultralytics) — optional, higher accuracy on GPU for offline batch extraction.
- **OpenPose** — optional where system libraries exist; highest fidelity, highest compute.

### 2.2 Comparison methodology

`model_comparison.py` and `run_full_comparison.py` benchmark **wall-clock time per frame**, **CPU/GPU usage**, and detector availability on sample clips. For coaching, **MediaPipe** is the default **production** choice; **YOLOv11** suits **offline** labelling and ablations.

### 2.3 Output format

Each processed frame yields **17 joints $\times$ 2** coordinates $(x,y)$ in image space (and confidence when available), stored as time series in `*_keypoints.npz`.

---

## 3. Preprocessing

### 3.1 Normalisation

We apply **skeleton-relative** normalisation (hip-centered, shoulder–hip / torso length scale) via `PosePreprocessor` and pipeline scripts (`keypoint_preprocessing_pipeline.py`, `apply_keypoint_preprocessing_pipeline` in `pose_estimation_core.py`), reducing global translation and apparent size effects. **Optional** steps include **bone proportion** scaling (BioPose-style limb/torso ratios) and **DWT** frequency-domain normalisation.

### 3.2 Missing and occluded joints

Low-confidence joints are **imputed** in space (COCO neighbor mean, or optional **graph Laplacian** harmonic relaxation on the skeleton) and in time (linear interpolation). Optional **batch KNN** (OPSTL-style) can refine sequences at dataset level. Angles that later require invalid triplets become **NaN** and are converted for learning (`nan_to_num` in dataset loaders).

### 3.3 Temporal alignment

Videos are associated with **source FPS** and optional **target FPS** in NPZ metadata; long sequences can be resampled for consistent step sizes across the dataset.

---

## 4. Biomechanical features

**Inputs** are the **preprocessed** keypoint streams from §3 (not raw detector coordinates). **Primary** features are **joint angles**: for each frame we compute **eight angles** (degrees) from 2D triplets—left/right elbow, knee, hip, shoulder (vertex at the joint). Arrays of shape $(T,8)$ are saved as `*_biomechanics.npz`. **Angles are preferred to raw coordinates** for exercise analysis because they describe **limb geometry** in a way that stays **consistent** as the person moves in the scene relative to the camera—after §3 normalisation, angles are also **invariant to translation** and **uniform scaling** in the plane, whereas raw $(x,y)$ still mix configuration with global position in the image.

**Ablation features:**

- **Angles only** — 8-D per frame.  
- **Coords only** — pelvis-centred, torso-scaled $(x,y)$ flattened to 34-D (`compute_coords_only_sequence_features`).  
- **Mixed** — 8 + 34 = 42-D (`compute_mixed_sequence_features`), following the “angles + coordinates” motivation in Riccio et al.

---

## 5. Sequence model: exercise classification

### 5.1 Formulation

We segment each video into **30-frame windows** with stride 15. Inputs are $(30, F)$ with $F \in \{8,34,42\}$ depending on mode. Optional **per-feature standardisation** is fit on **training windows only** and applied to validation and test.

### 5.2 Architecture

**ExerciseBiLSTM:** two-layer **bidirectional LSTM**, hidden size configurable (default 128; **Riccio preset**: 73 units, dropout $\approx 0.22$, batch 54, lr $4\times 10^{-4}$). **Mean pooling** over time; linear heads for **class logits** and **quality** scalar.

### 5.3 Labels

**Exercise class:** derived from `fine_grained_labels.json` by taking the **text before the first “ - ”** in the label string, normalised (`exercise_label_utils.py`). **Pseudo-classes** (e.g. `pseudo_k`) can be assigned for pipeline tests on long-range video when metadata lacks fine names.

---

## 6. Quality regression

The same trunk feeds a **regression head** (single linear output). Targets are **heuristic scores** in $[0,1]$ combining label counts and optional textual feedback (`heuristic_quality_score`). This yields **graded** feedback rather than a single correct/incorrect bit.

Joint training loss:

$$\mathcal{L} = \lambda_{\mathrm{cls}} \mathrm{CE} + \lambda_{\mathrm{reg}} \mathrm{MSE}$$

with configurable weights (`train_exercise_bilstm.py`).

---

## 7. Experimental setup and evaluation

### 7.1 Data splits

`split_exercise_index.py` assigns each **unique `video_stem`** to **train**, **val**, or **test** with approximate ratios (default 70% / 15% / 15%), **stratified by exercise class** when possible. All windows inherit the stem’s split, preventing **within-video** leakage across splits.

**Critical data note:** Short-clip indices (`00000000`, …) require **`{stem}_keypoints.npz`** for every stem. Local experiments often use **long-range** indices (`0000`, …) where only a few NPZ files exist—indices and NPZ prefixes must **match**.

### 7.2 Metrics

`evaluate_exercise_models.py` reports on the **held-out test** split:

- **Classification:** accuracy, **macro F1**, **per-class F1**.  
- **Quality:** **MAE**, **$R^2$** (undefined if targets are nearly constant).

### 7.3 Ablation protocol

Train separate checkpoints:

1. `--feature-mode angles`  
2. `--feature-mode coords`  
3. `--feature-mode mixed`  

Evaluate each on `split=test` and compare. Optionally train **GCNSequenceExerciseNet** (`train_gcn_supervised.py`) on keypoint windows for a **topology-aware** baseline.

### 7.4 Representative results (fill from your runs)

| Model / features | Accuracy | F1 (macro) | MAE (quality) | $R^2$ (quality) |
|-------------------|----------|------------|---------------|-----------------|
| BiLSTM angles   | *run*    | *run*      | *run*         | *run*           |
| BiLSTM coords   | *run*    | *run*      | *run*         | *run*           |
| BiLSTM mixed    | *run*    | *run*      | *run*         | *run*           |
| GCN supervised  | *run*    | *run*      | *run*         | *run*           |

Export JSON with `--json-out`, then generate figures:

```bash
./venv/bin/python docs/export_evaluation_plots.py --json results/eval_test.json --out-dir docs/figures
```

### 7.5 Visualisations

- **Skeleton overlay / trajectory:** `extract_skeleton_visualizations.py`.  
- **Metric bars:** `docs/export_evaluation_plots.py` → `docs/figures/eval_classification.png`, `eval_quality.png`.

---

## 8. Secondary models and related ideas

- **GCN–PSN-style** encoder (`gcn_pose_topology.py`, `gcn_pose_model.py`): normalised adjacency on COCO-17, **contrastive regression** for pose similarity (`train_gcn_pose_similarity.py`).  
- **Structure-aware sequence modelling** (SasMamba, arXiv:2511.08872): motivates **avoiding naive flattening** of $(T \times V)$ sequences; our GCN + temporal pooling aligns with preserving **skeletal topology** before classification.

---

## 9. Personalisation (roadmap)

**Goal:** keep a **frozen** base BiLSTM and adapt to each user with a **small** module (e.g. low-rank adapters on the last LSTM layer or a linear calibration on the quality head).

**Suggested protocol:**

1. Onboarding: collect $N$ labelled or weakly labelled windows.  
2. Optimise adapter only with **small learning rate** and **weight decay**; optional **EWC** to limit catastrophic forgetting.  
3. Store per-user weights in `results/adapters/{user_id}.pt`.  
4. At inference: `output = base(x) + adapter(x)` or gated combination.

*Implementation status:* **not included** in the current repository; section reserved for future work / thesis extension.

---

## 10. Application integration (roadmap)

**Target features:**

- **Live** webcam or file input → MediaPipe → windows → BiLSTM → **exercise label + quality**.  
- **Skeleton overlay** on the video frame.  
- **Post-rep feedback** using threshold rules on angles (`biomechanical_features`) plus model quality score.  
- **Session history** (SQLite or cloud) for progress curves.

*Implementation status:* inference scripts exist (`inference_exercise_bilstm.py`); a full **Streamlit / web** shell is **out of scope** of this repository snapshot and can be cited as **future work**.

---

## 11. Limitations

1. **2D only** — depth ambiguity remains; 3D lifting (e.g. lifting networks on Human3.6M) is not trained here.  
2. **Heuristic quality** — not expert scores; Spearman correlation to human ratings is future work.  
3. **Scale** — full short-clip training requires processing **thousands** of MP4s to NPZ; local demos use **three** long-range videos.  
4. **Class imbalance** — rare exercises need weighted loss or oversampling.

---

## 12. Conclusion

We described and implemented a **complete pipeline** from RGB video to **exercise recognition** and **quality estimation**, with **reproducible splits**, **ablations** on angle vs coordinate representations, and **optional GCN** baselines. The codebase is structured for extension toward **user personalisation** and a **real-time coaching UI**. Filling Table 7.4 with your full-scale runs and adding screenshots from `docs/figures/` completes the capstone submission.

---

## References (selected)

1. Riccio, R. *Real-Time Fitness Exercise Classification and Counting from Video Frames.* arXiv:2411.11548, 2024.  
2. Zeng, M. *A Topology-Aware Graph Convolutional Network for Human Pose Similarity and Action Quality Assessment.* arXiv:2511.01194, 2025.  
3. Cui, H. et al. *SasMamba: A Lightweight Structure-Aware Stride State Space Model for 3D Human Pose Estimation.* arXiv:2511.08872, 2025.  
4. QEVD-FIT-COACH dataset layout and pipeline map: `docs/CAPSTONE_PIPELINE.md`.

---

## Appendix A. Repository map

See **`docs/CAPSTONE_PIPELINE.md`** for file-level mapping.

## Appendix B. Commands checklist

```text
# Index + split
./venv/bin/python build_exercise_training_index.py --output results/exercise_training_index.csv
./venv/bin/python split_exercise_index.py --input results/exercise_training_index.csv --output results/exercise_training_index_split.csv

# Features (example)
./venv/bin/python setup_local_training_data.py
./venv/bin/python batch_compute_angles_for_index.py --max-videos 100

# Train BiLSTM
./venv/bin/python train_exercise_bilstm.py --index-csv results/exercise_training_index_split.csv ...

# Evaluate
./venv/bin/python evaluate_exercise_models.py --index-csv ... --split test --checkpoint bilstm ... --json-out results/eval_test.json

# Figures
./venv/bin/python docs/export_evaluation_plots.py --json results/eval_test.json --out-dir docs/figures
```

---

*End of draft — replace italicised “run” entries with your experimental numbers and add figure captions in your final PDF.*
