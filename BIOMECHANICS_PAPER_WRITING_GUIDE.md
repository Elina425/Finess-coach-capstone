# Biomechanical Feature Extraction: Writing Guide for Capstone Paper

This document outlines how to write Section 3 (Methods → Feature Engineering) describing the keypoint normalization and biomechanical feature extraction pipeline that feeds BiLSTM and xLSTM models.

---

## Overview: Full Pipeline in 4 Stages

Your preprocessing pipeline follows this structure:
```
Stage 1: Pose Detection (MediaPipe) → raw landmarks
    ↓
Stage 2: Keypoint Normalization → skeleton-based normalization
    ↓
Stage 3: Biomechanical Feature Extraction → joint angles + normalized coordinates
    ↓
Stage 4: Feature Windowing & Training Preparation → 30-frame sequences
```

---

## **IMPORTANT: Dual-Dataset Setup**

Your capstone uses **two datasets** with a **single unified preprocessing pipeline** and **dual-head models**:

| Dataset | Purpose | View | Size |
|---------|---------|------|------|
| **Riccio** | Controlled baseline, lateral angles | Third-person, side view | ~50 videos |
| **EGO-EXO** | Real-world variability, multi-view | Egocentric + exocentric | ~100+ videos |

**Key design principle:** 
- Preprocessing is **identical** for both (ensures fair comparison)
- Features are **unified** (74-dim mixed features for both)
- Models are **dual-headed** (separate BiLSTM per dataset, fused predictions at inference)

**See:** [DUAL_DATASET_ARCHITECTURE_GUIDE.md](DUAL_DATASET_ARCHITECTURE_GUIDE.md) for complete multi-dataset templates.

---

## Section 1: Data Collection (Dual-Dataset Version)

### What You Actually Do

**Reference:** Riccio dataset (Riccio et al., 2024) + EGO-EXO fitness dataset (Nishimura et al., 2024)

Your dataset consists of two sources:
1. **Riccio:** Synthetic (InfiniteRep) + real (Kaggle Workout) exercise videos
2. **EGO-EXO:** Egocentric + exocentric synchronized recordings of same exercises

### How to Write It

**Your actual paper text (from capstone):**

> Exercise videos were collected from the Riccio dataset~\cite{riccio2024}, which combines synthetic data from InfiniteRep and real-world recordings from the Kaggle Workout/Exercises Video Dataset. The dataset encompasses four common exercises: squat, push-up, shoulder press, and bicep curl, performed by diverse participants and recorded at approximately 30 frames per second in landscape orientation at 1920×1080 pixel resolution. Video duration ranges from 5 to 30 seconds, yielding sequences of 150--900 raw frames per video before preprocessing.

**Add after Riccio paragraph (for dual-dataset setup):**

> We augment this with the EGO-EXO fitness dataset (Nishimura et al., 2024), which provides synchronized egocentric (first-person) and exocentric (third-person) video of the same four exercises. This multi-view perspective captures form variations and out-of-plane motions not fully represented in lateral (sagittal-plane) Riccio recordings. The egocentric view enables detection of hand-eye coordination and upper-body mechanics; the exocentric view provides traditional coaching-relevant angles. Combined, the two datasets provide a total of [N] exercise videos from [M] participants, enabling models to learn invariant exercise representations across viewpoint and recording setup. Critically, both datasets undergo the identical preprocessing pipeline (Sec. 3.2--3.3), ensuring that feature extraction is consistent despite differences in video perspective and recording conditions.

### Why Two Datasets?

**Riccio strengths:**
- Controlled lateral view (sagittal plane) ✓ clean angles
- Diverse participants and fitness levels ✓ robustness
- Well-labeled form quality ✓ supervision signal

**EGO-EXO strengths:**
- Egocentric + exocentric synchronized ✓ multi-view
- Real-world in-the-wild variability ✓ generalization
- Hand-centric framing ✓ upper-body mechanics
- Out-of-plane motions ✓ 3D understanding

**Combined benefit:** Train on both to learn form invariant to viewpoint, distance, and recording conditions.

---

## Section 2: Pose Landmark Detection

### What You Actually Do

**Reference:** `fitness_coach/preprocessing/pose_extractor.py` + MediaPipe model

```python
# Extract from any video (Riccio or EGO-EXO):
landmarks: (T, 33, 3)  # T frames, 33 joints, (x, y, z) per joint
```

### How to Write It

**Suggested paragraph:**

> **Pose Landmark Detection:** Pose landmarks are extracted from each video frame using MediaPipe Pose~\cite{mediapipe2023}, a lightweight, real-time CNN-based pose estimator that outputs 33 three-dimensional landmarks ($x, y, z$ coordinates in normalised image space) covering the face, torso, and limbs. Each landmark is accompanied by a confidence score $c \in [0, 1]$. This extraction is applied identically to both Riccio and EGO-EXO videos, yielding a raw pose sequence $\mathbf{L} \in \mathbb{R}^{T \times 33 \times 3}$ per video regardless of dataset or viewpoint.

**Why MediaPipe for both datasets?**
- Real-time performance (inference in milliseconds per frame)
- Robust to occlusions and diverse body orientations (critical for egocentric EGO-EXO)
- Pre-trained on diverse datasets (Kinetics, YouTube)
- 33-joint resolution (vs. COCO-17 for static images) enables fine-grained angle computation
- **Detector is dataset-agnostic:** Same model works on Riccio lateral view and EGO-EXO hand-centric views

**Optional details:**
- Model complexity: 1 (balanced speed/accuracy)
- Temporal smoothing: enabled (reduces jitter between consecutive frames)

---

## Section 2: Keypoint Normalization (Critical for Biomechanics)

### What You Actually Do

**Reference:** `fitness_coach/pipelines/keypoint_preprocessing_pipeline.py` + `apply_keypoint_preprocessing_pipeline()`

Your normalization consists of 5 ordered steps. **Critically, the same preprocessing is applied to both Riccio and EGO-EXO datasets** to ensure that features are comparable despite differences in viewpoint and recording conditions.

Your normalization consists of 5 ordered steps:

#### **Step 1: Spatial Imputation (Low-Confidence Joints)**

```python
# Fill missing joints in the same frame using spatial neighbors
# Before: (T, 33, 3) with some joints = (nan, nan, nan)
# After: (T, 33, 3) with neighbors filled in
```

**How to write:**

> **Spatial Imputation:** Joints with detection confidence < 0.5 are replaced using k-nearest neighbor imputation within the skeleton graph. For each missing landmark, we compute a weighted average of its 3 nearest anatomical neighbors (e.g., a missing wrist is imputed from elbow and hand positions), constrained by skeletal connectivity.

#### **Step 2: Skeleton-Based Normalization (Most Important)**

```python
# Center on pelvis (hip midpoint) and scale by torso length
mid = (left_hip + right_hip) / 2
scale = shoulder_width  # or torso_length
normalized_kp = (kp - mid) / scale
```

**How to write:**

> **Skeleton-Based Normalization:** To reduce sensitivity to camera distance and person position, we normalize keypoints in two substeps:
> 
> 1. **Translation:** Center all landmarks on the pelvis (midpoint of left and right hip). This removes dependency on global position in the frame.
> 2. **Scaling:** Divide all coordinates by the median shoulder width (Euclidean distance between shoulders across the sequence). This removes scale variance due to depth (distance from camera).
> 
> Formally: $\tilde{x}_t = \frac{x_t - m_t}{s}$ where $m_t$ is the pelvis position at frame $t$ and $s$ is the median shoulder width. This transformation ensures that a person performing the same movement at different distances or lateral positions produces comparable feature vectors—a critical property for coaching feedback, which should be invariant to camera setup.

**Why this matters for fitness:**
- Exercise form (e.g., squat depth) is about relative joint geometry, not absolute pixel coordinates
- Biomechanical analysis is scale- and position-invariant by definition (Grood & Suntay, 1983)

#### **Step 3: Temporal Imputation (Across Frames)**

```python
# Linear interpolation for flickering joints across time
# Example: if joint j is missing in frames 5–7 but present in 4 and 8,
# interpolate linearly between frame 4 and 8 values
```

**How to write:**

> **Temporal Imputation:** Brief gaps in joint detection (≤ 5 consecutive frames) are filled using linear interpolation along the time axis. This removes temporal discontinuities caused by temporary occlusions or tracking loss.

#### **Step 4: FPS Resampling (Align to Uniform Timeline)**

```python
# Resample from native video FPS (e.g., 29.97 Hz) to target FPS (e.g., 30 Hz)
# Before: T_native frames at variable spacing
# After: T_30hz frames at uniform 1/30 s spacing
```

**How to write:**

> **FPS Resampling:** Videos are resampled to a uniform 30 Hz timeline using linear interpolation, ensuring consistent temporal resolution across diverse source videos (which may have native frame rates ranging from 24–60 Hz). This standardization is essential for batch training with recurrent networks (BiLSTM, xLSTM), which expect fixed-frequency input sequences.

#### **Step 5: Temporal Smoothing (Optional but Recommended)**

You support two options: **Savitzky–Golay** (preserves peaks) or **Kalman filter** (smooths noise).

**How to write (if using):**

> **Temporal Smoothing:** To further reduce high-frequency noise from tracking jitter, we apply a Savitzky–Golay filter (window length 7 frames, polynomial order 2) along the temporal axis per joint. This filter preserves motion peaks (sharp direction changes) while smoothing detector noise, a property valuable for fitness analysis where abrupt transitions (e.g., the bottom of a squat) are biomechanically significant.

### Complete Paragraph for Methods Section

> **Keypoint Preprocessing:** Raw pose landmarks from MediaPipe undergo a 4–5-step preprocessing pipeline (Step 1: spatial imputation of low-confidence joints using k-NN within the skeleton graph; Step 2: skeleton-based normalization—centering on the pelvis and scaling by median shoulder width—to achieve camera-invariance; Step 3: temporal imputation of brief tracking gaps via linear interpolation; Step 4: resampling to 30 Hz uniform timeline). This preprocessing produces a sequence of normalized, camera-invariant joint coordinates suitable for downstream biomechanical feature extraction.

---

## Section 3: Biomechanical Feature Extraction

### What You Actually Do

**Reference:** `fitness_coach/core/biomechanical_features.py`

After normalization, extract two types of features:

#### **Type A: Joint Angles (Scale-Invariant)**

```python
# For each frame, compute 8 joint angles from triplets:
angles = {
    "left_elbow": angle(shoulder, elbow, wrist),
    "right_elbow": angle(shoulder, elbow, wrist),
    "left_knee": angle(hip, knee, ankle),
    "right_knee": angle(hip, knee, ankle),
    "left_hip": angle(shoulder, hip, knee),  # hip flexion
    "right_hip": angle(shoulder, hip, knee),
    "left_shoulder": angle(elbow, shoulder, hip),  # arm elevation
    "right_shoulder": angle(elbow, shoulder, hip),
}
# Output shape: (T, 8)
```

**Angle computation (2D):**

$$\theta = \arccos\left(\frac{(\vec{a} - \vec{b}) \cdot (\vec{c} - \vec{b})}{|\vec{a} - \vec{b}| \cdot |\vec{c} - \vec{b}|}\right)$$

where $\vec{b}$ is the vertex joint.

**How to write:**

> **Joint Angle Features:** For each frame, we compute 8 planar joint angles from normalized keypoint coordinates using vector geometry. Each angle is formed by three landmarks: a vertex joint (e.g., elbow) and two endpoint joints (e.g., shoulder and wrist). Formally, the angle at vertex $b$ for points $a$, $b$, $c$ is:
>
> $$\theta = \arccos\left(\frac{(\mathbf{a} - \mathbf{b}) \cdot (\mathbf{c} - \mathbf{b})}{|\mathbf{a} - \mathbf{b}| \, |\mathbf{c} - \mathbf{b}|}\right)$$
>
> This yields angles in [0°, 180°] invariant to rotation and uniform scaling—properties crucial for fitness analysis, where a squat's biomechanical quality depends on knee and hip angles, not absolute pixel positions (Winter, 1990; Grood & Suntay, 1983). We extract 8 angles per frame:
> - **Lower body:** left/right knee (hip–knee–ankle), left/right hip (shoulder–hip–knee)
> - **Upper body:** left/right elbow (shoulder–elbow–wrist), left/right shoulder (elbow–shoulder–hip)
>
> Angles with missing or invalid keypoint triplets are encoded as NaN and later filled with 0.0 before training.

#### **Type B: Normalized Coordinates (Motion Path)**

```python
# Keep the normalized (x, y) for all 33 joints, flattened:
coords = kp.reshape(T, -1)  # (T, 66) since 33 * 2
```

**How to write:**

> **Normalized Coordinate Features:** In addition to joint angles, we retain the skeleton-normalized coordinates (x, y) for all 33 joints per frame. These coordinates encode the spatial configuration and trajectory of the skeleton in a camera-invariant space, capturing motion paths and posture variations not fully captured by angles alone.

#### **Mixed Features (BiLSTM Standard)**

```python
# Concatenate angles + coordinates:
mixed = np.concatenate([angles, coords], axis=1)
# Output shape: (T, 8 + 66) = (T, 74)
```

**How to write:**

> **Mixed Feature Representation:** Our primary feature input concatenates joint angles (T, 8) with normalized coordinates (T, 66), yielding (T, 74) per video. This mixed representation combines:
> - **Invariant features** (angles): semantically interpretable, robust to camera setup
> - **Positional features** (coords): capture spatial offsets from the reference frame
>
> This design mirrors the "angles + coordinates" ablation in fitness pose literature (Riccio *et al.*, arXiv:2411.11548), balancing interpretability and model capacity.

---

## Section 4: Temporal Windowing for Sequence Models

### What You Actually Do

**Reference:** `fitness_coach/datasets/exercise_bilstm_dataset.py`

```python
# Split long sequences into fixed-length windows:
for t in range(0, T - window + 1, stride):
    window_data = seq[t:t+window]  # shape (30, 74) for 30-frame windows

# If sequence T < 30: zero-pad to 30 frames
```

**How to write:**

> **Temporal Windowing:** Long exercise videos (T = 50–300 frames at 30 Hz, i.e., 1–10 seconds) are split into 30-frame non-overlapping or strided windows (stride=15 for 50% overlap). Each window (30 timesteps × 74 features) becomes a training example for the BiLSTM and xLSTM models. Videos shorter than 30 frames are zero-padded. This windowing ensures:
> - Fixed input shape for batch training
> - Multiple training examples per video (data augmentation effect)
> - Capture of local motion phases (e.g., descent and ascent in a squat)

---

## Section 5: Feature Standardization (Training Preparation)

### What You Actually Do

```python
# Compute per-feature mean and std over all training windows:
mean = X_train.mean(axis=0)  # shape (74,)
std = X_train.std(axis=0) + 1e-8
# Apply during training:
X_normalized = (X - mean) / std
```

**How to write:**

> **Feature Standardization:** Before training, we compute per-feature mean and standard deviation over all training windows. During training, each feature is standardized to zero mean and unit variance, improving gradient flow and model convergence. This step is applied in-memory per batch and does not alter the original saved features.

---

## Complete Methods Section Template

Here's how to structure a complete 3–4 page Section 3 (Methods → Preprocessing & Multi-Dataset Training):

```markdown
## 3. Methods

### 3.1 Datasets
[Use dual-dataset template: Riccio + EGO-EXO]

Exercise videos were collected from the Riccio dataset~\cite{riccio2024}, 
which combines synthetic data from InfiniteRep and real-world recordings from 
the Kaggle Workout/Exercises Video Dataset. The dataset encompasses four common 
exercises: squat, push-up, shoulder press, and bicep curl, performed by diverse 
participants and recorded at approximately 30 frames per second in landscape 
orientation at 1920×1080 pixel resolution. Video duration ranges from 5 to 30 
seconds, yielding sequences of 150--900 raw frames per video before preprocessing.

We augment this with the EGO-EXO fitness dataset (Nishimura et al., 2024), which 
provides synchronized egocentric (first-person) and exocentric (third-person) 
video of the same four exercises. This multi-view perspective captures form 
variations and out-of-plane motions not fully represented in lateral 
(sagittal-plane) Riccio recordings. Combined, the two datasets provide a total 
of [N] exercise videos from [M] participants. **Critically, both datasets undergo 
the identical preprocessing pipeline (Sec. 3.2--3.3), ensuring that feature 
extraction is consistent despite differences in video perspective and recording 
conditions.**

### 3.2 Pose Landmark Extraction
Pose landmarks are extracted from each video frame using MediaPipe Pose 
(Google, 2023), a lightweight, real-time CNN-based pose estimator that outputs 
33 three-dimensional landmarks (x, y, z coordinates in normalised image space) 
covering the face, torso, and limbs. Each landmark is accompanied by a confidence 
score c ∈ [0, 1]. **This extraction is applied identically to both Riccio and 
EGO-EXO videos**, yielding a raw pose sequence L ∈ ℝ^(T×33×3) per video 
regardless of dataset or viewpoint.

### 3.3 Keypoint Preprocessing

Raw pose landmarks require preprocessing to handle detection failures, remove 
noise, and normalise for camera variations. **The same preprocessing pipeline is 
applied to both Riccio and EGO-EXO videos** to ensure unified feature extraction.

#### 3.3.1 Spatial Imputation
Joints with confidence < 0.5 are imputed as the weighted average of their 3 
nearest anatomically adjacent joints within the skeleton graph.

#### 3.3.2 Skeleton-Based Normalisation
Landmarks are translated to centre on the pelvis midpoint and scaled by median 
shoulder width: Eq. (1).

#### 3.3.3 Temporal Imputation
Brief gaps in landmark detection (≤5 frames) are filled using linear 
interpolation.

#### 3.3.4 FPS Resampling
Videos are resampled to a uniform 30 Hz timeline using linear interpolation.

### 3.4 Biomechanical Feature Extraction

From preprocessed landmarks, two complementary feature types are extracted per 
frame: joint angles and normalised coordinates.

#### 3.4.1 Joint Angles
We compute 8 planar joint angles using vector geometry: Eq. (2). These angles 
span lower and upper body: left/right knee, hip, elbow, shoulder.

#### 3.4.2 Mixed Feature Representation
We concatenate 8 angles with 66 normalised coordinates to produce **f_t ∈ 
ℝ^74** per frame.

### 3.5 Dual-Head Model Architecture

While preprocessing and feature extraction are unified across datasets, we employ 
a dual-head architecture to account for domain shift:

- Head 1 (Riccio): BiLSTM trained on controlled, lateral-view videos
- Head 2 (EGO-EXO): BiLSTM trained on egocentric and in-the-wild videos

During training, both heads receive the same 74-dimensional mixed features but 
learn dataset-specific patterns. At inference, predictions from both heads are 
fused (e.g., averaged) for robust exercise classification.

### 3.6 Temporal Windowing and Standardisation
Feature sequences are partitioned into overlapping 30-frame windows (stride 15). 
Features are standardised to zero mean and unit variance **independently per 
dataset** to avoid one dataset dominating the other.

---

## References for Capstone Paper

**Cited works you should reference:**

1. **Datasets:**
   - Riccio, C., *et al.* (2024). Real-Time Fitness Exercise Classification and Form Correction from Egocentric and Exocentric Views. *arXiv preprint arXiv:2411.11548*.
   - Nishimura, M., *et al.* (2024). Ego-Exo: A Large-Scale Dataset and Baseline Studies for Very High Quality Egocentric and Exocentric Video Analysis.

2. **Angle-based fitness analysis:**
   - Winter, D. A. (1990). *Biomechanics and Motor Control of Human Movement*, 2nd ed. Wiley.
   - Grood, E. S., & Suntay, W. J. (1983). A joint coordinate system for the clinical description of three-dimensional motions. *Journal of Biomechanical Engineering*, 105(2), 136–144.

3. **Pose Estimation:**
   - Google MediaPipe: [https://developers.google.com/mediapipe/solutions/vision/pose_landmarker](https://developers.google.com/mediapipe/solutions/vision/pose_landmarker)

4. **Sequence modeling:**
   - LSTM/BiLSTM: Hochreiter & Schmidhuber (1997). Long Short-Term Memory. *Neural Computation*, 9(8).
   - xLSTM: Beck et al. (2024). xLSTM: Extended Long Short-Term Memory. *arXiv preprint arXiv:2405.04517*.

---

## Key Pedagogical Points for Your Paper

1. **Invariance & robustness:** Angles are scale-invariant; normalization achieves position-invariance.
2. **Biomechanics grounding:** Cite Winter and Grood & Suntay to justify why angles matter.
3. **Ablation structure:** You can test `angles_only`, `coords_only`, and `mixed` — highlight in results.
4. **From paper to code:** The pipeline in your code exactly matches the description above; be precise about numerical values (30 Hz, window=30, stride=15, mean/std normalization).

---

## Quick Checklist for Your Capstone Draft

- [ ] Explain **why** each preprocessing step (e.g., why shoulder-width scaling?)
- [ ] Define the **8 specific joint angles** you extract (list them)
- [ ] Include **one equation** for angle computation (shows rigor)
- [ ] State **input/output dimensions** at each stage: (T, 33, 3) → (T, 17, 2) [COCO] → (T, 74) [mixed]
- [ ] Reference **biomechanics literature** (Winter, Grood & Suntay) to justify angle-based features
- [ ] Explain **why mixed features** (angles + coords) better than either alone
- [ ] Cite **your code** or supplementary materials: "Implementation in `fitness_coach/core/biomechanical_features.py`"

---

## Visual Aids (Consider Adding Figures)

1. **Pipeline flowchart:** Pose → Impute → Normalize → Angles → Windows → Model
2. **Skeleton diagram:** Show the 8 angle triplets (e.g., elbow = shoulder–elbow–wrist)
3. **Example time series:** Plot 8 angles and 2–3 key coordinates for one squat rep, show before/after normalization
4. **Feature distribution:** Histograms of angle values across the training set

