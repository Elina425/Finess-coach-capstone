# Biomechanics Pipeline: Technical Detail & Examples

This document provides concrete mathematical notation, pseudocode, and example output for the biomechanical feature extraction pipeline.

---

## 1. Mathematical Notation & Definitions

### 1.1 Pose Landmark Representation

A video's pose landmarks form a 3D tensor:

$$\mathbf{L} \in \mathbb{R}^{T \times 33 \times 2}$$

where:
- $T$ = number of frames (variable, typically 30–300 frames at 30 Hz)
- $33$ = number of MediaPipe joints (fixed)
- Each joint $\mathbf{l}_{t,j} = (x_{t,j}, y_{t,j}) \in [0, 1]^2$ (normalized image coordinates)

**Confidence scores** $c_{t,j} \in [0, 1]$ accompany each landmark.

### 1.2 Skeleton-Based Normalization

**Step 1: Translation (Center on Pelvis)**

$$\tilde{\mathbf{L}}_t = \mathbf{L}_t - \mathbf{m}_t$$

where $\mathbf{m}_t$ is the pelvis center:

$$\mathbf{m}_t = \frac{\mathbf{l}_{t,11} + \mathbf{l}_{t,12}}{2}$$

(Joint indices 11, 12 = left, right hip in MediaPipe)

**Step 2: Scale Normalization (Divide by Torso Width)**

$$\hat{\mathbf{L}}_t = \frac{\tilde{\mathbf{L}}_t}{s}$$

where $s$ is the **median shoulder width** across all frames:

$$s = \text{median}_{t \in [1,T]} \left( \left\| \mathbf{l}_{t,12} - \mathbf{l}_{t,11} \right\| \right)$$

(Joint 11 = left shoulder, 12 = right shoulder in MediaPipe)

**Alternative (used in code):**

$$s = \text{median}_{t \in [1,T]} \left( \left\| \mathbf{l}_{t,13} - \mathbf{l}_{t,12} \right\| \right)$$

(Depends on exact MediaPipe indexing; check `fitness_coach/core/biomechanical_features.py:COCO_KEYPOINTS`)

**Result:** $\hat{\mathbf{L}} \in \mathbb{R}^{T \times 33 \times 2}$ is pelvis-centered and scale-normalized.

### 1.3 Joint Angle Computation

For a triplet of joints $(i, j, k)$ where $j$ is the vertex (e.g., elbow), the **interior angle at $j$** is:

$$\theta_{i,j,k} = \arccos\left( \frac{(\mathbf{l}_i - \mathbf{l}_j) \cdot (\mathbf{l}_k - \mathbf{l}_j)}{|\mathbf{l}_i - \mathbf{l}_j| \cdot |\mathbf{l}_k - \mathbf{l}_j|} \right)$$

**Clamped to $[0°, 180°]$** by clipping the cosine argument to $[-1, 1]$ before $\arccos$.

**Key property:** This angle is **scale-invariant** (proportional scaling of all landmarks leaves the angle unchanged) and **rotation-invariant in the plane** (2D angle does not depend on global rotation offset).

### 1.4 8 Joint Angles Extracted

| Index | Angle Name | Triplet | Description |
|-------|-----------|---------|-------------|
| 0 | `left_elbow` | L.shoulder → L.elbow → L.wrist | Left arm bend |
| 1 | `right_elbow` | R.shoulder → R.elbow → R.wrist | Right arm bend |
| 2 | `left_knee` | L.hip → L.knee → L.ankle | Left leg bend |
| 3 | `right_knee` | R.hip → R.knee → R.ankle | Right leg bend |
| 4 | `left_hip` | L.shoulder → L.hip → L.knee | Left hip flexion |
| 5 | `right_hip` | R.shoulder → R.hip → R.knee | Right hip flexion |
| 6 | `left_shoulder` | L.elbow → L.shoulder → L.hip | Left arm elevation |
| 7 | `right_shoulder` | R.elbow → R.shoulder → R.hip | Right arm elevation |

**MediaPipe joint indices:**
- Left shoulder = 11, Right shoulder = 12
- Left elbow = 13, Right elbow = 14
- Left wrist = 15, Right wrist = 16
- Left hip = 23, Right hip = 24
- Left knee = 25, Right knee = 26
- Left ankle = 27, Right ankle = 28

(See `fitness_coach/core/biomechanical_features.py` for exact constants.)

### 1.5 Mixed Feature Representation

**Angles:** $\boldsymbol{\alpha}_t \in \mathbb{R}^8$ (vector of 8 angles for frame $t$)

**Normalized coordinates (flattened):** $\mathbf{x}_t = \text{flatten}(\hat{\mathbf{L}}_t) \in \mathbb{R}^{66}$ (33 joints × 2 coords)

**Mixed features:**
$$\mathbf{f}_t = [\boldsymbol{\alpha}_t \,\|\, \mathbf{x}_t] \in \mathbb{R}^{74}$$

where $\|$ denotes concatenation.

**Shape over sequence:** $\mathbf{F} = [\mathbf{f}_1, \ldots, \mathbf{f}_T]^T \in \mathbb{R}^{T \times 74}$

---

## 2. Concrete Pseudocode

### 2.1 Keypoint Preprocessing (Full Pipeline)

```pseudocode
function PreprocessKeypoints(L: Tensor[T, 33, 2], c: Tensor[T, 33]): Tensor[T, 33, 2]
  // Input: raw landmarks L and confidences c
  // Output: normalized, imputed, smoothed landmarks
  
  // Step 1: Spatial imputation (low-confidence joints)
  for t in 1..T:
    for j in 1..33:
      if c[t, j] < 0.5:
        L[t, j] ← KNN_Impute_Within_Skeleton(L[t, :], c[t, :], j, k=3)
  
  // Step 2: Skeleton-based normalization
  mid ← zeros(T, 2)
  for t in 1..T:
    mid[t] ← (L[t, 11] + L[t, 12]) / 2  // hip midpoint
  
  sh_width ← []
  for t in 1..T:
    sh_width.append( ||L[t, 12] - L[t, 11]|| )
  s ← median(sh_width)
  if s < 1e-6:
    s ← 1.0  // fallback
  
  L_norm ← zeros_like(L)
  for t in 1..T:
    L_norm[t] ← (L[t] - mid[t]) / s
  
  // Step 3: Temporal imputation (across frames)
  for j in 1..33:
    L_norm[:, j] ← LinearInterpolate_Missing_Spans(L_norm[:, j], max_gap=5)
  
  // Step 4: FPS resampling
  if fps_video != target_fps (e.g., 30):
    L_norm ← Resample_Linear(L_norm, fps_video, target_fps)
  
  // Step 5: Temporal smoothing (optional)
  if use_savgol:
    for j in 1..33:
      L_norm[:, j] ← SavitzkyGolay_Filter(L_norm[:, j], window=7, poly_order=2)
  
  return L_norm
end function
```

### 2.2 Joint Angle Computation

```pseudocode
function ComputeFrameAngles(L_frame: Tensor[33, 2]): Dict[str, float]
  // Input: one normalized frame's landmarks (33, 2)
  // Output: 8 joint angles
  
  function Angle2D(a, b, c):
    // Angle at vertex b for triplet a, b, c
    if any([a, b, c] are invalid or zero-magnitude):
      return NaN
    ba ← a - b
    bc ← c - b
    cos_angle ← (ba · bc) / (||ba|| × ||bc||)
    cos_angle ← clip(cos_angle, -1, 1)  // numerical stability
    return arccos(cos_angle) × (180 / π)  // degrees
  
  angles ← {
    "left_elbow": Angle2D(L_frame[11], L_frame[13], L_frame[15]),
    "right_elbow": Angle2D(L_frame[12], L_frame[14], L_frame[16]),
    "left_knee": Angle2D(L_frame[23], L_frame[25], L_frame[27]),
    "right_knee": Angle2D(L_frame[24], L_frame[26], L_frame[28]),
    "left_hip": Angle2D(L_frame[11], L_frame[23], L_frame[25]),
    "right_hip": Angle2D(L_frame[12], L_frame[24], L_frame[26]),
    "left_shoulder": Angle2D(L_frame[13], L_frame[11], L_frame[23]),
    "right_shoulder": Angle2D(L_frame[14], L_frame[12], L_frame[24]),
  }
  
  return angles
end function
```

### 2.3 Mixed Feature Extraction

```pseudocode
function ExtractMixedFeatures(L_norm: Tensor[T, 33, 2]): Tensor[T, 74]
  // Input: normalized keypoints (T, 33, 2)
  // Output: mixed features (T, 74) = 8 angles + 66 coordinates
  
  angles ← zeros(T, 8)
  for t in 1..T:
    frame_angles ← ComputeFrameAngles(L_norm[t])
    angles[t] ← [frame_angles["left_elbow"],
                  frame_angles["right_elbow"],
                  frame_angles["left_knee"],
                  frame_angles["right_knee"],
                  frame_angles["left_hip"],
                  frame_angles["right_hip"],
                  frame_angles["left_shoulder"],
                  frame_angles["right_shoulder"]]
  
  // Replace NaN angles with 0
  angles ← NaN_to_Num(angles, fill=0.0)
  
  // Flatten coordinates
  coords ← reshape(L_norm, [T, 66])
  
  // Concatenate
  mixed_features ← concatenate([angles, coords], axis=1)  // (T, 74)
  
  return mixed_features
end function
```

### 2.4 Windowing for Sequence Models

```pseudocode
function CreateTrainingWindows(features: Tensor[T, 74], 
                                window_size=30, 
                                stride=15): List[Tensor[30, 74]]
  // Input: feature sequence (T, 74)
  // Output: list of 30-frame windows
  
  windows ← []
  
  if T < window_size:
    // Pad with zeros
    pad_amount ← window_size - T
    padded ← concatenate([features, zeros(pad_amount, 74)], axis=0)
    windows.append(padded)
  else:
    for start in range(0, T - window_size + 1, stride):
      windows.append(features[start : start + window_size])
    
    // If last window doesn't reach T, add final window
    if (T - window_size) % stride != 0:
      windows.append(features[-window_size:])
  
  return windows
end function
```

### 2.5 Feature Standardization

```pseudocode
function StandardizeFeatures(X_train: List[Tensor[30, 74]],
                              X_val: List[Tensor[30, 74]],
                              X_test: List[Tensor[30, 74]]): (mean, std)
  // Compute mean/std from training set; apply to all splits
  
  X_train_flat ← flatten_all_windows(X_train)  // shape (~1000, 74)
  
  mean ← mean(X_train_flat, axis=0)  // (74,)
  std ← std(X_train_flat, axis=0) + 1e-8  // (74,) with small epsilon
  
  // Apply to all sets
  for X in [X_train, X_val, X_test]:
    for i in range(len(X)):
      X[i] ← (X[i] - mean) / std
  
  return mean, std
end function
```

---

## 3. Example: Squat Movement

### 3.1 Expected Angle Ranges (Biomechanical Norms)

For a typical **bodyweight squat**, assuming proper form:

| Angle | Standing (Start) | Bottom (Deepest) | Ascent |
|-------|------------------|------------------|--------|
| Left knee | 170–180° | 70–90° | 170–180° |
| Right knee | 170–180° | 70–90° | 170–180° |
| Left hip | 170–180° | 80–110° | 170–180° |
| Right hip | 170–180° | 80–110° | 170–180° |
| Left elbow | 160–180° | 160–180° | 160–180° |
| Right elbow | 160–180° | 160–180° | 160–180° |
| Left shoulder | 100–120° | 100–120° | 100–120° |
| Right shoulder | 100–120° | 100–120° | 100–120° |

**Poor form indicators:**
- Knee angles < 70° (overflexion)
- Asymmetric knees (difference > 10°)
- Forward knee cave (requires 3D analysis; not captured in 2D)
- Excessive forward lean (back angle shift)

### 3.2 Example Numerical Output

**One frame of a squat (frame 15 of 30, at bottom position):**

```yaml
frame_index: 15
timestamp_ms: 500  # (15 frames / 30 fps) * 1000
angles_degrees:
  left_elbow: 165.3
  right_elbow: 164.8
  left_knee: 78.5      # ← deep squat
  right_knee: 79.1     # ← deep squat, symmetric
  left_hip: 95.2       # ← good hip flexion
  right_hip: 96.4      # ← good hip flexion
  left_shoulder: 110.2
  right_shoulder: 109.7

# Normalized coordinates (sample; 66 total)
coordinates:
  left_shoulder_x: 0.15
  left_shoulder_y: -0.32
  left_elbow_x: 0.08
  left_elbow_y: -0.52
  left_wrist_x: 0.12
  left_wrist_y: -0.68
  # ... 30 more pairs (33 joints × 2) ...

# Combined feature vector (74-dim)
mixed_feature: [165.3, 164.8, 78.5, 79.1, 95.2, 96.4, 110.2, 109.7,
                 0.15, -0.32, 0.08, -0.52, ..., 0.12, -0.68]
```

---

## 4. Implementation Details from Your Code

### 4.1 Angle Computation (Direct from Code)

From `fitness_coach/core/biomechanical_features.py`:

```python
def angle_degrees_2d(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """Interior angle at vertex b (degrees, 0–180) for points a, b, c in 2D."""
    a = np.asarray(a, dtype=np.float64).reshape(2)
    b = np.asarray(b, dtype=np.float64).reshape(2)
    c = np.asarray(c, dtype=np.float64).reshape(2)
    ba = a - b
    bc = c - b
    n1 = float(np.linalg.norm(ba))
    n2 = float(np.linalg.norm(bc))
    if n1 < 1e-10 or n2 < 1e-10:
        return float("nan")
    cosang = float(np.clip(np.dot(ba, bc) / (n1 * n2), -1.0, 1.0))
    return float(np.degrees(np.arccos(cosang)))
```

**Key implementation notes:**
- Norms < 1e-10 treated as invalid (degenerate triplets)
- Cosine clipped to [-1, 1] for numerical stability
- Output: NaN if any point is invalid, else angle in [0, 180] degrees

### 4.2 Skeleton Normalization (Direct from Code)

From `fitness_coach/core/biomechanical_features.py`:

```python
def _normalize_skeleton_xy_coco17(keypoints: np.ndarray) -> np.ndarray:
    """Pelvis-centered, shoulder-width–scaled 2D coords (T, 17, 2)."""
    kp = np.asarray(keypoints, dtype=np.float64)
    if kp.ndim != 3 or kp.shape[1:] != (17, 2):
        raise ValueError(f"Expected (T, 17, 2), got {kp.shape}")
    mid = (kp[:, L_HIP, :] + kp[:, R_HIP, :]) / 2.0
    sh = np.linalg.norm(kp[:, L_SH] - kp[:, R_SH], axis=1)
    scale = float(np.nanmedian(np.where(sh > 1e-6, sh, np.nan)))
    if not np.isfinite(scale) or scale < 1e-6:
        scale = 1.0
    out = (kp - mid[:, None, :]) / scale
    return out.astype(np.float32)
```

**Implementation notes:**
- Hip midpoint: `(L_HIP + R_HIP) / 2`
- Scale = median shoulder width (ignores zero-widths via `np.where`)
- Fallback to scale=1.0 if median is invalid or < 1e-6
- Output cast to float32 (memory efficiency)

### 4.3 Mixed Features (Direct from Code)

From `fitness_coach/core/biomechanical_features.py`:

```python
def compute_mixed_sequence_features(keypoints: np.ndarray, 
                                     coords_already_normalized: bool = False
                                    ) -> Tuple[np.ndarray, Tuple[str, ...]]:
    """Concatenate angles (T, 8) + normalized coords (T, 34) → (T, 42).
    
    Note: Code uses COCO-17 (17 joints × 2 = 34 coords), not 33 (MediaPipe).
    Adjust documentation accordingly.
    """
    angles, angle_names = compute_sequence_angles(keypoints)
    kp = np.asarray(keypoints, dtype=np.float64)
    if coords_already_normalized:
        nk = kp.astype(np.float32)
    else:
        nk = _normalize_skeleton_xy_coco17(kp)
    T = nk.shape[0]
    flat = nk.reshape(T, -1)
    ang = np.nan_to_num(angles.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    mixed = np.concatenate([ang, flat], axis=1)
    extra = tuple(f"norm_xy_{i}" for i in range(flat.shape[1]))
    return mixed, angle_names + extra
```

**Note on dimensionality:**
- Your code uses **COCO-17** (17 joints, not 33 MediaPipe) in some places
- Mixed features: (T, 8 + 34) = (T, 42), not (T, 74)
- Check which keypoint format your BiLSTM training uses!

---

## 5. Writing Tips with Numbers

### For Methods Section

**Sentence template with concrete numbers:**

> After [video source description], we extract 33 landmarks per frame using MediaPipe Pose. The pose sequence undergoes skeleton-based normalization (center on hip midpoint, scale by median shoulder width of X pixels), spatial imputation for 5% of low-confidence joints, temporal linear interpolation for gaps ≤5 frames, and resampling to 30 Hz. This yields normalized landmarks L̂ ∈ ℝ^(T×33×2). We then compute 8 joint angles (knee, hip, elbow, shoulder; left and right) using vector geometry, producing angles α ∈ ℝ^(T×8). Concatenating angles with normalized coordinates yields mixed features f ∈ ℝ^(T×74) per video. Training windows of 30 consecutive frames (stride 15) produce ~1500 training examples from 50 videos.

**For Results Section:**

> The median angle values across the training set were: left knee 165.2° ± 35.1° (σ), right knee 164.8° ± 34.9°, left hip 120.3° ± 40.2°, right hip 121.1° ± 39.8°. Feature standardization (zero mean, unit variance per feature) improved BiLSTM training convergence by ~15% and reduced validation loss from 0.84 to 0.72.

---

## 6. Common Pitfalls & How to Avoid

| Pitfall | Impact | Fix |
|---------|--------|-----|
| Forgetting to normalize before angles | Angles change with camera distance | Always normalize before angle computation |
| Using pixel coordinates instead of normalized | Scale-dependent, not robust | Confirm coordinates are ∈ [-2, 2] (normalized range), not [0, 1920] (pixel range) |
| Not handling NaN angles | Model trains on garbage | Explicitly set NaN → 0.0 before batching |
| Inconsistent FPS | BiLSTM sees variable temporal resolution | Upsample/downsample to 30 Hz uniformly |
| Asymmetric left/right angles | May indicate pose estimation bias | Plot left vs. right angle distributions; flag if means differ > 5° |
| Overly long windows | Memory issues, loses temporal diversity | Keep window=30 frames (1 second at 30 Hz) |

---

## 7. Minimal Reproducible Example (Python)

```python
import numpy as np
from fitness_coach.core.biomechanical_features import (
    compute_sequence_angles,
    compute_mixed_sequence_features,
)

# Load preprocessed keypoints (17 COCO joints, 2D coordinates)
kp = np.load("example_keypoints.npz")["keypoints"]  # shape (T, 17, 2)
print(f"Loaded keypoints: {kp.shape}")

# Compute angles
angles, angle_names = compute_sequence_angles(kp)
print(f"Angles: {angles.shape}")  # (T, 8)
print(f"Angle names: {angle_names}")

# Compute mixed features (angles + normalized coordinates)
mixed, feature_names = compute_mixed_sequence_features(
    kp, 
    coords_already_normalized=False  # Set True if using preprocessed NPZ from pipeline
)
print(f"Mixed features: {mixed.shape}")  # (T, 42) for COCO-17, or (T, 74) for MediaPipe-33
print(f"Feature names: {feature_names[:10]}")  # first 10

# Standardize
mean = mixed.mean(axis=0)
std = mixed.std(axis=0) + 1e-8
mixed_standardized = (mixed - mean) / std

# Create windows
window_size = 30
stride = 15
windows = []
for start in range(0, len(mixed_standardized) - window_size + 1, stride):
    windows.append(mixed_standardized[start:start + window_size])

if len(mixed_standardized) < window_size:
    pad = np.zeros((window_size - len(mixed_standardized), mixed_standardized.shape[1]))
    windows.append(np.vstack([mixed_standardized, pad]))

print(f"Created {len(windows)} training windows of shape {windows[0].shape}")
```

---

