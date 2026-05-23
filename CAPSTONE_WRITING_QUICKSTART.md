# Capstone Paper Writing Guide: Quick Start

This is your entry point for writing Section 3 (Methods) of your capstone paper on biomechanical feature extraction for fitness exercise analysis.

---

## Documents Available

I've created three comprehensive guides in your workspace:

1. **[BIOMECHANICS_PAPER_WRITING_GUIDE.md](BIOMECHANICS_PAPER_WRITING_GUIDE.md)**
   - High-level overview of each pipeline stage
   - Why each step matters (biomechanical grounding)
   - Complete Methods section template
   - Pedagogical points and reference suggestions
   - **Best for:** Understanding the full pipeline conceptually before writing

2. **[BIOMECHANICS_TECHNICAL_DETAILS.md](BIOMECHANICS_TECHNICAL_DETAILS.md)**
   - Mathematical notation and equations
   - Detailed pseudocode for each step
   - Concrete numerical examples (e.g., squat angle ranges)
   - Implementation details directly from your code
   - Common pitfalls and how to avoid them
   - **Best for:** Adding rigor to your Methods section; including equations

3. **[BIOMECHANICS_COPY_PASTE_READY.md](BIOMECHANICS_COPY_PASTE_READY.md)**
   - Pre-written paragraph templates
   - Multiple versions of each section (brief, detailed, with theory)
   - Example Results section text
   - Table and figure captions
   - Ready-to-cite references
   - **Best for:** Rapid drafting—copy templates and adapt with your actual numbers

---

## Writing Workflow (Recommended Steps)

### Step 1: Understand the Pipeline (30 min)
1. Read **BIOMECHANICS_PAPER_WRITING_GUIDE.md** sections 1–5 to grasp the conceptual flow.
2. Look at the visual "Overview: Full Pipeline in 4 Stages" at the top.

### Step 2: Choose Your Tone & Depth (10 min)
Decide: Does your capstone prefer **brief, readable** Methods or **detailed, rigorous** Methods?
- **Brief:** Use the "Template A" or "Template B (Concise)" versions from COPY_PASTE_READY.md
- **Detailed:** Use "Template B" or "Template C" versions; add equations from TECHNICAL_DETAILS.md

### Step 3: Draft Methods Section (60–90 min)
1. Open COPY_PASTE_READY.md
2. For each subsection (3.1 Data, 3.2 Pose, 3.3 Preprocessing, etc.), copy the template you prefer
3. Replace placeholders [N], [M], [frame rate], etc. with your actual dataset numbers
4. Add 1–2 equations (if detailed) from TECHNICAL_DETAILS.md Section 1
5. Keep TON consistent: use past tense, third person, active voice

### Step 4: Add References & Grounding (30 min)
1. From BIOMECHANICS_PAPER_WRITING_GUIDE.md, copy the key references (Winter, Grood & Suntay, Escamilla, etc.)
2. Cite them where you discuss **why** (e.g., "Angles are scale-invariant (Grood & Suntay, 1983)")
3. Format according to your institution's citation style (APA, IEEE, etc.)

### Step 5: Add Results & Discussion (if needed) (30 min)
1. Compute your actual feature statistics from your training set:
   ```python
   angles = np.concatenate(all_angle_sequences)  # (N, 8)
   print(f"Left knee: {angles[:, 2].mean():.1f}° ± {angles[:, 2].std():.1f}°")
   ```
2. Copy a Results template from COPY_PASTE_READY.md → Results section
3. Insert your numbers

### Step 6: Polish & Visualize (30 min)
1. Create 2–3 figures:
   - **Figure A:** Pipeline flowchart (sketch: raw video → pose → preprocess → angles → windows)
   - **Figure B:** Example angle time series (plot angles over 30 frames for one exercise)
   - **Figure C:** Angle distributions (histogram of left vs. right knee angles)
2. Add figure captions from COPY_PASTE_READY.md, adapt to your figures

---

## Quick Reference: Key Numbers to Gather

Before drafting, collect these from your dataset:

```python
# From your training data:
print(f"Number of videos: {len(video_list)}")
print(f"Number of participants: {len(set(participant_ids))}")
print(f"Video frame rates: {set(fps_list)}")
print(f"Video durations (frames): min={min(Ts)}, max={max(Ts)}, median={median(Ts)}")

# From preprocessed features:
angles = np.concatenate(all_angle_sequences)  # (total_frames, 8)
for i, name in enumerate(["left_elbow", "right_elbow", "left_knee", "right_knee",
                           "left_hip", "right_hip", "left_shoulder", "right_shoulder"]):
    print(f"{name}: {angles[:, i].mean():.1f}° ± {angles[:, i].std():.1f}°")

# Feature shapes:
print(f"Mixed features shape: {mixed_sequence.shape}")  # Should be (T, 74) or (T, 42) depending on joint format
print(f"Training windows created: {num_windows}")
print(f"Window shape: (30, {mixed_sequence.shape[1]})")
```

---

## Section-by-Section Guide

### Section 3.1: Data Collection

**From COPY_PASTE_READY.md:**
- Use **Template A** (Brief) for 2–3 sentences
- Use **Template B** (Detailed) for a full paragraph

**Update with your numbers:**
- [source]: Your dataset name (e.g., "the QEVD-Fitness dataset, containing 50 videos of squat exercises")
- [frame rate]: 29.97 fps or 30 fps (typical for most cameras)
- [resolution]: 1920×1080 or your actual resolution
- [duration]: "5 to 30 seconds" or your range

### Section 3.2: Pose Landmark Extraction

**From COPY_PASTE_READY.md:**
- Use **Template A** (Minimal) if brief
- Use **Template B** (With Detail) if comprehensive

**No numbers to update** — just MediaPipe constants (33 landmarks, 33×3 for (x, y, z))

### Section 3.3: Keypoint Preprocessing

**From COPY_PASTE_READY.md:**
- Use **Template A: Full Overview (2 paragraphs)** for complete coverage
- Use **Template B** to emphasize biomechanical justification

**Update with your numbers:**
- Confidence threshold: 0.5 (standard)
- k-NN imputation: k=3 (standard)
- Max interpolation gap: ≤5 frames (standard)
- Target FPS: 30 Hz (standard)
- Savitzky-Goyal: window=7, poly=2 (standard, if used)

### Section 3.4: Biomechanical Feature Extraction

**From COPY_PASTE_READY.md:**
- Use **Template A** (Comprehensive) if your paper emphasizes biomechanics
- Use **Template B** (Concise) for brevity
- Use **Template C** (With Literature) if emphasizing academic rigor

**Include:**
- The angle equation (from TECHNICAL_DETAILS.md Section 1.3)
- The 8 angle names (explicitly list them)
- Mention that NaN angles → 0.0 before training

### Section 3.5: Temporal Windowing

**From COPY_PASTE_READY.md:**
- Use **Template A** for 2–3 sentences
- Use **Template B** for a full paragraph

**Update with your numbers:**
- Window size: 30 frames (standard)
- Stride: 15 frames (standard, for 50% overlap)
- Padding: yes (for sequences T < 30)
- Example: "~1500 training examples from 50 videos" — compute from your data

### Section 3.6: Feature Standardization

**From COPY_PASTE_READY.md:**
- Use **Template A** (all you need)

**No updates** — this is standard practice

---

## Equations to Include (Optional but Recommended)

From **TECHNICAL_DETAILS.md Section 1**, include these equations:

**Equation 1: Joint Angle (2D)**
$$\theta_{i,j,k} = \arccos\left(\frac{(\mathbf{l}_i - \mathbf{l}_j) \cdot (\mathbf{l}_k - \mathbf{l}_j)}{|\mathbf{l}_i - \mathbf{l}_j| \cdot |\mathbf{l}_k - \mathbf{l}_j|}\right)$$

Where: i, j, k = joint indices; j is the vertex; θ ∈ [0°, 180°]

**Equation 2: Skeleton Normalization**
$$\hat{\mathbf{L}}_t = \frac{\mathbf{L}_t - \mathbf{m}_t}{s}$$

Where: $\mathbf{m}_t$ = pelvis center (hip midpoint); $s$ = median shoulder width

**Equation 3: Feature Standardization**
$$X_{\text{std}} = \frac{X - \mu}{\sigma}$$

Where: μ = per-feature mean (from training set); σ = per-feature std + 1e-8

---

## Common Writing Patterns (Use These Repeatedly)

### Pattern A: Explaining a Technical Step
> [Step Name] [does what]: [concrete detail about input/output]. This is important because [biomechanical or ML justification]. [Optional: cite relevant paper].

**Example:**
> Skeleton-based normalization achieves camera-invariance by translating to the pelvis and scaling by median shoulder width. This ensures that a person performing the same movement at different distances produces nearly identical features, a property critical for form coaching which depends on relative limb geometry, not absolute pixel coordinates (Grood & Suntay, 1983).

### Pattern B: Specifying Hyperparameters
> We use [specific value] for [parameter], [optional: because ...]. This yields [output shape or effect].

**Example:**
> We use a window size of 30 frames (stride 15, yielding 50% overlap). This corresponds to ~1 second of video at 30 Hz and captures one complete movement phase (e.g., squat descent + ascent), yielding ~1500 training examples from 50 videos.

### Pattern C: Justifying a Design Choice
> Our mixed feature design (angles + coordinates) balances [tradeoff 1] and [tradeoff 2]. Prior work (Riccio et al., 2024) shows that [empirical evidence].

**Example:**
> Our mixed design balances interpretability (angles are semantically meaningful to coaches) and model capacity (coordinates capture spatial detail). Prior work (Riccio et al., 2024) shows that mixed features outperform either angles-only or coordinates-only baselines.

---

## Figures to Create (Examples in TECHNICAL_DETAILS.md)

### Figure 1: Pipeline Flowchart
Create a flowchart showing:
```
Raw Video (T frames)
    ↓
MediaPipe Pose Extraction (33 landmarks per frame)
    ↓
Keypoint Preprocessing:
  - Spatial imputation
  - Skeleton normalization
  - Temporal imputation
  - FPS resampling
    ↓
Feature Extraction:
  - Joint angles (8)
  - Normalized coordinates (66)
  - Mixed features (74)
    ↓
Windowing into 30-frame sequences
    ↓
BiLSTM / xLSTM Training
```

### Figure 2: Angle Time Series
Plot your actual angle data from one video:
- X-axis: frame number (0–30)
- Y-axis: angle in degrees (0–180)
- 8 lines, one per joint angle
- Highlight where key motion phases occur (e.g., "Squat descent," "Squat bottom," "Squat ascent")

### Figure 3: Feature Distributions
Create histograms of your angle values:
- Subplots for each of the 8 angles
- X-axis: angle in degrees
- Y-axis: frequency
- Show symmetry between left/right (e.g., left knee vs. right knee should look similar)

---

## How to Write the Abstract (Short Summary)

Your abstract should mention:
1. **Problem:** Exercise form assessment from video
2. **Method:** Pose detection → preprocessing → biomechanical features → BiLSTM/xLSTM
3. **Features:** Joint angles (8) + normalized coordinates (66) = mixed (74)
4. **Key result:** Accuracy, improvement over baselines, or interpretability claim

**Template:**
> We develop a biomechanical feature extraction pipeline for real-time exercise form assessment from monocular video. Starting from MediaPipe pose landmarks, we apply skeleton-based normalization to achieve camera-invariance. We extract 8 planar joint angles (knees, hips, elbows, shoulders) and 66 normalized coordinates, combining them into 74-dimensional mixed features. These features feed BiLSTM and xLSTM models for exercise classification and quality assessment. Experiments on [dataset] show that mixed features achieve [X]% accuracy, outperforming angles-only and coordinates-only baselines by [Y]%.

---

## Checklist Before Submitting

- [ ] Section 3.1: Data source, number of videos, participant count, frame rate
- [ ] Section 3.2: Briefly mention MediaPipe (33 landmarks, 33 = (x, y, z))
- [ ] Section 3.3: Explain 4–5 preprocessing steps with WHY (camera-invariance, noise reduction, etc.)
- [ ] Section 3.3: Mention specific values (confidence threshold 0.5, k=3 for imputation, 30 Hz target)
- [ ] Section 3.4: Define joint angle equation with concrete triplet examples
- [ ] Section 3.4: List the 8 angles explicitly
- [ ] Section 3.4: Explain mixed features (74 = 8 + 66) and cite prior work
- [ ] Section 3.5: Explain 30-frame windowing (stride 15, ~1 sec at 30 Hz)
- [ ] Section 3.6: Mention zero-mean, unit-variance standardization
- [ ] All sections: Past tense, third person, active voice
- [ ] References: At least 3–5 foundational biomechanics papers (Winter, Grood & Suntay, Escamilla)
- [ ] References: At least 2 recent fitness AI papers (e.g., Riccio et al., 2024)
- [ ] Figures: 2–3 figures with clear captions
- [ ] Results section: Include your actual angle statistics (mean ± std per angle)
- [ ] Results section: Ablation study (if done) or feature importance
- [ ] Discussion: Biomechanical validity, limitations, future work

---

## If You Get Stuck

1. **"How do I explain skeleton normalization?"** → See COPY_PASTE_READY.md, Section 3.3, Template A, Step 2.
2. **"What equation should I use for angles?"** → TECHNICAL_DETAILS.md Section 1.3, or COPY_PASTE_READY.md Section 3.4, Template A.
3. **"What numbers should I report?"** → Use your actual dataset; example numbers in TECHNICAL_DETAILS.md Section 3.2.
4. **"How do I justify mixed features?"** → COPY_PASTE_READY.md Section 3.4, Template B (Concise).
5. **"What references do I cite?"** → BIOMECHANICS_PAPER_WRITING_GUIDE.md Section "References for Capstone Paper"

---

## Final Tips

1. **Use consistent notation:** If you write L for landmarks, use L everywhere. If you write f for features, use f everywhere.
2. **Be precise with dimensions:** Say "(T, 74)" not "74 features per frame"—it clarifies whether T is variable.
3. **Cite your code:** E.g., "Implementation details are in `fitness_coach/core/biomechanical_features.py`" (put in supplementary materials).
4. **Think like a coach:** When explaining why angles matter, say "a coach observes knee angle to assess squat depth" (concrete) instead of "angles capture joint geometry" (abstract).
5. **Every number needs a source:** Where did you get "30 Hz"? → It's your target FPS. Say so.

---

**Good luck with your capstone! 🎓**

