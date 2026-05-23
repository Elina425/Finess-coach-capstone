# EgoExo-Fitness — xLSTM Multi-Task Architecture

This document describes the recommended architecture for the EgoExo-Fitness leg of the project,
designed as an **enhancement of the Riccio classifier** rather than a replacement: the same
xLSTM[7:1] backbone is reused, but the input pipeline and the output head set are different.

Diagram: [`docs/diagrams/xlstm_egoexo_multitask.svg`](diagrams/xlstm_egoexo_multitask.svg)

---

## 1. Why a different input pipeline than Riccio

The Riccio path was built around raw YouTube/Kaggle videos, so it needs:

```
RGB video → YOLO26 detector → ViTPose-S (frozen + DoRA) → 17 kpts/frame
        → KNN imputation → biomechanical features (56-dim joint angles)
```

EgoExo-Fitness ships a **pre-extracted CLIP ViT-B/32 frame-feature tensor** for every record/view
(`features_open/visual/EgoExo_Fitness_CLIP_Vid_Feat_w_Rotate/<record_id>/<view>/clip_vit_b32_vid_frame_feat.pth`),
plus action-level temporal boundaries (`interpretable_action_judgement.json` → `st_ed_frame`).
Therefore for EgoExo we **skip the entire pose pipeline**:

| Step | Riccio | EgoExo |
| --- | --- | --- |
| Detector / pose model | YOLO26 + ViTPose-S | ❌ none |
| KNN imputation | ✔ | ❌ |
| Biomechanical features | 56-dim angles | ❌ |
| Frame embedding | derived from kpts | **CLIP ViT-B/32, 512-dim/frame, frozen** |
| Sequence slicing | sliding window (60 / stride 30) | **action-clip slice** from `st_ed_frame` |
| Subsampling | none | stride 3, ≤ 300 frames |

This matches the existing `EgoExoXLSTMDataset` in
[`fitness_coach/datasets/egoexo_xlstm_dataset.py`](../fitness_coach/datasets/egoexo_xlstm_dataset.py)
when `feature_mode="clip"`.

## 2. Backbone (shared with Riccio)

`xLSTMExerciseClassifier` from [`fitness_coach/models/xlstm_model.py`](../fitness_coach/models/xlstm_model.py),
configured exactly as in the Riccio diagram:

- `block_pattern = "mmmmmmms"` → **xLSTM[7:1]** (7 mLSTM + 1 sLSTM)
- `hidden_size = 256`, `num_heads = 4`, `conv_kernel_size = 4`, `projection_factor = 4/3`
- `dropout ≈ 0.15`
- The only thing that changes vs. Riccio is `input_size = 512` (CLIP) instead of `56` (angles).

Each block: `pre-LN → causal Conv1d(k=4, SiLU) → matrix-memory recurrence → head-wise GroupNorm → gated MLP → residual`.
Summarise the time dimension after the stack either with **`LayerNorm + global average pool`** or, as in Figure~10 /
`xlstm_egoexo_multitask.svg`, with **`LayerNorm + attention pooling`** (`--use-attention-pool`) → 256-dim clip embedding
**e**.

When **`--use-fusion`** is on, **e** is split into additive multitask fusion branches (**a**, **b**) whose sum
**z = a + b** can feed optional auxiliary heads while classification and regression read only **a** and **b**
respectively (see [`fitness_coach/models/xlstm_model.py`](../fitness_coach/models/xlstm_model.py) `fuse` /
`infer`).

## 3. Task heads & supervision

The EgoExo annotations expose these principal signals per action clip:

| Annotation field | Used by |
| --- | --- |
| `action_name`            | Classification head |
| `action_quality_score` (1–5) | Quality regression head |
| `key_point_verification` (list of `(text, "True"/"False")`) | Error-tag head (when `--error-weight > 0`) |
| `comment` / `action_guidance` | **Retrieval tables** built once from metadata (defaults in `train_xlstm_egoexo_multitask.py`) |

### 3.1 Classification head
`pre-LN → Linear(256→256) → GELU → Dropout → Linear(256→C)` with class-balanced cross-entropy.
`C` = number of distinct `action_name` values in the index split.

### 3.2 Quality regression head
`pre-LN → Linear(256→256) → GELU → Dropout → Linear(256→1) → σ`.
Targets are normalised: `y = (score − 1) / 4 ∈ [0, 1]`. Loss = `SmoothL1` (Huber, robust to noisy
annotator scores).

### 3.3 Interpretable error-tag head
`pre-LN → Linear(256→256) → GELU → Dropout → Linear(256→K)`.
`K = 11` predefined tags from `ERROR_TAGS` in `egoexo_xlstm_dataset.py`
(`alignment, balance_stability, back_not_straight, elbows_flared, hip_position,
incomplete_extension, insufficient_depth, knees_too_far_forward, range_of_motion,
shoulder_instability, tempo_control`). Targets are built by keyword-matching the failed
verifications (`ok = False`) — see `_collect_error_text` in the dataset. Loss = `BCEWithLogits`
with `pos_weight = neg / max(pos, 1)` per tag (handles severe label imbalance).

### 3.4 Retrieval-based coaching outputs (defaults — replaces LM comment generation)

Production training **`train_xlstm_egoexo_multitask.py` no longer attaches a frozen Flan‑T5 / Gemma
comment decoder** (`comment_head = None`). User-facing **`guidance`** and **`comment`** strings are pure
**retrieval**:

1. **Guidance lookup (class → text)** — one **zero-parameter** dict built from CSV column
   **`action_guidance`**, keyed only by classifier argmax (**ŷ_class**).

2. **Corrective-comment lookup ((class, quality-bucket) → text)** — a second **zero-parameter** dict
   built from annotated **`comment`** text. At inference **ŷ_class** selects the dictionary row and
   the regression head scalar **q̂** is discretised with **histogram bucket edges fitted on training**
   (**`--comment-quality-buckets`**, default **3**, equal-frequency on quality labels).

No cross-entropy is back-propagated into language generation; lookups are persisted in checkpoint
artifacts so deployment is deterministic and **transformers‑free**. Implementation:
``EgoExoXLSTMDataset.build_guidance_table`` / ``build_comment_table``;
``xLSTMExerciseClassifier.lookup_guidance`` / ``lookup_comment``
in [`fitness_coach/models/xlstm_model.py`](../fitness_coach/models/xlstm_model.py).

Optional **legacy** Approach~B (**`--comment-head`**, **`--class-conditioned-comment`**) still exists in
[`fitness_coach/models/xlstm_model.py`](../fitness_coach/models/xlstm_model.py) for experiments that want LM
generation + soft prefixes.

### 3.5 Optional multitask-loss weighting (DeepMTL / phased curriculum)

Aside from scalar weights **α**, **β**, … the trainer exposes **`--mtl-method`** (`ls`, `phase_ls`, `dwa`)
for classification vs regression weight dynamics (DeepMTL-style DWA, etc.).
See `train_xlstm_egoexo_multitask.py` and DeepMTL helper module.

## 4. Loss & optimisation

Training loss is a weighted sum over **differentiable heads only**:

```
L = α · L_cls + β · L_quality + γ · L_error   (+ LM cross-entropy if --comment-head is enabled)

Typical retrieval defaults:
  γ = L_error coefficient from --error-weight  (often 0.0 — error head inactive)
```

With **default retrieval**, there is **no** **δ · L_comment** term; textual outputs are keyed off **ŷ_class**
and **q̂** after training.


Matches the instructor's emphasis (classification vs quality trade-off via `--mtl-method phase_ls`,
`ls`, or `dwa`, etc.). Auxiliary losses (`L_error`) act as encoder regularisers when enabled.

**Optimiser & schedule (matches the instructor's spec):**

- AdamW (default; ``--optimizer adam`` available as ablation)
- Linear **warmup over the first 10 %** of total steps, then cosine decay to
  ``min_lr_ratio · base_lr`` (``min_lr_ratio = 0.05`` by default).
- Gradient clipping at L2 norm 1.5.
- **Attention pooling** over time replaces global average pooling
  (``--use-attention-pool``) — small gain but consistent.
- Optional backbone freeze for a personalisation ablation
  (``--freeze-backbone --unfreeze-last-n 2``): trains heads + last 2 xLSTM blocks only.

**Tuning (typical order):**

1. **Unfreeze capacity first** — largest expected gains for retrieval quality F1: either increase
   ``--unfreeze-last-n`` (often **6**) while keeping ``--freeze-backbone``, or **omit**
   ``--freeze-backbone`` entirely so the full backbone updates (more compute).
2. **Bucket rebalancing** — if some Likert levels are rare, add ``--balanced-quality-weights``
   (inverse-frequency weights on the **quality** cross-entropy over ``K`` buckets; only applies with
   ``--quality-head-mode classification``).
3. **Discrete unit ratings {0.25, 0.5, 0.75, 1.0}** — use ``--quality-encoding unit``,
   ``--comment-quality-buckets 4``, and fixed thresholds
   ``--quality-bucket-edges 0.375,0.625,0.875`` (midpoints between adjacent labels). This replaces
   Likert ordinal / quantile bucket discovery.

## 5. Why this is an *enhancement* of the classifier, not just a parallel task

1. **Same backbone, different input modality** — anything you learn about xLSTM[7:1]
   (depth, heads, conv kernel) transfers between the two legs.
2. **Auxiliary supervision** — quality + error tags are dense, multi-bit signals on the same
   embedding the classifier uses. Multi-task learning typically improves classification
   accuracy by 1–3 pp on small datasets like EgoExo (≈ 1k clips).
3. **No preprocessing surface area** — CLIP features arrive ready, so iteration speed is much
   higher than on Riccio (training a single epoch on the full set is < 1 min on CPU).
4. **Interpretability for the capstone** — **retrieval grounding** (+ optional error-tag logits)
   attaches human-readable critiques to (**ŷ**, **q̂**), unlike a pure classifier without rationales,
   without requiring a heavyweight LM service at inference.

## 6. Concrete command

Training expects a CSV with columns including ``split``, ``video_stem``, ``exercise_class``,
``quality``, and (for CLIP mode) ``judgement_key`` / frame bounds — produced by
``build_egoexo_fitness_index.py`` and ``split_exercise_index.py`` (see
``fitness_coach/training/build_egoexo_fitness_index.py``). The canonical paths are
``results/egoexo_fitness_index.csv`` (unsplit) and ``results/egoexo_fitness_index_split.csv``
(with ``train``/``val``/``test``). A convenience symlink ``results/egoexo_index.csv`` may point
at the split file so older notes that used the short name still work.

```bash
pip install -r requirements.txt   # transformers only if enabling --comment-head / LM baselines

python train_xlstm_egoexo_multitask.py \
  --index-csv results/egoexo_fitness_index_split.csv \
  --feature-mode clip \
  --clip-features-root notebooks/data/egoexo_fitness_full/features_open/visual \
  --clip-view ego_l --clip-max-frames 300 --clip-subsample-stride 3 \
  --hidden 256 --layers 8 --num-heads 4 --conv-kernel-size 4 --projection-factor 1.333 \
  --block-pattern mmmmmmms \
  --use-attention-pool \
  --use-fusion \
  --dropout 0.15 \
  --optimizer adamw --lr 3e-4 --weight-decay 1e-4 \
  --warmup-frac 0.1 --min-lr-ratio 0.05 --grad-clip 1.5 \
  --cls-weight 0.9 --reg-weight 0.1 \
  --balanced-class-weights --balanced-quality-weights --standardize \
  --epochs 40 --batch-size 32 \
  --output-dir results/xlstm_egoexo_multitask
```

Add ``--comment-head`` (+ ``--lm-name google/flan-t5-small``, etc.) **only** for legacy LM-generation
runs. Tune ``--error-weight`` if enabling the auxiliary error-tag head. ``--freeze-backbone --unfreeze-last-n`` supports personalisation A/B tests.
