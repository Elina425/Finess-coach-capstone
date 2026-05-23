#!/usr/bin/env python3
"""
Train an xLSTM on EgoExo-Fitness for classification plus quality prediction
(bucket classification aligned with retrieval comments by default; optional regression).
Deterministic retrieval feedback (guidance keyed by exercise; comment keyed by
predicted exercise + predicted quality bucket) replaces LM comment heads by default.

Loss combination uses ``--mtl-method``: fixed linear scalarisation (``ls``),
DeepMTL2R-style Dynamic Weight Average (``dwa`` — data-driven scheduling of
scalar weights), or phased linear scheduling (``phase_ls``: e.g. class-heavy
epochs then regression-heavy epochs, as alternating manual weights).

Optional: frozen-LM comments, error-tags (``--error-weight > 0``).

Optimizer: AdamW with linear warmup + cosine decay to ``min_lr_ratio·lr``.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

from fitness_coach.datasets.egoexo_xlstm_dataset import (
    CLIP_SUBDIR,
    ERROR_TAGS,
    EgoExoXLSTMDataset,
    LIKERT_Q_MAX,
    LIKERT_Q_MIN,
    apply_feature_standardizer,
    egoexo_collate_fn,
    fit_feature_standardizer,
)
from fitness_coach.datasets.exercise_bilstm_dataset import load_index_rows
from fitness_coach.evaluation.classification_metrics import detailed_classification_metrics
from fitness_coach.training.deepmtl_loss_weighting import DynamicWeightAverage
from fitness_coach.models.xlstm_model import (
    BranchedFusionT5CommentHead,
    ClassConditionedCommentHead,
    CommentGenerationHead,
    xLSTMExerciseClassifier,
)

# Comment heads that consume fused ``z`` (not legacy pooled + error-tag prompts).
_BRANCHED_FUSION_COMMENT_HEADS = (ClassConditionedCommentHead, BranchedFusionT5CommentHead)


def _comment_head_needs_branched_z(head: Optional[Any]) -> bool:
    return isinstance(head, _BRANCHED_FUSION_COMMENT_HEADS)


_REPO_ROOT = Path(__file__).resolve().parent
_RESULTS_DIR = _REPO_ROOT / "results"


def normalize_clip_features_root(root: Path) -> Path:
    """``load_clip_segment`` uses ``root / CLIP_SUBDIR / record_id / view / …``.

    If the user passes ``…/visual/EgoExo_Fitness_CLIP_Vid_Feat_w_Rotate`` (one segment
    too deep), strip the final directory so paths resolve correctly.
    """
    p = root.expanduser()
    try:
        p = p.resolve()
    except OSError:
        p = p.absolute()
    if p.name == CLIP_SUBDIR:
        parent = p.parent
        print(
            f"[info] --clip-features-root ends with '{CLIP_SUBDIR}/'; "
            f"using parent directory: {parent}",
            flush=True,
        )
        return parent
    return p


def resolve_egoexo_index_csv(user_path: Path) -> Path:
    """
    Locate the EgoExo index CSV:
    - Honour ``--index-csv`` relative to **cwd** first.
    - Retry the same relative path under the **repo root** (script parent) so commands work
      when the shell is not cd'd into the repo.
    - Fall back to canonical ``results/egoexo_fitness_index_split.csv``, then
      ``results/egoexo_index.csv`` (symlink), if the requested name is missing or a broken link.
    """
    user_path = user_path.expanduser()
    candidates: List[Path] = []

    if user_path.is_absolute():
        candidates.append(user_path)
    else:
        candidates.append(Path.cwd() / user_path)
        candidates.append(_REPO_ROOT / user_path)

    base = user_path.name.lower()
    parent = user_path.parent
    if parent and str(parent) != ".":
        if not user_path.is_absolute():
            candidates.append(_REPO_ROOT / parent / user_path.name)

    if base in ("egoexo_index.csv",):
        candidates.extend(
            [
                _RESULTS_DIR / "egoexo_fitness_index_split.csv",
                _RESULTS_DIR / "egoexo_index.csv",
            ]
        )

    candidates.extend(
        [
            _RESULTS_DIR / "egoexo_fitness_index_split.csv",
            _RESULTS_DIR / "egoexo_index.csv",
        ]
    )

    seen: set[str] = set()
    for raw in candidates:
        try:
            c = raw.resolve()
        except OSError:
            c = raw
        key = str(c)
        if key in seen:
            continue
        seen.add(key)
        if c.is_file():
            print(f"[info] index CSV: {c}", flush=True)
            return c

    raise FileNotFoundError(
        f"Missing index CSV for {user_path}. Expected under {_RESULTS_DIR}: "
        "egoexo_fitness_index_split.csv (or egoexo_index.csv → that file). "
        "Build with build_egoexo_fitness_index.py and split_exercise_index.py."
    )


# --------------------------------------------------------------------------- CLI


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train xLSTM on EgoExo: cls + quality + error-tag + comment")
    p.add_argument("--index-csv", type=Path, required=True)
    p.add_argument("--angles-dir", type=Path, default=Path("results/egoexo_exercise_angles"))
    p.add_argument("--keypoints-dir", type=Path, default=Path("results/egoexo_exercise_angles"))
    p.add_argument("--feature-mode", choices=("clip", "annotation", "angles", "coords", "mixed"), default="clip")
    p.add_argument(
        "--clip-features-root",
        type=Path,
        default=Path("notebooks/data/egoexo_fitness_full/features_open/visual"),
        help="Directory that **contains** the %s folder (do not append that folder "
        "yourself — the loader adds it). Example: …/features_open/visual"
        % CLIP_SUBDIR,
    )
    p.add_argument("--clip-view", default="ego_l")
    p.add_argument("--clip-max-frames", type=int, default=300)
    p.add_argument("--clip-subsample-stride", type=int, default=3)
    p.add_argument("--output-dir", type=Path, default=Path("results/xlstm_egoexo_multitask"))

    # window/stride only used for pose feature modes
    p.add_argument("--window", type=int, default=60)
    p.add_argument("--stride", type=int, default=30)

    # optimisation
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--min-lr-ratio", type=float, default=0.05,
                   help="Floor of the cosine schedule, expressed as a fraction of --lr.")
    p.add_argument("--warmup-frac", type=float, default=0.1,
                   help="Fraction of total training steps used for linear warmup. 0.1 == 10%%.")
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--optimizer", choices=("adamw", "adam"), default="adamw")
    p.add_argument("--grad-clip", type=float, default=1.5)

    # backbone shape
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--layers", type=int, default=8)
    p.add_argument("--dropout", type=float, default=0.15)
    p.add_argument("--num-heads", type=int, default=4)
    p.add_argument("--conv-kernel-size", type=int, default=4)
    p.add_argument("--projection-factor", type=float, default=4.0 / 3.0)
    p.add_argument("--block-pattern", default="mmmmmmms",
                   help="String of m/s for the xLSTM stack. Default is xLSTM[7:1].")
    p.add_argument("--use-attention-pool", action="store_true",
                   help="Replace global average pooling with learned attention pooling over time.")
    p.add_argument("--task-specific-pools", action="store_true",
                   help="Use a separate attention pool per task (cls + quality) so each task can "
                        "pick its own frame weighting over the same 8-block backbone. Requires "
                        "--use-attention-pool. Adds ~10K params, expected to help quality the most.")
    p.add_argument("--hard-class-conditioning", action="store_true",
                   help="Use legacy argmax(class_logits) → class_emb[id] for the quality-head "
                        "conditioning. Default is SOFT conditioning (softmax @ embedding matrix) "
                        "which is differentiable end-to-end and lets quality gradients flow back "
                        "into the classifier branch.")
    p.add_argument("--use-fusion", action="store_true",
                   help="Enable the additive multi-task fusion tower (the '+' from the whiteboard).")
    p.add_argument("--fusion-dim", type=int, default=128,
                   help="Output dim of each fusion branch (cls + quality heads consume this).")

    # multitask loss combination (manual, phased manual, DeepMTL2R DWA)
    p.add_argument(
        "--mtl-method",
        choices=("ls", "dwa", "phase_ls"),
        default="phase_ls",
        help="How to combine classification vs regression scalar losses (DeepMTL2R-aligned). "
             "'ls' = fixed weights (--cls-weight, --reg-weight); "
             "'dwa' = Dynamic Weight Average (loss-ratio-driven schedule; Liu et al.); "
             "'phase_ls' = alternate two linear-weight regimes by epoch (--phase-*). "
             "Default phase_ls matches common curriculum: emphasize cls then regress (or the reverse).",
    )
    p.add_argument(
        "--dwa-window",
        type=int,
        default=25,
        help="DWA FIFO window length in **optimizer steps** (batches): compare "
             "mean loss in the recent half vs the earlier half.",
    )
    p.add_argument(
        "--dwa-temp",
        type=float,
        default=2.0,
        help="Temperature in exp(loss_ratio/temp) for DWA softmax-style weights.",
    )
    # task weights (linear scalarisation: --mtl-method ls, or phases B / comment/err)
    p.add_argument("--cls-weight", type=float, default=0.9)
    p.add_argument("--reg-weight", type=float, default=0.1)
    p.add_argument("--error-weight", type=float, default=0.0,
                   help="Auxiliary error-tag head weight. Default 0 (head disabled, "
                        "matches the user-facing spec of cls + quality + comment).")
    p.add_argument("--comment-weight", type=float, default=0.5,
                   help="LM-CE loss weight for the comment head (default 0.5). "
                        "Originally 0.1 — bumped because too-weak gradient caused the "
                        "decoder to ignore the soft prefix and collapse onto squat-style "
                        "generic critiques. 0.5 gives the prefix enough signal to learn "
                        "exercise-discriminative features for all 12 EgoExo actions.")
    p.add_argument("--balanced-class-weights", action="store_true")
    p.add_argument(
        "--balanced-quality-weights",
        action="store_true",
        help="With --quality-head-mode classification: inverse-frequency weights on the "
             "quality CE loss over K buckets (helps rare Likert levels). Ignored for regression.",
    )
    p.add_argument("--standardize", action="store_true")

    # Retrieval-based feedback (replaces the legacy LM comment head)
    p.add_argument("--filter-null-comments", action="store_true", default=True,
                   help="Drop training clips whose `comment` field is null/empty. "
                        "On by default in the retrieval-based feedback design: "
                        "null-comment clips contribute nothing to the comment "
                        "lookup and only waste compute. Val/test are never filtered.")
    p.add_argument("--no-filter-null-comments", dest="filter_null_comments",
                   action="store_false",
                   help="Disable null-comment filtering (legacy behaviour).")
    p.add_argument("--comment-quality-buckets", type=int, default=5,
                   help="K quality buckets keying retrieval comments (default 5 = paper Likert levels). "
                        "Must match discrete classes when --quality-head-mode classification.")
    p.add_argument(
        "--quality-bucket-edges",
        default=None,
        metavar="E1,E2,...",
        help="Optional fixed comma-separated thresholds on the supervised quality axis "
             "(exactly K-1 floats for --comment-quality-buckets). Uses the same upper-exclusive "
             "bin rule as quality_score_to_bucket. Example for discrete unit ratings "
             "{0.25, 0.5, 0.75, 1.0}: --quality-encoding unit --comment-quality-buckets 4 "
             "--quality-bucket-edges 0.375,0.625,0.875 (midpoints between labels). "
             "Overrides Likert 1.5–4.5 ordinal defaults and train-quantile edges.",
    )
    p.add_argument(
        "--quality-head-mode",
        choices=("regression", "classification"),
        default="classification",
        help="Regression vs K-way bucket classification aligned with retrieval comments.",
    )
    p.add_argument(
        "--quality-encoding",
        choices=("unit", "likert"),
        default="likert",
        help="Training quality axis: unit=[0,1] (legacy fold) vs likert [1,5] (EgoExo paper ratings). "
             "Likert uses ordinal edges at 1.5..4.5 when K=5.",
    )
    p.add_argument(
        "--teacher-force-quality",
        action="store_true",
        help="Class-conditional quality head: add a learned class embedding (same dim as branch‑B) "
             "to branch‑B features. Training uses ground-truth exercise id for that embedding; "
             "validation / inference use the predicted class (argmax logits).",
    )

    # comment head
    p.add_argument("--comment-head", action="store_true",
                   help="Enable Approach-B comment generation (frozen Flan-T5-small + soft prefix).")
    p.add_argument("--lm-name", default="google/flan-t5-small")
    p.add_argument("--n-prefix", type=int, default=16)
    p.add_argument("--max-target-len", type=int, default=48)
    p.add_argument("--max-prompt-len", type=int, default=96)

    # class-conditioned comment head (new — supervisor review May 2026)
    p.add_argument("--class-conditioned-comment", action="store_true",
                   help="Branched-fusion comment head on fused z = a + b. Uses "
                        "Google Gemma (-2b, etc.) via ClassConditionedCommentHead, or "
                        "Flan-T5 (--lm-name google/flan-t5-*) via BranchedFusionT5CommentHead.")
    p.add_argument("--class-emb-dim", type=int, default=0,
                   help="Dimension of the optional learned class-embedding table E_cls. "
                        "0 (default) → branched-fusion design: comment head consumes z = a + b "
                        "and Branch A already carries the class signal. Set to e.g. 64 to also "
                        "feed an explicit class embedding (legacy class-conditioned variant).")
    p.add_argument("--hf-token", default=None,
                   help="HuggingFace token for gated models (Gemma requires license accept). "
                        "Or set the HUGGING_FACE_HUB_TOKEN env var.")

    # phased linear multitask (--mtl-method phase_ls): swap cls/reg emphasis by epoch slice
    p.add_argument(
        "--phase-a-cls-weight",
        type=float,
        default=0.9,
        help="With --mtl-method phase_ls: classification weight during phase A (first segment of training).",
    )
    p.add_argument(
        "--phase-a-reg-weight",
        type=float,
        default=0.1,
        help="With --mtl-method phase_ls: regression weight during phase A.",
    )
    p.add_argument(
        "--phase-b-cls-weight",
        type=float,
        default=0.1,
        help="With --mtl-method phase_ls: classification weight during phase B (after switch).",
    )
    p.add_argument(
        "--phase-b-reg-weight",
        type=float,
        default=0.9,
        help="With --mtl-method phase_ls: regression weight during phase B.",
    )
    p.add_argument(
        "--phase-a-fraction",
        type=float,
        default=0.5,
        help="Fraction of total epochs spent in phase A (default 0.5 = split training in half).",
    )

    # backbone freezing (optional ablation)
    p.add_argument("--freeze-backbone", action="store_true",
                   help="Freeze xLSTM stack; only heads + last --unfreeze-last-n blocks update.")
    p.add_argument(
        "--unfreeze-last-n",
        type=int,
        default=2,
        help="With --freeze-backbone: number of final xLSTM blocks left trainable. "
             "Try 6 for more capacity before dropping freeze; omit --freeze-backbone to train the full stack.",
    )

    # misc
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--eval-test", action="store_true")
    p.add_argument("--num-workers", type=int, default=0)
    return p


# ------------------------------------------------------------------- helpers


def compute_inverse_frequency_class_weights(samples: List[Any], num_classes: int) -> torch.Tensor:
    counts = Counter(int(sample[1]) for sample in samples)
    total = max(1, sum(counts.values()))
    weights = torch.zeros(num_classes, dtype=torch.float32)
    for cls_idx in range(num_classes):
        count = counts.get(cls_idx, 0)
        weights[cls_idx] = 1.0 if count <= 0 else total / (num_classes * count)
    return weights / weights.mean().clamp_min(1e-8)


def parse_optional_quality_bucket_edges(
    raw: Optional[str],
    *,
    num_buckets: int,
    domain_lo: float,
    domain_hi: float,
) -> Optional[Tuple[float, ...]]:
    """Parse ``--quality-bucket-edges`` into a sorted strictly-increasing tuple inside the domain."""
    if raw is None:
        return None
    s = str(raw).strip()
    if not s:
        return None
    edges = tuple(float(x.strip()) for x in s.split(",") if x.strip())
    need = int(num_buckets) - 1
    if len(edges) != need:
        raise ValueError(
            f"--quality-bucket-edges: need exactly K-1={need} edges for "
            f"--comment-quality-buckets={num_buckets}; got {len(edges)}"
        )
    edges_t = tuple(sorted(edges))
    if tuple(edges) != edges_t:
        print("[info] --quality-bucket-edges sorted ascending for binning.", flush=True)
    if len(set(edges_t)) != len(edges_t):
        raise ValueError("--quality-bucket-edges: values must be distinct")
    eps = 1e-5 * max(1.0, abs(domain_hi - domain_lo))
    for e in edges_t:
        if not (domain_lo + eps <= e <= domain_hi - eps):
            raise ValueError(
                f"--quality-bucket-edges: edge {e} must lie strictly inside "
                f"[{domain_lo}, {domain_hi}] (margin eps={eps:g})"
            )
    return edges_t


def compute_inverse_frequency_quality_bucket_weights(samples: List[Any], num_buckets: int) -> torch.Tensor:
    """Inverse-frequency vector for quality-bucket CE (shape ``[K]``, mean-normalised)."""
    counts: Counter = Counter()
    for sample in samples:
        if len(sample) < 5:
            continue
        b = int(sample[3])
        if b < 0:
            continue
        counts[b] += 1
    total = max(1, sum(counts.values()))
    kb = int(num_buckets)
    weights = torch.zeros(kb, dtype=torch.float32)
    for b in range(kb):
        c = counts.get(b, 0)
        weights[b] = 1.0 if c <= 0 else float(total) / (float(kb) * float(c))
    return weights / weights.mean().clamp_min(1e-8)


def compute_error_pos_weight(samples: List[Tuple[np.ndarray, int, float, np.ndarray]]) -> torch.Tensor:
    targets = np.stack([sample[-1] for sample in samples], axis=0)
    positives = targets.sum(axis=0)
    negatives = float(targets.shape[0]) - positives
    weight = negatives / np.maximum(positives, 1.0)
    return torch.tensor(weight, dtype=torch.float32)


def warmup_cosine_schedule(
    optimizer: torch.optim.Optimizer,
    total_steps: int,
    warmup_frac: float,
    min_lr_ratio: float,
) -> LambdaLR:
    """Linear warmup over the first ``warmup_frac`` of steps, then cosine decay
    down to ``min_lr_ratio·base_lr``."""
    total_steps = max(1, int(total_steps))
    warmup_steps = max(1, int(round(total_steps * max(0.0, min(1.0, warmup_frac)))))
    min_ratio = float(max(0.0, min(1.0, min_lr_ratio)))

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        progress = min(1.0, max(0.0, progress))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_ratio + (1.0 - min_ratio) * cosine

    return LambdaLR(optimizer, lr_lambda=lr_lambda)


def phased_linear_cls_reg_weights(
    epoch: int,
    total_epochs: int,
    phase_a_frac: float,
    w_cls_a: float,
    w_reg_a: float,
    w_cls_b: float,
    w_reg_b: float,
) -> Tuple[float, float, str]:
    """Classification / regression loss weights for 1-indexed ``epoch``.

    Phase A runs for the first ``round(phase_a_frac * total_epochs)`` epochs
    (at least one epoch, at most ``total_epochs``); subsequent epochs use phase B.
    """

    total_epochs = max(1, int(total_epochs))
    frac = float(max(0.0, min(1.0, phase_a_frac)))
    n_a = max(1, int(round(frac * total_epochs)))
    n_a = min(n_a, total_epochs)
    if epoch <= n_a:
        return float(w_cls_a), float(w_reg_a), "A"
    return float(w_cls_b), float(w_reg_b), "B"


# ------------------------------------------------------------------- train/eval


def _forward_heads(
    model: xLSTMExerciseClassifier,
    xb: torch.Tensor,
    *,
    quality_explicit_class_one_hot: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run encoder once and apply each head with task-specific routing.

    Returns ``(pooled, z, logits, quality, err_logits)`` where:
        pooled  — raw 256-d clip embedding from the xLSTM (kept for legacy
                  callers that read it; not used by the cls/quality/comment heads).
        z       — fused 128-d vector ``a + b``. **This is what the comment head
                  consumes.** Carries both class-discriminative features (via
                  Branch A) and quality-discriminative features (via Branch B).
        logits  — classification logits computed from Branch A only.
        quality — logits ``[B, K]`` if ``model.quality_is_classification`` else ``[B, 1]`` sigmoid regression.
        err_logits — auxiliary error-tag logits from ``z`` (zero-width when disabled).

        ``quality_explicit_class_one_hot``: when the model uses class conditioning on the
        quality branch, pass ground-truth one-hot rows during supervised training steps;
        omit (``None``) so conditioning uses ``argmax(logits)`` (evaluation / inference).

    Branch-A-only and Branch-B-only routing is what forces specialisation: each
    branch is the *only* path from the encoder to its respective task head, so
    its parameters are updated exclusively by that task's loss.
    """
    pooled = model.encode(xb)
    a, b, z = model.fuse(pooled)
    class_logits = model.class_head(a)
    b_q = model.quality_branch_feat(b, class_logits, quality_explicit_class_one_hot)
    qual_raw = model.quality_head(b_q)
    if model.quality_is_classification:
        pred_q = qual_raw
    else:
        pred_q = torch.sigmoid(qual_raw) * model.quality_scale + model.quality_output_low
    err_logits = model.error_head(z) if model.error_head is not None else pooled.new_zeros((pooled.size(0), 0))
    return pooled, z, class_logits, pred_q, err_logits




def train_one_epoch(
    model: xLSTMExerciseClassifier,
    comment_head: Optional[CommentGenerationHead],
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[LambdaLR],
    device: torch.device,
    weights: Dict[str, float],
    class_weights: Optional[torch.Tensor],
    error_pos_weight: Optional[torch.Tensor],
    grad_clip: float,
    mtl_method: str,
    task_order: Sequence[str],
    dwa_state: Optional[DynamicWeightAverage] = None,
    teacher_force_quality: bool = False,
    quality_bucket_weights: Optional[torch.Tensor] = None,
) -> Dict[str, Any]:
    model.train()
    if comment_head is not None:
        comment_head.train()
    ce = nn.CrossEntropyLoss(weight=class_weights.to(device) if class_weights is not None else None)
    totals = {"loss": 0.0, "cls": 0.0, "reg": 0.0, "err": 0.0, "cmt": 0.0}
    seen = 0
    dwa_w_sum = torch.zeros(len(task_order), dtype=torch.float32)
    nb = 0
    for batch in loader:
        xb, y_cls, y_qf, y_q_bucket, y_err, comments, class_names = batch
        xb = xb.to(device)
        y_cls = y_cls.to(device)
        y_qf = y_qf.to(device)
        y_q_bucket = y_q_bucket.to(device)
        y_err = y_err.to(device)

        optimizer.zero_grad()
        # NOTE: `z` is the fused (a + b) vector — what the comment head consumes
        # in the branched-fusion design. `pooled` is kept for the legacy
        # Flan-T5-style comment head and for error-tag inputs.
        q_teacher: Optional[torch.Tensor] = None
        if teacher_force_quality and model.quality_class_conditioning:
            q_teacher = F.one_hot(y_cls, num_classes=model.num_classes).to(device=device, dtype=xb.dtype)
        pooled, z, logits, pred_q, pred_err = _forward_heads(
            model, xb, quality_explicit_class_one_hot=q_teacher,
        )

        cls_loss = ce(logits, y_cls)
        if model.quality_is_classification:
            qb_w = quality_bucket_weights.to(device) if quality_bucket_weights is not None else None
            reg_loss = F.cross_entropy(pred_q, y_q_bucket, weight=qb_w)
        else:
            reg_loss = F.smooth_l1_loss(pred_q.squeeze(-1), y_qf)
        if pred_err.numel() > 0 and weights["err"] > 0:
            err_loss = F.binary_cross_entropy_with_logits(
                pred_err, y_err,
                pos_weight=error_pos_weight.to(device) if error_pos_weight is not None else None,
            )
        else:
            err_loss = pooled.new_zeros(())

        cmt_loss_value = 0.0
        cmt_loss: Optional[torch.Tensor] = None
        if comment_head is not None:
            if _comment_head_needs_branched_z(comment_head):
                # Branched-fusion design: comment head consumes z = a + b which
                # already carries both class- and quality-specialised features.
                # The class signal arrives implicitly via Branch A's contribution
                # to z, so no explicit class-embedding lookup is needed at this
                # point. We still pass y_cls / class_names so the head can build
                # the text-prompt portion that pairs with the soft prefix.
                cmt_loss = comment_head.compute_loss(z, y_cls, class_names, comments)
            else:
                # Legacy Flan-T5 path: uses gold error tags + class names.
                cmt_loss = comment_head.compute_loss(pooled, y_err, class_names, comments)
            if torch.is_tensor(cmt_loss) and cmt_loss.requires_grad:
                cmt_loss_value = float(cmt_loss.item())
            else:
                cmt_loss = None  # skip in loss combination

        stacked_tensors: List[torch.Tensor] = []
        for task in task_order:
            if task == "cls":
                stacked_tensors.append(cls_loss)
            elif task == "reg":
                stacked_tensors.append(reg_loss)
            elif task == "err":
                stacked_tensors.append(err_loss)
            elif task == "cmt":
                if cmt_loss is None or not cmt_loss.requires_grad:
                    raise RuntimeError("comment loss missing but 'cmt' is in multitask ``task_order``")
                stacked_tensors.append(cmt_loss)
            else:
                raise ValueError(f"unknown multitask name {task!r}")
        stacked = torch.stack(stacked_tensors)

        if mtl_method == "dwa":
            if dwa_state is None:
                raise RuntimeError("--mtl-method dwa requires DynamicWeightAverage state")
            loss, tw = dwa_state.weighted_loss_mean(stacked)
            dwa_w_sum += tw.detach().cpu()
            nb += 1
        elif mtl_method in ("ls", "phase_ls"):
            loss = (
                weights["cls"] * cls_loss
                + weights["reg"] * reg_loss
                + weights["err"] * err_loss
            )
            if cmt_loss is not None and weights["cmt"] > 0.0:
                loss = loss + weights["cmt"] * cmt_loss
        else:
            raise ValueError(f"unknown --mtl-method {mtl_method!r}")

        loss.backward()
        if grad_clip and grad_clip > 0:
            params = [p for p in model.parameters() if p.requires_grad]
            if comment_head is not None:
                params += [p for p in comment_head.parameters() if p.requires_grad]
            torch.nn.utils.clip_grad_norm_(params, grad_clip)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        bs = xb.size(0)
        totals["loss"] += float(loss.item()) * bs
        totals["cls"] += float(cls_loss.item()) * bs
        totals["reg"] += float(reg_loss.item()) * bs
        totals["err"] += float(err_loss.item()) * bs
        totals["cmt"] += cmt_loss_value * bs
        seen += bs
    averaged = {k: v / max(seen, 1) for k, v in totals.items()}
    if mtl_method == "dwa" and nb > 0:
        averaged["dwa_w_mean"] = {
            task_order[i]: float(dwa_w_sum[i] / nb)
            for i in range(len(task_order))
        }
    return averaged


@torch.no_grad()
def evaluate(
    model: xLSTMExerciseClassifier,
    comment_head: Optional[CommentGenerationHead],
    loader: Optional[DataLoader],
    device: torch.device,
    class_names_all: List[str],
    sample_comments: int = 0,
) -> Dict[str, Any]:
    if loader is None:
        return {
            "accuracy": 0.0, "f1_macro": 0.0, "f1_per_class": {}, "mae": float("nan"),
            "r2": float("nan"), "error_f1_macro": 0.0, "comment_samples": [],
            "quality": {"n_samples": 0},
        }

    model.eval()
    if comment_head is not None:
        comment_head.eval()

    all_true, all_pred = [], []
    all_q_true_scalar: List[float] = []
    all_q_pred_scalar: List[float] = []
    all_q_true_bucket: List[int] = []
    all_q_pred_bucket: List[int] = []
    all_err_true, all_err_pred = [], []
    comment_samples: List[Dict[str, str]] = []

    centres = getattr(model, "_quality_bucket_centres", ())

    for batch in loader:
        xb, y_cls, y_qf, y_q_bucket, y_err, comments, cls_names = batch
        xb = xb.to(device)
        pooled, z, logits, pred_q, pred_err = _forward_heads(model, xb)
        preds = logits.argmax(dim=1).cpu().numpy()
        all_true.extend(y_cls.numpy().tolist())
        all_pred.extend(preds.tolist())

        if model.quality_is_classification:
            pred_np = pred_q.argmax(dim=1).detach().cpu().numpy().astype(np.int64)
            true_np = y_q_bucket.cpu().numpy().astype(np.int64)
            qfn = y_qf.cpu().numpy().astype(np.float64)
            m = true_np >= 0
            if np.any(m):
                all_q_pred_bucket.extend(pred_np[m].tolist())
                all_q_true_bucket.extend(true_np[m].tolist())
                all_q_true_scalar.extend(qfn[m].tolist())
                if centres and len(centres) == model.num_quality_classes:
                    cnp = np.asarray(centres, dtype=np.float64)
                    all_q_pred_scalar.extend(cnp[pred_np[m]].tolist())
                else:
                    den = float(max(model.num_quality_classes - 1, 1))
                    all_q_pred_scalar.extend((pred_np[m].astype(np.float64) / den).tolist())
        else:
            pq = pred_q.squeeze(-1).detach().cpu().numpy().astype(np.float64).tolist()
            all_q_true_scalar.extend(y_qf.numpy().astype(np.float64).tolist())
            all_q_pred_scalar.extend(pq)

        if pred_err.numel() > 0:
            all_err_true.append(y_err.numpy().astype(np.float32))
            all_err_pred.append((torch.sigmoid(pred_err).cpu().numpy() >= 0.5).astype(np.float32))

        if comment_head is not None and len(comment_samples) < sample_comments:
            if _comment_head_needs_branched_z(comment_head):
                # Branched-fusion design: comment head consumes z = a + b.
                pred_cls_idx = logits.argmax(dim=1)
                pred_cls_names = [class_names_all[int(i)] for i in pred_cls_idx.cpu().tolist()]
                generated = comment_head.generate(z, pred_cls_idx, pred_cls_names)
            else:
                generated = comment_head.generate(pooled, pred_err, cls_names)
            for gen, ref, cls in zip(generated, comments, cls_names):
                if len(comment_samples) >= sample_comments:
                    break
                comment_samples.append({"class": cls, "reference": ref, "generated": gen})

    cls_report = detailed_classification_metrics(
        np.asarray(all_true, dtype=np.int64),
        np.asarray(all_pred, dtype=np.int64),
        class_names_all,
    )
    q_true_np = np.asarray(all_q_true_scalar, dtype=np.float64)
    q_pred_np = np.asarray(all_q_pred_scalar, dtype=np.float64)

    if model.quality_is_classification:
        yt = np.asarray(all_q_true_bucket, dtype=np.int64)
        yp = np.asarray(all_q_pred_bucket, dtype=np.int64)
        if len(yt) != len(yp):
            raise RuntimeError(f"quality bucket mismatch: true={len(yt)} pred={len(yp)}")
        quality_metrics = compute_quality_classification_metrics(
            yt,
            yp,
            num_bins=model.num_quality_classes,
            q_true_scalar=np.asarray(all_q_true_scalar, dtype=np.float64),
            q_pred_scalar=np.asarray(all_q_pred_scalar, dtype=np.float64),
        )
        mae_fallback = float(quality_metrics["mae_proxy"])
        r2_fallback = float(quality_metrics.get("scalar_r2", float("nan")))
    else:
        quality_metrics = compute_quality_regression_metrics(
            q_true_np,
            q_pred_np,
            class_indices=np.asarray(all_true, dtype=np.int64),
            class_names=class_names_all,
        )
        mae_fallback = float(quality_metrics["mae"])
        r2_fallback = float(quality_metrics["r2"])
    try:
        from sklearn.metrics import f1_score
        err_true = np.concatenate(all_err_true, axis=0) if all_err_true else np.zeros((0, len(ERROR_TAGS)))
        err_pred = np.concatenate(all_err_pred, axis=0) if all_err_pred else np.zeros((0, len(ERROR_TAGS)))
        error_f1_macro = float(f1_score(err_true, err_pred, average="macro", zero_division=0))
    except ValueError:
        error_f1_macro = 0.0

    return {
        **cls_report,
        "mae": mae_fallback,
        "r2": r2_fallback,
        "quality": quality_metrics,
        "error_f1_macro": error_f1_macro,
        "comment_samples": comment_samples,
    }


def compute_quality_classification_metrics(
    y_true_bucket: np.ndarray,
    y_pred_bucket: np.ndarray,
    *,
    num_bins: int,
    q_true_scalar: np.ndarray,
    q_pred_scalar: np.ndarray,
) -> Dict[str, Any]:
    """Discrete quality bucket metrics (+ scalar MAE/R² on representative bucket centres)."""
    n = int(len(y_true_bucket))
    labels = list(range(max(1, int(num_bins))))
    out: Dict[str, Any] = {
        "n_samples": n,
        "task": "classification",
        "num_bins": int(num_bins),
        "accuracy": float("nan"),
        "f1_macro": float("nan"),
        "f1_weighted": float("nan"),
        "bucket_confusion_note": "rows=true bucket, cols=predicted bucket",
    }
    if n == 0:
        nan = float("nan")
        out.update(mae_proxy=nan, scalar_r2=nan, confusion_matrix=[], bucket_support={})
        return out

    from sklearn.metrics import accuracy_score, confusion_matrix as sk_confusion_matrix, f1_score

    out["accuracy"] = float(accuracy_score(y_true_bucket, y_pred_bucket))
    out["f1_macro"] = float(f1_score(y_true_bucket, y_pred_bucket, average="macro", zero_division=0, labels=labels))
    out["f1_weighted"] = float(f1_score(y_true_bucket, y_pred_bucket, average="weighted", zero_division=0, labels=labels))
    cm = sk_confusion_matrix(y_true_bucket, y_pred_bucket, labels=labels)
    out["confusion_matrix"] = cm.tolist()
    support = [int(np.sum(y_true_bucket == i)) for i in labels]
    out["bucket_support"] = {str(i): support[i] for i in labels}

    mae_proxy = float("nan")
    scalar_r2 = float("nan")
    if q_true_scalar.size >= n and q_pred_scalar.size >= n:
        qt = q_true_scalar[:n]
        qp = q_pred_scalar[:n]
        diff = qp - qt
        mae_proxy = float(np.mean(np.abs(diff)))
        try:
            from sklearn.metrics import r2_score
            scalar_r2 = float(r2_score(qt, qp)) if n > 1 else float("nan")
        except Exception:
            scalar_r2 = float("nan")
        out.update(
            mae_proxy=mae_proxy,
            scalar_r2=scalar_r2,
            q_true_mean=float(np.mean(qt)),
            q_pred_mean=float(np.mean(qp)),
        )
    else:
        out.setdefault("mae_proxy", float("nan"))
        out.setdefault("scalar_r2", float("nan"))
    return out


def compute_quality_regression_metrics(
    q_true: np.ndarray,
    q_pred: np.ndarray,
    class_indices: Optional[np.ndarray] = None,
    class_names: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Full quality-regression report.

    Standard Action-Quality-Assessment metrics used across the AQA literature
    (Parmar et al. ECCV'22, FLEX, EgoExo-Fitness Table 5): MSE, RMSE, MAE,
    R^2, Pearson and Spearman rank correlation. Spearman is the canonical
    headline metric for AQA because it measures how well the model *ranks*
    quality across clips, independently of absolute scale.
    """
    n = int(len(q_true))
    out: Dict[str, Any] = {"n_samples": n}
    if n == 0:
        nan = float("nan")
        out.update(mse=nan, rmse=nan, mae=nan, r2=nan,
                   pearson_r=nan, spearman_rho=nan, q_true_std=nan, q_pred_std=nan)
        return out

    diff = q_pred - q_true
    mse = float(np.mean(diff ** 2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(diff)))
    q_true_std = float(np.std(q_true))
    q_pred_std = float(np.std(q_pred))

    # R^2 (coefficient of determination)
    try:
        from sklearn.metrics import r2_score
        r2 = float(r2_score(q_true, q_pred)) if n > 1 else float("nan")
    except Exception:
        r2 = float("nan")

    # Pearson + Spearman — robust to zero-variance ground truth
    pearson_r = float("nan")
    spearman_rho = float("nan")
    if n > 1 and q_true_std > 1e-9 and q_pred_std > 1e-9:
        try:
            from scipy.stats import pearsonr, spearmanr
            pearson_r = float(pearsonr(q_true, q_pred)[0])
            spearman_rho = float(spearmanr(q_true, q_pred)[0])
        except Exception:
            # Manual fallback if SciPy is unavailable
            pearson_r = float(np.corrcoef(q_true, q_pred)[0, 1])

    out.update(
        mse=mse, rmse=rmse, mae=mae, r2=r2,
        pearson_r=pearson_r, spearman_rho=spearman_rho,
        q_true_mean=float(np.mean(q_true)), q_true_std=q_true_std,
        q_pred_mean=float(np.mean(q_pred)), q_pred_std=q_pred_std,
    )

    # Per-class MAE — reveals which exercises are easier/harder to score
    if class_indices is not None and class_names is not None and len(class_indices) == n:
        per_class: Dict[str, Dict[str, float]] = {}
        for idx, name in enumerate(class_names):
            mask = (class_indices == idx)
            k = int(mask.sum())
            if k == 0:
                continue
            per_class[name] = {
                "n": k,
                "mae": float(np.mean(np.abs(diff[mask]))),
                "rmse": float(np.sqrt(np.mean(diff[mask] ** 2))),
                "true_mean": float(np.mean(q_true[mask])),
                "pred_mean": float(np.mean(q_pred[mask])),
            }
        out["per_class"] = per_class

    return out


# ------------------------------------------------------------------- driver


def train_from_args(args: argparse.Namespace) -> Dict[str, Any]:
    idx = resolve_egoexo_index_csv(Path(args.index_csv))
    args.index_csv = idx

    if args.feature_mode == "clip":
        args.clip_features_root = normalize_clip_features_root(Path(args.clip_features_root))

    rows = load_index_rows(args.index_csv)
    train_rows = [row for row in rows if (row.get("split") or "train") == "train"]
    if not train_rows:
        raise ValueError("No train rows found in index CSV")

    classes = sorted({row["exercise_class"] for row in train_rows})
    class_to_idx = {name: idx for idx, name in enumerate(classes)}
    idx_to_class = {idx: name for name, idx in class_to_idx.items()}

    needs_pose = args.feature_mode in ("angles", "coords", "mixed")
    filter_null = bool(getattr(args, "filter_null_comments", True))
    enc_norm = str(args.quality_encoding).strip().lower()
    ds_kwargs = dict(
        feature_mode=args.feature_mode,
        quality_encoding=enc_norm,
        angles_dir=args.angles_dir if needs_pose else None,
        keypoints_dir=args.keypoints_dir if args.feature_mode in ("coords", "mixed") else None,
        clip_features_root=args.clip_features_root if args.feature_mode == "clip" else None,
        clip_view=args.clip_view, clip_max_frames=args.clip_max_frames,
        clip_subsample_stride=args.clip_subsample_stride,
        window=args.window if needs_pose else 0,
        stride=args.stride if needs_pose else 0,
        filter_null_comments=filter_null,
    )
    train_ds = EgoExoXLSTMDataset(args.index_csv, class_to_idx, "train", **ds_kwargs)
    val_split = "val" if any((row.get("split") or "") == "val" for row in rows) else "test"
    # Val / test sets keep every clip (no filtering) so per-class support
    # remains directly comparable across training runs.
    eval_ds_kwargs = {**ds_kwargs, "filter_null_comments": False}
    val_ds = EgoExoXLSTMDataset(args.index_csv, class_to_idx, val_split, **eval_ds_kwargs)
    test_ds = None
    if val_split == "val" and any((row.get("split") or "") == "test" for row in rows):
        test_ds = EgoExoXLSTMDataset(args.index_csv, class_to_idx, "test", **eval_ds_kwargs)
    print(f"[data] filter_null_comments={filter_null} · "
          f"train={len(train_ds)} val={len(val_ds)} "
          f"test={len(test_ds) if test_ds is not None else 0}",
          flush=True)

    if len(train_ds) == 0:
        hint = ""
        if args.feature_mode == "clip":
            hint = (
                " For --feature-mode clip, each row needs CLIP ViT-B/32 features under "
                f"{args.clip_features_root}/{CLIP_SUBDIR}/<record_id>/<view>/clip_vit_b32_vid_frame_feat.pth. "
                "Pass --clip-features-root as the parent of "
                f"{CLIP_SUBDIR} (typically …/features_open/visual), not …/visual/{CLIP_SUBDIR}. "
                "Alternatively use --feature-mode annotation if archives are not extracted."
            )
        elif args.feature_mode in ("angles", "coords", "mixed"):
            hint = (
                f" For {args.feature_mode!r}, ensure --angles-dir / --keypoints-dir contain per-stem "
                "*_biomechanics.npz / *_keypoints.npz files matching video_stem in the index."
            )
        raise ValueError(
            "No train samples found; check index CSV, split column, exercise_class names, and feature files."
            + hint
        )

    scale_mean = scale_std = None
    if args.standardize:
        scale_mean, scale_std = fit_feature_standardizer(train_ds.samples)
        apply_feature_standardizer(train_ds.samples, scale_mean, scale_std)
        if len(val_ds) > 0:
            apply_feature_standardizer(val_ds.samples, scale_mean, scale_std)
        if test_ds is not None and len(test_ds) > 0:
            apply_feature_standardizer(test_ds.samples, scale_mean, scale_std)

    # Device selection: CUDA → MPS (Apple Silicon) → CPU.  On macOS there is no
    # CUDA; MPS uses the integrated Apple GPU and gives a large speed-up over CPU
    # for everything except a handful of unimplemented ops (which fall back
    # silently when PYTORCH_ENABLE_MPS_FALLBACK=1).
    if args.cpu:
        device = torch.device("cpu")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"[setup] device = {device}", flush=True)
    feature_dim = train_ds.feature_dim()
    # Error-tag head is built only when its weight is > 0; otherwise the head is omitted
    # entirely so it can't be accidentally consumed at inference.
    n_error_tags = len(ERROR_TAGS) if args.error_weight > 0.0 else 0

    q_domain_lo = float(LIKERT_Q_MIN) if enc_norm == "likert" else 0.0
    q_domain_hi = float(LIKERT_Q_MAX) if enc_norm == "likert" else 1.0
    span = float(LIKERT_Q_MAX - LIKERT_Q_MIN) if enc_norm == "likert" else 1.0
    q_out_low = float(LIKERT_Q_MIN) if enc_norm == "likert" else 0.0

    # Comment / quality buckets FIRST (classification uses the same edges).
    fixed_q_edges = parse_optional_quality_bucket_edges(
        getattr(args, "quality_bucket_edges", None),
        num_buckets=args.comment_quality_buckets,
        domain_lo=q_domain_lo,
        domain_hi=q_domain_hi,
    )
    comment_table, q_bucket_edges = train_ds.build_comment_table(
        class_to_idx,
        num_quality_buckets=args.comment_quality_buckets,
        bucket_edges=fixed_q_edges,
    )
    kb = len(q_bucket_edges) + 1
    if getattr(args, "quality_head_mode", "classification") == "classification":
        train_ds.apply_quality_bucket_labels(q_bucket_edges)
        if len(val_ds) > 0:
            val_ds.apply_quality_bucket_labels(q_bucket_edges)
        if test_ds is not None and len(test_ds) > 0:
            test_ds.apply_quality_bucket_labels(q_bucket_edges)
        if int(args.comment_quality_buckets) != kb:
            raise ValueError(
                f"--comment-quality-buckets ({args.comment_quality_buckets}) mismatches edge-derived K={kb}."
            )
        n_quality_out = kb
        if fixed_q_edges is not None:
            bucket_note = f"fixed edges {tuple(round(e, 4) for e in q_bucket_edges)}"
        elif enc_norm == "likert" and int(args.comment_quality_buckets) == 5:
            bucket_note = "ordinal Likert 1–5 edges (1.5…4.5)"
        else:
            bucket_note = "train-quantile buckets"
        print(
            f"[data] quality = classification · K={n_quality_out} · encoding={args.quality_encoding!r} "
            f"(axis [{q_domain_lo:.1f}, {q_domain_hi:.1f}]) · buckets: {bucket_note}",
            flush=True,
        )
    else:
        n_quality_out = 1
        print(
            f"[data] quality = regression on continuous axis [{q_domain_lo:.1f}, {q_domain_hi:.1f}] · "
            f"encoding={args.quality_encoding!r}",
            flush=True,
        )

    model = xLSTMExerciseClassifier(
        input_size=feature_dim,
        hidden_size=args.hidden,
        num_layers=args.layers,
        num_classes=len(classes),
        dropout=args.dropout,
        num_heads=args.num_heads,
        conv_kernel_size=args.conv_kernel_size,
        projection_factor=args.projection_factor,
        num_error_tags=n_error_tags,
        quality_scale=span,
        quality_output_low=q_out_low,
        num_quality_classes=n_quality_out,
        block_pattern=args.block_pattern,
        use_attention_pool=args.use_attention_pool,
        use_fusion=args.use_fusion,
        fusion_dim=args.fusion_dim,
        quality_class_conditioning=bool(args.teacher_force_quality),
        task_specific_pools=bool(args.task_specific_pools),
        soft_class_conditioning=not bool(args.hard_class_conditioning),
    ).to(device)
    if args.task_specific_pools and not args.use_attention_pool:
        print("[warn] --task-specific-pools requires --use-attention-pool; falling back to single pool.")
    print(
        f"[setup] task-specific-pools={bool(args.task_specific_pools) and bool(args.use_attention_pool)} "
        f"· class-conditioning={'soft (softmax@emb)' if not args.hard_class_conditioning else 'hard (argmax→emb)'}",
        flush=True,
    )

    if args.teacher_force_quality:
        print(
            "[setup] teacher-force-quality: add learned class embedding (GT id in train, predicted argmax "
            "in val/test/infer) to branch‑B before the quality head.",
            flush=True,
        )

    guidance_table = train_ds.build_guidance_table(class_to_idx)
    model.set_guidance_table(guidance_table, idx_to_class)
    model.set_comment_table(comment_table, q_bucket_edges, domain_lo=q_domain_lo, domain_hi=q_domain_hi)
    print(f"[comment] retrieval table · {len(comment_table)} cells · "
          f"quality bucket edges = {tuple(round(e, 3) for e in q_bucket_edges)}",
          flush=True)
    if not guidance_table:
        print("[warn] guidance lookup is empty — verify your index CSV has an 'action_guidance' column.")
    if not comment_table:
        print("[warn] comment lookup is empty — check that 'comment' column has non-empty entries.")

    if args.freeze_backbone:
        model.freeze_backbone(freeze=True, last_n_unfrozen=args.unfreeze_last_n)

    # The comment head (Flan-T5 / Gemma-2-2B) is removed in this design — its
    # role is replaced by a zero-parameter (class, quality-bucket) lookup
    # populated above. Any legacy comment-related CLI flags are ignored.
    comment_head = None

    task_order_list: List[str] = ["cls", "reg"]
    if args.error_weight > 0:
        task_order_list.append("err")
    if comment_head is not None:
        task_order_list.append("cmt")
    task_order: Tuple[str, ...] = tuple(task_order_list)

    dwa_state: Optional[DynamicWeightAverage] = None
    if args.mtl_method == "dwa":
        dwa_state = DynamicWeightAverage(
            n_tasks=len(task_order),
            iteration_window=args.dwa_window,
            temp=args.dwa_temp,
        )
        print(
            f"[mtl] dwa (DeepMTL2R, loss-guided schedule) · tasks={list(task_order)} · "
            f"window={args.dwa_window} temp={args.dwa_temp}",
            flush=True,
        )
    elif args.mtl_method == "phase_ls":
        print(
            f"[mtl] phase_ls · phase A ({args.phase_a_fraction:.0%} of epochs): "
            f"w_cls={args.phase_a_cls_weight} w_reg={args.phase_a_reg_weight} · "
            f"phase B: w_cls={args.phase_b_cls_weight} w_reg={args.phase_b_reg_weight}",
            flush=True,
        )
    elif args.mtl_method == "ls":
        print(
            f"[mtl] ls (fixed) · w_cls={args.cls_weight} w_reg={args.reg_weight} "
            f"w_err={args.error_weight} w_cmt={args.comment_weight if comment_head else 0.0}",
            flush=True,
        )
    else:
        raise ValueError(f"unknown --mtl-method {args.mtl_method!r}")

    # ----- optimizer (trainable backbone + heads only) -----
    trainable = [p for p in model.parameters() if p.requires_grad]
    if comment_head is not None:
        trainable += [p for p in comment_head.parameters() if p.requires_grad]
    opt_cls = torch.optim.AdamW if args.optimizer == "adamw" else torch.optim.Adam
    optimizer = opt_cls(trainable, lr=args.lr, weight_decay=args.weight_decay)

    # ----- data loaders -----
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, drop_last=False,
        collate_fn=egoexo_collate_fn, num_workers=args.num_workers,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=egoexo_collate_fn,
        num_workers=args.num_workers,
    ) if len(val_ds) > 0 else None
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False, collate_fn=egoexo_collate_fn,
        num_workers=args.num_workers,
    ) if test_ds is not None and len(test_ds) > 0 else None

    # ----- scheduler: linear warmup (10% by default) + cosine decay -----
    steps_per_epoch = max(1, math.ceil(len(train_ds) / max(1, args.batch_size)))
    total_steps = steps_per_epoch * args.epochs
    scheduler = warmup_cosine_schedule(
        optimizer, total_steps=total_steps,
        warmup_frac=args.warmup_frac, min_lr_ratio=args.min_lr_ratio,
    )

    class_weights = compute_inverse_frequency_class_weights(train_ds.samples, len(classes)) \
        if args.balanced_class_weights else None
    quality_bucket_weights: Optional[torch.Tensor] = None
    if getattr(args, "balanced_quality_weights", False):
        if model.quality_is_classification:
            quality_bucket_weights = compute_inverse_frequency_quality_bucket_weights(
                train_ds.samples, model.num_quality_classes,
            )
            print(
                "[data] balanced-quality-weights "
                + "[" + ", ".join(f"{float(w):.3f}" for w in quality_bucket_weights) + "]",
                flush=True,
            )
        else:
            print(
                "[warn] --balanced-quality-weights ignored (--quality-head-mode regression).",
                flush=True,
            )
    error_pos_weight = compute_error_pos_weight(train_ds.samples)
    err_w = float(args.error_weight)
    cmt_w = float(args.comment_weight) if comment_head is not None else 0.0
    weights: Dict[str, float] = {
        "cls": float(args.cls_weight),
        "reg": float(args.reg_weight),
        "err": err_w,
        "cmt": cmt_w,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    best_state = None
    best_metrics = None
    best_score = -float("inf")
    history: List[Dict[str, Any]] = []

    print(
        f"[setup] device={device} feature_dim={feature_dim} train={len(train_ds)} val={len(val_ds)} "
        f"steps/epoch={steps_per_epoch} total_steps={total_steps} "
        f"warmup_steps={int(round(total_steps * args.warmup_frac))} "
        f"block_pattern={args.block_pattern} mtl_method={args.mtl_method}",
        flush=True,
    )

    for epoch in range(1, args.epochs + 1):
        if args.mtl_method == "phase_ls":
            w_cls_cur, w_reg_cur, phase_id = phased_linear_cls_reg_weights(
                epoch,
                args.epochs,
                args.phase_a_fraction,
                args.phase_a_cls_weight,
                args.phase_a_reg_weight,
                args.phase_b_cls_weight,
                args.phase_b_reg_weight,
            )
            weights["cls"] = w_cls_cur
            weights["reg"] = w_reg_cur
            weights["err"] = err_w
            weights["cmt"] = cmt_w
            phase_tag = phase_id
        else:
            phase_tag = ""

        train_metrics = train_one_epoch(
            model, comment_head, train_loader, optimizer, scheduler, device,
            weights=weights, class_weights=class_weights, error_pos_weight=error_pos_weight,
            grad_clip=args.grad_clip,
            mtl_method=args.mtl_method,
            task_order=task_order,
            dwa_state=dwa_state,
            teacher_force_quality=bool(args.teacher_force_quality),
            quality_bucket_weights=quality_bucket_weights,
        )
        train_metrics_flat = dict(train_metrics)
        dwa_wm = train_metrics_flat.pop("dwa_w_mean", None)
        val_metrics = evaluate(
            model, comment_head, val_loader, device, classes,
            sample_comments=4 if comment_head is not None and (epoch % 5 == 0 or epoch == args.epochs) else 0,
        )
        cur_lr = optimizer.param_groups[0]["lr"]
        hist_row: Dict[str, Any] = {
            "epoch": float(epoch), "lr": float(cur_lr),
            "train_loss": float(train_metrics_flat["loss"]),
            "train_cls": float(train_metrics_flat["cls"]),
            "train_reg": float(train_metrics_flat["reg"]),
            "train_err": float(train_metrics_flat["err"]),
            "train_cmt": float(train_metrics_flat["cmt"]),
            "val_accuracy": float(val_metrics["accuracy"]),
            "val_f1_macro": float(val_metrics["f1_macro"]),
            "val_mae": float(val_metrics["mae"]),
            "val_error_f1_macro": float(val_metrics["error_f1_macro"]),
        }
        vq = val_metrics.get("quality") or {}
        if model.quality_is_classification:
            hist_row["val_quality_accuracy"] = float(vq.get("accuracy", float("nan")))
            hist_row["val_quality_f1_macro"] = float(vq.get("f1_macro", float("nan")))
        if args.mtl_method == "phase_ls":
            hist_row["mtl_phase"] = phase_tag
            hist_row["w_cls"] = float(weights["cls"])
            hist_row["w_reg"] = float(weights["reg"])
        if dwa_wm is not None:
            hist_row["dwa_w_mean"] = dwa_wm
        history.append(hist_row)
        if model.quality_is_classification:
            qa = float(vq.get("accuracy", 0.0))
            score = float(val_metrics["f1_macro"]) + float(val_metrics["error_f1_macro"]) - (1.0 - qa)
        else:
            score = float(val_metrics["f1_macro"]) + float(val_metrics["error_f1_macro"]) - float(val_metrics["mae"])
        if score > best_score:
            best_score = score
            best_metrics = val_metrics
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            if comment_head is None:
                best_comment_state = None
            elif isinstance(comment_head, ClassConditionedCommentHead):
                best_comment_state = {
                    "prefix_proj": {
                        k: v.detach().cpu().clone()
                        for k, v in comment_head.prefix_proj.state_dict().items()
                    },
                }
                if comment_head.class_embeddings is not None:
                    best_comment_state["class_embeddings"] = {
                        k: v.detach().cpu().clone()
                        for k, v in comment_head.class_embeddings.state_dict().items()
                    }
            else:
                best_comment_state = {
                    k: v.detach().cpu().clone()
                    for k, v in comment_head.prefix_proj.state_dict().items()
                }
        if model.quality_is_classification:
            # val_metrics["mae"] duplicates quality["mae_proxy"]: scalar error vs CSV quality using
            # predicted-bucket centres (not optimized when quality trains with CE).
            val_quality_log = (
                f" q_acc={float(vq.get('accuracy', float('nan'))):.4f} "
                f"q_f1={float(vq.get('f1_macro', float('nan'))):.4f} "
                f"proxy_mae={float(val_metrics['mae']):.4f}"
            )
        else:
            val_quality_log = f" mae={val_metrics['mae']:.4f}"
        print(
            f"epoch {epoch:03d} lr={cur_lr:.2e} "
            f"loss={train_metrics_flat['loss']:.4f} (cls={train_metrics_flat['cls']:.3f} reg={train_metrics_flat['reg']:.3f} "
            f"err={train_metrics_flat['err']:.3f} cmt={train_metrics_flat['cmt']:.3f}) | "
            f"val acc={val_metrics['accuracy']:.4f} f1={val_metrics['f1_macro']:.4f}{val_quality_log} "
            f"err_f1={val_metrics['error_f1_macro']:.4f}"
            + (f" | dwa_w={dwa_wm}" if dwa_wm is not None else "")
            + (f" | phase={phase_tag} w_cls={weights['cls']:.2f} w_reg={weights['reg']:.2f}"
               if args.mtl_method == "phase_ls" else ""),
            flush=True,
        )
        if val_metrics["comment_samples"]:
            for s in val_metrics["comment_samples"][:2]:
                print(f"  ↳ [{s['class']}] gen: {s['generated']}")

    if best_state is None:
        raise RuntimeError("Training did not produce a checkpoint state.")

    checkpoint_path = args.output_dir / "xlstm_egoexo_multitask_best.pt"
    torch.save({
        "model": best_state,
        "comment_prefix_proj": best_comment_state if comment_head is not None else None,
        "feature_mode": args.feature_mode, "window": args.window, "stride": args.stride,
        "input_size": feature_dim,
        "hidden": args.hidden, "layers": args.layers, "dropout": args.dropout,
        "num_heads": model.num_heads, "conv_kernel_size": args.conv_kernel_size,
        "projection_factor": args.projection_factor,
        "block_pattern": args.block_pattern,
        "use_attention_pool": bool(args.use_attention_pool),
        "use_fusion": bool(args.use_fusion),
        "fusion_dim": int(args.fusion_dim),
        "guidance_table": guidance_table,
        "comment_table": {f"{k[0]}|{k[1]}": v for k, v in comment_table.items()},
        "comment_quality_bucket_edges": list(q_bucket_edges),
        "quality_bucket_edges_cli": getattr(args, "quality_bucket_edges", None),
        "classes": classes, "class_to_idx": class_to_idx, "idx_to_class": idx_to_class,
        "error_tags": list(ERROR_TAGS),
        "quality_encoding": str(args.quality_encoding),
        "quality_domain_lo": q_domain_lo,
        "quality_domain_hi": q_domain_hi,
        "quality_output_low": float(model.quality_output_low),
        "quality_scale": float(model.quality_scale),
        "mean": scale_mean, "std": scale_std,
        "comment_head": {
            "enabled": False,
            "mode": "retrieval",
            "note": "Comment is a zero-parameter lookup, no LM is used.",
        },
        "weights": weights,
        "mtl_method": args.mtl_method,
        "mtl_task_order": list(task_order),
        "dwa_window": int(args.dwa_window),
        "dwa_temp": float(args.dwa_temp),
        "mtl_phase_schedule": {
            "phase_a_fraction": float(args.phase_a_fraction),
            "phase_a": {
                "cls": float(args.phase_a_cls_weight),
                "reg": float(args.phase_a_reg_weight),
            },
            "phase_b": {
                "cls": float(args.phase_b_cls_weight),
                "reg": float(args.phase_b_reg_weight),
            },
        },
        "warmup_frac": args.warmup_frac, "min_lr_ratio": args.min_lr_ratio,
        "optimizer": args.optimizer,
        "num_quality_classes": int(model.num_quality_classes),
        "quality_head_mode": str(args.quality_head_mode),
        "teacher_force_quality": bool(args.teacher_force_quality),
        "balanced_quality_weights": bool(getattr(args, "balanced_quality_weights", False)),
    }, checkpoint_path)

    history_path = args.output_dir / "training_history.json"
    history_path.write_text(json.dumps(history, indent=2))

    model.load_state_dict(best_state)
    if comment_head is not None and best_comment_state is not None:
        if isinstance(comment_head, ClassConditionedCommentHead):
            comment_head.prefix_proj.load_state_dict(best_comment_state["prefix_proj"])
            emb_sd = best_comment_state.get("class_embeddings")
            if emb_sd is not None and getattr(comment_head, "class_embeddings", None) is not None:
                comment_head.class_embeddings.load_state_dict(emb_sd)
        else:
            comment_head.prefix_proj.load_state_dict(best_comment_state)
    final_metrics = {
        "best_val": best_metrics,
        "test": evaluate(model, comment_head, test_loader, device, classes,
                          sample_comments=8 if comment_head is not None else 0)
                if args.eval_test else None,
    }
    metrics_path = args.output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(final_metrics, indent=2, default=str))
    print(f"saved checkpoint to {checkpoint_path}")
    _print_final_report(final_metrics)
    return {
        "checkpoint_path": str(checkpoint_path),
        "history_path": str(history_path),
        "metrics_path": str(metrics_path),
        "feature_mode": args.feature_mode,
        "best_val": best_metrics,
        "test": final_metrics["test"],
    }


def _print_final_report(final_metrics: Dict[str, Any]) -> None:
    """Pretty-print the test-set metrics at end of training."""
    print("\n" + "=" * 72)
    print(" FINAL EVALUATION — EgoExo-Fitness multi-task")
    print("=" * 72)
    for split in ("best_val", "test"):
        block = final_metrics.get(split)
        if block is None:
            continue
        title = "VALIDATION (best epoch)" if split == "best_val" else "TEST"
        print(f"\n[{title}]")
        # Classification head
        acc = block.get("accuracy", float("nan"))
        f1m = block.get("f1_macro", float("nan"))
        print(f"  classification : accuracy={acc:.4f}  f1_macro={f1m:.4f}")
        # Quality regression
        q = block.get("quality") or {
            "mae": block.get("mae"), "r2": block.get("r2"),
            "rmse": float("nan"), "pearson_r": float("nan"),
            "spearman_rho": float("nan"), "n_samples": 0,
        }
        print(f"  quality (n={q.get('n_samples', 0)}):")
        if q.get("task") == "classification":
            print(f"    [bucket classification] accuracy={q.get('accuracy', float('nan')):.4f}  "
                  f"f1_macro={q.get('f1_macro', float('nan')):.4f}  "
                  f"f1_weighted={q.get('f1_weighted', float('nan')):.4f}")
            print(f"    scalar proxy on bucket centres · MAE={q.get('mae_proxy', float('nan')):.4f}  "
                  f"R²={q.get('scalar_r2', float('nan')):.4f}")
            cm = q.get("confusion_matrix")
            if cm is not None:
                print(f"    bucket confusion rows=true cols=pred:")
                for row in cm:
                    print(f"      {row}")
        else:
            print(f"    MAE        = {q.get('mae', float('nan')):.4f}")
            print(f"    RMSE       = {q.get('rmse', float('nan')):.4f}")
            print(f"    R^2        = {q.get('r2', float('nan')):.4f}")
            print(f"    Pearson r  = {q.get('pearson_r', float('nan')):.4f}")
            print(f"    Spearman ρ = {q.get('spearman_rho', float('nan')):.4f}  "
                  f"(AQA-standard headline metric)")
            per_class = q.get("per_class")
            if per_class:
                print(f"    per-class MAE:")
                for cls, d in sorted(per_class.items(), key=lambda kv: -kv[1]["mae"]):
                    print(f"      {cls:<35s}  n={d['n']:>4d}  MAE={d['mae']:.4f}  RMSE={d['rmse']:.4f}")
        # Error tags (optional)
        ef1 = block.get("error_f1_macro")
        if ef1 is not None and ef1 > 0:
            print(f"  error tags     : F1_macro={ef1:.4f}")
    print("=" * 72)


def main() -> int:
    args = build_parser().parse_args()
    try:
        train_from_args(args)
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
