#!/usr/bin/env python3
"""
Train a BiLSTM on 30-frame windows for:
  (1) exercise classification (which base exercise)
  (2) regression head for quality (CSV ``quality`` column; EgoExo defaults to raw ``action_quality_score`` scale)

Aligned with Riccio (arXiv:2411.11548): BiLSTM on sequences, optional mixed angles+coords,
standardized features (§3.3.1), 30-frame windows. Use --preset riccio for paper Table 4 BiLSTM hyperparameters.
Hyperparameter search (same paper-style ranges): tune_exercise_bilstm.py (Optuna TPE).

Primary path (Riccio / Kaggle-style NPZs):
  - --kaggle-angles-dir: folder with {stem}_biomechanics.npz + {stem}_labels.npz
  - Prefer labels.npz with video_id (from riccio_kaggle_video_pipeline) so train/val/test split **per video**.

Optional (index CSV + angles dir): build_exercise_training_index.py or EgoExo index — see --index-csv / --angles-dir.

Examples:
  # PosePulse-style: mixed (T,42), train-only standardization, 2×128 BiLSTM→BN+Conv+GAP, classification only
  PYTHONPATH=. python train_exercise_bilstm.py --preset posepulse --eval-test \\
    --kaggle-angles-dir results/riccio_realtime_exercise_recognition \\
    --kaggle-stem riccio_realtime_exercise_recognition --epochs 30 \\
    --output-dir results/exercise_bilstm_posepulse

  # Default capstone: Riccio NPZs
  ./venv/bin/python train_exercise_bilstm.py --preset riccio --standardize --eval-test \\
    --kaggle-angles-dir results/riccio_realtime_exercise_recognition \\
    --kaggle-stem riccio_realtime_exercise_recognition --epochs 30

  ./venv/bin/python train_exercise_bilstm.py --preset riccio --standardize --eval-test \\
    --kaggle-angles-dir results/kaggle_exercise_recognition --kaggle-stem kaggle_exercise_recognition

  ./venv/bin/python tune_exercise_bilstm.py --standardize --n-trials 30 --tune-epochs 15 \\
    --kaggle-angles-dir results/riccio_realtime_exercise_recognition \\
    --kaggle-stem riccio_realtime_exercise_recognition
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from fitness_coach.datasets.exercise_bilstm_dataset import (
    ExerciseAngleWindowDataset,
    build_kaggle_angle_datasets,
    build_kaggle_frame_feature_datasets,
    build_kaggle_mixed_datasets,
    coarse_exclusion_normalized_set,
    fit_standardizer_from_dataset,
    load_index_rows,
    normalized_coarse_class,
)
from fitness_coach.datasets.text_coaching_features import TextCoachingEncoder, fit_text_coaching_encoder
from fitness_coach.models.exercise_bilstm_model import build_exercise_bilstm, unpack_bilstm_outputs


def build_bilstm_optimizer(
    model: nn.Module,
    *,
    kind: str,
    lr: float,
    weight_decay: float,
) -> torch.optim.Optimizer:
    """``adam`` or ``adamw`` (default BiLSTM trainer previously used AdamW only)."""
    k = (kind or "adamw").strip().lower()
    if k == "adam":
        return torch.optim.Adam(model.parameters(), lr=float(lr), weight_decay=float(weight_decay))
    if k == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=float(lr), weight_decay=float(weight_decay))
    raise ValueError(f"Unknown optimizer {kind!r}; use 'adam' or 'adamw'.")


@dataclass
class BiLSTMTrainContext:
    """Datasets and metadata after index/Kaggle resolution (before model / loaders)."""

    train_ds: Dataset
    val_ds: Dataset
    test_ds: Optional[Dataset]
    classes: List[str]
    class_to_idx: Dict[str, int]
    idx_to_class: Dict[int, str]
    feat_dim: int
    scale_mean: Optional[Any]
    scale_std: Optional[Any]
    feature_mode: str
    kaggle_dir: str
    window: int
    stride: int
    window_label: str
    excluded_coarse_classes: Optional[List[str]]
    rows: List[dict]


def apply_text_coaching_supervision(
    ctx: BiLSTMTrainContext,
    args: argparse.Namespace,
) -> Tuple[int, Optional[TextCoachingEncoder]]:
    """
    Fit TF–IDF + SVD on training ``comment`` + ``action_guidance`` and attach embeddings to
    index-CSV datasets. No-op for Kaggle NPZ mode (no text in labels) or when ``--no-text-supervision``.
    """
    if not getattr(args, "text_supervision", True):
        return 0, None
    if ctx.kaggle_dir:
        return 0, None
    if len(ctx.train_ds) == 0:
        return 0, None
    enc, z_tr = fit_text_coaching_encoder(
        ctx.train_ds._text_raw,
        svd_dim=int(getattr(args, "text_svd_dim", 64)),
        max_features=int(getattr(args, "text_max_features", 8192)),
    )
    ctx.train_ds.set_text_features(z_tr)
    if len(ctx.val_ds) > 0:
        ctx.val_ds.set_text_features(enc.transform(ctx.val_ds._text_raw))
    if ctx.test_ds is not None and len(ctx.test_ds) > 0:
        if hasattr(ctx.test_ds, "_text_raw") and hasattr(ctx.test_ds, "set_text_features"):
            ctx.test_ds.set_text_features(enc.transform(ctx.test_ds._text_raw))  # type: ignore[attr-defined]
    return enc.dim, enc


def _maybe_report_prune(trial: Any, step: int, value: float) -> None:
    if trial is None:
        return
    try:
        from optuna.exceptions import TrialPruned

        trial.report(float(value), step)
        if trial.should_prune():
            raise TrialPruned()
    except ImportError:
        pass


def parse_exclude_coarse_classes_cli(raw: str) -> Optional[List[str]]:
    """Comma-separated coarse names; empty string → exclude nothing (``None``)."""
    t = (raw or "").strip()
    if not t:
        return None
    return [p.strip() for p in t.split(",") if p.strip()]


def build_bilstm_train_context(args: argparse.Namespace) -> Optional[BiLSTMTrainContext]:
    """
    Build train/val/(test) datasets from --index-csv or --kaggle-angles-dir.
    Returns None if training set is empty or paths invalid.
    """
    kaggle_dir = (args.kaggle_angles_dir or "").strip()
    scale_mean = scale_std = None
    test_ds = None
    rows: List[dict] = []
    exc = parse_exclude_coarse_classes_cli(getattr(args, "exclude_classes", "") or "")

    if kaggle_dir:
        kpath = Path(kaggle_dir)
        if not kpath.is_dir():
            print(f"Not a directory: {kpath}", file=sys.stderr)
            return None
        wl = str(getattr(args, "window_label", "last") or "last").strip().lower()
        if wl not in ("first", "last"):
            print(f"Invalid --window-label {wl!r}; use first or last.", file=sys.stderr)
            return None
        try:
            if args.feature_mode == "mixed":
                train_ds, val_ds, test_ds, class_to_idx, idx_to_class, scale_mean, scale_std = (
                    build_kaggle_mixed_datasets(
                        kpath,
                        stem=args.kaggle_stem,
                        window=args.window,
                        stride=args.stride,
                        test_ratio=args.kaggle_test_ratio,
                        val_ratio=args.kaggle_val_ratio,
                        seed=args.kaggle_seed,
                        standardize=args.standardize,
                        window_label=wl,
                        exclude_coarse_classes=exc,
                    )
                )
            elif args.feature_mode in ("vit_backbone", "resnet_backbone"):
                train_ds, val_ds, test_ds, class_to_idx, idx_to_class, scale_mean, scale_std = (
                    build_kaggle_frame_feature_datasets(
                        kpath,
                        stem=args.kaggle_stem,
                        window=args.window,
                        stride=args.stride,
                        test_ratio=args.kaggle_test_ratio,
                        val_ratio=args.kaggle_val_ratio,
                        seed=args.kaggle_seed,
                        standardize=args.standardize,
                        window_label=wl,
                        exclude_coarse_classes=exc,
                    )
                )
            else:
                if args.feature_mode not in ("angles", "mixed"):
                    print(
                        "Kaggle mode supports feature-mode angles, mixed, vit_backbone, or resnet_backbone; using angles.",
                        file=sys.stderr,
                    )
                    args.feature_mode = "angles"
                train_ds, val_ds, test_ds, class_to_idx, idx_to_class, scale_mean, scale_std = (
                    build_kaggle_angle_datasets(
                        kpath,
                        stem=args.kaggle_stem,
                        window=args.window,
                        stride=args.stride,
                        test_ratio=args.kaggle_test_ratio,
                        val_ratio=args.kaggle_val_ratio,
                        seed=args.kaggle_seed,
                        standardize=args.standardize,
                        window_label=wl,
                        exclude_coarse_classes=exc,
                    )
                )
        except (FileNotFoundError, ValueError, RuntimeError) as e:
            print(str(e), file=sys.stderr)
            return None
        classes = [idx_to_class[i] for i in range(len(class_to_idx))]
    else:
        index_path = Path(args.index_csv)
        angles_dir = Path(args.angles_dir)
        if not index_path.is_file():
            print(f"Missing {index_path} — run build_exercise_training_index.py first", file=sys.stderr)
            return None

        rows = load_index_rows(index_path)
        train_rows = [r for r in rows if r.get("split") == "train"]
        val_rows = [r for r in rows if r.get("split") == "val"]
        if not val_rows:
            val_rows = [r for r in rows if r.get("split") == "test"]
        if not train_rows:
            print("No train split in index", file=sys.stderr)
            return None

        ex_norm = coarse_exclusion_normalized_set(exc)
        classes = sorted(
            {
                r["exercise_class"]
                for r in train_rows
                if normalized_coarse_class(r["exercise_class"]) not in ex_norm
            }
        )
        if len(classes) < 2:
            print(
                "Fewer than 2 exercise classes in train split after --exclude-classes; "
                "clear exclusions or broaden data.",
                file=sys.stderr,
            )
            return None
        class_to_idx = {c: i for i, c in enumerate(classes)}
        idx_to_class = {i: c for c, i in class_to_idx.items()}

        kp_dir = Path(args.keypoints_dir) if args.feature_mode in ("mixed", "coords") else None

        train_ds = ExerciseAngleWindowDataset(
            index_path,
            angles_dir,
            class_to_idx,
            split="train",
            window=args.window,
            stride=args.stride,
            feature_mode=args.feature_mode,
            keypoints_dir=kp_dir,
        )
        val_split = "val" if any(r.get("split") == "val" for r in rows) else (
            "test" if any(r.get("split") == "test" for r in rows) else "train"
        )
        val_ds = ExerciseAngleWindowDataset(
            index_path,
            angles_dir,
            class_to_idx,
            split=val_split,
            window=args.window,
            stride=args.stride,
            feature_mode=args.feature_mode,
            keypoints_dir=kp_dir,
        )
        # Held-out test (e.g. EgoExo split_exercise_index.py): only if val uses "val" rows.
        if val_split == "val" and any(r.get("split") == "test" for r in rows):
            test_ds = ExerciseAngleWindowDataset(
                index_path,
                angles_dir,
                class_to_idx,
                split="test",
                window=args.window,
                stride=args.stride,
                feature_mode=args.feature_mode,
                keypoints_dir=kp_dir,
            )

        if args.standardize and len(train_ds) > 0:
            scale_mean, scale_std = fit_standardizer_from_dataset(train_ds)
            train_ds.apply_standardizer(scale_mean, scale_std)
            val_ds.apply_standardizer(scale_mean, scale_std)
            if test_ds is not None and len(test_ds) > 0:
                test_ds.apply_standardizer(scale_mean, scale_std)

    if len(train_ds) == 0:
        if not kaggle_dir:
            angles_dir = Path(args.angles_dir)
            kp_dir = Path(args.keypoints_dir) if args.feature_mode in ("mixed", "coords") else None
            _diagnose_empty_train(rows, "train", args.feature_mode, angles_dir, kp_dir)
            hint = (
                "For feature-mode=angles: *_biomechanics.npz under --angles-dir.\n"
                "For feature-mode=mixed|coords: *_keypoints.npz under --keypoints-dir (same video_stem as index).\n"
                "Short-clip indices (00000000…) need matching NPZ per stem, or use a long_range index with 0000,0001,…\n"
                "Run: batch_compute_angles_for_index.py (with videos) OR setup_local_training_data.py"
            )
            print(f"No training windows — {hint}", file=sys.stderr)
        else:
            print(
                "No training windows from Kaggle NPZs — check angles length, window/stride, and label alignment.",
                file=sys.stderr,
            )
        return None

    samples = getattr(train_ds, "samples", None)
    feat_dim = int(samples[0][0].shape[1]) if samples else 8
    return BiLSTMTrainContext(
        train_ds=train_ds,
        val_ds=val_ds,
        test_ds=test_ds,
        classes=classes,
        class_to_idx=class_to_idx,
        idx_to_class=idx_to_class,
        feat_dim=feat_dim,
        scale_mean=scale_mean,
        scale_std=scale_std,
        feature_mode=args.feature_mode,
        kaggle_dir=kaggle_dir,
        window=args.window,
        stride=args.stride,
        window_label=str(getattr(args, "window_label", "last") or "last"),
        excluded_coarse_classes=list(exc) if exc else None,
        rows=rows,
    )


def compute_inverse_frequency_class_weights(ctx: BiLSTMTrainContext) -> torch.Tensor:
    """
    Mean-normalized inverse-frequency weights: w_c = N / (K * n_c).
    Upweights rare classes (e.g. hammer curl) in cross-entropy during training.
    """
    samples = getattr(ctx.train_ds, "samples", None)
    if not samples:
        raise ValueError("empty train dataset")
    ys = [int(s[1]) for s in samples]
    cnt = Counter(ys)
    n = len(ys)
    k = len(ctx.classes)
    raw: List[float] = []
    for c in range(k):
        nc = int(cnt.get(c, 0))
        if nc <= 0:
            raw.append(1.0)
        else:
            raw.append(n / (k * nc))
    w = torch.tensor(raw, dtype=torch.float32)
    m = float(w.mean().clamp_min(1e-8))
    w = w / m
    return w


def run_bilstm_training(
    ctx: BiLSTMTrainContext,
    *,
    hidden: int,
    layers: int,
    dropout: float,
    lr: float,
    batch_size: int,
    weight_decay: float,
    epochs: int,
    cls_weight: float,
    reg_weight: float,
    device: torch.device,
    out_dir: Optional[Path],
    save_checkpoint: bool,
    verbose: bool,
    trial: Any = None,
    text_dim: int = 0,
    text_coaching_encoder: Optional[TextCoachingEncoder] = None,
    architecture: str = "plain",
    cnn_hidden: int = 64,
    attn_heads: int = 4,
    ce_class_weights: Optional[torch.Tensor] = None,
    optimizer_kind: str = "adamw",
    dora_head_rank: int = 0,
    dora_head_alpha: float = 8.0,
    label_smoothing: float = 0.0,
    grad_clip_max_norm: float = 2.0,
    use_cosine_schedule: bool = False,
    lr_eta_min: float = 0.0,
) -> Dict[str, Any]:
    """
    Train BiLSTM; maximize validation accuracy for checkpointing.
    If ``trial`` is set (Optuna), reports val acc per epoch and may prune.
    """
    train_loader = DataLoader(
        ctx.train_ds, batch_size=batch_size, shuffle=True, drop_last=False
    )
    val_loader = (
        DataLoader(ctx.val_ds, batch_size=batch_size, shuffle=False)
        if len(ctx.val_ds) > 0
        else None
    )
    test_loader = (
        DataLoader(ctx.test_ds, batch_size=batch_size, shuffle=False)
        if ctx.test_ds is not None and len(ctx.test_ds) > 0
        else None
    )

    model = build_exercise_bilstm(
        architecture,
        input_dim=ctx.feat_dim,
        num_classes=len(ctx.classes),
        hidden=hidden,
        num_layers=layers,
        dropout=dropout,
        text_dim=text_dim,
        cnn_hidden=cnn_hidden,
        attn_heads=attn_heads,
        sequence_len=int(ctx.window),
        dora_head_rank=int(dora_head_rank),
        dora_head_alpha=float(dora_head_alpha),
    ).to(device)
    opt = build_bilstm_optimizer(
        model, kind=optimizer_kind, lr=lr, weight_decay=weight_decay
    )
    scheduler = None
    if use_cosine_schedule:
        from torch.optim.lr_scheduler import CosineAnnealingLR

        scheduler = CosineAnnealingLR(opt, T_max=int(epochs), eta_min=float(lr_eta_min))

    best_acc = 0.0
    best_rmse = 0.0
    best_state: Optional[Dict[str, Any]] = None
    last_tr_loss = 0.0

    for epoch in range(1, epochs + 1):
        last_tr_loss = train_one_epoch(
            model,
            train_loader,
            opt,
            cls_weight,
            reg_weight,
            device,
            ce_class_weights=ce_class_weights,
            label_smoothing=float(label_smoothing),
            grad_clip_max_norm=float(grad_clip_max_norm),
        )
        if val_loader is not None:
            va, vrmse, vreg = evaluate(model, val_loader, device, len(ctx.classes))
        else:
            va, vrmse, vreg = 0.0, 0.0, {"rmse": 0.0, "mae": 0.0, "r2": float("nan")}

        _maybe_report_prune(trial, epoch - 1, va)

        should_save = (val_loader is not None and va > best_acc) or (
            val_loader is None and epoch == epochs
        )
        if should_save:
            if val_loader is not None:
                best_acc = va
                best_rmse = vrmse
            if save_checkpoint:
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        if verbose:
            if getattr(model, "has_regression_head", True):
                qtail = (
                    f"val_q_rmse={vrmse:.4f}  val_q_mae={vreg['mae']:.4f}  val_q_r2={vreg['r2']:.4f}"
                )
            else:
                qtail = "val_quality=n/a (classification-only)"
            lr_cur = opt.param_groups[0].get("lr", float("nan"))
            print(
                f"epoch {epoch:03d}  train_loss={last_tr_loss:.4f}  "
                f"val_acc={va:.4f}  {qtail}  lr={lr_cur:.2e}"
            )
        if scheduler is not None:
            scheduler.step()

    if save_checkpoint and best_state is None and val_loader is None:
        best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        best_acc = 0.0
        best_rmse = 0.0

    ckpt_payload: Optional[Dict[str, Any]] = None
    if save_checkpoint and out_dir is not None and best_state is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        ckpt_payload = {
            "model": best_state,
            "window": ctx.window,
            "stride": ctx.stride,
            "feat_dim": ctx.feat_dim,
            "num_classes": len(ctx.classes),
            "classes": ctx.classes,
            "feature_mode": ctx.feature_mode,
            "scale_mean": ctx.scale_mean,
            "scale_std": ctx.scale_std,
            "hidden": hidden,
            "layers": layers,
            "dropout": dropout,
            "lr": lr,
            "batch_size": batch_size,
            "weight_decay": weight_decay,
            "text_dim": int(text_dim),
            "architecture": str(architecture),
            "cnn_hidden": int(cnn_hidden) if str(architecture) == "cnn_attn" else 0,
            "attn_heads": int(getattr(model, "attn_heads", 0)),
            "posepulse_sequence_len": int(ctx.window),
            "window_label": str(ctx.window_label),
            "optimizer": str(optimizer_kind),
            "dora_head_rank": int(dora_head_rank),
            "dora_head_alpha": float(dora_head_alpha),
            "excluded_coarse_classes": list(getattr(ctx, "excluded_coarse_classes", None) or []),
            "has_regression_head": bool(getattr(model, "has_regression_head", True)),
            "label_smoothing": float(label_smoothing),
            "grad_clip_max_norm": float(grad_clip_max_norm),
            "lr_scheduler": "cosine" if use_cosine_schedule else "none",
            "lr_eta_min": float(lr_eta_min),
        }
        if text_coaching_encoder is not None and int(text_dim) > 0:
            text_coaching_encoder.save(out_dir / "text_coaching_encoder.pkl")
        with open(out_dir / "class_map.json", "w") as f:
            json.dump(
                {"class_to_idx": ctx.class_to_idx, "idx_to_class": ctx.idx_to_class},
                f,
                indent=2,
            )
        torch.save(ckpt_payload, out_dir / "exercise_bilstm_best.pt")

    return {
        "best_val_acc": best_acc,
        "best_val_rmse": best_rmse,
        "last_train_loss": last_tr_loss,
        "test_loader": test_loader,
        "model": model,
        "checkpoint": ckpt_payload,
    }


def _diagnose_empty_train(
    rows: list,
    split: str,
    feature_mode: str,
    angles_dir: Path,
    keypoints_dir: Path | None,
) -> None:
    stems = sorted({(r.get("video_stem") or "").strip() for r in rows if r.get("split") == split})
    if not stems:
        print(f"No rows with split={split!r} in index.", file=sys.stderr)
        return
    sample = stems[:5]
    print(f"Train split has {len(stems)} unique video_stem values (e.g. {sample}).", file=sys.stderr)
    if feature_mode in ("mixed", "coords") and keypoints_dir is not None:
        have = sum(1 for s in stems if (keypoints_dir / f"{s}_keypoints.npz").is_file())
        print(
            f"Under {keypoints_dir}: {have}/{len(stems)} matching *_keypoints.npz files.",
            file=sys.stderr,
        )
        if have == 0 and len(stems) > 0:
            ex = stems[0]
            print(
                f"  Expected e.g. {keypoints_dir / (ex + '_keypoints.npz')!s} — "
                "stems in CSV must match filenames (short_clips 00000000 vs long_range 0000).",
                file=sys.stderr,
            )
    if feature_mode == "angles" or feature_mode == "mixed":
        have_a = sum(1 for s in stems if (angles_dir / f"{s}_biomechanics.npz").is_file())
        print(
            f"Under {angles_dir}: {have_a}/{len(stems)} matching *_biomechanics.npz files.",
            file=sys.stderr,
        )
        if have_a == 0 and stems:
            ex = stems[0]
            looks_short_clip = len(ex) >= 8 and ex.isdigit()
            if looks_short_clip:
                print(
                    "  Index stems look like short clips (e.g. 00000000); local demo NPZs often use "
                    "long_range names (0000, 0001). Example:",
                    file=sys.stderr,
                )
                print(
                    "  ./venv/bin/python train_exercise_bilstm.py \\\n"
                    "    --index-csv results/exercise_training_index_long_range_split.csv \\\n"
                    "    --feature-mode mixed \\\n"
                    "    --keypoints-dir results/processed_keypoints_mediapipe \\\n"
                    "    --angles-dir results/exercise_angles",
                    file=sys.stderr,
                )


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    opt: torch.optim.Optimizer,
    cls_weight: float,
    reg_weight: float,
    device: torch.device,
    ce_class_weights: Optional[torch.Tensor] = None,
    label_smoothing: float = 0.0,
    grad_clip_max_norm: float = 2.0,
):
    model.train()
    total = 0.0
    n = 0
    w = ce_class_weights.to(device) if ce_class_weights is not None else None
    ce = nn.CrossEntropyLoss(weight=w, label_smoothing=float(label_smoothing))
    mse = nn.MSELoss()
    for batch in loader:
        if len(batch) == 4:
            xb, y_cls, y_q, tb = batch
            tb = tb.to(device)
        else:
            xb, y_cls, y_q = batch
            tb = None
        xb = xb.to(device)
        y_cls = y_cls.to(device)
        y_q = y_q.to(device)
        opt.zero_grad()
        logits, pred_q = unpack_bilstm_outputs(model(xb, tb))
        if pred_q is None:
            loss = cls_weight * ce(logits, y_cls)
        else:
            loss = cls_weight * ce(logits, y_cls) + reg_weight * mse(pred_q, y_q)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), float(grad_clip_max_norm))
        opt.step()
        total += float(loss.item()) * xb.size(0)
        n += xb.size(0)
    return total / max(n, 1)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    num_classes: int,
    *,
    detailed: bool = False,
    class_names: Optional[List[str]] = None,
) -> Union[
    Tuple[float, float, Dict[str, float]],
    Tuple[float, float, Dict[str, float], Dict[str, Any], np.ndarray, np.ndarray],
]:
    model.eval()
    ce = nn.CrossEntropyLoss()
    mse = nn.MSELoss()
    correct = 0
    tot = 0
    reg_err = 0.0
    abs_err = 0.0
    q_true: List[float] = []
    q_pred: List[float] = []
    ys_list: List[int] = []
    pred_list: List[int] = []
    probs_list: List[np.ndarray] = []
    for batch in loader:
        if len(batch) == 4:
            xb, y_cls, y_q, tb = batch
            tb = tb.to(device)
        else:
            xb, y_cls, y_q = batch
            tb = None
        xb = xb.to(device)
        y_cls = y_cls.to(device)
        y_q = y_q.to(device)
        logits, pred_q = unpack_bilstm_outputs(model(xb, tb))
        pred = logits.argmax(dim=1)
        if detailed:
            ys_list.extend(y_cls.cpu().numpy().tolist())
            pred_list.extend(pred.cpu().numpy().tolist())
            probs_list.append(torch.softmax(logits, dim=1).detach().cpu().numpy())
        correct += int((pred == y_cls).sum().item())
        tot += xb.size(0)
        if pred_q is not None:
            reg_err += float(mse(pred_q, y_q).item()) * xb.size(0)
            abs_err += float(torch.abs(pred_q - y_q).sum().item())
            q_true.extend(y_q.detach().cpu().numpy().astype(np.float64).tolist())
            q_pred.extend(pred_q.detach().cpu().numpy().astype(np.float64).tolist())
    acc = correct / max(tot, 1)
    if not q_pred:
        reg_metrics = {"rmse": float("nan"), "mae": float("nan"), "r2": float("nan")}
    else:
        rmse = (reg_err / max(tot, 1)) ** 0.5
        mae = abs_err / max(tot, 1)
        r2 = float("nan")
        if tot > 0 and len(q_true) > 1:
            try:
                from sklearn.metrics import r2_score

                yt = np.asarray(q_true, dtype=np.float64)
                yp = np.asarray(q_pred, dtype=np.float64)
                r2 = float(r2_score(yt, yp))
            except ValueError:
                r2 = float("nan")
        reg_metrics = {"rmse": float(rmse), "mae": float(mae), "r2": r2}
    if detailed:
        if class_names is None:
            raise ValueError("class_names required when detailed=True")
        from fitness_coach.evaluation.classification_metrics import detailed_classification_metrics

        y_true_arr = np.array(ys_list, dtype=np.int64)
        metrics = detailed_classification_metrics(
            y_true_arr,
            np.array(pred_list, dtype=np.int64),
            class_names,
        )
        y_prob_arr = np.vstack(probs_list) if probs_list else np.zeros((0, len(class_names)), dtype=np.float64)
        return acc, reg_metrics["rmse"], reg_metrics, metrics, y_true_arr, y_prob_arr
    return acc, reg_metrics["rmse"], reg_metrics


def add_bilstm_train_args(ap: argparse.ArgumentParser) -> None:
    """Data paths, Kaggle options, and model hyperparameters (shared with tune_exercise_bilstm.py)."""
    ap.add_argument("--index-csv", default="./results/exercise_training_index.csv")
    ap.add_argument("--angles-dir", default="./results/exercise_angles")
    ap.add_argument(
        "--keypoints-dir",
        default="./results/processed_keypoints_mediapipe",
        help="For --feature-mode mixed: folder with {stem}_keypoints.npz",
    )
    ap.add_argument(
        "--feature-mode",
        choices=("angles", "coords", "mixed", "vit_backbone", "resnet_backbone"),
        default="angles",
        help="angles=(T,8); coords=(T,34); mixed=(T,42); vit_backbone=(T,D) ViT/ViTPose crop embeddings (typ. D=256); "
        "resnet_backbone=(T,D) torchvision ResNet GAP embeddings from --representation resnet_backbone (D=512 or 2048).",
    )
    ap.add_argument(
        "--standardize",
        action="store_true",
        help="Fit StandardScaler on training windows only, apply train+val (paper §3.3.1)",
    )
    ap.add_argument(
        "--preset",
        choices=("none", "riccio", "posepulse", "paper_riccio", "paper_posepulse_vit", "paper_posepulse_resnet"),
        default="none",
        help="riccio: Table 4 BiLSTM — units 73, dropout ~0.22, lr 4e-4, batch 54. "
        "posepulse: mixed (T,42), window label first, standardize, diagram BiLSTM–CNN "
        "(2×128 + BN + GAP), Adam, **no** quality head / DoRA. Exclude hammer curl by default "
        "(--exclude-classes). Use --exclude-classes \"\" to keep all classes. "
        "paper_riccio: capstone paper BiLSTM–CNN on (1,30,8) map, AdamW, cosine LR, label smoothing 0.1, "
        "grad clip 1.0, batch 64, 50 epochs, DoRA rank 8 on classifier, exclude hammer curl by default. "
        "paper_posepulse_vit: same as paper_riccio but inputs are ViTPose-S 256-D frame_features "
        "(export NPZ with riccio_kaggle_video_pipeline --representation vit_backbone). "
        "paper_posepulse_resnet: same hyperparameters but ResNet crop embeddings "
        "(export with --representation resnet_backbone).",
    )
    ap.add_argument("--output-dir", default="./results/exercise_bilstm")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--dropout", type=float, default=0.35)
    ap.add_argument("--window", type=int, default=30)
    ap.add_argument("--stride", type=int, default=15)
    ap.add_argument("--cls-weight", type=float, default=1.0)
    ap.add_argument("--reg-weight", type=float, default=0.5)
    ap.add_argument(
        "--label-smoothing",
        type=float,
        default=0.0,
        help="Cross-entropy label smoothing (paper capstone: 0.1 with --preset paper_riccio).",
    )
    ap.add_argument(
        "--grad-clip",
        type=float,
        default=2.0,
        dest="grad_clip",
        help="Max gradient norm (paper: 1.0 with --preset paper_riccio).",
    )
    ap.add_argument(
        "--lr-scheduler",
        choices=("none", "cosine"),
        default="none",
        help="Learning-rate schedule; paper uses cosine to η_min with --preset paper_riccio.",
    )
    ap.add_argument(
        "--lr-eta-min",
        type=float,
        default=0.0,
        help="Cosine schedule floor (paper: 3e-6). Used when --lr-scheduler cosine.",
    )
    ap.add_argument(
        "--weight-decay",
        type=float,
        default=1e-4,
        help="L2 weight decay for Adam / AdamW.",
    )
    ap.add_argument(
        "--optimizer",
        choices=("adam", "adamw"),
        default="adamw",
        help="Optimization algorithm. Preset posepulse uses Adam (plain heads, no DoRA).",
    )
    ap.add_argument(
        "--dora-head-rank",
        type=int,
        default=0,
        help="If >0, wrap classifier linears with DoRA (plain / cnn_attn / posepulse_bilstm_cnn / paper_riccio_bilstm_cnn). "
        "Ignored for posepulse_diagram (single Linear classifier).",
    )
    ap.add_argument(
        "--dora-head-alpha",
        type=float,
        default=8.0,
        help="LoRA scaling alpha for DoRA head adapters when --dora-head-rank > 0.",
    )
    ap.add_argument(
        "--no-dora-head",
        action="store_true",
        help="Legacy: with older posepulse presets, keep plain Linear heads. posepulse_diagram never uses DoRA.",
    )
    ap.add_argument("--cpu", action="store_true", help="Force CPU even if CUDA is available.")
    ap.add_argument(
        "--kaggle-angles-dir",
        default="",
        help="If set, train from Kaggle pipeline NPZs ({stem}_biomechanics.npz + {stem}_labels.npz; "
        "for --feature-mode mixed also {stem}_keypoints.npz) instead of --index-csv; "
        "windows labeled by coarse exercise (see --window-label).",
    )
    ap.add_argument(
        "--kaggle-stem",
        default="kaggle_exercise_recognition",
        help="Filename prefix for *_biomechanics.npz and *_labels.npz under --kaggle-angles-dir.",
    )
    ap.add_argument("--kaggle-test-ratio", type=float, default=0.15)
    ap.add_argument("--kaggle-val-ratio", type=float, default=0.15)
    ap.add_argument("--kaggle-seed", type=int, default=42)
    ap.add_argument(
        "--text-supervision",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="TF–IDF/SVD on comment+action_guidance (index CSV only; no effect in Kaggle NPZ mode). Default: on.",
    )
    ap.add_argument(
        "--text-svd-dim",
        type=int,
        default=64,
        help="TruncatedSVD components for coaching text (fit on train split only).",
    )
    ap.add_argument(
        "--text-max-features",
        type=int,
        default=8192,
        help="TfidfVectorizer max_features for coaching text.",
    )
    ap.add_argument(
        "--window-label",
        choices=("first", "last"),
        default="last",
        help="Kaggle sliding windows: class from first or last frame in the window (PosePulse uses first).",
    )
    ap.add_argument(
        "--architecture",
        choices=("plain", "cnn_attn", "posepulse_bilstm_cnn", "posepulse_diagram", "paper_riccio_bilstm_cnn"),
        default="plain",
        help="plain: BiLSTM+mean pool; cnn_attn: dilated1D CNN → BiLSTM → MHA; "
        "posepulse_bilstm_cnn: legacy small BiLSTM→Conv (quality head); "
        "posepulse_diagram: 2×128 BiLSTM, BN+Conv+GAP, classification only (cleaner ONNX / Netron). "
        "paper_riccio_bilstm_cnn: 1×BiLSTM (4/dir), (1,30,8) Conv2D stack 128→256→64→1, CE-only.",
    )
    ap.add_argument(
        "--cnn-hidden",
        type=int,
        default=64,
        help="Dilated CNN hidden channels (cnn_attn only).",
    )
    ap.add_argument(
        "--attn-heads",
        type=int,
        default=4,
        help="MultiheadAttention heads; adjusted if embed_dim is not divisible (cnn_attn only).",
    )
    ap.add_argument(
        "--classification-only",
        action="store_true",
        help="Optimize classification only (sets --reg-weight 0).",
    )
    ap.add_argument(
        "--exclude-classes",
        type=str,
        default="hammer curl",
        help="Comma-separated coarse exercise names to omit from windows (Kaggle) or index rows. "
        "Matching is case-insensitive; spaces/underscores normalized. Empty string = exclude none.",
    )
    ap.add_argument(
        "--balanced-class-weights",
        action="store_true",
        help="Inverse-frequency weights in cross-entropy (helps rare classes; Riccio/Kaggle).",
    )


def main() -> int:
    ap = argparse.ArgumentParser(description="Train BiLSTM exercise classifier + quality regressor")
    add_bilstm_train_args(ap)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument(
        "--eval-test",
        action="store_true",
        help="After training, load best checkpoint and evaluate on test windows "
        "(Kaggle mode, or index CSV with split=test when split=val exists).",
    )
    args = ap.parse_args()

    if args.preset == "riccio":
        args.batch_size = 54
        args.lr = 0.0004
        args.hidden = 73
        args.dropout = 0.2174
    elif args.preset == "posepulse":
        args.architecture = "posepulse_diagram"
        args.feature_mode = "mixed"
        args.window_label = "first"
        args.standardize = True
        args.reg_weight = 0.0
        args.dropout = min(float(args.dropout), 0.2)
        args.optimizer = "adam"
        args.dora_head_rank = 0
        args.dora_head_alpha = 8.0
    elif args.preset == "paper_riccio":
        args.architecture = "paper_riccio_bilstm_cnn"
        args.feature_mode = "mixed"
        args.window_label = "first"
        args.standardize = True
        args.reg_weight = 0.0
        args.dropout = 0.0
        args.optimizer = "adamw"
        args.lr = 3e-4
        args.weight_decay = 1e-4
        args.batch_size = 64
        args.epochs = 50
        args.dora_head_rank = 8
        args.dora_head_alpha = 8.0
        args.label_smoothing = 0.1
        args.grad_clip = 1.0
        args.lr_scheduler = "cosine"
        args.lr_eta_min = 3e-6
    elif args.preset == "paper_posepulse_vit":
        args.architecture = "paper_riccio_bilstm_cnn"
        args.feature_mode = "vit_backbone"
        args.window_label = "first"
        args.standardize = True
        args.reg_weight = 0.0
        args.dropout = 0.0
        args.optimizer = "adamw"
        args.lr = 3e-4
        args.weight_decay = 1e-4
        args.batch_size = 64
        args.epochs = 50
        args.dora_head_rank = 8
        args.dora_head_alpha = 8.0
        args.label_smoothing = 0.1
        args.grad_clip = 1.0
        args.lr_scheduler = "cosine"
        args.lr_eta_min = 3e-6
    elif args.preset == "paper_posepulse_resnet":
        args.architecture = "paper_riccio_bilstm_cnn"
        args.feature_mode = "resnet_backbone"
        args.window_label = "first"
        args.standardize = True
        args.reg_weight = 0.0
        args.dropout = 0.0
        args.optimizer = "adamw"
        args.lr = 3e-4
        args.weight_decay = 1e-4
        args.batch_size = 64
        args.epochs = 50
        args.dora_head_rank = 8
        args.dora_head_alpha = 8.0
        args.label_smoothing = 0.1
        args.grad_clip = 1.0
        args.lr_scheduler = "cosine"
        args.lr_eta_min = 3e-6

    if getattr(args, "no_dora_head", False):
        args.dora_head_rank = 0

    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")

    ctx = build_bilstm_train_context(args)
    if ctx is None:
        return 1

    text_dim, text_enc = apply_text_coaching_supervision(ctx, args)

    ce_w: Optional[torch.Tensor] = None
    if getattr(args, "balanced_class_weights", False):
        ce_w = compute_inverse_frequency_class_weights(ctx)
        print(
            "Balanced CE weights (mean=1): "
            + ", ".join(f"{ctx.classes[i]}={ce_w[i].item():.3f}" for i in range(len(ctx.classes)))
        )

    out_dir = Path(args.output_dir)
    res = run_bilstm_training(
        ctx,
        hidden=args.hidden,
        layers=args.layers,
        dropout=args.dropout,
        lr=args.lr,
        batch_size=args.batch_size,
        weight_decay=args.weight_decay,
        epochs=args.epochs,
        cls_weight=args.cls_weight,
        reg_weight=args.reg_weight,
        device=device,
        out_dir=out_dir,
        save_checkpoint=True,
        verbose=True,
        trial=None,
        text_dim=text_dim,
        text_coaching_encoder=text_enc,
        architecture=args.architecture,
        cnn_hidden=args.cnn_hidden,
        attn_heads=args.attn_heads,
        ce_class_weights=ce_w,
        optimizer_kind=str(args.optimizer),
        dora_head_rank=int(args.dora_head_rank),
        dora_head_alpha=float(args.dora_head_alpha),
        label_smoothing=float(args.label_smoothing),
        grad_clip_max_norm=float(args.grad_clip),
        use_cosine_schedule=(str(args.lr_scheduler) == "cosine"),
        lr_eta_min=float(args.lr_eta_min),
    )
    best_acc = res["best_val_acc"]
    ckpt_path = out_dir / "exercise_bilstm_best.pt"
    print(f"Best val acc ≈ {best_acc:.4f}; checkpoint: {ckpt_path}")

    kaggle_dir = ctx.kaggle_dir
    test_loader = res.get("test_loader")
    model = res["model"]
    classes = ctx.classes

    if args.eval_test and test_loader is not None and ckpt_path.is_file():
        try:
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        except TypeError:
            ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model"])
        ta, trmse, treg, test_cls, y_true_t, y_prob_t = evaluate(
            model,
            test_loader,
            device,
            len(classes),
            detailed=True,
            class_names=classes,
        )
        has_q = getattr(model, "has_regression_head", True)
        if has_q:
            print(
                f"Test acc={ta:.4f}  quality RMSE={trmse:.4f}  MAE={treg['mae']:.4f}  R²={treg['r2']:.4f}"
            )
        else:
            print(f"Test acc={ta:.4f}  (classification-only; no quality head)")
        print(f"  F1 (macro)={test_cls['f1_macro']:.4f}  recall (macro)={test_cls['recall_macro']:.4f}")
        print("  Per-class F1:", test_cls["f1_per_class"])
        from fitness_coach.evaluation.classification_metrics import format_confusion_matrix_text

        print("  Confusion matrix (rows=true, cols=predicted):")
        print(
            format_confusion_matrix_text(
                test_cls["confusion_matrix"],
                test_cls["confusion_matrix_row_labels"],
            )
        )
        metrics_path = out_dir / "test_classification_metrics.json"
        probs_path = out_dir / "test_classification_probs.npz"
        with open(metrics_path, "w") as f:
            json.dump(
                {
                    "split": "test",
                    "class_names": classes,
                    "accuracy": test_cls["accuracy"],
                    "f1_macro": test_cls["f1_macro"],
                    "f1_weighted": test_cls["f1_weighted"],
                    "recall_macro": test_cls["recall_macro"],
                    "precision_macro": test_cls["precision_macro"],
                    "f1_per_class": test_cls["f1_per_class"],
                    "recall_per_class": test_cls["recall_per_class"],
                    "precision_per_class": test_cls["precision_per_class"],
                    "confusion_matrix": test_cls["confusion_matrix"],
                    "confusion_matrix_note": test_cls["confusion_matrix_note"],
                    **(
                        {
                            "quality_rmse": treg["rmse"],
                            "quality_mae": treg["mae"],
                            "quality_r2": treg["r2"],
                        }
                        if has_q
                        else {
                            "quality_rmse": None,
                            "quality_mae": None,
                            "quality_r2": None,
                        }
                    ),
                },
                f,
                indent=2,
            )
        np.savez(
            probs_path,
            y_true=y_true_t,
            y_prob=y_prob_t,
            class_names=np.array(classes, dtype=object),
        )
        print(f"  Wrote {metrics_path}")
        print(f"  Wrote {probs_path} (for ROC plots)")
    elif args.eval_test and test_loader is None:
        print(
            "Skipping --eval-test: no test DataLoader (Kaggle: check split ratios; "
            "index CSV: need both split=val and split=test rows).",
            file=sys.stderr,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
