#!/usr/bin/env python3
"""
Train both classification models from the PosePulse paper on identical inputs:

* BiLSTM-CNN          (Riccio §3.3): BiLSTM(4 hidden/direction) → 3 × Conv2D
                                     (128, 256, 64) → Conv2D(1) → Flatten → Linear → 4-class
* xLSTM[7:1]                       : 7 mLSTM + 1 sLSTM blocks, hidden=256, mean-pool → Linear → 4-class

Hyperparameters per paper §3.3.3:
  optimiser           = AdamW
  learning rate       = 3e-4   (cosine to 3e-6)
  betas               = (0.9, 0.999)
  eps                 = 1e-8
  weight decay        = 1e-4
  grad clip           = 1.0
  epochs              = 50
  batch size          = 64
  loss                = CrossEntropy with label smoothing 0.1
  best-ckpt           = max val accuracy

Both models receive the same 30-frame windows of ViTPose-S features ``(N, 30, 256)``.
Pass ``--feature-dim 42`` to train on the legacy angle-only feature set instead.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

from fitness_coach.datasets.exercise_bilstm_dataset import (
    ExerciseAngleWindowDataset, build_class_map, load_index_rows,
    fit_standardizer_from_dataset, make_windows, load_angles_npz,
    build_kaggle_frame_feature_datasets,
)
from fitness_coach.evaluation.classification_metrics import (
    classification_report_with_roc_proba,
    format_confusion_matrix_text,
)
from fitness_coach.models.exercise_bilstm_model import ExerciseBiLSTMCNN
from fitness_coach.models.xlstm_model import xLSTMExerciseClassifier


# ──────────────────────────────────────────────────────────────────── CLI


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train BiLSTM-CNN and xLSTM[7:1] per the PosePulse paper")
    p.add_argument("--index-csv", type=Path, default=None,
                   help="Riccio index CSV with split column (train/val/test). "
                        "Either --index-csv or --kaggle-angles-dir is required.")
    p.add_argument("--features-dir", type=Path, default=None,
                   help="Directory of per-clip *.npz feature files (angles or ViTPose embeddings).")
    p.add_argument("--kaggle-angles-dir", type=Path, default=None,
                   help="Combined Riccio-pipeline output dir containing {stem}_biomechanics.npz "
                        "and {stem}_labels.npz (frame_features + video_id).")
    p.add_argument("--kaggle-stem", type=str, default="riccio_realtime_exercise_recognition",
                   help="Stem of the combined NPZ files in --kaggle-angles-dir.")
    p.add_argument("--exclude-classes", type=str, default="hammer curl",
                   help="Comma-separated coarse exercise names to exclude (case-insensitive). "
                        "Empty string = exclude none.")
    p.add_argument("--kaggle-test-ratio", type=float, default=0.15)
    p.add_argument("--kaggle-val-ratio",  type=float, default=0.15)
    p.add_argument("--kaggle-seed",       type=int,   default=42)
    p.add_argument("--feature-dim", type=int, default=256,
                   help="Per-frame feature dim. 256 = ViTPose-S, 42 = angles. Default 256.")
    p.add_argument("--seq-len",    type=int, default=30)
    p.add_argument("--stride",     type=int, default=15)
    p.add_argument("--num-classes", type=int, default=4)
    p.add_argument("--output-dir", type=Path, default=Path("results/paper_classification"))

    p.add_argument("--epochs",     type=int,   default=50)
    p.add_argument("--batch-size", type=int,   default=64)
    p.add_argument("--lr",         type=float, default=3e-4)
    p.add_argument("--min-lr",     type=float, default=3e-6)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--grad-clip",  type=float, default=1.0)
    p.add_argument("--label-smoothing", type=float, default=0.1)
    p.add_argument("--num-workers", type=int, default=0)

    p.add_argument("--bilstm-hidden", type=int, default=4,
                   help="Hidden units per direction in BiLSTM (paper: 4 → 8 concatenated)")
    p.add_argument("--xlstm-hidden",  type=int, default=256)
    p.add_argument("--xlstm-num-heads", type=int, default=4)
    p.add_argument("--xlstm-block-pattern", default="mmmmmmms")  # xLSTM[7:1]
    p.add_argument("--xlstm-conv-kernel-size", type=int, default=4)
    p.add_argument("--xlstm-projection-factor", type=float, default=4.0 / 3.0)
    p.add_argument("--dropout",       type=float, default=0.15)

    p.add_argument("--models", nargs="+", choices=("bilstm", "xlstm"),
                   default=("bilstm", "xlstm"))
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--warmup-frac", type=float, default=0.1,
                   help="Fraction of total steps used for linear LR warmup before "
                        "cosine decay (Beck et al. xLSTM §B.1). 0 disables warmup.")
    p.add_argument("--ema-decay", type=float, default=0.999,
                   help="Exponential moving average decay for evaluation weights. "
                        "Set to 0 to disable EMA (vanilla eval).")
    p.add_argument("--no-eval-plots", action="store_true",
                   help="Skip saving ROC and confusion-matrix figures (matplotlib).")
    p.add_argument("--no-save-test-probs", action="store_true",
                   help="Skip saving test_probs.npz (y_true, y_pred, softmax probabilities).")
    return p


# ──────────────────────────────────────────────────────── data + scheduler


def cosine_to_min(optimizer: torch.optim.Optimizer, total_steps: int, base_lr: float, min_lr: float) -> LambdaLR:
    """Cosine schedule from ``base_lr`` to ``min_lr`` over ``total_steps``."""
    floor = max(0.0, float(min_lr) / max(1e-12, float(base_lr)))
    def lr_lambda(step: int) -> float:
        progress = min(1.0, step / max(1, total_steps))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return floor + (1.0 - floor) * cosine
    return LambdaLR(optimizer, lr_lambda=lr_lambda)


def warmup_cosine_to_min(
    optimizer: torch.optim.Optimizer,
    total_steps: int,
    base_lr: float,
    min_lr: float,
    warmup_frac: float,
) -> LambdaLR:
    """Linear warmup over ``warmup_frac`` of steps, then cosine decay to ``min_lr``.

    Standard recipe in Beck et al. (xLSTM, 2024 §B.1) and most modern sequence-model
    training pipelines. Stabilises the first few hundred steps where exponential
    gates can saturate at full lr.
    """
    total_steps = max(1, int(total_steps))
    warmup_steps = max(1, int(round(total_steps * max(0.0, min(1.0, warmup_frac)))))
    floor = max(0.0, float(min_lr) / max(1e-12, float(base_lr)))

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        progress = min(1.0, max(0.0, progress))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return floor + (1.0 - floor) * cosine

    return LambdaLR(optimizer, lr_lambda=lr_lambda)


class ModelEMA:
    """Exponential moving average of model parameters (Polyak averaging).

    Maintains shadow weights updated as ``shadow = decay·shadow + (1-decay)·param``
    after every optimiser step. Use ``ema.apply_to(model)`` to swap shadow weights
    into the model for evaluation, then restore via the saved backup.

    Reference: Tarvainen & Valpola, "Mean Teachers" (NeurIPS 2017).
    """

    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = float(decay)
        self.shadow: Dict[str, torch.Tensor] = {
            k: v.detach().clone() for k, v in model.state_dict().items()
            if v.dtype.is_floating_point
        }

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        d = self.decay
        for k, v in model.state_dict().items():
            if k in self.shadow:
                self.shadow[k].mul_(d).add_(v.detach(), alpha=1.0 - d)

    def state_dict(self) -> Dict[str, torch.Tensor]:
        return {k: v.clone() for k, v in self.shadow.items()}


def build_datasets(args: argparse.Namespace):
    """Returns (train_ds, val_ds, test_ds, class_map). Supports two input modes:

    1. Riccio combined-NPZ (``--kaggle-angles-dir``): one ``*_biomechanics.npz``
       holding ``frame_features (T, D)`` plus ``*_labels.npz`` with ``pose`` +
       ``video_id``. Splits are stratified per-video.
    2. Per-clip index CSV (``--index-csv`` + ``--features-dir``).
    """
    if args.kaggle_angles_dir is not None:
        exc = [s.strip() for s in (args.exclude_classes or "").split(",") if s.strip()]
        train_ds, val_ds, test_ds, class_to_idx, _idx_to_class, _mean, _std = (
            build_kaggle_frame_feature_datasets(
                Path(args.kaggle_angles_dir),
                stem=args.kaggle_stem,
                window=args.seq_len,
                stride=args.stride,
                test_ratio=args.kaggle_test_ratio,
                val_ratio=args.kaggle_val_ratio,
                seed=args.kaggle_seed,
                standardize=True,
                window_label="first",
                exclude_coarse_classes=exc or None,
            )
        )
        if len(class_to_idx) != args.num_classes:
            print(f"[warn] kaggle data has {len(class_to_idx)} classes — overriding --num-classes")
            args.num_classes = len(class_to_idx)
        return train_ds, val_ds, test_ds, class_to_idx

    if args.index_csv is None or args.features_dir is None:
        raise ValueError("Provide either --kaggle-angles-dir or both --index-csv and --features-dir.")

    rows = load_index_rows(args.index_csv)
    train_rows = [r for r in rows if (r.get("split") or "train") == "train"]
    val_rows   = [r for r in rows if (r.get("split") or "")     == "val"]
    test_rows  = [r for r in rows if (r.get("split") or "")     == "test"]
    if not train_rows:
        raise ValueError("No training rows; check the split column of the index CSV.")
    class_map = build_class_map(train_rows)
    if len(class_map) != args.num_classes:
        print(f"[warn] index has {len(class_map)} classes — overriding --num-classes")
        args.num_classes = len(class_map)

    def _ds(rows):
        return ExerciseAngleWindowDataset(
            rows, class_map, args.features_dir,
            window=args.seq_len, stride=args.stride,
        )
    train_ds = _ds(train_rows)
    val_ds   = _ds(val_rows)  if val_rows  else None
    test_ds  = _ds(test_rows) if test_rows else None

    mean, std = fit_standardizer_from_dataset(train_ds)
    train_ds.set_standardizer(mean, std)
    if val_ds:  val_ds.set_standardizer(mean, std)
    if test_ds: test_ds.set_standardizer(mean, std)
    return train_ds, val_ds, test_ds, class_map


# ──────────────────────────────────────────────────────── train / eval


def train_one(
    model: nn.Module,
    name: str,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    class_names: List[str],
    args: argparse.Namespace,
    device: torch.device,
) -> Dict[str, Any]:
    optim = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr, betas=(0.9, 0.999), eps=1e-8,
        weight_decay=args.weight_decay,
    )
    steps = max(1, math.ceil(len(train_loader)) * args.epochs)
    warmup_frac = float(getattr(args, "warmup_frac", 0.0) or 0.0)
    if warmup_frac > 0.0:
        sched = warmup_cosine_to_min(optim, steps, args.lr, args.min_lr, warmup_frac)
    else:
        sched = cosine_to_min(optim, steps, args.lr, args.min_lr)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

    ema_decay = float(getattr(args, "ema_decay", 0.0) or 0.0)
    ema = ModelEMA(model, decay=ema_decay) if ema_decay > 0.0 else None
    print(f"[{name}] schedule={'warmup+cosine' if warmup_frac > 0 else 'cosine'} "
          f"warmup_steps={int(round(steps * warmup_frac)) if warmup_frac > 0 else 0}  "
          f"ema_decay={ema_decay if ema else 'off'}")

    best_acc = -1.0
    best_state = None
    history: List[Dict[str, float]] = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        running, seen, correct = 0.0, 0, 0
        for batch in train_loader:
            x, y = _unpack(batch, device)
            optim.zero_grad()
            logits = _logits(model(x))
            loss = loss_fn(logits, y)
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optim.step(); sched.step()
            if ema is not None:
                ema.update(model)
            running += float(loss.item()) * x.size(0)
            seen    += x.size(0)
            correct += int((logits.argmax(1) == y).sum().item())
        train_loss = running / max(1, seen)
        train_acc  = correct / max(1, seen)

        val_acc, val_f1 = _evaluate_with_ema(model, ema, val_loader, device)
        history.append({"epoch": epoch, "train_loss": train_loss, "train_acc": train_acc,
                        "val_acc": val_acc, "val_f1": val_f1, "lr": optim.param_groups[0]["lr"]})
        print(f"[{name}] epoch {epoch:03d} loss={train_loss:.4f} train_acc={train_acc:.4f} "
              f"val_acc={val_acc:.4f} val_f1={val_f1:.4f}  lr={optim.param_groups[0]['lr']:.2e}")
        if val_acc > best_acc:
            best_acc = val_acc
            # Snapshot whichever weights produced the val_acc we just measured.
            # With EMA on, that's the EMA shadow merged with raw buffers from the
            # live model (BN running stats etc.); without EMA, the live model.
            snapshot_src = model.state_dict()
            if ema is not None:
                snapshot_src = {**snapshot_src, **ema.state_dict()}
            best_state = {k: v.detach().cpu().clone() for k, v in snapshot_src.items()}

    assert best_state is not None
    model.load_state_dict(best_state)
    test_acc, test_f1 = _evaluate(model, test_loader, device)

    y_true, y_pred, probs = np.array([]), np.array([]), np.empty((0, args.num_classes))
    if test_loader is not None and len(test_loader.dataset) > 0:
        yt, pr, pb = collect_test_predictions(model, test_loader, device)
        y_true, y_pred, probs = yt, pr, pb
        report = classification_report_with_roc_proba(
            yt, pr, pb, class_names, split="test",
        )
        report["quality_rmse"] = None
        report["quality_mae"] = None
        report["quality_r2"] = None
    else:
        report = {
            "split": "test",
            "class_names": class_names,
            "accuracy": float("nan"),
            "note": "empty_test_loader",
            "quality_rmse": None,
            "quality_mae": None,
            "quality_r2": None,
        }

    out_dir = args.output_dir / name
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"model": best_state, "name": name, "args": vars(args)}, out_dir / "best.pt")
    (out_dir / "history.json").write_text(json.dumps(history, indent=2))
    (out_dir / "metrics.json").write_text(json.dumps({
        "best_val_acc": best_acc, "test_acc": test_acc, "test_f1": test_f1,
    }, indent=2))
    (out_dir / "test_classification_metrics.json").write_text(json.dumps(report, indent=2))
    if len(y_true) and not args.no_save_test_probs:
        np.savez(
            out_dir / "test_probs.npz",
            y_true=y_true.astype(np.int64),
            y_pred=y_pred.astype(np.int64),
            y_proba=probs.astype(np.float32),
            class_names=np.array(class_names, dtype=object),
        )
    if len(y_true) and "confusion_matrix" in report:
        cm_txt = format_confusion_matrix_text(report["confusion_matrix"], class_names)
        (out_dir / "test_confusion_matrix.txt").write_text(cm_txt + "\n", encoding="utf-8")
        if not args.no_eval_plots:
            save_test_eval_plots(out_dir, report, class_names)

    print(f"[{name}] best_val_acc={best_acc:.4f}  test_acc={test_acc:.4f}  test_f1={test_f1:.4f}")
    if len(y_true):
        roc_m = report.get("roc_auc_ovr_macro")
        if roc_m is not None and roc_m == roc_m:
            print(f"[{name}] wrote {out_dir / 'test_classification_metrics.json'}  "
                  f"roc_auc_ovr_macro={roc_m:.4f}")
    return {"name": name, "best_val_acc": best_acc, "test_acc": test_acc, "test_f1": test_f1}


def _unpack(batch, device):
    if isinstance(batch, (list, tuple)):
        x, y = batch[0], batch[1]
    else:
        x, y = batch["x"], batch["y"]
    return x.to(device), y.to(device)


@torch.no_grad()
def collect_test_predictions(
    model: nn.Module, loader: DataLoader, device: torch.device,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """All test windows: true label, predicted label, row-wise softmax probabilities."""
    if loader is None or len(loader.dataset) == 0:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64), np.zeros((0, 1))
    model.eval()
    yt: List[int] = []; pr: List[int] = []; pb: List[np.ndarray] = []
    for batch in loader:
        x, y = _unpack(batch, device)
        logits = _logits(model(x))
        probs = torch.softmax(logits, dim=1).cpu().numpy()
        pred = probs.argmax(axis=1)
        yt.extend(int(t) for t in y.cpu().numpy().tolist())
        pr.extend(int(p) for p in pred.tolist())
        pb.append(probs)
    return np.asarray(yt, dtype=np.int64), np.asarray(pr, dtype=np.int64), np.vstack(pb)


def save_test_eval_plots(
    out_dir: Path,
    report: Dict[str, Any],
    class_names: List[str],
) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt  # noqa: WPS433
    except ImportError:
        print("[warn] matplotlib not installed — skip test_confusion_matrix.png / test_roc_ovr.png")
        return

    cm = report.get("confusion_matrix")
    if cm:
        arr = np.asarray(cm, dtype=np.float64)
        n = len(class_names)
        fig, ax = plt.subplots(figsize=(max(6.5, n * 0.85), max(5.2, n * 0.7)))
        im = ax.imshow(arr, interpolation="nearest")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_xticks(np.arange(n))
        ax.set_yticks(np.arange(n))
        short = [(c[:14] + "\u2026") if len(c) > 15 else c for c in class_names]
        ax.set_xticklabels(short, rotation=42, ha="right", fontsize=8)
        ax.set_yticklabels(short, fontsize=8)
        ax.set_xlabel("predicted"); ax.set_ylabel("true")
        ax.set_title("Test confusion matrix (rows=true, cols=predicted)")
        plt.tight_layout()
        plt.savefig(out_dir / "test_confusion_matrix.png", dpi=168)
        plt.close()

    roc_data = report.get("roc_ovr_curves") or {}
    if not roc_data:
        return

    fig, ax = plt.subplots(figsize=(8.2, 6.2))
    for name in class_names:
        curve = roc_data.get(name, {})
        fpr, tpr = curve.get("fpr"), curve.get("tpr")
        if fpr and tpr:
            lab = name if len(name) <= 26 else name[:23] + "\u2026"
            pauc = report.get("roc_auc_ovr_per_class", {}).get(name)
            if pauc is None or (isinstance(pauc, float) and pauc != pauc):
                auc_piece = "n/a"
            else:
                auc_piece = f"{float(pauc):.3f}"
            ax.plot(fpr, tpr, lw=2, label=f"{lab} (AUROC={auc_piece})")
    ax.plot([0.0, 1.0], [0.0, 1.0], linestyle="--", color="gray", linewidth=1, label="random")
    ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("False positive rate (OvR)")
    ax.set_ylabel("True positive rate (OvR)")
    rm = report.get("roc_auc_ovr_macro")
    auc_s = "n/a" if rm is None or rm != rm else f"{rm:.4f}"
    ax.set_title(f"Test ROC one-vs-rest  macro AUROC={auc_s}")
    ax.legend(fontsize=7, loc="lower right")
    plt.tight_layout()
    plt.savefig(out_dir / "test_roc_ovr.png", dpi=168)
    plt.close()


def _logits(out):
    if isinstance(out, tuple):
        return out[0]
    return out


def _evaluate_with_ema(
    model: nn.Module,
    ema: Optional["ModelEMA"],
    loader: DataLoader,
    device: torch.device,
) -> Tuple[float, float]:
    """Evaluate with EMA weights if available, otherwise the live model.

    Swaps EMA shadow weights into the model for the eval pass, then restores
    the live training weights so the next epoch resumes normally.
    """
    if ema is None:
        return _evaluate(model, loader, device)
    backup = {k: v.detach().clone() for k, v in model.state_dict().items()
              if v.dtype.is_floating_point}
    try:
        merged = {**model.state_dict(), **ema.state_dict()}
        model.load_state_dict(merged, strict=False)
        return _evaluate(model, loader, device)
    finally:
        full = {**model.state_dict(), **backup}
        model.load_state_dict(full, strict=False)


@torch.no_grad()
def _evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> Tuple[float, float]:
    if loader is None or len(loader.dataset) == 0:
        return float("nan"), float("nan")
    model.eval()
    y_true, y_pred = [], []
    for batch in loader:
        x, y = _unpack(batch, device)
        logits = _logits(model(x))
        y_true.extend(y.cpu().numpy().tolist())
        y_pred.extend(logits.argmax(1).cpu().numpy().tolist())
    y_true = np.asarray(y_true); y_pred = np.asarray(y_pred)
    acc = float((y_true == y_pred).mean())
    try:
        from sklearn.metrics import f1_score
        f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    except Exception:
        f1 = float("nan")
    return acc, f1


# ──────────────────────────────────────────────────────── driver


def main() -> int:
    args = build_parser().parse_args()
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    print(f"[setup] device={device} models={list(args.models)}")

    train_ds, val_ds, test_ds, class_map = build_datasets(args)
    print(f"[setup] classes={class_map}  train={len(train_ds)} val={len(val_ds) if val_ds else 0} test={len(test_ds) if test_ds else 0}")
    class_names_ordered = [name for name, _ in sorted(class_map.items(), key=lambda kv: kv[1])]

    def _loader(ds, shuffle):
        if ds is None or len(ds) == 0: return None
        return DataLoader(ds, batch_size=args.batch_size, shuffle=shuffle,
                          drop_last=False, num_workers=args.num_workers)
    train_loader = _loader(train_ds, True)
    val_loader   = _loader(val_ds,   False)
    test_loader  = _loader(test_ds,  False)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    runs: List[Dict[str, Any]] = []

    if "bilstm" in args.models:
        model = ExerciseBiLSTMCNN(
            input_dim=args.feature_dim, num_classes=args.num_classes,
            bilstm_hidden=args.bilstm_hidden, seq_len=args.seq_len,
            dropout=args.dropout,
        ).to(device)
        print(f"[bilstm] params={sum(p.numel() for p in model.parameters()):,}")
        runs.append(train_one(
            model, "bilstm_cnn", train_loader, val_loader, test_loader, class_names_ordered, args, device,
        ))

    if "xlstm" in args.models:
        model = xLSTMExerciseClassifier(
            input_size=args.feature_dim, hidden_size=args.xlstm_hidden,
            num_classes=args.num_classes, dropout=args.dropout,
            num_heads=args.xlstm_num_heads, conv_kernel_size=args.xlstm_conv_kernel_size,
            projection_factor=args.xlstm_projection_factor,
            block_pattern=args.xlstm_block_pattern,
            num_error_tags=0,
        ).to(device)
        print(f"[xlstm ] params={sum(p.numel() for p in model.parameters()):,}")
        runs.append(train_one(
            model, "xlstm_7_1", train_loader, val_loader, test_loader, class_names_ordered, args, device,
        ))

    summary = {"runs": runs, "class_map": class_map, "args": {k: str(v) for k, v in vars(args).items()}}
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    print("\n==== final ====")
    print(f"{'model':<14s} {'val_acc':>10s} {'test_acc':>10s} {'test_f1':>10s}")
    for r in runs:
        print(f"{r['name']:<14s} {r['best_val_acc']:>10.4f} {r['test_acc']:>10.4f} {r['test_f1']:>10.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
