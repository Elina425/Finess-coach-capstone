#!/usr/bin/env python3
"""
Train a BiLSTM on **annotation-only** sequences (EgoExo interpretable judgements).

No frames or pose NPZs: uses ``verification_json`` + ``comment`` / ``action_guidance`` from
``build_egoexo_fitness_index.py``. Dual heads: exercise class + quality regression.

Example:
  ./venv/bin/python build_egoexo_fitness_index.py ... --output results/egoexo_fitness_index.csv
  ./venv/bin/python split_exercise_index.py --input results/egoexo_fitness_index.csv \\
    --output results/egoexo_fitness_index_split.csv
  ./venv/bin/python train_exercise_annotation_bilstm.py \\
    --index-csv results/egoexo_fitness_index_split.csv \\
    --output-dir results/exercise_annotation_bilstm_egoexo
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from fitness_coach.datasets.exercise_annotation_bilstm_dataset import (
    STEP_DIM,
    ExerciseAnnotationSequenceDataset,
    annotation_collate_fn,
    apply_annotation_standardizer,
    fit_annotation_standardizer,
)
from fitness_coach.datasets.exercise_bilstm_dataset import load_index_rows
from fitness_coach.models.exercise_annotation_bilstm import ExerciseAnnotationBiLSTM


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    opt: torch.optim.Optimizer,
    cls_weight: float,
    reg_weight: float,
    device: torch.device,
    ce_class_weights: Optional[torch.Tensor] = None,
) -> float:
    model.train()
    total = 0.0
    n = 0
    w = ce_class_weights.to(device) if ce_class_weights is not None else None
    ce = nn.CrossEntropyLoss(weight=w)
    mse = nn.MSELoss()
    for xb, lens, y_cls, y_q in loader:
        xb = xb.to(device)
        y_cls = y_cls.to(device)
        y_q = y_q.to(device)
        opt.zero_grad()
        logits, pred_q = model(xb, lens)
        loss = cls_weight * ce(logits, y_cls) + reg_weight * mse(pred_q, y_q)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
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
):
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
    for xb, lens, y_cls, y_q in loader:
        xb = xb.to(device)
        y_cls = y_cls.to(device)
        y_q = y_q.to(device)
        logits, pred_q = model(xb, lens)
        pred = logits.argmax(dim=1)
        if detailed:
            ys_list.extend(y_cls.cpu().numpy().tolist())
            pred_list.extend(pred.cpu().numpy().tolist())
        correct += int((pred == y_cls).sum().item())
        tot += xb.size(0)
        reg_err += float(mse(pred_q, y_q).item()) * xb.size(0)
        abs_err += float(torch.abs(pred_q - y_q).sum().item())
        q_true.extend(y_q.detach().cpu().numpy().astype(np.float64).tolist())
        q_pred.extend(pred_q.detach().cpu().numpy().astype(np.float64).tolist())
    acc = correct / max(tot, 1)
    rmse = (reg_err / max(tot, 1)) ** 0.5
    mae = abs_err / max(tot, 1)
    r2 = float("nan")
    if tot > 0 and len(q_true) > 1:
        try:
            from sklearn.metrics import r2_score

            r2 = float(r2_score(np.asarray(q_true), np.asarray(q_pred)))
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
        return acc, rmse, reg_metrics, metrics
    return acc, rmse, reg_metrics


def compute_inverse_frequency_class_weights(
    samples: List[Tuple[Any, int, float]], num_classes: int
) -> torch.Tensor:
    ys = [int(s[1]) for s in samples]
    cnt = Counter(ys)
    n = len(ys)
    raw: List[float] = []
    for c in range(num_classes):
        nc = int(cnt.get(c, 0))
        raw.append(1.0 if nc <= 0 else n / (num_classes * nc))
    w = torch.tensor(raw, dtype=torch.float32)
    w = w / float(w.mean().clamp_min(1e-8))
    return w


def main() -> int:
    ap = argparse.ArgumentParser(description="BiLSTM on EgoExo annotation sequences (no pose)")
    ap.add_argument("--index-csv", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, default=Path("results/exercise_annotation_bilstm"))
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--dropout", type=float, default=0.35)
    ap.add_argument("--max-steps", type=int, default=96, help="Max sequence length (truncate tail)")
    ap.add_argument("--cls-weight", type=float, default=1.0)
    ap.add_argument("--reg-weight", type=float, default=0.5)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--standardize", action="store_true")
    ap.add_argument("--balanced-class-weights", action="store_true")
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument(
        "--eval-test",
        action="store_true",
        help="Evaluate on split=test when val and test rows exist",
    )
    args = ap.parse_args()

    if not args.index_csv.is_file():
        print(f"Missing {args.index_csv}", file=sys.stderr)
        return 1

    rows = load_index_rows(args.index_csv)
    train_rows = [r for r in rows if r.get("split") == "train"]
    if not train_rows:
        print("No train split in index", file=sys.stderr)
        return 1
    classes = sorted({r["exercise_class"] for r in train_rows})
    class_to_idx = {c: i for i, c in enumerate(classes)}
    idx_to_class = {i: c for c, i in class_to_idx.items()}

    train_ds = ExerciseAnnotationSequenceDataset(
        args.index_csv,
        class_to_idx,
        split="train",
        max_steps=args.max_steps,
    )
    val_split = "val" if any(r.get("split") == "val" for r in rows) else (
        "test" if any(r.get("split") == "test" for r in rows) else "train"
    )
    val_ds = ExerciseAnnotationSequenceDataset(
        args.index_csv,
        class_to_idx,
        split=val_split,
        max_steps=args.max_steps,
    )
    test_ds = None
    if val_split == "val" and any(r.get("split") == "test" for r in rows):
        test_ds = ExerciseAnnotationSequenceDataset(
            args.index_csv,
            class_to_idx,
            split="test",
            max_steps=args.max_steps,
        )

    if len(train_ds) == 0:
        print(
            "No training samples — rebuild index with current build_egoexo_fitness_index.py "
            "(needs verification_json for best results; comment/guidance still give one step).",
            file=sys.stderr,
        )
        return 1

    if args.standardize:
        mean, std = fit_annotation_standardizer(train_ds.samples)
        apply_annotation_standardizer(train_ds.samples, mean, std)
        apply_annotation_standardizer(val_ds.samples, mean, std)
        if test_ds is not None:
            apply_annotation_standardizer(test_ds.samples, mean, std)
    else:
        mean = std = None

    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=annotation_collate_fn,
    )
    val_loader = (
        DataLoader(
            val_ds,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=annotation_collate_fn,
        )
        if len(val_ds) > 0
        else None
    )
    test_loader = (
        DataLoader(
            test_ds,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=annotation_collate_fn,
        )
        if test_ds is not None and len(test_ds) > 0
        else None
    )

    model = ExerciseAnnotationBiLSTM(
        input_dim=STEP_DIM,
        num_classes=len(classes),
        hidden=args.hidden,
        num_layers=args.layers,
        dropout=args.dropout,
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    ce_w = None
    if args.balanced_class_weights:
        ce_w = compute_inverse_frequency_class_weights(train_ds.samples, len(classes)).to(device)
        print(
            "Balanced CE weights: "
            + ", ".join(f"{classes[i]}={ce_w[i].item():.3f}" for i in range(len(classes)))
        )

    best_acc = 0.0
    best_state = None
    for epoch in range(1, args.epochs + 1):
        tr = train_one_epoch(
            model,
            train_loader,
            opt,
            args.cls_weight,
            args.reg_weight,
            device,
            ce_class_weights=ce_w,
        )
        if val_loader is not None:
            va, vrmse, vreg = evaluate(model, val_loader, device, len(classes))
        else:
            va, vrmse, vreg = 0.0, 0.0, {"rmse": 0.0, "mae": 0.0, "r2": float("nan")}
        save_now = (val_loader is not None and va > best_acc) or (
            val_loader is None and epoch == args.epochs
        )
        if save_now:
            if val_loader is not None:
                best_acc = va
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        print(
            f"epoch {epoch:03d}  train_loss={tr:.4f}  val_acc={va:.4f}  "
            f"val_q_rmse={vrmse:.4f}  val_q_mae={vreg['mae']:.4f}  val_q_r2={vreg['r2']:.4f}"
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    if best_state is None:
        best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    ckpt = {
        "model": best_state,
        "model_type": "annotation_bilstm",
        "feat_dim": STEP_DIM,
        "num_classes": len(classes),
        "classes": classes,
        "hidden": args.hidden,
        "layers": args.layers,
        "dropout": args.dropout,
        "max_steps": args.max_steps,
        "scale_mean": mean.tolist() if mean is not None else None,
        "scale_std": std.tolist() if std is not None else None,
    }
    torch.save(ckpt, args.output_dir / "exercise_annotation_bilstm_best.pt")
    with open(args.output_dir / "class_map.json", "w") as f:
        json.dump({"class_to_idx": class_to_idx, "idx_to_class": idx_to_class}, f, indent=2)
    print(f"Wrote {args.output_dir / 'exercise_annotation_bilstm_best.pt'}")

    if args.eval_test and test_loader is not None:
        model.load_state_dict(best_state)
        ta, trmse, treg, tcls = evaluate(
            model,
            test_loader,
            device,
            len(classes),
            detailed=True,
            class_names=classes,
        )
        print(
            f"Test acc={ta:.4f}  quality RMSE={trmse:.4f}  MAE={treg['mae']:.4f}  R²={treg['r2']:.4f}"
        )
        metrics_path = args.output_dir / "test_classification_metrics.json"
        with open(metrics_path, "w") as f:
            json.dump(
                {
                    "split": "test",
                    "accuracy": tcls["accuracy"],
                    "f1_macro": tcls["f1_macro"],
                    "f1_per_class": tcls["f1_per_class"],
                    "quality_rmse": treg["rmse"],
                    "quality_mae": treg["mae"],
                    "quality_r2": treg["r2"],
                },
                f,
                indent=2,
            )
        print(f"Wrote {metrics_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
