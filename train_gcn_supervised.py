#!/usr/bin/env python3
"""
Supervised GCN sequence model: same exercise classification + quality regression as BiLSTM
for fair comparison (per-frame topology-aware GCN, mean-pool over 30 frames).

Requires *_keypoints.npz per indexed stem. Trains with CE + MSE like train_exercise_bilstm.py.

Example:
  ./venv/bin/python train_gcn_supervised.py \\
    --index-csv results/exercise_training_index_long_range.csv \\
    --keypoints-dir results/processed_keypoints_mediapipe \\
    --epochs 30
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from exercise_keypoint_window_dataset import ExerciseKeypointWindowDataset
from exercise_bilstm_dataset import load_index_rows
from gcn_pose_model import GCNSequenceExerciseNet


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    opt: torch.optim.Optimizer,
    cls_weight: float,
    reg_weight: float,
    device: torch.device,
):
    model.train()
    total = 0.0
    n = 0
    ce = nn.CrossEntropyLoss()
    mse = nn.MSELoss()
    for xb, y_cls, y_q in loader:
        xb = xb.to(device)
        y_cls = y_cls.to(device)
        y_q = y_q.to(device)
        opt.zero_grad()
        logits, pred_q = model(xb)
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
):
    model.eval()
    mse = nn.MSELoss()
    correct = 0
    tot = 0
    reg_err = 0.0
    for xb, y_cls, y_q in loader:
        xb = xb.to(device)
        y_cls = y_cls.to(device)
        y_q = y_q.to(device)
        logits, pred_q = model(xb)
        pred = logits.argmax(dim=1)
        correct += int((pred == y_cls).sum().item())
        tot += xb.size(0)
        reg_err += float(mse(pred_q, y_q).item()) * xb.size(0)
    acc = correct / max(tot, 1)
    rmse = (reg_err / max(tot, 1)) ** 0.5
    return acc, rmse


def main() -> int:
    ap = argparse.ArgumentParser(description="Train supervised GCN exercise + quality model")
    ap.add_argument("--index-csv", default="./results/exercise_training_index.csv")
    ap.add_argument("--keypoints-dir", default="./results/processed_keypoints_mediapipe")
    ap.add_argument("--output-dir", default="./results/gcn_supervised")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--gcn-hidden", type=int, default=64)
    ap.add_argument("--embed-dim", type=int, default=50)
    ap.add_argument("--window", type=int, default=30)
    ap.add_argument("--stride", type=int, default=15)
    ap.add_argument("--cls-weight", type=float, default=1.0)
    ap.add_argument("--reg-weight", type=float, default=0.5)
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()

    index_path = Path(args.index_csv)
    kp_dir = Path(args.keypoints_dir)
    if not index_path.is_file():
        print(f"Missing {index_path}", file=sys.stderr)
        return 1

    rows = load_index_rows(index_path)
    train_rows = [r for r in rows if r.get("split") == "train"]
    if not train_rows:
        print("No train split in index", file=sys.stderr)
        return 1

    classes = sorted({r["exercise_class"] for r in train_rows})
    class_to_idx: Dict[str, int] = {c: i for i, c in enumerate(classes)}
    idx_to_class: Dict[int, str] = {i: c for c, i in class_to_idx.items()}

    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")

    train_ds = ExerciseKeypointWindowDataset(
        index_path, kp_dir, class_to_idx, "train", args.window, args.stride
    )
    val_split = "val" if any(r.get("split") == "val" for r in rows) else (
        "test" if any(r.get("split") == "test" for r in rows) else "train"
    )
    val_ds = ExerciseKeypointWindowDataset(
        index_path, kp_dir, class_to_idx, val_split, args.window, args.stride
    )

    if len(train_ds) == 0:
        print(
            "No training windows — need *_keypoints.npz for indexed stems.",
            file=sys.stderr,
        )
        return 1

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = (
        DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
        if len(val_ds) > 0
        else None
    )

    model = GCNSequenceExerciseNet(
        num_classes=len(classes),
        gcn_hidden=args.gcn_hidden,
        embed_dim=args.embed_dim,
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "class_map.json", "w") as f:
        json.dump({"class_to_idx": class_to_idx, "idx_to_class": idx_to_class}, f, indent=2)

    best_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        tr = train_one_epoch(
            model, train_loader, opt, args.cls_weight, args.reg_weight, device
        )
        if val_loader is not None:
            va, vrmse = evaluate(model, val_loader, device)
        else:
            va, vrmse = 0.0, 0.0
        if va > best_acc:
            best_acc = va
            torch.save(
                {
                    "model": model.state_dict(),
                    "model_type": "gcn_supervised",
                    "window": args.window,
                    "stride": args.stride,
                    "num_classes": len(classes),
                    "classes": classes,
                    "gcn_hidden": args.gcn_hidden,
                    "embed_dim": args.embed_dim,
                },
                out_dir / "gcn_supervised_best.pt",
            )
        print(
            f"epoch {epoch:03d}  train_loss={tr:.4f}  val_acc={va:.4f}  val_quality_rmse={vrmse:.4f}"
        )

    print(f"Best val acc ≈ {best_acc:.4f}; checkpoint: {out_dir / 'gcn_supervised_best.pt'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
