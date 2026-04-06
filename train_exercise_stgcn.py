#!/usr/bin/env python3
"""
Train an ST-GCN (spatial–temporal graph CNN) on COCO-17 keypoint sequences in sliding windows:
  (1) exercise classification
  (2) quality regression in [0, 1]

Uses normalized skeleton graph convolution + temporal conv (see exercise_stgcn_model.py).

Example (Kaggle CSV pipeline NPZs):
  ./venv/bin/python train_exercise_stgcn.py \\
    --kaggle-keypoints-dir results/kaggle_exercise_recognition --standardize --eval-test

Riccio video dataset (needs *_keypoints.npz from riccio_kaggle_video_pipeline — same stem as BiLSTM):
  ./venv/bin/python train_exercise_stgcn.py --standardize --eval-test \\
    --kaggle-keypoints-dir results/riccio_realtime_exercise_recognition \\
    --kaggle-stem riccio_realtime_exercise_recognition
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from exercise_stgcn_dataset import (
    ExerciseKeypointIndexDataset,
    apply_stgcn_standardizer,
    build_kaggle_stgcn_datasets,
    fit_stgcn_standardizer,
    load_index_rows,
)
from exercise_stgcn_model import ExerciseSTGCN


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
    num_classes: int,
):
    model.eval()
    ce = nn.CrossEntropyLoss()
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
    ap = argparse.ArgumentParser(description="Train ST-GCN exercise classifier + quality regressor")
    ap.add_argument("--index-csv", default="./results/exercise_training_index.csv")
    ap.add_argument(
        "--keypoints-dir",
        default="./results/processed_keypoints_mediapipe",
        help="Folder with {stem}_keypoints.npz (COCO-17, T,17,2)",
    )
    ap.add_argument("--standardize", action="store_true")
    ap.add_argument("--output-dir", default="./results/exercise_stgcn")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--base-channels", type=int, default=64)
    ap.add_argument("--dropout", type=float, default=0.25)
    ap.add_argument("--window", type=int, default=30)
    ap.add_argument("--stride", type=int, default=15)
    ap.add_argument("--cls-weight", type=float, default=1.0)
    ap.add_argument("--reg-weight", type=float, default=0.5)
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument(
        "--kaggle-keypoints-dir",
        default="",
        help="If set, train from {stem}_keypoints.npz + {stem}_labels.npz (coarse exercise labels).",
    )
    ap.add_argument("--kaggle-stem", default="kaggle_exercise_recognition")
    ap.add_argument("--kaggle-test-ratio", type=float, default=0.15)
    ap.add_argument("--kaggle-val-ratio", type=float, default=0.15)
    ap.add_argument("--kaggle-seed", type=int, default=42)
    ap.add_argument("--eval-test", action="store_true")
    args = ap.parse_args()

    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")

    kaggle_dir = (args.kaggle_keypoints_dir or "").strip()
    scale_mean = scale_std = None
    test_ds = None
    rows: list = []

    if kaggle_dir:
        kpath = Path(kaggle_dir)
        if not kpath.is_dir():
            print(f"Not a directory: {kpath}", file=sys.stderr)
            return 1
        try:
            train_ds, val_ds, test_ds, class_to_idx, idx_to_class, scale_mean, scale_std = (
                build_kaggle_stgcn_datasets(
                    kpath,
                    stem=args.kaggle_stem,
                    window=args.window,
                    stride=args.stride,
                    test_ratio=args.kaggle_test_ratio,
                    val_ratio=args.kaggle_val_ratio,
                    seed=args.kaggle_seed,
                    standardize=args.standardize,
                )
            )
        except (FileNotFoundError, ValueError, RuntimeError) as e:
            print(str(e), file=sys.stderr)
            return 1
        classes = [idx_to_class[i] for i in range(len(class_to_idx))]
    else:
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
        class_to_idx = {c: i for i, c in enumerate(classes)}
        idx_to_class = {i: c for c, i in class_to_idx.items()}

        train_ds = ExerciseKeypointIndexDataset(
            index_path,
            kp_dir,
            class_to_idx,
            split="train",
            window=args.window,
            stride=args.stride,
        )
        val_split = "val" if any(r.get("split") == "val" for r in rows) else (
            "test" if any(r.get("split") == "test" for r in rows) else "train"
        )
        val_ds = ExerciseKeypointIndexDataset(
            index_path,
            kp_dir,
            class_to_idx,
            split=val_split,
            window=args.window,
            stride=args.stride,
        )

        if args.standardize and len(train_ds) > 0:
            scale_mean, scale_std = fit_stgcn_standardizer(train_ds.samples)
            apply_stgcn_standardizer(train_ds, scale_mean, scale_std)
            apply_stgcn_standardizer(val_ds, scale_mean, scale_std)

    if len(train_ds) == 0:
        if kaggle_dir:
            print(
                "No training windows from Kaggle keypoints — check files and window/stride.",
                file=sys.stderr,
            )
        else:
            print(
                f"No training windows — need *_keypoints.npz under {args.keypoints_dir} matching index stems.",
                file=sys.stderr,
            )
        return 1

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, drop_last=False
    )
    val_loader = (
        DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
        if len(val_ds) > 0
        else None
    )
    test_loader = (
        DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)
        if test_ds is not None and len(test_ds) > 0
        else None
    )

    model = ExerciseSTGCN(
        num_classes=len(classes),
        num_joints=17,
        in_channels=2,
        base_channels=args.base_channels,
        dropout=args.dropout,
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "class_map.json", "w") as f:
        json.dump({"class_to_idx": class_to_idx, "idx_to_class": idx_to_class}, f, indent=2)

    best_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        tr_loss = train_one_epoch(
            model,
            train_loader,
            opt,
            args.cls_weight,
            args.reg_weight,
            device,
        )
        if val_loader is not None:
            va, vrmse = evaluate(model, val_loader, device, len(classes))
        else:
            va, vrmse = 0.0, 0.0
        should_save = (val_loader is not None and va > best_acc) or (
            val_loader is None and epoch == args.epochs
        )
        if should_save:
            if val_loader is not None:
                best_acc = va
            torch.save(
                {
                    "model": model.state_dict(),
                    "model_type": "stgcn",
                    "window": args.window,
                    "stride": args.stride,
                    "num_classes": len(classes),
                    "classes": classes,
                    "num_joints": 17,
                    "in_channels": 2,
                    "base_channels": args.base_channels,
                    "dropout": args.dropout,
                    "scale_mean": scale_mean,
                    "scale_std": scale_std,
                },
                out_dir / "exercise_stgcn_best.pt",
            )
        print(
            f"epoch {epoch:03d}  train_loss={tr_loss:.4f}  "
            f"val_acc={va:.4f}  val_quality_rmse={vrmse:.4f}"
        )

    ckpt_path = out_dir / "exercise_stgcn_best.pt"
    print(f"Best val acc ≈ {best_acc:.4f}; checkpoint: {ckpt_path}")

    if args.eval_test and kaggle_dir and test_loader is not None and ckpt_path.is_file():
        try:
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        except TypeError:
            ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model"])
        ta, trmse = evaluate(model, test_loader, device, len(classes))
        print(f"Test acc={ta:.4f}  test_quality_rmse={trmse:.4f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
