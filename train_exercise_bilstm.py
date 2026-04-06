#!/usr/bin/env python3
"""
Train a BiLSTM on 30-frame windows for:
  (1) exercise classification (which base exercise)
  (2) regression head for quality score in [0, 1]

Aligned with Riccio (arXiv:2411.11548): BiLSTM on sequences, optional mixed angles+coords,
standardized features (§3.3.1), 30-frame windows. Use --preset riccio for paper Table 4 BiLSTM hyperparameters.

Requires:
  - Index CSV from build_exercise_training_index.py or setup_local_training_data.py
  - angles: per-video *_biomechanics.npz under --angles-dir, or mixed: *_keypoints.npz under --keypoints-dir

Example:
  ./venv/bin/python train_exercise_bilstm.py --epochs 20 --preset riccio --standardize
  ./venv/bin/python train_exercise_bilstm.py --feature-mode mixed --keypoints-dir results/processed_keypoints_mediapipe

  # Riccio Kaggle dataset (after kaggle_exercise_recognition_pipeline.py --download --riccio):
  ./venv/bin/python train_exercise_bilstm.py --preset riccio --standardize --eval-test \\
    --kaggle-angles-dir results/riccio_realtime_exercise_recognition \\
    --kaggle-stem riccio_realtime_exercise_recognition

  ./venv/bin/python train_exercise_bilstm.py --kaggle-angles-dir results/kaggle_exercise_recognition --standardize --eval-test
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

from exercise_bilstm_dataset import (
    ExerciseAngleWindowDataset,
    build_kaggle_angle_datasets,
    fit_standardizer_from_dataset,
    load_index_rows,
)
from exercise_bilstm_model import ExerciseBiLSTM


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
    ap = argparse.ArgumentParser(description="Train BiLSTM exercise classifier + quality regressor")
    ap.add_argument("--index-csv", default="./results/exercise_training_index.csv")
    ap.add_argument("--angles-dir", default="./results/exercise_angles")
    ap.add_argument(
        "--keypoints-dir",
        default="./results/processed_keypoints_mediapipe",
        help="For --feature-mode mixed: folder with {stem}_keypoints.npz",
    )
    ap.add_argument(
        "--feature-mode",
        choices=("angles", "coords", "mixed"),
        default="angles",
        help="angles=(T,8) biomechanics; coords=(T,34) normalized xy only; mixed=(T,42) angles+xy",
    )
    ap.add_argument(
        "--standardize",
        action="store_true",
        help="Fit StandardScaler on training windows only, apply train+val (paper §3.3.1)",
    )
    ap.add_argument(
        "--preset",
        choices=("none", "riccio"),
        default="none",
        help="riccio: Table 4 BiLSTM — units 73, dropout ~0.22, lr 4e-4, batch 54",
    )
    ap.add_argument("--output-dir", default="./results/exercise_bilstm")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--dropout", type=float, default=0.35)
    ap.add_argument("--window", type=int, default=30)
    ap.add_argument("--stride", type=int, default=15)
    ap.add_argument("--cls-weight", type=float, default=1.0)
    ap.add_argument("--reg-weight", type=float, default=0.5)
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument(
        "--kaggle-angles-dir",
        default="",
        help="If set, train from Kaggle pipeline NPZs ({stem}_biomechanics.npz + {stem}_labels.npz) "
        "instead of --index-csv; windows are labeled by coarse exercise (phase suffix stripped).",
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
        "--eval-test",
        action="store_true",
        help="After training, load best checkpoint and evaluate on held-out test windows (Kaggle mode).",
    )
    args = ap.parse_args()

    if args.preset == "riccio":
        args.batch_size = 54
        args.lr = 0.0004
        args.hidden = 73
        args.dropout = 0.2174

    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")

    kaggle_dir = (args.kaggle_angles_dir or "").strip()
    scale_mean = scale_std = None
    test_ds = None
    rows: list = []

    if kaggle_dir:
        kpath = Path(kaggle_dir)
        if not kpath.is_dir():
            print(f"Not a directory: {kpath}", file=sys.stderr)
            return 1
        if args.feature_mode != "angles":
            print(
                "Kaggle mode uses precomputed biomechanics angles only; using feature-mode=angles.",
                file=sys.stderr,
            )
            args.feature_mode = "angles"
        try:
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
                )
            )
        except (FileNotFoundError, ValueError, RuntimeError) as e:
            print(str(e), file=sys.stderr)
            return 1
        classes = [idx_to_class[i] for i in range(len(class_to_idx))]
    else:
        index_path = Path(args.index_csv)
        angles_dir = Path(args.angles_dir)
        if not index_path.is_file():
            print(f"Missing {index_path} — run build_exercise_training_index.py first", file=sys.stderr)
            return 1

        rows = load_index_rows(index_path)
        train_rows = [r for r in rows if r.get("split") == "train"]
        val_rows = [r for r in rows if r.get("split") == "val"]
        if not val_rows:
            val_rows = [r for r in rows if r.get("split") == "test"]
        if not train_rows:
            print("No train split in index", file=sys.stderr)
            return 1

        classes = sorted({r["exercise_class"] for r in train_rows})
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

        if args.standardize and len(train_ds) > 0:
            scale_mean, scale_std = fit_standardizer_from_dataset(train_ds)
            train_ds.apply_standardizer(scale_mean, scale_std)
            val_ds.apply_standardizer(scale_mean, scale_std)

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

    feat_dim = train_ds.samples[0][0].shape[1] if train_ds.samples else 8
    model = ExerciseBiLSTM(
        input_dim=feat_dim,
        num_classes=len(classes),
        hidden=args.hidden,
        num_layers=args.layers,
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
                    "window": args.window,
                    "stride": args.stride,
                    "feat_dim": feat_dim,
                    "num_classes": len(classes),
                    "classes": classes,
                    "feature_mode": args.feature_mode,
                    "scale_mean": scale_mean,
                    "scale_std": scale_std,
                    "hidden": args.hidden,
                    "layers": args.layers,
                    "dropout": args.dropout,
                },
                out_dir / "exercise_bilstm_best.pt",
            )
        print(
            f"epoch {epoch:03d}  train_loss={tr_loss:.4f}  "
            f"val_acc={va:.4f}  val_quality_rmse={vrmse:.4f}"
        )

    ckpt_path = out_dir / "exercise_bilstm_best.pt"
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
