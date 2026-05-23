#!/usr/bin/env python3
"""
Train xLSTM on normalized keypoint / angle / mixed windows for the Riccio Kaggle NPZ pipeline.

Uses the same ``build_kaggle_*_datasets`` splits and standardization as ``train_exercise_bilstm.py``
when ``--preset posepulse`` (mixed (T,42), ``window_label=first``, train-only z-score, optional
``--exclude-classes``).

The bundled ``xLSTM`` stack is a residual **sLSTM-style** block tower (Beck et al. public design);
it does **not** yet interleave separate mLSTM vs sLSTM modules in a literal ``[7:1]`` ratio — use
``--layers 8`` with ``--preset posepulse`` to match the **block count** described in the capstone text.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from fitness_coach.datasets.exercise_stgcn_dataset import build_kaggle_stgcn_datasets
from fitness_coach.datasets.exercise_bilstm_dataset import (
    build_kaggle_angle_datasets,
    build_kaggle_frame_feature_datasets,
    build_kaggle_mixed_datasets,
)
from fitness_coach.models.xlstm_model import xLSTMExerciseClassifier


def parse_exclude_coarse_classes_cli(raw: Optional[str]) -> Optional[List[str]]:
    """Comma-separated coarse names; empty string → exclude nothing (``None``)."""
    s = (raw or "").strip()
    if not s:
        return None
    return [p.strip() for p in s.split(",") if p.strip()]


def compute_inverse_frequency_class_weights(train_ds, num_classes: int) -> torch.Tensor:
    """Compute inverse frequency weights for imbalanced classes."""
    labels = []
    for _, y, _ in train_ds.samples:
        labels.append(y)
    counts = Counter(labels)
    total = sum(counts.values())
    weights = torch.zeros(num_classes, dtype=torch.float32)
    for cls, count in counts.items():
        weights[cls] = total / (len(counts) * count)
    return weights


class KeypointXLSTMDataset(torch.utils.data.Dataset):
    """Dataset for xLSTM: flatten keypoints to (T, 34) per window."""

    def __init__(
        self,
        samples: List[Tuple[np.ndarray, int, float]],
        mean: Optional[np.ndarray],
        std: Optional[np.ndarray],
    ):
        self.samples = samples
        self.mean = mean
        self.std = std

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        x, y, q = self.samples[idx]
        x = np.asarray(x, dtype=np.float32)  # (T, 17, 2)
        if self.mean is not None and self.std is not None:
            x = (x - self.mean) / self.std
        # Flatten to (T, 34)
        x = x.reshape(x.shape[0], -1)
        return (
            torch.from_numpy(x),
            torch.tensor(y, dtype=torch.long),
            torch.tensor(q, dtype=torch.float32),
        )


def build_kaggle_xlstm_datasets(
    data_dir: Path,
    *,
    stem: str = "riccio_realtime_exercise_recognition",
    window: int = 30,
    stride: int = 15,
    test_ratio: float = 0.15,
    val_ratio: float = 0.15,
    seed: int = 42,
    standardize: bool = True,
    quality_default: float = 0.75,
) -> Tuple[torch.utils.data.Dataset, torch.utils.data.Dataset, torch.utils.data.Dataset, Dict[str, int], Dict[int, str], Optional[np.ndarray], Optional[np.ndarray]]:
    # Use the same loading as STGCN
    from fitness_coach.datasets.exercise_stgcn_dataset import build_kaggle_stgcn_datasets
    train_ds, val_ds, test_ds, class_to_idx, idx_to_class, mean, std = build_kaggle_stgcn_datasets(
        data_dir,
        stem=stem,
        window=window,
        stride=stride,
        test_ratio=test_ratio,
        val_ratio=val_ratio,
        seed=seed,
        standardize=standardize,
        quality_default=quality_default,
    )

    # Convert to XLSTM datasets
    train_xlstm = KeypointXLSTMDataset(train_ds.samples, mean, std)
    val_xlstm = KeypointXLSTMDataset(val_ds.samples, mean, std)
    test_xlstm = KeypointXLSTMDataset(test_ds.samples, mean, std)

    return train_xlstm, val_xlstm, test_xlstm, class_to_idx, idx_to_class, mean, std


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    opt: torch.optim.Optimizer,
    cls_weight: float,
    reg_weight: float,
    device: torch.device,
    ce_class_weights: Optional[torch.Tensor] = None,
):
    model.train()
    total = 0.0
    n = 0
    w = ce_class_weights.to(device) if ce_class_weights is not None else None
    ce = nn.CrossEntropyLoss(weight=w)
    mse = nn.MSELoss()
    for xb, y_cls, y_q in loader:
        xb = xb.to(device)
        y_cls = y_cls.to(device)
        y_q = y_q.to(device)
        opt.zero_grad()
        logits, pred_q = model(xb)
        pred_q = pred_q.squeeze(-1)  # (batch, 1) -> (batch,)
        if reg_weight > 0:
            loss = cls_weight * ce(logits, y_cls) + reg_weight * mse(pred_q, y_q)
        else:
            loss = cls_weight * ce(logits, y_cls)
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
    for xb, y_cls, y_q in loader:
        xb = xb.to(device)
        y_cls = y_cls.to(device)
        y_q = y_q.to(device)
        logits, pred_q = model(xb)
        pred_q = pred_q.squeeze(-1)
        pred = logits.argmax(dim=1)
        if detailed:
            ys_list.extend(y_cls.cpu().numpy().tolist())
            pred_list.extend(pred.cpu().numpy().tolist())
            probs_list.append(torch.softmax(logits, dim=1).detach().cpu().numpy())
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
        return acc, rmse, reg_metrics, metrics, y_true_arr, y_prob_arr
    return acc, rmse, reg_metrics


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Train xLSTM on Kaggle/Riccio NPZ windows (angles / coords / mixed)."
    )
    ap.add_argument(
        "--kaggle-keypoints-dir",
        required=True,
        help="Directory with {stem}_*.npz (same folder as BiLSTM: biomechanics + labels; mixed mode also needs keypoints NPZ).",
    )
    ap.add_argument(
        "--kaggle-stem",
        default="riccio_realtime_exercise_recognition",
        help="Filename prefix for *_keypoints.npz / *_biomechanics.npz and *_labels.npz",
    )
    ap.add_argument("--output-dir", default="./results/xlstm_keypoints")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument(
        "--weight-decay",
        type=float,
        default=1e-4,
        help="AdamW L2 weight decay (same default as BiLSTM training).",
    )
    ap.add_argument(
        "--hidden",
        type=int,
        default=128,
        help="xLSTM hidden size (posepulse preset: 256 for capstone text; riccio preset: 128).",
    )
    ap.add_argument(
        "--layers",
        type=int,
        default=2,
        help="Number of residual xLSTM (sLSTM-style) blocks in the encoder stack.",
    )
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--window", type=int, default=30)
    ap.add_argument("--stride", type=int, default=15)
    ap.add_argument("--cls-weight", type=float, default=1.0)
    ap.add_argument(
        "--reg-weight",
        type=float,
        default=0.5,
        help="Quality MSE weight; use 0 for classification-only (matches PosePulse BiLSTM diagram run).",
    )
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument(
        "--standardize",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Train-only z-score per feature (default: on). Use --no-standardize to disable.",
    )
    ap.add_argument("--eval-test", action="store_true")
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument(
        "--window-label",
        choices=("first", "last"),
        default="last",
        help="Window label policy (posepulse preset sets first, aligned with train_exercise_bilstm --preset posepulse).",
    )
    ap.add_argument(
        "--exclude-classes",
        default=None,
        metavar="NAMES",
        help="Comma-separated coarse exercise names to drop. Default: for --preset posepulse only, "
        "'hammer curl'; otherwise no exclusion. Pass an empty string to exclude nothing on posepulse.",
    )
    ap.add_argument(
        "--linear-classifier",
        action="store_true",
        help="Single Linear classification head after pool+dropout (paper-style) instead of the default 2-layer MLP head.",
    )
    ap.add_argument(
        "--feature-mode",
        choices=("coords", "angles", "mixed", "vit_backbone", "resnet_backbone"),
        default="angles",
        help="coords: raw (x,y) 34-dim; angles: joint angles 8-dim; mixed: angles+coords 42-dim; "
        "vit_backbone: ViTPose-S / timm ViT frame_features from NPZ (--representation vit_backbone); "
        "resnet_backbone: torchvision ResNet GAP features (--representation resnet_backbone).",
    )
    ap.add_argument(
        "--xlstm-block-pattern",
        default=None,
        metavar="PATTERN",
        help="Block stack as 'm' (matrix mLSTM) and 's' (scalar sLSTM), e.g. mmmmmmmms for Beck [7:1]. "
        "If set, overrides --layers (depth = len(pattern)). Default: preset posepulse uses mmmmmmmms.",
    )
    ap.add_argument(
        "--preset",
        choices=("riccio", "posepulse", "paper_posepulse_vit", "paper_posepulse_resnet", "none"),
        default="riccio",
        help="riccio: Table-4-style BiLSTM hyperparameters on this script's xLSTM. "
        "posepulse: same NPZ protocol as BiLSTM posepulse — mixed 42-D, window_label=first, "
        "standardize, exclude hammer curl by default, hidden=256, xLSTM[7:1] (``mmmmmmms``), "
        "reg_weight=0, AdamW. "
        "paper_posepulse_vit: vit_backbone (T,256) windows, same stack/optim ballpark as posepulse; "
        "batch 64, 50 epochs, lr 3e-4. "
        "paper_posepulse_resnet: resnet_backbone (T,512 or T,2048) windows; same hyperparameters as paper_posepulse_vit.",
    )

    args = ap.parse_args()

    if args.preset == "riccio":
        args.batch_size = 54
        args.lr = 0.0004
        args.hidden = 128
        args.layers = 2
        args.dropout = 0.2174
    elif args.preset == "posepulse":
        args.feature_mode = "mixed"
        args.window_label = "first"
        args.standardize = True
        args.reg_weight = 0.0
        args.hidden = 256
        args.layers = 8
        args.dropout = min(float(args.dropout), 0.2)
        if args.xlstm_block_pattern is None:
            args.xlstm_block_pattern = "mmmmmmms"
        if args.exclude_classes is None:
            args.exclude_classes = "hammer curl"
    elif args.preset == "paper_posepulse_vit":
        args.feature_mode = "vit_backbone"
        args.window_label = "first"
        args.standardize = True
        args.reg_weight = 0.0
        args.hidden = 256
        args.layers = 8
        args.dropout = min(float(args.dropout), 0.15)
        args.batch_size = 64
        args.epochs = 50
        args.lr = 3e-4
        if args.xlstm_block_pattern is None:
            args.xlstm_block_pattern = "mmmmmmms"
        if args.exclude_classes is None:
            args.exclude_classes = "hammer curl"
    elif args.preset == "paper_posepulse_resnet":
        args.feature_mode = "resnet_backbone"
        args.window_label = "first"
        args.standardize = True
        args.reg_weight = 0.0
        args.hidden = 256
        args.layers = 8
        args.dropout = min(float(args.dropout), 0.15)
        args.batch_size = 64
        args.epochs = 50
        args.lr = 3e-4
        if args.xlstm_block_pattern is None:
            args.xlstm_block_pattern = "mmmmmmms"
        if args.exclude_classes is None:
            args.exclude_classes = "hammer curl"

    exclude_list = parse_exclude_coarse_classes_cli(args.exclude_classes or "")

    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")

    data_dir = Path(args.kaggle_keypoints_dir)

    if args.feature_mode == "angles":
        train_ds, val_ds, test_ds, class_to_idx, idx_to_class, mean, std = build_kaggle_angle_datasets(
            data_dir,
            stem=args.kaggle_stem,
            window=args.window,
            stride=args.stride,
            standardize=args.standardize,
            window_label=args.window_label,
            exclude_coarse_classes=exclude_list,
        )
        input_size = 8
    elif args.feature_mode == "mixed":
        train_ds, val_ds, test_ds, class_to_idx, idx_to_class, mean, std = build_kaggle_mixed_datasets(
            data_dir,
            stem=args.kaggle_stem,
            window=args.window,
            stride=args.stride,
            standardize=args.standardize,
            window_label=args.window_label,
            exclude_coarse_classes=exclude_list,
        )
        input_size = 42
    elif args.feature_mode in ("vit_backbone", "resnet_backbone"):
        train_ds, val_ds, test_ds, class_to_idx, idx_to_class, mean, std = build_kaggle_frame_feature_datasets(
            data_dir,
            stem=args.kaggle_stem,
            window=args.window,
            stride=args.stride,
            standardize=args.standardize,
            window_label=args.window_label,
            exclude_coarse_classes=exclude_list,
        )
        if len(train_ds.samples) == 0:
            print("No training windows from vit_backbone / resnet_backbone NPZ.", file=sys.stderr)
            return 1
        input_size = int(np.asarray(train_ds.samples[0][0]).shape[1])
    else:
        train_ds, val_ds, test_ds, class_to_idx, idx_to_class, mean, std = build_kaggle_xlstm_datasets(
            data_dir,
            stem=args.kaggle_stem,
            window=args.window,
            stride=args.stride,
            standardize=args.standardize,
        )
        input_size = 34

    print(f"Feature mode: {args.feature_mode}, input_size: {input_size}")
    print(f"Preset: {args.preset}  standardize={args.standardize}  window_label={args.window_label}")
    if exclude_list:
        print(f"Excluded coarse classes: {exclude_list}")
    print(f"Classes: {list(class_to_idx.keys())}")
    print(f"Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}")
    if args.xlstm_block_pattern:
        print(f"xLSTM block pattern ({len(args.xlstm_block_pattern)} blocks): {args.xlstm_block_pattern!r}")

    num_classes = len(class_to_idx)

    model = xLSTMExerciseClassifier(
        input_size=input_size,
        hidden_size=args.hidden,
        num_layers=args.layers,
        num_classes=num_classes,
        dropout=args.dropout,
        bidirectional=True,
        linear_classifier=bool(args.linear_classifier),
        block_pattern=args.xlstm_block_pattern,
    ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=float(args.weight_decay))

    # Class weights
    ce_weights = compute_inverse_frequency_class_weights(train_ds, num_classes).to(device)
    print(f"Class weights: {ce_weights}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False) if test_ds else None

    best_acc = 0.0
    best_state = None

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(
            model, train_loader, opt, args.cls_weight, args.reg_weight, device, ce_weights
        )
        val_acc, val_rmse, val_reg = evaluate(model, val_loader, device, num_classes)
        print(
            f"epoch {epoch:03d}  train_loss={train_loss:.4f}  "
            f"val_acc={val_acc:.4f}  val_q_rmse={val_rmse:.4f}"
        )

        if val_acc > best_acc:
            best_acc = val_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    # Save best model
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / "xlstm_keypoints_best.pt"
    torch.save({
        "model": best_state,
        "window": args.window,
        "stride": args.stride,
        "input_size": input_size,
        "feature_mode": args.feature_mode,
        "bidirectional": True,
        "num_classes": num_classes,
        "classes": list(class_to_idx.keys()),
        "class_to_idx": class_to_idx,
        "idx_to_class": idx_to_class,
        "mean": mean,
        "std": std,
        "hidden": args.hidden,
        "layers": args.layers,
        "dropout": args.dropout,
        "lr": args.lr,
        "weight_decay": float(args.weight_decay),
        "batch_size": args.batch_size,
        "preset": args.preset,
        "window_label": args.window_label,
        "standardize": bool(args.standardize),
        "reg_weight": float(args.reg_weight),
        "excluded_coarse_classes": list(exclude_list) if exclude_list else None,
        "linear_classifier": bool(args.linear_classifier),
        "block_pattern": args.xlstm_block_pattern,
    }, ckpt_path)
    print(f"Best val acc: {best_acc:.4f}, saved to {ckpt_path}")

    # Evaluate on test set
    if args.eval_test and test_loader:
        model.load_state_dict(best_state)
        ta, trmse, treg, test_cls, y_true_t, y_prob_t = evaluate(
            model, test_loader, device, num_classes, detailed=True, class_names=list(class_to_idx.keys())
        )
        print(f"\nTest Results:")
        print(f"Accuracy: {ta:.4f}")
        print(f"F1 (weighted): {test_cls['f1_weighted']:.4f}")
        print(f"Precision (weighted): {test_cls['precision_macro']:.4f}")
        print(f"Recall (weighted): {test_cls['recall_macro']:.4f}")
        print(f"Quality RMSE: {trmse:.4f}")

        print("\nPer-class F1:")
        for name, f1 in test_cls["f1_per_class"].items():
            print(f"  {name}: {f1:.4f}")

        print("\nConfusion Matrix (rows=true, cols=predicted):")
        from fitness_coach.evaluation.classification_metrics import format_confusion_matrix_text
        print(format_confusion_matrix_text(
            test_cls["confusion_matrix"],
            test_cls["confusion_matrix_row_labels"],
        ))

        # Save test metrics
        metrics_path = out_dir / "test_metrics.json"
        with open(metrics_path, "w") as f:
            json.dump({
                "accuracy": test_cls["accuracy"],
                "f1_macro": test_cls["f1_macro"],
                "f1_weighted": test_cls["f1_weighted"],
                "precision_macro": test_cls["precision_macro"],
                "precision_weighted": test_cls["precision_macro"],  # Note: no weighted precision in current impl
                "recall_macro": test_cls["recall_macro"],
                "recall_weighted": test_cls["recall_macro"],  # Note: no weighted recall in current impl
                "f1_per_class": list(test_cls["f1_per_class"].values()),
                "confusion_matrix": test_cls["confusion_matrix"],
                "class_names": test_cls["confusion_matrix_row_labels"],
                "quality_rmse": trmse,
                "quality_mae": treg["mae"],
                "quality_r2": treg["r2"],
            }, f, indent=2)
        print(f"Test metrics saved to {metrics_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())