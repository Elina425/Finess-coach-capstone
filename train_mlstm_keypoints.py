#!/usr/bin/env python3
"""
Train residual mLSTM on normalized keypoint windows from the Riccio dataset.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from fitness_coach.datasets.exercise_stgcn_dataset import (
    build_kaggle_stgcn_datasets,
    fit_stgcn_standardizer,
)
from fitness_coach.models.mlstm_model import MLSTMExerciseClassifier


class KeypointMLSTMDataset(Dataset):
    """Dataset for mLSTM: flatten normalized keypoints to sequence frames."""

    def __init__(
        self,
        samples: List[Tuple[np.ndarray, int, float]],
        mean: Optional[np.ndarray],
        std: Optional[np.ndarray],
        use_confidence: bool = False,
    ):
        self.samples = samples
        self.mean = mean
        self.std = std
        self.use_confidence = use_confidence

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        x, y, q = self.samples[idx]
        x = np.asarray(x, dtype=np.float32)
        if self.mean is not None and self.std is not None:
            x = (x - self.mean) / self.std

        if self.use_confidence:
            if x.shape[-1] == 2:
                conf = np.ones((x.shape[0], x.shape[1], 1), dtype=np.float32)
                x = np.concatenate([x, conf], axis=-1)
            elif x.shape[-1] == 3:
                pass
            else:
                raise ValueError(f"Unsupported keypoint channel dimension: {x.shape[-1]}")

        x = x.reshape(x.shape[0], -1)
        return (
            torch.from_numpy(x),
            torch.tensor(y, dtype=torch.long),
            torch.tensor(q, dtype=torch.float32),
        )


def compute_inverse_frequency_class_weights(train_ds: Dataset, num_classes: int) -> torch.Tensor:
    labels = []
    for _, y, _ in train_ds:
        labels.append(int(y.item()))
    counts = Counter(labels)
    total = sum(counts.values())
    weights = torch.zeros(num_classes, dtype=torch.float32)
    for cls, count in counts.items():
        weights[cls] = total / (len(counts) * count)
    return weights


def build_kaggle_mlstm_datasets(
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
    use_confidence: bool = False,
) -> Tuple[Dataset, Dataset, Dataset, Dict[str, int], Dict[int, str], Optional[np.ndarray], Optional[np.ndarray]]:
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

    train_mlstm = KeypointMLSTMDataset(train_ds.samples, mean, std, use_confidence=use_confidence)
    val_mlstm = KeypointMLSTMDataset(val_ds.samples, mean, std, use_confidence=use_confidence)
    test_mlstm = KeypointMLSTMDataset(test_ds.samples, mean, std, use_confidence=use_confidence)

    return train_mlstm, val_mlstm, test_mlstm, class_to_idx, idx_to_class, mean, std


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    opt: torch.optim.Optimizer,
    cls_weight: float,
    reg_weight: float,
    device: torch.device,
    class_weights: Optional[torch.Tensor] = None,
) -> float:
    model.train()
    total_loss = 0.0
    total_samples = 0

    ce = nn.CrossEntropyLoss(weight=class_weights.to(device) if class_weights is not None else None)
    mse = nn.MSELoss()

    for xb, y_cls, y_q in loader:
        xb = xb.to(device)
        y_cls = y_cls.to(device)
        y_q = y_q.to(device)

        opt.zero_grad()
        logits, pred_q = model(xb)
        loss = cls_weight * ce(logits, y_cls) + reg_weight * mse(pred_q.squeeze(-1), y_q)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
        opt.step()

        total_loss += float(loss.item()) * xb.size(0)
        total_samples += xb.size(0)

    return total_loss / max(total_samples, 1)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    num_classes: int,
    *,
    detailed: bool = False,
    class_names: Optional[List[str]] = None,
) -> Tuple[float, float, Dict[str, float], Optional[Dict[str, float]], Optional[np.ndarray], Optional[np.ndarray]]:
    model.eval()
    ce = nn.CrossEntropyLoss()
    mse = nn.MSELoss()
    correct = 0
    total = 0
    reg_loss = 0.0
    abs_err = 0.0

    y_true = []
    y_pred = []
    prob_list = []

    for xb, y_cls, y_q in loader:
        xb = xb.to(device)
        y_cls = y_cls.to(device)
        y_q = y_q.to(device)

        logits, pred_q = model(xb)
        pred = logits.argmax(dim=1)
        pred_q = pred_q.squeeze(-1)

        correct += int((pred == y_cls).sum().item())
        total += xb.size(0)
        reg_loss += float(mse(pred_q, y_q).item()) * xb.size(0)
        abs_err += float(torch.abs(pred_q - y_q).sum().item())

        if detailed:
            y_true.extend(y_cls.cpu().numpy().tolist())
            y_pred.extend(pred.cpu().numpy().tolist())
            prob_list.append(torch.softmax(logits, dim=1).cpu().numpy())

    accuracy = correct / max(total, 1)
    rmse = (reg_loss / max(total, 1)) ** 0.5
    mae = abs_err / max(total, 1)
    r2 = float("nan")
    if detailed and total > 1:
        try:
            from sklearn.metrics import r2_score
            r2 = float(r2_score(np.array(y_q.cpu()), np.array(pred_q.cpu())))
        except Exception:
            r2 = float("nan")

    metrics = None
    y_true_arr = None
    y_prob_arr = None
    if detailed:
        if class_names is None:
            raise ValueError("class_names required when detailed=True")
        from fitness_coach.evaluation.classification_metrics import detailed_classification_metrics

        metrics = detailed_classification_metrics(
            np.array(y_true, dtype=np.int64),
            np.array(y_pred, dtype=np.int64),
            class_names,
        )
        y_prob_arr = np.vstack(prob_list) if prob_list else np.zeros((0, num_classes), dtype=np.float32)
        y_true_arr = np.array(y_true, dtype=np.int64)

    return accuracy, rmse, {"rmse": rmse, "mae": mae, "r2": r2}, metrics, y_true_arr, y_prob_arr


def main() -> int:
    parser = argparse.ArgumentParser(description="Train residual mLSTM on Riccio keypoints")
    parser.add_argument("--kaggle-keypoints-dir", required=True)
    parser.add_argument("--kaggle-stem", default="riccio_realtime_exercise_recognition")
    parser.add_argument("--output-dir", default="./results/mlstm_keypoints")
    parser.add_argument("--batch-size", type=int, default=48)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--embed", type=int, default=256)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.25)
    parser.add_argument("--window", type=int, default=30)
    parser.add_argument("--stride", type=int, default=15)
    parser.add_argument("--cls-weight", type=float, default=1.0)
    parser.add_argument("--reg-weight", type=float, default=0.5)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--standardize", action="store_true", default=True)
    parser.add_argument("--eval-test", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--use-confidence", action="store_true", help="Append confidence channel if available")
    args = parser.parse_args()

    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    data_dir = Path(args.kaggle_keypoints_dir)

    train_ds, val_ds, test_ds, class_to_idx, idx_to_class, mean, std = build_kaggle_mlstm_datasets(
        data_dir,
        stem=args.kaggle_stem,
        window=args.window,
        stride=args.stride,
        standardize=args.standardize,
        use_confidence=args.use_confidence,
    )

    input_tensor, _, _ = train_ds[0]
    input_size = input_tensor.shape[-1]
    num_classes = len(class_to_idx)

    print(f"Classes: {list(class_to_idx.keys())}")
    print(f"Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}")
    print(f"Input size: {input_size}, layers: {args.layers}, hidden: {args.hidden}, embed: {args.embed}")

    model = MLSTMExerciseClassifier(
        input_size=input_size,
        embed_dim=args.embed,
        hidden_size=args.hidden,
        num_layers=args.layers,
        num_classes=num_classes,
        dropout=args.dropout,
        bidirectional=True,
    ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    class_weights = compute_inverse_frequency_class_weights(train_ds, num_classes).to(device)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    best_acc = 0.0
    best_state = None
    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(
            model,
            train_loader,
            opt,
            args.cls_weight,
            args.reg_weight,
            device,
            class_weights,
        )
        val_acc, val_rmse, _, _, _, _ = evaluate(
            model,
            val_loader,
            device,
            num_classes,
            detailed=False,
        )
        print(f"epoch {epoch:03d} train_loss={train_loss:.4f} val_acc={val_acc:.4f} val_q_rmse={val_rmse:.4f}")

        if val_acc > best_acc:
            best_acc = val_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / "mlstm_keypoints_best.pt"
    torch.save(
        {
            "model": best_state,
            "input_size": input_size,
            "num_classes": num_classes,
            "classes": list(class_to_idx.keys()),
            "class_to_idx": class_to_idx,
            "idx_to_class": idx_to_class,
            "mean": mean,
            "std": std,
            "hidden": args.hidden,
            "embed": args.embed,
            "layers": args.layers,
            "dropout": args.dropout,
            "lr": args.lr,
            "batch_size": args.batch_size,
            "window": args.window,
            "stride": args.stride,
        },
        ckpt_path,
    )
    print(f"Best val acc: {best_acc:.4f}, saved to {ckpt_path}")

    if args.eval_test:
        model.load_state_dict(best_state)
        ta, trmse, treg, test_cls, _, _ = evaluate(
            model,
            test_loader,
            device,
            num_classes,
            detailed=True,
            class_names=list(class_to_idx.keys()),
        )
        print("\nTest Results:")
        print(f"Accuracy: {ta:.4f}")
        print(f"F1 (weighted): {test_cls['f1_weighted']:.4f}")
        print(f"Precision (macro): {test_cls['precision_macro']:.4f}")
        print(f"Recall (macro): {test_cls['recall_macro']:.4f}")
        print(f"Quality RMSE: {trmse:.4f}")

        print("\nPer-class F1:")
        for name, f1 in test_cls['f1_per_class'].items():
            print(f"  {name}: {f1:.4f}")

        from fitness_coach.evaluation.classification_metrics import format_confusion_matrix_text
        print("\nConfusion Matrix (rows=true, cols=predicted):")
        print(format_confusion_matrix_text(test_cls['confusion_matrix'], test_cls['confusion_matrix_row_labels']))

        metrics_path = out_dir / "test_metrics.json"
        with open(metrics_path, "w") as f:
            json.dump(
                {
                    "accuracy": test_cls["accuracy"],
                    "f1_macro": test_cls["f1_macro"],
                    "f1_weighted": test_cls["f1_weighted"],
                    "precision_macro": test_cls["precision_macro"],
                    "precision_weighted": test_cls["precision_macro"],
                    "recall_macro": test_cls["recall_macro"],
                    "recall_weighted": test_cls["recall_macro"],
                    "f1_per_class": list(test_cls["f1_per_class"].values()),
                    "confusion_matrix": test_cls["confusion_matrix"],
                    "class_names": list(class_to_idx.keys()),
                    "quality_rmse": trmse,
                    "quality_mae": treg["mae"],
                    "quality_r2": treg["r2"],
                },
                f,
                indent=2,
            )
        print(f"Metrics saved to {metrics_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
