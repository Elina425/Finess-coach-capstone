#!/usr/bin/env python3
"""
Rigorous evaluation: classification accuracy, per-class F1, quality MAE and R².

Supports:
  - BiLSTM checkpoints (train_exercise_bilstm.py): feature mode inferred from feat_dim or ckpt.
  - GCN supervised checkpoints (train_gcn_supervised.py).

Use the same --index-csv with a held-out split (e.g. split=test or val). Window-level metrics
(aggregating all 30-frame windows in that split).

Ablation (angles vs coords vs mixed): train three BiLSTM runs with --feature-mode, then:
  ./venv/bin/python evaluate_exercise_models.py \\
    --index-csv results/exercise_training_index.csv \\
    --split test \\
    --checkpoint bilstm results/bilstm_angles/exercise_bilstm_best.pt \\
    --checkpoint bilstm results/bilstm_coords/exercise_bilstm_best.pt \\
    --checkpoint bilstm results/bilstm_mixed/exercise_bilstm_best.pt \\
    --checkpoint gcn_supervised results/gcn_supervised/gcn_supervised_best.pt
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    r2_score,
)

from exercise_bilstm_dataset import ExerciseAngleWindowDataset, load_index_rows
from exercise_bilstm_model import ExerciseBiLSTM
from exercise_keypoint_window_dataset import ExerciseKeypointWindowDataset
from gcn_pose_model import GCNSequenceExerciseNet


def _resolve_bilstm_feature_mode(feat_dim: int, ckpt_mode: Optional[str]) -> str:
    if ckpt_mode in ("angles", "coords", "mixed"):
        return str(ckpt_mode)
    return {8: "angles", 34: "coords", 42: "mixed"}.get(feat_dim, "angles")


@torch.no_grad()
def eval_bilstm(
    ckpt_path: Path,
    index_csv: Path,
    angles_dir: Path,
    keypoints_dir: Path,
    split: str,
    class_to_idx: Dict[str, int],
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, str]:
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    if isinstance(ckpt, dict) and "model" not in ckpt:
        raise ValueError(f"Unexpected checkpoint format: {ckpt_path}")

    feat_dim = int(ckpt.get("feat_dim", 8))
    num_classes = int(ckpt["num_classes"])
    classes: List[str] = list(ckpt.get("classes", []))
    window = int(ckpt.get("window", 30))
    stride = int(ckpt.get("stride", 15))
    feature_mode = _resolve_bilstm_feature_mode(feat_dim, ckpt.get("feature_mode"))
    scale_mean = ckpt.get("scale_mean")
    scale_std = ckpt.get("scale_std")
    hidden = int(ckpt.get("hidden", 128))
    layers = int(ckpt.get("layers", 2))
    dropout = float(ckpt.get("dropout", 0.35))

    kp_dir = Path(keypoints_dir) if feature_mode in ("mixed", "coords") else None

    ds = ExerciseAngleWindowDataset(
        index_csv,
        Path(angles_dir),
        class_to_idx,
        split=split,
        window=window,
        stride=stride,
        feature_mode=feature_mode,
        keypoints_dir=kp_dir,
    )
    if scale_mean is not None and scale_std is not None and len(ds) > 0:
        ds.apply_standardizer(
            np.asarray(scale_mean, dtype=np.float32),
            np.asarray(scale_std, dtype=np.float32),
        )

    if len(ds) == 0:
        raise RuntimeError(
            f"No windows for split={split!r} — check index, paths, and feature_mode={feature_mode!r}."
        )

    model = ExerciseBiLSTM(
        input_dim=feat_dim,
        num_classes=num_classes,
        hidden=hidden,
        num_layers=layers,
        dropout=dropout,
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    ys: List[int] = []
    preds: List[int] = []
    qs: List[float] = []
    qhats: List[float] = []

    loader = torch.utils.data.DataLoader(ds, batch_size=64, shuffle=False)
    for xb, y_cls, y_q in loader:
        xb = xb.to(device)
        logits, pred_q = model(xb)
        pr = logits.argmax(dim=1).cpu().numpy()
        ys.extend(y_cls.numpy().tolist())
        preds.extend(pr.tolist())
        qs.extend(y_q.numpy().tolist())
        qhats.extend(pred_q.cpu().numpy().tolist())

    tag = f"bilstm_{feature_mode}"
    return (
        np.array(ys),
        np.array(preds),
        np.array(qs, dtype=np.float64),
        np.array(qhats, dtype=np.float64),
        tag,
    )


@torch.no_grad()
def eval_gcn_supervised(
    ckpt_path: Path,
    index_csv: Path,
    keypoints_dir: Path,
    split: str,
    class_to_idx: Dict[str, int],
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, str]:
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    num_classes = int(ckpt["num_classes"])
    window = int(ckpt.get("window", 30))
    stride = int(ckpt.get("stride", 15))
    gcn_hidden = int(ckpt.get("gcn_hidden", 64))
    embed_dim = int(ckpt.get("embed_dim", 50))

    ds = ExerciseKeypointWindowDataset(
        index_csv,
        Path(keypoints_dir),
        class_to_idx,
        split=split,
        window=window,
        stride=stride,
    )
    if len(ds) == 0:
        raise RuntimeError(f"No GCN windows for split={split!r}.")

    model = GCNSequenceExerciseNet(
        num_classes=num_classes,
        gcn_hidden=gcn_hidden,
        embed_dim=embed_dim,
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    ys: List[int] = []
    preds: List[int] = []
    qs: List[float] = []
    qhats: List[float] = []

    loader = torch.utils.data.DataLoader(ds, batch_size=64, shuffle=False)
    for xb, y_cls, y_q in loader:
        xb = xb.to(device)
        logits, pred_q = model(xb)
        pr = logits.argmax(dim=1).cpu().numpy()
        ys.extend(y_cls.numpy().tolist())
        preds.extend(pr.tolist())
        qs.extend(y_q.numpy().tolist())
        qhats.extend(pred_q.cpu().numpy().tolist())

    return (
        np.array(ys),
        np.array(preds),
        np.array(qs, dtype=np.float64),
        np.array(qhats, dtype=np.float64),
        "gcn_supervised_coords",
    )


def classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str],
) -> Dict:
    n_classes = len(class_names)
    acc = float(accuracy_score(y_true, y_pred))
    f1_macro = float(
        f1_score(y_true, y_pred, average="macro", labels=range(n_classes), zero_division=0)
    )
    f1_each = f1_score(y_true, y_pred, average=None, labels=range(n_classes), zero_division=0)
    per_class = {
        class_names[i]: float(f1_each[i]) for i in range(n_classes) if i < len(f1_each)
    }
    return {"accuracy": acc, "f1_macro": f1_macro, "f1_per_class": per_class}


def regression_metrics(q_true: np.ndarray, q_pred: np.ndarray) -> Dict:
    mae = float(mean_absolute_error(q_true, q_pred))
    if np.var(q_true) < 1e-12:
        r2 = float("nan")
    else:
        r2 = float(r2_score(q_true, q_pred))
    return {"mae": mae, "r2": r2}


def main() -> int:
    ap = argparse.ArgumentParser(description="Evaluate exercise models (cls + quality metrics)")
    ap.add_argument("--index-csv", required=True)
    ap.add_argument("--angles-dir", default="./results/exercise_angles")
    ap.add_argument("--keypoints-dir", default="./results/processed_keypoints_mediapipe")
    ap.add_argument(
        "--split",
        default="test",
        help="CSV split column value (e.g. test, val)",
    )
    ap.add_argument(
        "--checkpoint",
        action="append",
        nargs=2,
        metavar=("TYPE", "PATH"),
        required=True,
        help="TYPE is bilstm or gcn_supervised; repeat for multiple models",
    )
    ap.add_argument("--json-out", type=Path, default=None)
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()

    index_path = Path(args.index_csv)
    if not index_path.is_file():
        print(f"Missing {index_path}", file=sys.stderr)
        return 1

    rows = load_index_rows(index_path)
    train_rows = [r for r in rows if r.get("split") == "train"]
    if not train_rows:
        print("No train rows in index (needed for class vocabulary).", file=sys.stderr)
        return 1

    classes = sorted({r["exercise_class"] for r in train_rows})
    class_to_idx: Dict[str, int] = {c: i for i, c in enumerate(classes)}

    split_rows = [r for r in rows if r.get("split") == args.split]
    if not split_rows:
        print(
            f"No rows with split={args.split!r}. Available: {set(r.get('split') for r in rows)}",
            file=sys.stderr,
        )
        return 1

    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")

    results: List[Dict] = []

    for model_type, ckpt_path in args.checkpoint:
        ckpt_path = Path(ckpt_path)
        if not ckpt_path.is_file():
            print(f"Skip missing checkpoint: {ckpt_path}", file=sys.stderr)
            continue
        try:
            if model_type == "bilstm":
                yt, yp, qt, qp, tag = eval_bilstm(
                    ckpt_path,
                    index_path,
                    Path(args.angles_dir),
                    Path(args.keypoints_dir),
                    args.split,
                    class_to_idx,
                    device,
                )
            elif model_type == "gcn_supervised":
                yt, yp, qt, qp, tag = eval_gcn_supervised(
                    ckpt_path,
                    index_path,
                    Path(args.keypoints_dir),
                    args.split,
                    class_to_idx,
                    device,
                )
            else:
                print(f"Unknown TYPE {model_type!r} (use bilstm or gcn_supervised)", file=sys.stderr)
                return 1
        except Exception as e:
            print(f"Error evaluating {ckpt_path}: {e}", file=sys.stderr)
            return 1

        cm = classification_metrics(yt, yp, classes)
        rm = regression_metrics(qt, qp)
        row = {
            "tag": tag,
            "checkpoint": str(ckpt_path.resolve()),
            "split": args.split,
            "n_windows": int(len(yt)),
            "classification": cm,
            "quality_regression": rm,
        }
        results.append(row)

    print("\n=== Exercise classification (window-level) ===\n")
    for r in results:
        c = r["classification"]
        print(f"Model: {r['tag']}  (n={r['n_windows']})")
        print(f"  accuracy:     {c['accuracy']:.4f}")
        print(f"  F1 (macro):   {c['f1_macro']:.4f}")
        print("  F1 per class:")
        for name, f1 in sorted(c["f1_per_class"].items()):
            print(f"    {name}: {f1:.4f}")
        print()

    print("=== Quality regression (window-level) ===\n")
    for r in results:
        q = r["quality_regression"]
        print(f"Model: {r['tag']}")
        print(f"  MAE:  {q['mae']:.4f}")
        print(f"  R²:   {q['r2'] if np.isfinite(q['r2']) else 'nan'}")
        print()

    print("Ablation note: compare bilstm_angles vs bilstm_coords vs bilstm_mixed vs gcn_supervised_coords.")
    print("  angles = joint angles only (8-D); coords = normalized xy only (34-D); mixed = 42-D.")

    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        with open(args.json_out, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Wrote {args.json_out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
