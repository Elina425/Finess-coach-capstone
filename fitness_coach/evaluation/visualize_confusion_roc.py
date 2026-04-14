#!/usr/bin/env python3
"""
Confusion matrix heatmap + multiclass ROC curves from training outputs.

**Option A — files from** ``train_exercise_bilstm.py --eval-test`` **(recommended)**:

  - ``test_classification_metrics.json`` — confusion matrix + ``class_names``
  - ``test_classification_probs.npz`` — ``y_true``, ``y_prob``, ``class_names``

**Option B — no** ``.npz`` **yet:** pass ``--checkpoint`` + ``--kaggle-angles-dir`` (same stem/ratios/seed
as training) to recompute test-set softmax probabilities and plot ROC (BiLSTM / angles Kaggle mode only).

Example::

  ./venv/bin/python visualize_confusion_roc.py \\
    --metrics-json results/exercise_bilstm/test_classification_metrics.json \\
    --probs-npz results/exercise_bilstm/test_classification_probs.npz \\
    --out-dir results/exercise_bilstm/figures

  # Without .npz file (recompute probs from checkpoint — must match training split args):
  ./venv/bin/python visualize_confusion_roc.py \\
    --metrics-json results/exercise_bilstm/test_classification_metrics.json \\
    --checkpoint results/exercise_bilstm/exercise_bilstm_best.pt \\
    --kaggle-angles-dir results/riccio_realtime_exercise_recognition \\
    --kaggle-stem riccio_realtime_exercise_recognition \\
    --out-dir results/exercise_bilstm/figures --prefix riccio_test
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import auc, roc_auc_score, roc_curve
from sklearn.preprocessing import label_binarize


def plot_confusion_matrix_heatmap(
    cm: Sequence[Sequence[float]],
    labels: List[str],
    out_path: Path,
    *,
    title: str = "Confusion matrix (rows=true, cols=predicted)",
) -> None:
    cm = np.asarray(cm, dtype=float)
    n = cm.shape[0]
    fig, ax = plt.subplots(figsize=(max(6.0, n * 1.2), max(5.0, n * 0.9)))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    tick_marks = np.arange(n)
    short = [str(s)[:22] + ("…" if len(str(s)) > 22 else "") for s in labels]
    ax.set_xticks(tick_marks)
    ax.set_yticks(tick_marks)
    ax.set_xticklabels(short, rotation=40, ha="right")
    ax.set_yticklabels(short)
    ax.set_ylabel("True class")
    ax.set_xlabel("Predicted class")
    vmax = float(cm.max()) if cm.size else 1.0
    thresh = vmax / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                f"{int(cm[i, j])}",
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=9,
            )
    ax.set_title(title)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_multiclass_roc(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    class_names: List[str],
    out_path: Path,
) -> None:
    y_true = np.asarray(y_true).astype(int).ravel()
    y_prob = np.asarray(y_prob, dtype=np.float64)
    n_classes = len(class_names)
    if y_prob.ndim != 2 or y_prob.shape[1] != n_classes:
        raise ValueError(
            f"y_prob expected shape (N, {n_classes}), got {y_prob.shape}; "
            "check class_names length matches model outputs."
        )
    y_bin = label_binarize(y_true, classes=np.arange(n_classes))

    fig, ax = plt.subplots(figsize=(8, 7))
    for i in range(n_classes):
        if np.unique(y_bin[:, i]).size < 2:
            continue
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_prob[:, i])
        roc_auc = auc(fpr, tpr)
        label = str(class_names[i])[:40]
        ax.plot(fpr, tpr, lw=2, label=f"{label} (AUC={roc_auc:.3f})")

    if y_bin.size and y_prob.size:
        fpr_m, tpr_m, _ = roc_curve(y_bin.ravel(), y_prob.ravel())
        micro_auc = auc(fpr_m, tpr_m)
        ax.plot(fpr_m, tpr_m, "k--", lw=2, label=f"Micro-average (AUC={micro_auc:.3f})")

    title_extra = ""
    try:
        macro_auc = float(
            roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro")
        )
        title_extra = f"  |  macro AUC (OvR) = {macro_auc:.3f}"
    except ValueError:
        pass

    ax.plot([0, 1], [0, 1], "k:", lw=1)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.05)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("Multiclass ROC (one-vs-rest)" + title_extra)
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def recompute_bilstm_test_probs_from_checkpoint(
    ckpt_path: Path,
    angles_dir: Path,
    *,
    stem: str,
    test_ratio: float,
    val_ratio: float,
    seed: int,
    window: int,
    stride: int,
    device: Any,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Rebuild Kaggle angle test windows and run the saved BiLSTM forward (softmax).
    Must match how ``train_exercise_bilstm.py`` was run (Kaggle angles mode).
    """
    import torch
    from torch.utils.data import DataLoader

    from fitness_coach.datasets.exercise_bilstm_dataset import build_kaggle_angle_datasets
    from fitness_coach.models.exercise_bilstm_model import build_exercise_bilstm_from_checkpoint

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    if not isinstance(ckpt, dict) or "model" not in ckpt:
        raise ValueError(f"Not a BiLSTM checkpoint from this repo: {ckpt_path}")

    train_ds, val_ds, test_ds, class_to_idx, idx_to_class, _, _ = build_kaggle_angle_datasets(
        angles_dir,
        stem=stem,
        window=window,
        stride=stride,
        test_ratio=test_ratio,
        val_ratio=val_ratio,
        seed=seed,
        standardize=True,
    )
    classes = [idx_to_class[i] for i in range(len(class_to_idx))]
    if len(test_ds) == 0:
        raise RuntimeError("Test split is empty — adjust ratios or check NPZ content.")

    num_classes = int(ckpt["num_classes"])
    if num_classes != len(class_to_idx):
        raise ValueError(
            f"Checkpoint num_classes={num_classes} != dataset {len(class_to_idx)} — "
            "wrong stem or stale checkpoint."
        )
    model = build_exercise_bilstm_from_checkpoint(ckpt).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    loader = DataLoader(test_ds, batch_size=64, shuffle=False)
    ys_list: List[int] = []
    probs_list: List[np.ndarray] = []
    with torch.no_grad():
        for xb, y_cls, _y_q in loader:
            xb = xb.to(device)
            logits, _ = model(xb)
            probs_list.append(torch.softmax(logits, dim=1).cpu().numpy())
            ys_list.extend(y_cls.numpy().tolist())
    y_true = np.array(ys_list, dtype=np.int64)
    y_prob = np.vstack(probs_list)
    return y_true, y_prob, classes


def main() -> int:
    ap = argparse.ArgumentParser(description="Plot confusion matrix heatmap and/or ROC from eval outputs")
    ap.add_argument(
        "--metrics-json",
        type=Path,
        default=None,
        help="test_classification_metrics.json (confusion matrix + class_names)",
    )
    ap.add_argument(
        "--probs-npz",
        type=Path,
        default=None,
        help="test_classification_probs.npz (y_true, y_prob, class_names). If missing, use --checkpoint + --kaggle-angles-dir.",
    )
    ap.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="exercise_bilstm_best.pt — recompute test softmax if .npz is missing (Kaggle angles mode).",
    )
    ap.add_argument(
        "--kaggle-angles-dir",
        type=Path,
        default=None,
        help="Folder with {stem}_biomechanics.npz + {stem}_labels.npz (same as training).",
    )
    ap.add_argument("--kaggle-stem", default="riccio_realtime_exercise_recognition")
    ap.add_argument("--kaggle-test-ratio", type=float, default=0.15)
    ap.add_argument("--kaggle-val-ratio", type=float, default=0.15)
    ap.add_argument("--kaggle-seed", type=int, default=42)
    ap.add_argument(
        "--save-probs-npz",
        type=Path,
        default=None,
        help="After recompute, write probabilities here (default: out-dir / prefix_test_probs.npz).",
    )
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory (default: same folder as inputs)",
    )
    ap.add_argument("--prefix", default="eval", help="Output filename prefix")
    args = ap.parse_args()

    if args.metrics_json is None and args.probs_npz is None and args.checkpoint is None:
        print(
            "Provide --metrics-json and/or --probs-npz, or --checkpoint + --kaggle-angles-dir for ROC.",
            file=sys.stderr,
        )
        return 1

    out_dir = args.out_dir
    if out_dir is None:
        if args.metrics_json is not None:
            out_dir = args.metrics_json.parent
        elif args.probs_npz is not None:
            out_dir = args.probs_npz.parent
        elif args.checkpoint is not None:
            out_dir = args.checkpoint.parent
        else:
            out_dir = Path(".")

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.metrics_json is not None:
        if not args.metrics_json.is_file():
            print(f"Missing {args.metrics_json}", file=sys.stderr)
            return 1
        with open(args.metrics_json) as f:
            data = json.load(f)
        if isinstance(data, list) and data:
            data = data[0]
        cm = data.get("confusion_matrix")
        if cm is None and isinstance(data.get("classification"), dict):
            cm = data["classification"].get("confusion_matrix")
        if cm is None:
            print("No confusion_matrix in JSON", file=sys.stderr)
            return 1
        labels = data.get("class_names")
        if not labels and isinstance(data.get("classification"), dict):
            labels = data["classification"].get("confusion_matrix_row_labels")
        if not labels:
            labels = list(data.get("f1_per_class", {}).keys())
        if not labels and isinstance(data.get("classification"), dict):
            labels = list(data["classification"].get("f1_per_class", {}).keys())
        if not labels or len(labels) != len(cm):
            print(
                "Could not align class names with confusion matrix. "
                "Re-run training with --eval-test to regenerate JSON with class_names.",
                file=sys.stderr,
            )
            return 1
        p = out_dir / f"{args.prefix}_confusion_matrix.png"
        plot_confusion_matrix_heatmap(cm, labels, p)
        print(f"Wrote {p}")

    y_true: np.ndarray | None = None
    y_prob: np.ndarray | None = None
    class_names: List[str] | None = None

    if args.probs_npz is not None and args.probs_npz.is_file():
        z = np.load(args.probs_npz, allow_pickle=True)
        y_true = z["y_true"]
        y_prob = z["y_prob"]
        class_names = [str(x) for x in z["class_names"].tolist()]
    elif args.probs_npz is not None and not args.probs_npz.is_file():
        print(f"Missing {args.probs_npz} — try --checkpoint + --kaggle-angles-dir to recompute.", file=sys.stderr)

    if y_true is None and args.checkpoint is not None and args.kaggle_angles_dir is not None:
        if not args.checkpoint.is_file():
            print(f"Missing checkpoint {args.checkpoint}", file=sys.stderr)
            return 1
        if not args.kaggle_angles_dir.is_dir():
            print(f"Not a directory: {args.kaggle_angles_dir}", file=sys.stderr)
            return 1
        import torch

        ckpt_meta = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        window = int(ckpt_meta.get("window", 30))
        stride = int(ckpt_meta.get("stride", 15))
        device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
        try:
            y_true, y_prob, class_names = recompute_bilstm_test_probs_from_checkpoint(
                args.checkpoint,
                args.kaggle_angles_dir,
                stem=args.kaggle_stem,
                test_ratio=args.kaggle_test_ratio,
                val_ratio=args.kaggle_val_ratio,
                seed=args.kaggle_seed,
                window=window,
                stride=stride,
                device=device,
            )
        except Exception as e:
            print(f"Recompute failed: {e}", file=sys.stderr)
            return 1
        save_npz = args.save_probs_npz or (out_dir / f"{args.prefix}_test_probs.npz")
        np.savez(
            save_npz,
            y_true=y_true,
            y_prob=y_prob,
            class_names=np.array(class_names, dtype=object),
        )
        print(f"Saved recomputed probabilities to {save_npz}")
    elif y_true is None and (args.probs_npz is not None or args.checkpoint is not None):
        print(
            "ROC skipped: use an existing --probs-npz, or pass both:\n"
            "  --checkpoint results/.../exercise_bilstm_best.pt\n"
            "  --kaggle-angles-dir results/riccio_realtime_exercise_recognition\n"
            "(Match training: --kaggle-stem, --kaggle-test-ratio, --kaggle-val-ratio, --kaggle-seed.)",
            file=sys.stderr,
        )

    if y_true is not None and y_prob is not None and class_names is not None:
        p = out_dir / f"{args.prefix}_roc.png"
        plot_multiclass_roc(y_true, y_prob, class_names, p)
        print(f"Wrote {p}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
