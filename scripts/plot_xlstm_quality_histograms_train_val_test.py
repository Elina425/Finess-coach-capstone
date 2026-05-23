#!/usr/bin/env python3
"""
Overlay histograms of **original (dataset) quality** vs **model-predicted quality**
for train, validation, and test splits — EgoExo multitask xLSTM checkpoint.

`metrics.json` only has aggregate regression stats; per-sample q_true/q_pred are
obtained by re-running the saved checkpoint on the index CSV (same pipeline as
training).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from fitness_coach.datasets.egoexo_xlstm_dataset import (  # noqa: E402
    CLIP_SUBDIR,
    EgoExoXLSTMDataset,
    apply_feature_standardizer,
    egoexo_collate_fn,
)
from fitness_coach.datasets.exercise_bilstm_dataset import load_index_rows  # noqa: E402
from fitness_coach.models.xlstm_model import xLSTMExerciseClassifier  # noqa: E402
from train_xlstm_egoexo_multitask import (  # noqa: E402
    _forward_heads,
    normalize_clip_features_root,
    resolve_egoexo_index_csv,
)


def load_model_from_checkpoint(ckpt_path: Path, device: torch.device) -> xLSTMExerciseClassifier:
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    n_err = len(ck.get("error_tags") or [])
    model = xLSTMExerciseClassifier(
        input_size=int(ck.get("input_size", 512)),
        hidden_size=int(ck["hidden"]),
        num_layers=int(ck["layers"]),
        num_classes=len(ck["classes"]),
        dropout=float(ck.get("dropout", 0.15)),
        num_heads=int(ck["num_heads"]),
        conv_kernel_size=int(ck["conv_kernel_size"]),
        projection_factor=float(ck["projection_factor"]),
        num_error_tags=n_err,
        num_quality_classes=int(ck.get("num_quality_classes", 1)),
        quality_scale=float(ck.get("quality_scale", 1.0)),
        quality_output_low=float(ck.get("quality_output_low", 0.0)),
        block_pattern=str(ck.get("block_pattern", "mmmmmmms")),
        use_attention_pool=bool(ck.get("use_attention_pool", True)),
        use_fusion=bool(ck.get("use_fusion", True)),
        fusion_dim=int(ck.get("fusion_dim", 128)),
        quality_class_conditioning=bool(ck.get("teacher_force_quality", False)),
    ).to(device)
    model.load_state_dict(ck["model"], strict=False)
    model.eval()

    idx_to_class = {int(k): str(v) for k, v in ck["idx_to_class"].items()}
    guidance_table = {int(k): str(v) for k, v in ck["guidance_table"].items()}
    model.set_guidance_table(guidance_table, idx_to_class)

    comment_table_raw = ck.get("comment_table") or {}
    comment_table = {}
    for key, text in comment_table_raw.items():
        cls_idx, bucket = key.split("|")
        comment_table[(int(cls_idx), int(bucket))] = str(text)
    bucket_edges = tuple(float(e) for e in (ck.get("comment_quality_bucket_edges") or (0.4, 0.7)))
    q_lo = float(ck.get("quality_domain_lo", 0.0))
    q_hi = float(ck.get("quality_domain_hi", 1.0))
    model.set_comment_table(comment_table, bucket_edges, domain_lo=q_lo, domain_hi=q_hi)
    return model


@torch.no_grad()
def collect_quality(
    model: xLSTMExerciseClassifier,
    loader: DataLoader | None,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    if loader is None:
        return np.zeros(0, dtype=np.float64), np.zeros(0, dtype=np.float64)
    q_true: List[float] = []
    q_pred: List[float] = []
    for batch in loader:
        xb, y_cls, y_qf, _y_q_bucket, _y_err, *_rest = batch
        xb = xb.to(device)
        _pooled, _z, _logits, pred_q, _pred_err = _forward_heads(model, xb)
        q_true.extend(y_qf.numpy().astype(np.float64).tolist())
        if model.quality_is_classification:
            probs = torch.softmax(pred_q, dim=-1)
            centres = pred_q.new_tensor(list(model._quality_bucket_centres))
            exp = (probs * centres).sum(dim=-1)
            q_pred.extend(exp.detach().cpu().numpy().astype(np.float64).tolist())
        else:
            q_pred.extend(pred_q.squeeze(-1).cpu().numpy().astype(np.float64).tolist())
    return np.asarray(q_true, dtype=np.float64), np.asarray(q_pred, dtype=np.float64)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument(
        "--ckpt",
        type=Path,
        default=Path("results/xlstm_egoexo_multitask_allviews/xlstm_egoexo_multitask_best.pt"),
    )
    p.add_argument("--index-csv", type=Path, default=Path("results/egoexo_fitness_index_split.csv"))
    p.add_argument(
        "--clip-features-root",
        type=Path,
        default=Path("notebooks/data/egoexo_fitness_full/features_open/visual"),
        help=f"Parent of `{CLIP_SUBDIR}` (same as training).",
    )
    p.add_argument(
        "--clip-view",
        default="all",
        help='EgoExo views to load: "ego_l", "all", comma list, etc. '
        'Match training (multitask allviews runs used --clip-view all).',
    )
    p.add_argument("--clip-max-frames", type=int, default=300)
    p.add_argument("--clip-subsample-stride", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--output-dir", type=Path, default=Path("results/xlstm_egoexo_multitask_allviews/plots"))
    p.add_argument("--bins", type=int, default=45)
    p.add_argument("--cpu", action="store_true", help="Force CPU even if CUDA is available.")
    args = p.parse_args()

    if args.cpu:
        device = torch.device("cpu")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    ck = torch.load(args.ckpt.expanduser(), map_location="cpu", weights_only=False)

    class_to_idx: dict = ck["class_to_idx"]
    if not all(isinstance(k, str) for k in class_to_idx):
        class_to_idx = {str(k): int(v) for k, v in class_to_idx.items()}
    feature_mode = str(ck["feature_mode"])
    window = int(ck.get("window", 60))
    stride = int(ck.get("stride", 30))
    needs_pose = feature_mode in ("angles", "coords", "mixed")

    idx_path = resolve_egoexo_index_csv(Path(args.index_csv))
    clip_root = normalize_clip_features_root(Path(args.clip_features_root))

    rows = load_index_rows(idx_path)
    val_split = "val" if any((r.get("split") or "") == "val" for r in rows) else "test"
    has_test = any((r.get("split") or "") == "test" for r in rows)

    base_kwargs = dict(
        feature_mode=feature_mode,
        quality_encoding=str(ck.get("quality_encoding", "unit")),
        angles_dir=Path("results/egoexo_exercise_angles") if needs_pose else None,
        keypoints_dir=Path("results/egoexo_exercise_angles") if feature_mode in ("coords", "mixed") else None,
        clip_features_root=clip_root if feature_mode == "clip" else None,
        clip_view=args.clip_view,
        clip_max_frames=args.clip_max_frames,
        clip_subsample_stride=args.clip_subsample_stride,
        window=window if needs_pose else 0,
        stride=stride if needs_pose else 0,
    )

    train_ds = EgoExoXLSTMDataset(idx_path, class_to_idx, "train", filter_null_comments=True, **base_kwargs)
    val_ds = EgoExoXLSTMDataset(idx_path, class_to_idx, val_split, filter_null_comments=False, **base_kwargs)
    test_ds = (
        EgoExoXLSTMDataset(idx_path, class_to_idx, "test", filter_null_comments=False, **base_kwargs)
        if has_test
        else None
    )

    mean, std = ck.get("mean"), ck.get("std")
    if mean is not None and std is not None:
        apply_feature_standardizer(train_ds.samples, mean, std)
        if len(val_ds) > 0:
            apply_feature_standardizer(val_ds.samples, mean, std)
        if test_ds is not None and len(test_ds) > 0:
            apply_feature_standardizer(test_ds.samples, mean, std)

    model = load_model_from_checkpoint(args.ckpt.expanduser(), device)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=egoexo_collate_fn,
        num_workers=args.num_workers,
    )
    val_loader = (
        DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=egoexo_collate_fn, num_workers=args.num_workers)
        if len(val_ds) > 0
        else None
    )
    test_loader = (
        DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, collate_fn=egoexo_collate_fn, num_workers=args.num_workers)
        if test_ds is not None and len(test_ds) > 0
        else None
    )

    splits = [
        ("train", collect_quality(model, train_loader, device)),
        ("val", collect_quality(model, val_loader, device)),
        ("test", collect_quality(model, test_loader, device)),
    ]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    png_path = args.output_dir / "quality_histogram_original_vs_predicted_train_val_test.png"

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    fig.suptitle("Quality: EgoExo dataset (original) vs model prediction", fontsize=12)

    lo = float(ck.get("quality_domain_lo", 0.0))
    hi = float(ck.get("quality_domain_hi", 1.0))
    for ax, (name, (qt, qp)) in zip(axes, splits):
        if qt.size == 0:
            ax.text(0.5, 0.5, "no samples", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(name)
            continue
        ax.hist(qt, bins=args.bins, range=(lo, hi), alpha=0.55, color="C0", label="Original (dataset)", density=True)
        ax.hist(qp, bins=args.bins, range=(lo, hi), alpha=0.55, color="C1", label="Predicted", density=True)
        ax.set_title(f"{name} (n={len(qt):,})")
        ax.set_xlabel(f"Quality [{lo:.2f}, {hi:.2f}] (checkpoint axis)")
        ax.set_ylabel("Density")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    npz_path = args.output_dir / "quality_original_vs_predicted_arrays.npz"
    np.savez_compressed(
        npz_path,
        train_q_true=splits[0][1][0],
        train_q_pred=splits[0][1][1],
        val_q_true=splits[1][1][0],
        val_q_pred=splits[1][1][1],
        test_q_true=splits[2][1][0],
        test_q_pred=splits[2][1][1],
        checkpoint=str(args.ckpt.resolve()),
        index_csv=str(idx_path.resolve()),
    )

    meta = {
        "checkpoint": str(args.ckpt.resolve()),
        "index_csv": str(idx_path.resolve()),
        "clip_features_root": str(clip_root.resolve()),
        "splits": {
            name: {"n": int(qt.size), "original_mean": float(qt.mean()), "predicted_mean": float(qp.mean())}
            for name, (qt, qp) in splits
            if qt.size > 0
        },
    }
    (args.output_dir / "quality_histogram_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"Wrote {png_path}")
    print(f"Wrote {npz_path}")
    print(f"Wrote {args.output_dir / 'quality_histogram_meta.json'}")


if __name__ == "__main__":
    main()
