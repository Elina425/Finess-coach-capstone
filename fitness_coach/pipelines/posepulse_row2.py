#!/usr/bin/env python3
"""
PosePulse **Row 2** (diagram): frozen ViTPose-S or ResNet on person crops → ``(T, D)`` embeddings,
then **sliding windows** (default 30 frames, stride 15 = 50%% overlap) and **train-split Z-score**
standardisation.

Row 1 (spatial impute → torso norm → temporal impute, bilateral filter, etc.) either:

- runs **inside** the ViTPose / ResNet helpers when you use ``--workflow index-csv`` with
  ``--encoder vit`` or ``--encoder resnet`` (they decode video again and rerun pose + preprocessing), or
- is assumed **already applied** when you pass NPZ files that already contain ``frame_features``
  from ``riccio_kaggle_video_pipeline.py --representation vit_backbone|resnet_backbone``.

Outputs one ``*_row2_windows.npz`` plus ``*_row2_summary.json`` (class map, window/stride, mean/std).

Examples::

    # Already merged Riccio export with frame_features + labels (same layout as train_exercise_bilstm Kaggle mode)
    python -m fitness_coach.pipelines.posepulse_row2 riccio \\
        --angles-dir results/riccio_npz \\
        --stem kaggle_exercise_recognition \\
        --out results/posepulse_row2/riccio_windows.npz

    # Per-video index: encode with ViTPose-S then window (needs video_path column)
    python -m fitness_coach.pipelines.posepulse_row2 index \\
        --index-csv results/my_index.csv \\
        --features-dir results/per_video_feats \\
        --encoder vit \\
        --out results/posepulse_row2/from_index_windows.npz

    # Index but embeddings already exported as ``{video_stem}_biomechanics.npz`` with key ``frame_features``
    python -m fitness_coach.pipelines.posepulse_row2 index \\
        --index-csv results/my_index.csv \\
        --features-dir results/per_video_feats \\
        --encoder skip \\
        --out results/posepulse_row2/from_index_windows.npz
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from fitness_coach.datasets.exercise_bilstm_dataset import (
    build_kaggle_frame_feature_datasets,
    make_windows,
)


def _tensor_window_ds_to_arrays(ds: torch.utils.data.Dataset) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    xs: List[np.ndarray] = []
    ys: List[int] = []
    qs: List[float] = []
    for i in range(len(ds)):
        x, y, q = ds[i]
        xs.append(x.detach().cpu().numpy().astype(np.float32, copy=False))
        ys.append(int(y.item()))
        qs.append(float(q.item()))
    return np.stack(xs, axis=0), np.asarray(ys, dtype=np.int64), np.asarray(qs, dtype=np.float32)


def run_riccio_combined(
    angles_dir: Path,
    *,
    stem: str,
    out_npz: Path,
    window: int,
    stride: int,
    test_ratio: float,
    val_ratio: float,
    seed: int,
    standardize: bool,
    window_label: str,
) -> Dict[str, Any]:
    angles_dir = angles_dir.expanduser().resolve()
    out_npz = out_npz.expanduser().resolve()
    out_npz.parent.mkdir(parents=True, exist_ok=True)

    train_ds, val_ds, test_ds, class_to_idx, idx_to_class, scale_mean, scale_std = (
        build_kaggle_frame_feature_datasets(
            angles_dir,
            stem=stem,
            window=window,
            stride=stride,
            test_ratio=test_ratio,
            val_ratio=val_ratio,
            seed=seed,
            standardize=standardize,
            window_label=window_label,
        )
    )

    X_tr, y_tr, q_tr = _tensor_window_ds_to_arrays(train_ds)
    X_va, y_va, q_va = _tensor_window_ds_to_arrays(val_ds)
    X_te, y_te, q_te = _tensor_window_ds_to_arrays(test_ds)

    kw: Dict[str, Any] = {
        "X_train": X_tr,
        "y_train": y_tr,
        "Q_train": q_tr,
        "X_val": X_va,
        "y_val": y_va,
        "Q_val": q_va,
        "X_test": X_te,
        "y_test": y_te,
        "Q_test": q_te,
        "window": np.int32(window),
        "stride": np.int32(stride),
        "standardized": np.array([bool(standardize)]),
    }
    if scale_mean is not None and scale_std is not None:
        kw["feature_mean"] = scale_mean.astype(np.float32)
        kw["feature_std"] = scale_std.astype(np.float32)

    np.savez_compressed(out_npz, **kw)

    summary = {
        "workflow": "riccio_combined",
        "angles_dir": str(angles_dir),
        "stem": stem,
        "out_npz": str(out_npz),
        "n_train": int(X_tr.shape[0]),
        "n_val": int(X_va.shape[0]),
        "n_test": int(X_te.shape[0]),
        "feature_dim": int(X_tr.shape[2]) if X_tr.size else 0,
        "window": window,
        "stride": stride,
        "standardize": standardize,
        "class_to_idx": class_to_idx,
        "idx_to_class": {str(k): v for k, v in idx_to_class.items()},
        "window_label": window_label,
    }
    with open(out_npz.with_suffix(".summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    return summary


def _load_index_rows(path: Path) -> List[Dict[str, str]]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def _encode_video_row5(
    video_path: Path,
    *,
    encoder: str,
    max_frames: Optional[int],
    yolo_pose_model: str,
    vitpose_checkpoint: Optional[str],
    vit_device: str,
    resnet_variant: str,
    resnet_device: str,
    bbox_margin: float,
    bilateral_filter: bool,
    bilateral_conf_tau: float,
    bilateral_include_ankles: bool,
    detection_stride: int,
    detection_max_long_edge: int,
) -> Optional[np.ndarray]:
    enc = encoder.strip().lower()
    if enc == "vit":
        from fitness_coach.preprocessing.vit_frame_features import vit_frame_features_from_yolo_video

        out = vit_frame_features_from_yolo_video(
            video_path,
            max_frames,
            yolo_pose_model=str(yolo_pose_model),
            bilateral_filter=bilateral_filter,
            bilateral_conf_tau=float(bilateral_conf_tau),
            bilateral_include_ankles=bilateral_include_ankles,
            detection_stride=int(detection_stride),
            detection_max_long_edge=int(detection_max_long_edge),
            vit_feature_encoder="paper",
            vitpose_checkpoint=vitpose_checkpoint,
            vit_device=str(vit_device),
            bbox_margin=float(bbox_margin),
        )
    elif enc == "resnet":
        from fitness_coach.preprocessing.resnet_frame_features import resnet_frame_features_from_yolo_video

        out = resnet_frame_features_from_yolo_video(
            video_path,
            max_frames,
            yolo_pose_model=str(yolo_pose_model),
            bilateral_filter=bilateral_filter,
            bilateral_conf_tau=float(bilateral_conf_tau),
            bilateral_include_ankles=bilateral_include_ankles,
            detection_stride=int(detection_stride),
            detection_max_long_edge=int(detection_max_long_edge),
            resnet_variant=str(resnet_variant),
            resnet_device=str(resnet_device),
            bbox_margin=float(bbox_margin),
        )
    else:
        raise ValueError(f"unknown encoder {encoder!r}")
    if out is None:
        return None
    fe, _meta = out
    return np.asarray(fe, dtype=np.float32)


def run_index_csv(
    index_csv: Path,
    *,
    features_dir: Path,
    out_npz: Path,
    encoder: str,
    window: int,
    stride: int,
    quality_default: float,
    max_frames: Optional[int],
    yolo_pose_model: str,
    vitpose_checkpoint: Optional[str],
    vit_device: str,
    resnet_variant: str,
    resnet_device: str,
    bbox_margin: float,
    bilateral_filter: bool,
    bilateral_conf_tau: float,
    bilateral_include_ankles: bool,
    detection_stride: int,
    detection_max_long_edge: int,
    standardize: bool,
) -> Dict[str, Any]:
    index_csv = index_csv.expanduser().resolve()
    features_dir = features_dir.expanduser().resolve()
    features_dir.mkdir(parents=True, exist_ok=True)
    out_npz = out_npz.expanduser().resolve()
    out_npz.parent.mkdir(parents=True, exist_ok=True)

    rows = _load_index_rows(index_csv)
    enc = encoder.strip().lower()

    train_w: List[Tuple[np.ndarray, str, float]] = []
    val_w: List[Tuple[np.ndarray, str, float]] = []
    test_w: List[Tuple[np.ndarray, str, float]] = []

    for row in rows:
        stem = (row.get("video_stem") or "").strip()
        vpath = (row.get("video_path") or "").strip()
        cls = (row.get("exercise_class") or "").strip()
        split = (row.get("split") or "train").strip().lower()
        try:
            q = float(row.get("quality", "").strip() or quality_default)
        except ValueError:
            q = float(quality_default)

        if not stem:
            print(f"posepulse_row2: skip row missing video_stem: {row!r}", file=sys.stderr)
            continue
        if not cls:
            print(f"posepulse_row2: skip row missing exercise_class stem={stem}", file=sys.stderr)
            continue

        bio_path = features_dir / f"{stem}_biomechanics.npz"

        if enc in ("vit", "resnet"):
            if not vpath:
                print(f"posepulse_row2: skip stem={stem} (encoder needs video_path)", file=sys.stderr)
                continue
            vp = Path(vpath).expanduser().resolve()
            if not vp.is_file():
                print(f"posepulse_row2: missing video {vp}", file=sys.stderr)
                continue
            fe = _encode_video_row5(
                vp,
                encoder=enc,
                max_frames=max_frames,
                yolo_pose_model=yolo_pose_model,
                vitpose_checkpoint=vitpose_checkpoint,
                vit_device=vit_device,
                resnet_variant=resnet_variant,
                resnet_device=resnet_device,
                bbox_margin=bbox_margin,
                bilateral_filter=bilateral_filter,
                bilateral_conf_tau=bilateral_conf_tau,
                bilateral_include_ankles=bilateral_include_ankles,
                detection_stride=detection_stride,
                detection_max_long_edge=detection_max_long_edge,
            )
            if fe is None or fe.shape[0] == 0:
                print(f"posepulse_row2: encoder failed stem={stem}", file=sys.stderr)
                continue
            np.savez_compressed(
                bio_path,
                frame_features=fe,
                feat_dim=np.int32(int(fe.shape[1])),
            )
        else:
            if not bio_path.is_file():
                print(f"posepulse_row2: missing {bio_path} (use --encoder vit|resnet or export first)", file=sys.stderr)
                continue
            d = np.load(bio_path, allow_pickle=True)
            if "frame_features" not in d.files:
                print(f"posepulse_row2: {bio_path} has no frame_features", file=sys.stderr)
                continue
            fe = np.asarray(d["frame_features"], dtype=np.float32)

        for w_arr in make_windows(fe, window, stride):
            trip = (w_arr.astype(np.float32), cls, q)
            if split == "train":
                train_w.append(trip)
            elif split == "val":
                val_w.append(trip)
            elif split == "test":
                test_w.append(trip)
            else:
                print(f"posepulse_row2: unknown split {split!r} for stem={stem}, using train", file=sys.stderr)
                train_w.append(trip)

    all_cls = sorted({c for _, c, _ in train_w + val_w + test_w})
    if not all_cls:
        raise RuntimeError("No windows produced — check index, videos, and embeddings.")

    def _feat_dim() -> int:
        for bucket in (train_w, val_w, test_w):
            if bucket:
                return int(bucket[0][0].shape[1])
        raise RuntimeError("internal: no windows but all_cls nonempty")

    fdim = _feat_dim()

    class_to_idx = {n: i for i, n in enumerate(all_cls)}
    idx_to_class = {i: n for n, i in class_to_idx.items()}

    def pack(samples: List[Tuple[np.ndarray, str, float]]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if not samples:
            z = np.zeros((0, window, fdim), dtype=np.float32)
            return z, np.zeros((0,), dtype=np.int64), np.zeros((0,), dtype=np.float32)
        X = np.stack([s[0] for s in samples], axis=0)
        y = np.array([class_to_idx[s[1]] for s in samples], dtype=np.int64)
        qq = np.array([s[2] for s in samples], dtype=np.float32)
        return X, y, qq

    X_tr, y_tr, q_tr = pack(train_w)
    X_va, y_va, q_va = pack(val_w)
    X_te, y_te, q_te = pack(test_w)

    scale_mean: Optional[np.ndarray] = None
    scale_std: Optional[np.ndarray] = None
    if standardize and X_tr.shape[0] > 0:
        flat = X_tr.reshape(-1, X_tr.shape[-1])
        scale_mean = flat.mean(axis=0).astype(np.float32)
        scale_std = (flat.std(axis=0) + np.float32(1e-8)).astype(np.float32)

    def maybe_std(X: np.ndarray) -> np.ndarray:
        if X.size == 0 or scale_mean is None or scale_std is None:
            return X
        return ((X - scale_mean) / scale_std).astype(np.float32)

    kw: Dict[str, Any] = {
        "X_train": maybe_std(X_tr),
        "y_train": y_tr,
        "Q_train": q_tr,
        "X_val": maybe_std(X_va),
        "y_val": y_va,
        "Q_val": q_va,
        "X_test": maybe_std(X_te),
        "y_test": y_te,
        "Q_test": q_te,
        "window": np.int32(window),
        "stride": np.int32(stride),
        "standardized": np.array([bool(standardize)]),
    }
    if scale_mean is not None and scale_std is not None:
        kw["feature_mean"] = scale_mean
        kw["feature_std"] = scale_std

    np.savez_compressed(out_npz, **kw)

    summary = {
        "workflow": "index_csv",
        "index_csv": str(index_csv),
        "features_dir": str(features_dir),
        "encoder": enc,
        "out_npz": str(out_npz),
        "n_train": int(X_tr.shape[0]),
        "n_val": int(X_va.shape[0]),
        "n_test": int(X_te.shape[0]),
        "feature_dim": fdim,
        "window": window,
        "stride": stride,
        "standardize": standardize,
        "class_to_idx": class_to_idx,
        "idx_to_class": {str(k): v for k, v in idx_to_class.items()},
    }
    with open(out_npz.with_suffix(".summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    return summary


def main(argv: Optional[List[str]] = None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    ap_root = argparse.ArgumentParser(
        description="PosePulse Row 2: CNN frame embeddings → 30-frame windows → train Z-score.",
    )
    sub = ap_root.add_subparsers(dest="workflow", required=True)

    pr = sub.add_parser("riccio", help="Merged *_biomechanics.npz + *_labels.npz (frame_features + video_id).")
    pr.add_argument("--angles-dir", type=Path, required=True, help="Directory containing stem_biomechanics.npz")
    pr.add_argument("--stem", type=str, default="kaggle_exercise_recognition", help="NPZ stem prefix")
    pr.add_argument("--out", type=Path, required=True, help="Output posepulse_row2_windows.npz")
    pr.add_argument("--window", type=int, default=30)
    pr.add_argument("--stride", type=int, default=15)
    pr.add_argument("--test-ratio", type=float, default=0.15)
    pr.add_argument("--val-ratio", type=float, default=0.15)
    pr.add_argument("--seed", type=int, default=42)
    pr.add_argument("--no-standardize", action="store_true", help="Skip train-split Z-score")
    pr.add_argument(
        "--window-label",
        type=str,
        default="last",
        choices=("first", "last"),
        help="Which frame's coarse pose labels sliding windows (Riccio loader only)",
    )

    pi = sub.add_parser("index", help="Per-video CSV with splits; optional ViTPose/ResNet encode.")
    pi.add_argument("--index-csv", type=Path, required=True)
    pi.add_argument("--features-dir", type=Path, required=True, help="Read/write per-video *_biomechanics.npz")
    pi.add_argument("--out", type=Path, required=True)
    pi.add_argument("--encoder", type=str, default="vit", choices=("vit", "resnet", "skip"))
    pi.add_argument("--window", type=int, default=30)
    pi.add_argument("--stride", type=int, default=15)
    pi.add_argument("--quality-default", type=float, default=0.75)
    pi.add_argument("--max-frames", type=int, default=0, help="Cap frames per video (0 = no cap)")
    pi.add_argument("--yolo-pose-model", type=str, default="yolo26n-pose.pt")
    pi.add_argument("--vitpose-checkpoint", type=str, default="", help="Optional path to ViTPose-S ckpt")
    pi.add_argument("--vit-device", type=str, default="cpu")
    pi.add_argument("--resnet-model", type=str, default="resnet50")
    pi.add_argument("--resnet-device", type=str, default="cpu")
    pi.add_argument("--bbox-margin", type=float, default=0.12)
    pi.add_argument("--bilateral-filter", action="store_true")
    pi.add_argument("--bilateral-conf-tau", type=float, default=0.3)
    pi.add_argument("--bilateral-include-ankles", action="store_true")
    pi.add_argument("--detection-stride", type=int, default=1)
    pi.add_argument("--detection-max-long-edge", type=int, default=0)
    pi.add_argument("--no-standardize", action="store_true")

    args = ap_root.parse_args(argv)

    if args.workflow == "riccio":
        run_riccio_combined(
            args.angles_dir,
            stem=args.stem,
            out_npz=args.out,
            window=int(args.window),
            stride=int(args.stride),
            test_ratio=float(args.test_ratio),
            val_ratio=float(args.val_ratio),
            seed=int(args.seed),
            standardize=not args.no_standardize,
            window_label=str(args.window_label),
        )
        print(json.dumps({"ok": True, "out": str(args.out.expanduser().resolve())}, indent=2))
        return 0

    mf = int(args.max_frames) if args.max_frames else None
    ck = args.vitpose_checkpoint.strip() or None

    run_index_csv(
        args.index_csv,
        features_dir=args.features_dir,
        out_npz=args.out,
        encoder=str(args.encoder),
        window=int(args.window),
        stride=int(args.stride),
        quality_default=float(args.quality_default),
        max_frames=mf,
        yolo_pose_model=str(args.yolo_pose_model),
        vitpose_checkpoint=ck,
        vit_device=str(args.vit_device),
        resnet_variant=str(args.resnet_model),
        resnet_device=str(args.resnet_device),
        bbox_margin=float(args.bbox_margin),
        bilateral_filter=bool(args.bilateral_filter),
        bilateral_conf_tau=float(args.bilateral_conf_tau),
        bilateral_include_ankles=bool(args.bilateral_include_ankles),
        detection_stride=int(args.detection_stride),
        detection_max_long_edge=int(args.detection_max_long_edge),
        standardize=not args.no_standardize,
    )
    print(json.dumps({"ok": True, "out": str(args.out.expanduser().resolve())}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
