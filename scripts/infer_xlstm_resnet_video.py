#!/usr/bin/env python3
"""Raw-video inference with **paper xLSTM trained on ResNet-50 crop embeddings** (Riccio 4-class).

Pipeline (matches ``riccio_kaggle_video_pipeline`` / ``train_paper_classification.py`` frame mode):

    Video → YOLO26 pose → torso-normalised crops → frozen torchvision **ResNet-50** GAP features
    (2048-D / frame) → per-window train-split **standardisation** (μ, σ recomputed from the same
    Kaggle NPZ folder used at training time — they are **not** stored inside ``best.pt``) →
    xLSTM → **squat / push-up / shoulder press / barbell biceps curl**.

Example::

    ./venv/bin/python scripts/infer_xlstm_resnet_video.py \\
        --ckpt results/paper_xlstm_seq60_resnet/xlstm_7_1/best.pt \\
        --video data/raw_clips/b1_bicepcurl_1.mp4 \\
        --json-out results/inference/bicep_xlstm_resnet.json

Override NPZ directory only when your files are not under the checkpoint's default::

    --kaggle-angles-dir /ABS/PATH/to/folder/with/riccio_*_biomechanics.npz
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from fitness_coach.datasets.exercise_bilstm_dataset import build_kaggle_frame_feature_datasets
from fitness_coach.models.xlstm_model import xLSTMExerciseClassifier
from fitness_coach.preprocessing.resnet_frame_features import resnet_frame_features_from_yolo_video

from inference_xlstm_complete import aggregate_model_predictions


def _parse_exclude(raw: Optional[str]) -> Optional[List[str]]:
    t = (raw or "").strip()
    if not t:
        return None
    return [p.strip() for p in t.split(",") if p.strip()]


def load_paper_resnet_xlstm(
    ckpt_path: Path,
    device: torch.device,
) -> Tuple[xLSTMExerciseClassifier, Dict[str, Any], Sequence[str]]:
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    ad = ck.get("args") or {}

    feature_dim = int(ad.get("feature_dim", 2048))
    classes = ck.get("classes") or ad.get("classes") or [
        "barbell biceps curl",
        "push-up",
        "shoulder press",
        "squat",
    ]
    if isinstance(classes, dict):
        classes = [classes[i] for i in sorted(classes.keys())]
    classes = list(classes)

    model = xLSTMExerciseClassifier(
        input_size=feature_dim,
        hidden_size=int(ad.get("xlstm_hidden", 256)),
        num_layers=8,
        num_classes=len(classes),
        dropout=float(ad.get("dropout", 0.25)),
        bidirectional=True,
        num_heads=int(ad.get("xlstm_num_heads", 4)),
        conv_kernel_size=int(ad.get("xlstm_conv_kernel_size", 4)),
        projection_factor=float(ad.get("xlstm_projection_factor", 4.0 / 3.0)),
        num_error_tags=0,
        quality_scale=1.0,
        block_pattern=str(ad.get("xlstm_block_pattern", "mmmmmmms")),
        use_attention_pool=bool(ad.get("xlstm_attention_pool", False)),
        temporal_pool=str(ad.get("xlstm_pool", "mean")),
        input_dropout=float(ad.get("xlstm_input_dropout", 0.0)),
        linear_classifier=bool(ad.get("xlstm_linear_classifier", False)),
        use_fusion=False,
    ).to(device)
    model.load_state_dict(ck["model"], strict=False)
    model.eval()
    return model, ad, classes


def fit_standardizer_like_training(ad: Dict[str, Any], base_override: Optional[Path]) -> Tuple[np.ndarray, np.ndarray]:
    """Rebuild μ, σ from **training windows only** (same logic as ``train_paper_classification``)."""
    base = Path(base_override or ad.get("kaggle_angles_dir") or "")
    stem_hint = str(ad.get("kaggle_stem", "riccio_realtime_exercise_recognition"))
    if not base.is_dir():
        raise FileNotFoundError(
            f"kaggle angles dir not found: {base.resolve()}\n"
            "Use the folder that contains "
            f"'{stem_hint}_biomechanics.npz' (and '{stem_hint}_labels.npz').\n"
            "If you copied `/path/to/riccio_resnet50_features` from an example, replace it "
            "with a real path — or omit `--kaggle-angles-dir` entirely so the checkpoint "
            "default is used (typically `results/riccio_resnet50_features` under the repo)."
        )
    stem = stem_hint
    wl = str(ad.get("kaggle_window_label") or "first")
    _, _, _, _, _, mean, std = build_kaggle_frame_feature_datasets(
        base.resolve(),
        stem=stem,
        window=int(ad["seq_len"]),
        stride=int(ad["stride"]),
        test_ratio=float(ad.get("kaggle_test_ratio", 0.15)),
        val_ratio=float(ad.get("kaggle_val_ratio", 0.15)),
        seed=int(ad.get("kaggle_seed", 42)),
        standardize=True,
        window_label=wl,
        exclude_coarse_classes=_parse_exclude(ad.get("exclude_classes")),
    )
    if mean is None or std is None:
        raise RuntimeError("standardizer is None — empty train windows from NPZ?")
    return np.asarray(mean, dtype=np.float32), np.asarray(std, dtype=np.float32)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument(
        "--ckpt",
        type=Path,
        default=Path("results/paper_xlstm_seq60_resnet/xlstm_7_1/best.pt"),
        help="Paper ``best.pt`` (model + args) from train_paper_classification.py",
    )
    p.add_argument("--video", type=Path, required=True)
    p.add_argument("--json-out", type=Path, default=None)
    p.add_argument("--device", default=None, help="cuda | cpu | mps (default: cuda if available)")
    p.add_argument(
        "--kaggle-angles-dir",
        type=Path,
        default=None,
        help="Override ``args.kaggle_angles_dir`` from checkpoint for μ/σ recomputation.",
    )
    p.add_argument("--kaggle-stem", type=str, default=None, help="Override NPZ stem (default from ckpt args).")
    p.add_argument("--yolo-pose-model", type=str, default="yolo26n-pose.pt")
    p.add_argument("--resnet-variant", type=str, default="resnet50")
    p.add_argument("--bilateral-filter", action="store_true")
    p.add_argument("--max-detection-frames", type=int, default=0,
                   help="Cap pose/ResNet frames (0 = full video at pipeline FPS sync).")
    p.add_argument(
        "--inference-stride",
        type=int,
        default=-1,
        help="-1 → use checkpoint stride; else sliding-window stride at inference.",
    )
    p.add_argument(
        "--aggregate",
        choices=("mean_softmax", "median_softmax", "vote_mode"),
        default="mean_softmax",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    video = args.video.expanduser().resolve()
    if not video.is_file():
        print(f"infer_xlstm_resnet_video: missing video {video}", file=sys.stderr)
        return 1

    dev = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(dev)

    ckpt_path = args.ckpt.expanduser()
    if not ckpt_path.is_file():
        ckpt_path = (REPO_ROOT / args.ckpt).resolve()
    if not ckpt_path.is_file():
        print(f"infer_xlstm_resnet_video: missing ckpt {args.ckpt}", file=sys.stderr)
        return 1

    print("[infer] loading xLSTM (paper ResNet checkpoint) …", flush=True)
    model, ad, classes = load_paper_resnet_xlstm(ckpt_path, device)

    ad_use = dict(ad)
    if args.kaggle_stem:
        ad_use["kaggle_stem"] = args.kaggle_stem

    print("[infer] recomputing train-split μ/σ from Kaggle NPZ (matches training) …", flush=True)
    mean, std = fit_standardizer_like_training(ad_use, args.kaggle_angles_dir)

    max_frames = None if args.max_detection_frames <= 0 else int(args.max_detection_frames)

    print(f"[infer] YOLO→ResNet({args.resnet_variant}) features from {video.name} …", flush=True)
    raw = resnet_frame_features_from_yolo_video(
        video,
        max_frames,
        yolo_pose_model=str(args.yolo_pose_model),
        bilateral_filter=bool(args.bilateral_filter),
        resnet_variant=str(args.resnet_variant),
        resnet_device=str(device),
    )
    if raw is None:
        print(
            "infer_xlstm_resnet_video: feature extraction failed "
            "(install ultralytics + torchvision; check video codec).",
            file=sys.stderr,
        )
        return 1
    feats, meta = raw
    seq_len = int(ad_use["seq_len"])
    if feats.shape[0] < seq_len:
        print(
            f"infer_xlstm_resnet_video: only T={feats.shape[0]} frames — "
            f"need ≥ seq_len={seq_len}",
            file=sys.stderr,
        )
        return 1

    pseudo_ckpt: Dict[str, Any] = {
        "classes": list(classes),
        "window": seq_len,
        "stride": int(ad_use["stride"]),
        "mean": mean,
        "std": std,
    }

    cls_probs, quality_score, _, agg_meta = aggregate_model_predictions(
        model,
        feats.astype(np.float32),
        pseudo_ckpt,
        device,
        inference_stride=args.inference_stride,
        aggregate=args.aggregate,
        classes=classes,
    )
    detected_idx = int(np.argmax(cls_probs))
    detected = classes[detected_idx]
    conf = float(cls_probs[detected_idx])

    payload: Dict[str, Any] = {
        "video": str(video),
        "checkpoint": str(ckpt_path.resolve()),
        "detected_exercise": detected,
        "confidence": round(conf, 6),
        "quality_score": round(float(quality_score), 6),
        "all_class_probs": {classes[i]: round(float(cls_probs[i]), 6) for i in range(len(classes))},
        "aggregation": agg_meta,
        "resnet_feature_meta": meta,
        "standardizer": {
            "kaggle_angles_dir": str(Path(args.kaggle_angles_dir or ad_use["kaggle_angles_dir"]).resolve()),
            "kaggle_stem": str(ad_use.get("kaggle_stem")),
            "seq_len": seq_len,
            "stride": int(ad_use["stride"]),
        },
    }

    print(json.dumps(payload, indent=2))
    if args.json_out is not None:
        outp = args.json_out.expanduser()
        outp.parent.mkdir(parents=True, exist_ok=True)
        outp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"[infer] wrote {outp}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
