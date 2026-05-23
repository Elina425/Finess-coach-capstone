#!/usr/bin/env python3
"""Run EgoExo multi-task xLSTM inference on a **raw video file**.

Pipeline:
    RGB frames (OpenCV + stride/cap, same defaults as training)
        → CLIP ViT-B/32 embeddings (512-d / frame, HuggingFace ``openai/clip-vit-base-patch32``)
        → optional per-dimension standardisation using ``mean`` / ``std`` stored in the checkpoint
        → ``xLSTMExerciseClassifier.infer`` → exercise, quality (trained axis), guidance, comment.

This checkpoint only knows the **12 EgoExo-Fitness** exercise names from training. Other
movements (e.g. barbell curl) are **not in the label set** — the model still picks one of 12,
often wrongly or repeatedly under **out-of-domain** video + CLIP embeddings. JSON fields
``classification_top_k`` and ``classification_entropy`` show whether softmax is peaked or flat.

Usage:
    ./venv/bin/python scripts/infer_multitask_from_video.py \\
        --ckpt results/xlstm_egoexo_multitask_allviews/xlstm_egoexo_multitask_best.pt \\
        --video path/to/clip.mp4

    ./venv/bin/python scripts/infer_multitask_from_video.py \\
        --ckpt ... --video clip.mp4 --json-out results/inference/my_clip.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from fitness_coach.models.xlstm_model import xLSTMExerciseClassifier
from fitness_coach.preprocessing.clip_frame_features import clip_vit_b32_frames_from_video


def load_multitask_model(ckpt_path: Path, device: torch.device) -> Tuple[xLSTMExerciseClassifier, Dict[str, Any]]:
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = xLSTMExerciseClassifier(
        input_size=ck.get("input_size", 512),
        hidden_size=int(ck["hidden"]),
        num_layers=int(ck["layers"]),
        num_classes=len(ck["classes"]),
        dropout=float(ck.get("dropout", 0.15)),
        num_heads=int(ck["num_heads"]),
        conv_kernel_size=int(ck["conv_kernel_size"]),
        projection_factor=float(ck["projection_factor"]),
        num_error_tags=len(ck.get("error_tags", [])) if ck.get("error_tags") else 0,
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
    comment_table: Dict = {}
    for key, text in comment_table_raw.items():
        cls_idx, bucket = key.split("|")
        comment_table[(int(cls_idx), int(bucket))] = str(text)
    bucket_edges = tuple(float(e) for e in (ck.get("comment_quality_bucket_edges") or (0.4, 0.7)))
    q_lo = float(ck.get("quality_domain_lo", 0.0))
    q_hi = float(ck.get("quality_domain_hi", 1.0))
    model.set_comment_table(comment_table, bucket_edges, domain_lo=q_lo, domain_hi=q_hi)

    meta_ckpt = {
        "classes": ck["classes"],
        "mean": ck.get("mean"),
        "std": ck.get("std"),
        "bucket_edges": list(bucket_edges),
        "quality_domain_lo": q_lo,
        "quality_domain_hi": q_hi,
        "quality_encoding": ck.get("quality_encoding"),
    }
    print(
        f"[infer] loaded {ckpt_path.name}: {len(ck['classes'])} classes · "
        f"{len(guidance_table)} guidance · {len(comment_table)} comment cells · "
        f"standardise={'mean/std in ckpt' if meta_ckpt['mean'] is not None else 'no'}",
        flush=True,
    )
    return model, meta_ckpt


def maybe_standardise_clip(feats: np.ndarray, ck_meta: Dict[str, Any]) -> np.ndarray:
    """Apply checkpoint train-set μ/σ per dimension when present."""
    mean = ck_meta.get("mean")
    std = ck_meta.get("std")
    if mean is None or std is None:
        return feats.astype(np.float32, copy=False)
    mu = np.asarray(mean, dtype=np.float32).reshape(1, -1)
    sig = np.asarray(std, dtype=np.float32).reshape(1, -1)
    x = feats.astype(np.float32)
    return ((x - mu) / sig).astype(np.float32)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--ckpt", type=Path, required=True, help="xlstm_egoexo_multitask_best.pt")
    p.add_argument("--video", type=Path, required=True)
    p.add_argument("--device", default=None,
                   help="Torch device for **model** (default: cuda if available else cpu). "
                        "CLIP runs here too unless --clip-device is set.")
    p.add_argument("--clip-device", default=None,
                   help="Separate device for CLIP extraction (e.g. cuda while debugging model on cpu).")
    p.add_argument("--clip-max-frames", type=int, default=300)
    p.add_argument("--clip-subsample-stride", type=int, default=3)
    p.add_argument("--clip-batch-size", type=int, default=16)
    p.add_argument("--json-out", type=Path, default=None)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if not args.video.is_file():
        print(f"infer_multitask_from_video: not a file: {args.video}", file=sys.stderr)
        return 1

    dev_model = args.device
    if dev_model is None:
        dev_model = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(dev_model)

    clip_dev = args.clip_device if args.clip_device is not None else str(device)

    print(f"[infer] extracting CLIP features from {args.video.name} …", flush=True)
    raw = clip_vit_b32_frames_from_video(
        args.video,
        max_frames=args.clip_max_frames,
        subsample_stride=args.clip_subsample_stride,
        device=clip_dev,
        batch_size=args.clip_batch_size,
    )
    if raw is None:
        print("infer_multitask_from_video: CLIP extraction failed.", file=sys.stderr)
        return 1
    feats, clip_meta = raw
    if feats.shape[0] == 0:
        print("infer_multitask_from_video: zero frames after sampling.", file=sys.stderr)
        return 1

    model, ck_meta = load_multitask_model(args.ckpt, device)
    feats_s = maybe_standardise_clip(feats, ck_meta)
    x = torch.from_numpy(feats_s).unsqueeze(0).to(device)

    classes: list = list(ck_meta["classes"])
    with torch.no_grad():
        pooled = model.encode(x)
        a, b, _ = model.fuse(pooled)
        logits = model.class_head(a)
        probs = torch.softmax(logits, dim=-1)[0]
        ent = float((-(probs * (probs.clamp_min(1e-12).log())).sum()).item())
        k = min(12, int(probs.numel()))
        top = torch.topk(probs, k=k)
        classification_top_k = [
            {"class": classes[int(idx)], "prob": float(p)}
            for idx, p in zip(top.indices.tolist(), top.values.tolist())
        ]

        inferred = model.infer(x)[0]
        q01 = float(inferred["quality"])
        qb = inferred.get("quality_bucket")
        out = {
            "exercise": inferred["exercise"],
            "quality": q01,
            "guidance": inferred["guidance"],
            "comment": inferred["comment"],
        }
        if qb is not None:
            out["quality_bucket"] = int(qb)

    payload = {
        "video": str(args.video.resolve()),
        "checkpoint": str(args.ckpt.resolve()),
        "exercise": out["exercise"],
        "predicted_class_confidence": float(
            probs[int(torch.argmax(probs))].item()
        ),
        "classification_entropy": ent,
        "classification_top_k": classification_top_k,
        "quality": q01,
        "quality_1_to_5": 1.0 + 4.0 * q01,
        "quality_bucket": out.get("quality_bucket"),
        "guidance": out["guidance"],
        "comment": out["comment"],
        "frames_used": int(x.shape[1]),
        "feature_dim": int(x.shape[2]),
        "label_space": classes,
        "clip_meta": clip_meta,
    }

    print(json.dumps(payload, indent=2))
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"[infer] wrote {args.json_out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
