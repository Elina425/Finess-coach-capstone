#!/usr/bin/env python3
"""
End-to-end exercise inference pipeline.

Stage 1: MediaPipe keypoint extraction from video
Stage 2: xLSTM exercise classification + quality regression
Stage 3: Rule-based biomechanical feedback from measured joint angles

Usage with a single model (Riccio or any keypoint-trained xLSTM):
  python3 inference_xlstm_complete.py \
    --video path/to/squat.mp4 \
    --model results/xlstm_riccio_mixed/xlstm_keypoints_best.pt

Optional temporal pooling (train with --feature-mode mixed for 42-dim angles+coords):
  --inference-stride 10 --aggregate vote_mode

The model provides classification + quality; feedback comes from deterministic
angle measurements over the extracted keypoints.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from collections import Counter
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch

from fitness_coach.core.biomechanical_features import (
    compute_coords_only_sequence_features,
    compute_mixed_sequence_features,
    compute_sequence_angles,
)
from fitness_coach.datasets.exercise_bilstm_dataset import make_windows
from fitness_coach.inference.feedback_rules import quality_label, select_feedback
from fitness_coach.models.xlstm_model import BottleneckAdapter, xLSTMExerciseClassifier
from fitness_coach.pipelines.batch_compute_angles_for_index import angles_and_keypoints_from_video


def infer_feature_mode(checkpoint: Dict[str, object]) -> str:
    if "feature_mode" in checkpoint:
        fm = str(checkpoint["feature_mode"])
        if fm not in ("clip", "annotation"):
            return fm
    input_size = int(checkpoint.get("input_size", 34))
    if input_size == 8:
        return "angles"
    if input_size == 42:
        return "mixed"
    return "coords"


def build_model_from_checkpoint(checkpoint: Dict[str, object]) -> xLSTMExerciseClassifier:
    classes = list(checkpoint["classes"])
    error_tags = list(checkpoint.get("error_tags", []))
    model = xLSTMExerciseClassifier(
        input_size=int(checkpoint["input_size"]),
        hidden_size=int(checkpoint.get("hidden", 128)),
        num_layers=int(checkpoint.get("layers", 2)),
        num_classes=len(classes),
        dropout=float(checkpoint.get("dropout", 0.1)),
        bidirectional=bool(checkpoint.get("bidirectional", True)),
        num_heads=int(checkpoint.get("num_heads", 4)),
        conv_kernel_size=int(checkpoint.get("conv_kernel_size", 4)),
        projection_factor=float(checkpoint.get("projection_factor", 4.0 / 3.0)),
        num_error_tags=len(error_tags),
        quality_scale=float(checkpoint.get("quality_scale", 1.0)),
        linear_classifier=bool(checkpoint.get("linear_classifier", False)),
        block_pattern=checkpoint.get("block_pattern") if isinstance(checkpoint.get("block_pattern"), str) else None,
    )
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model


def compute_feature_sequence(feature_mode: str, keypoints: np.ndarray, angles: np.ndarray) -> np.ndarray:
    if feature_mode == "angles":
        return angles.astype(np.float32)
    if feature_mode == "coords":
        return compute_coords_only_sequence_features(keypoints, coords_already_normalized=True).astype(np.float32)
    mixed, _ = compute_mixed_sequence_features(keypoints, coords_already_normalized=True)
    return mixed.astype(np.float32)


def _combine_class_probabilities(
    class_probs_stack: np.ndarray,
    aggregate: str,
    classes: Sequence[str],
) -> Tuple[np.ndarray, List[str]]:
    """
    class_probs_stack: (N_windows, N_classes).

    Returns (distribution over classes, per-window argmax class names).
    """
    n_w, n_c = class_probs_stack.shape
    per_idx = [int(np.argmax(class_probs_stack[i])) for i in range(n_w)]
    per_names = [classes[j] for j in per_idx]

    agg = aggregate.lower().strip()
    if agg == "mean_softmax":
        out = class_probs_stack.mean(axis=0)
    elif agg == "median_softmax":
        out = np.median(class_probs_stack, axis=0).astype(np.float64)
        s = float(out.sum())
        if s > 1e-12:
            out = (out / s).astype(np.float32)
        else:
            out = np.ones(n_c, dtype=np.float32) / n_c
    elif agg == "vote_mode":
        counts = Counter(per_idx)
        max_v = max(counts.values()) if counts else 0
        tied = [c for c, v in counts.items() if v == max_v and max_v > 0]
        if not tied:
            winner = int(np.argmax(class_probs_stack.mean(axis=0)))
        elif len(tied) == 1:
            winner = tied[0]
        else:
            mean_p = class_probs_stack.mean(axis=0)
            winner = max(tied, key=lambda c: float(mean_p[c]))
        out = np.zeros(n_c, dtype=np.float32)
        out[winner] = 1.0
    else:
        raise ValueError(f"Unknown aggregate={aggregate!r}; use mean_softmax, median_softmax, vote_mode")

    out = np.asarray(out, dtype=np.float32)
    if agg != "vote_mode":
        s = float(out.sum())
        if s > 1e-12:
            out = out / s
    return out, per_names


def aggregate_model_predictions(
    model: xLSTMExerciseClassifier,
    feature_sequence: np.ndarray,
    checkpoint: Dict[str, object],
    device: torch.device,
    *,
    inference_stride: int = -1,
    aggregate: str = "mean_softmax",
    classes: Optional[Sequence[str]] = None,
) -> Tuple[np.ndarray, float, Optional[np.ndarray], Dict[str, Any]]:
    """
    inference_stride: -1 → use checkpoint training stride; else sliding stride for more/fewer windows.
    aggregate: mean_softmax (legacy), median_softmax, or vote_mode (plurality over per-window argmax).
    """
    window = int(checkpoint.get("window", min(60, feature_sequence.shape[0])))
    if window <= 0:
        window = feature_sequence.shape[0]
    ckpt_stride = int(checkpoint.get("stride", max(1, window // 2)))
    if ckpt_stride <= 0:
        ckpt_stride = max(1, window // 2)
    stride = ckpt_stride if inference_stride < 0 else max(1, int(inference_stride))
    windows = make_windows(feature_sequence, window=window, stride=stride)
    mean = checkpoint.get("mean")
    std = checkpoint.get("std")
    class_probs: List[np.ndarray] = []
    quality_preds: List[float] = []
    error_probs: List[np.ndarray] = []

    for window_features in windows:
        x = np.asarray(window_features, dtype=np.float32)
        if mean is not None and std is not None:
            m = np.asarray(mean, dtype=np.float32).reshape(-1)
            s = np.asarray(std, dtype=np.float32).reshape(-1)
            if m.shape[0] == x.shape[-1]:
                x = (x - m) / s
        xb = torch.from_numpy(x).unsqueeze(0).to(device)
        with torch.no_grad():
            outputs = model(xb)
        if len(outputs) == 2:
            logits, quality = outputs
            err = None
        else:
            logits, quality, err = outputs
        class_probs.append(torch.softmax(logits, dim=1).cpu().numpy()[0])
        quality_preds.append(float(quality.squeeze().cpu().item()))
        if err is not None:
            error_probs.append(torch.sigmoid(err).cpu().numpy()[0])

    cls_list = list(classes) if classes is not None else list(checkpoint["classes"])
    n_classes = len(cls_list)
    if not class_probs:
        uniform = np.ones(n_classes, dtype=np.float32) / max(n_classes, 1)
        meta = {
            "aggregation": aggregate,
            "inference_window": window,
            "inference_stride": stride,
            "checkpoint_stride": ckpt_stride,
            "num_windows": 0,
            "per_window_argmax": [],
            "mean_softmax_probs": {cls_list[i]: round(float(uniform[i]), 6) for i in range(n_classes)},
        }
        return uniform, 0.0, None, meta
    stack = np.stack(class_probs, axis=0)
    combined, per_window_names = _combine_class_probabilities(stack, aggregate, cls_list)
    mean_soft = stack.mean(axis=0)

    avg_quality = float(np.mean(quality_preds)) if quality_preds else 0.0
    avg_error = np.mean(np.stack(error_probs, axis=0), axis=0) if error_probs else None
    meta: Dict[str, Any] = {
        "aggregation": aggregate,
        "inference_window": window,
        "inference_stride": stride,
        "checkpoint_stride": ckpt_stride,
        "num_windows": len(windows),
        "per_window_argmax": per_window_names,
        "mean_softmax_probs": {cls_list[i]: round(float(mean_soft[i]), 6) for i in range(len(cls_list))},
    }
    return combined, avg_quality, avg_error, meta


def create_annotated_video(input_video: Path, output_video: Path, lines: Sequence[str]) -> None:
    cap = cv2.VideoCapture(str(input_video))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(str(output_video), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        for idx, line in enumerate(lines):
            cv2.putText(frame, line, (20, 50 + 40 * idx), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
        writer.write(frame)
    cap.release()
    writer.release()


def main() -> int:
    parser = argparse.ArgumentParser(description="xLSTM exercise inference: classify + quality + feedback")
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True, help="xLSTM checkpoint (e.g. Riccio classifier)")
    parser.add_argument("--personalization-adapter", type=Path, default=None, help="Legacy bottleneck adapter .pt")
    parser.add_argument(
        "--personalization",
        type=Path,
        default=None,
        help="Unified checkpoint (LoRA/DoRA/bottleneck) from personalize_xlstm_kinematics.py or updated saves",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("results/inference"))
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--annotate-video", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument(
        "--inference-stride",
        type=int,
        default=-1,
        help="Sliding-window stride at inference. -1 = use checkpoint stride; smaller values = more overlap.",
    )
    parser.add_argument(
        "--aggregate",
        choices=("mean_softmax", "median_softmax", "vote_mode"),
        default="mean_softmax",
        help="How to combine per-window predictions: mean_softmax (default), median_softmax, or vote_mode (plurality).",
    )
    args = parser.parse_args()

    video_path = args.video.expanduser()
    if not video_path.is_file():
        video_path = (Path.cwd() / args.video).expanduser()
    video_path = video_path.resolve()
    if not video_path.is_file():
        print(f"ERROR: Video not found: {args.video} (resolved: {video_path})", file=sys.stderr)
        return 1

    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")

    print(f"Loading model from {args.model} ...")
    checkpoint = torch.load(args.model, map_location="cpu", weights_only=False)
    model = build_model_from_checkpoint(checkpoint).to(device)
    adapter_meta = None
    if args.personalization:
        from fitness_coach.models.personalization import apply_personalization_from_file

        pl = torch.load(args.personalization, map_location="cpu", weights_only=False)
        apply_personalization_from_file(model, args.personalization, map_location=str(device))
        adapter_meta = {
            "user_id": pl.get("user_id"),
            "personalization": str(args.personalization),
            "method": pl.get("personalization_method", pl.get("method")),
        }
    elif args.personalization_adapter:
        adapter_ckpt = torch.load(args.personalization_adapter, map_location="cpu", weights_only=False)
        adapter = BottleneckAdapter(
            dim=int(adapter_ckpt["hidden_size"]),
            bottleneck_dim=int(adapter_ckpt["adapter_dim"]),
            dropout=float(adapter_ckpt.get("dropout", 0.1)),
        )
        adapter.load_state_dict(adapter_ckpt["adapter"])
        adapter.eval()
        model.attach_adapter(adapter)
        adapter_meta = {"user_id": adapter_ckpt.get("user_id"), "adapter_dim": adapter_ckpt.get("adapter_dim")}

    print(f"Extracting pose from {video_path} ...")
    processed = angles_and_keypoints_from_video(
        video_path,
        args.max_frames if args.max_frames > 0 else None,
        preprocess=True,
        return_keypoints=True,
    )
    if processed is None:
        print(
            "ERROR: Could not extract keypoints from input video (MediaPipe failed or no frames). "
            "Try: python -m pip install --force-reinstall 'mediapipe>=0.10.14,<0.11' "
            "and ensure the file path has no stray space after a line-ending \\.",
            file=sys.stderr,
        )
        return 1
    angles, keypoints, _meta = processed
    if angles.shape[0] == 0 or keypoints is None or keypoints.shape[0] == 0:
        print("ERROR: Empty pose sequence extracted from input video.")
        return 1
    print(f"  Extracted {angles.shape[0]} frames, {keypoints.shape[1]} joints")

    feature_mode = infer_feature_mode(checkpoint)
    features = compute_feature_sequence(feature_mode, keypoints, angles)
    print(f"  Feature mode: {feature_mode}, shape: {features.shape}")

    classes = list(checkpoint["classes"])
    cls_probs, quality_score, error_probs, agg_meta = aggregate_model_predictions(
        model,
        features,
        checkpoint,
        device,
        inference_stride=args.inference_stride,
        aggregate=args.aggregate,
        classes=classes,
    )
    error_tags = list(checkpoint.get("error_tags", []))
    detected_idx = int(np.argmax(cls_probs))
    detected_exercise = classes[detected_idx]
    if args.aggregate == "vote_mode" and agg_meta.get("num_windows", 0) > 0:
        votes = Counter(agg_meta["per_window_argmax"])
        detected_conf = float(votes[detected_exercise] / agg_meta["num_windows"])
    else:
        detected_conf = float(cls_probs[detected_idx])

    feedback = select_feedback(
        detected_exercise,
        quality_score,
        keypoints,
        angles=angles,
        error_tag_names=error_tags if error_probs is not None else None,
        error_probabilities=error_probs.tolist() if error_probs is not None else None,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    result = {
        "video": str(video_path),
        "detected_exercise": detected_exercise,
        "confidence": round(detected_conf, 4),
        "quality_score": round(float(quality_score), 4),
        "quality_label": quality_label(quality_score),
        "feedback": feedback["feedback"],
        "selected_error_tag": feedback["selected_tag"],
        "measurement": feedback["measurement"],
        "all_class_probs": {classes[i]: round(float(cls_probs[i]), 4) for i in range(len(classes))},
        "aggregation": agg_meta.get("aggregation"),
        "inference_stride": agg_meta.get("inference_stride"),
        "num_windows": agg_meta.get("num_windows"),
        "per_window_argmax": agg_meta.get("per_window_argmax"),
        "mean_softmax_probs": agg_meta.get("mean_softmax_probs"),
        "adapter": adapter_meta,
    }
    json_path = args.output_dir / f"{video_path.stem}_results.json"
    json_path.write_text(json.dumps(result, indent=2))

    if args.annotate_video:
        vid_out = args.output_dir / f"{video_path.stem}_annotated.mp4"
        create_annotated_video(
            video_path,
            vid_out,
            [
                f"Detected: {detected_exercise} ({detected_conf:.0%})",
                f'Quality: {quality_score:.2f} -> "{quality_label(quality_score)}"',
                feedback["feedback"],
            ],
        )
        result["annotated_video"] = str(vid_out)
        json_path.write_text(json.dumps(result, indent=2))
        print(f"Annotated video saved to {vid_out}")

    print()
    print(f'Detected: {detected_exercise} ({detected_conf:.0%})')
    print(f'Quality: {quality_score:.2f} -> "{quality_label(quality_score)}"')
    print(f'Feedback: "{feedback["feedback"]}"')
    print(f"Results saved to {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
