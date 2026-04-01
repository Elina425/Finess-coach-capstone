#!/usr/bin/env python3
"""
Run trained BiLSTM on a new video: sliding 30-frame windows over angles →
mean exercise probabilities + mean quality score.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

from batch_compute_angles_for_index import angles_from_video, mixed_features_from_video
from exercise_bilstm_model import ExerciseBiLSTM


@torch.no_grad()
def predict_video(
    ckpt_path: Path,
    video_path: Path,
    window: int = 30,
    stride: int = 15,
    max_frames: int = 0,
) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    except TypeError:
        ckpt = torch.load(ckpt_path, map_location=device)
    feat_dim = int(ckpt.get("feat_dim", 8))
    num_classes = int(ckpt["num_classes"])
    classes: list = ckpt.get("classes", [])
    feature_mode = str(ckpt.get("feature_mode", "angles"))
    hidden = int(ckpt.get("hidden", 128))
    num_layers = int(ckpt.get("layers", 2))
    dropout = float(ckpt.get("dropout", 0.35))
    scale_mean = ckpt.get("scale_mean")
    scale_std = ckpt.get("scale_std")
    stride = int(ckpt.get("stride", stride))
    window = int(ckpt.get("window", window))

    model = ExerciseBiLSTM(
        input_dim=feat_dim,
        num_classes=num_classes,
        hidden=hidden,
        num_layers=num_layers,
        dropout=dropout,
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    mf = max_frames if max_frames > 0 else None
    if feature_mode == "mixed":
        ang = mixed_features_from_video(video_path, mf)
    else:
        ang = angles_from_video(video_path, mf)
    if ang is None:
        print("Could not extract pose features", file=sys.stderr)
        return

    T = ang.shape[0]
    ang = np.nan_to_num(np.asarray(ang, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    if T < window:
        pad = np.zeros((window - T, ang.shape[1]), dtype=np.float32)
        ang = np.vstack([ang, pad])
        windows = [ang]
    else:
        windows = []
        for s in range(0, T - window + 1, stride):
            windows.append(ang[s : s + window].copy())

    if not windows:
        windows = [ang[-window:].copy()]

    sm = np.asarray(scale_mean, dtype=np.float32) if scale_mean is not None else None
    ss = np.asarray(scale_std, dtype=np.float32) if scale_std is not None else None

    probs_sum = np.zeros(num_classes, dtype=np.float64)
    q_sum = 0.0
    for w in windows:
        if sm is not None and ss is not None and sm.shape[0] == w.shape[1]:
            w = (w - sm) / ss
        x = torch.from_numpy(w).float().unsqueeze(0).to(device)
        logits, q = model(x)
        p = torch.softmax(logits, dim=1).cpu().numpy()[0]
        probs_sum += p
        q_sum += float(q.cpu().item())

    n = len(windows)
    probs = probs_sum / n
    top = int(np.argmax(probs))
    print("Predicted exercise:", classes[top] if top < len(classes) else top, f"(p={probs[top]:.3f})")
    print("Mean quality estimate:", q_sum / n)
    print("Top-3 classes:")
    order = np.argsort(-probs)[:3]
    for j in order:
        print(f"  {classes[j] if j < len(classes) else j}: {probs[j]:.3f}")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="./results/exercise_bilstm/exercise_bilstm_best.pt")
    p.add_argument("--video", required=True)
    p.add_argument("--window", type=int, default=30)
    p.add_argument("--stride", type=int, default=15)
    p.add_argument("--max-frames", type=int, default=0)
    args = p.parse_args()
    predict_video(
        Path(args.checkpoint),
        Path(args.video),
        window=args.window,
        stride=args.stride,
        max_frames=args.max_frames,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
