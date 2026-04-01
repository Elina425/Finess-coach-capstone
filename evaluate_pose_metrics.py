#!/usr/bin/env python3
"""
Quantitative pose-model comparison on the same videos used for skeleton PNG grids.

The skeleton PNGs do not store keypoint arrays; this script re-runs detectors on video
frames and aggregates metrics.

PCK / MPJPE / OKS require a reference pose. This project has no per-frame ground-truth
labels for QEVD clips, so we use ViTPose (RTMPose) predictions as pseudo–ground truth.
Interpret inter-model gaps as agreement with that reference, not absolute accuracy.

Metrics (per model vs reference):
  - fps: num_frames / total_time  (frames per second; also reports sec_per_frame)
  - pck_0.05: fraction of keypoints with L2 error < 0.05 * max(h, w) in pixels
  - mpjpe: mean per-joint L2 distance in pixels (masked where ref conf > 0.3)
  - oks: mean COCO-style OKS using bbox area from reference keypoints
  - num_keypoints: 17 (COCO)

Usage:
  ./venv/bin/python evaluate_pose_metrics.py --random-split 5 0 --seed 42 --max-frames 80
  ./venv/bin/python evaluate_pose_metrics.py --videos path/a.mp4 path/b.mp4 --max-frames 50
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from pose_estimation_core import iter_default_comparison_detectors

# COCO 17 keypoint sigmas (OKS), see https://cocodataset.org/#keypoints-eval
COCO_SIGMAS = np.array(
    [0.26, 0.25, 0.25, 0.35, 0.35, 0.79, 0.79, 0.72, 0.72, 0.62, 0.62, 1.07, 1.07, 0.87, 0.87, 0.89, 0.89],
    dtype=np.float64,
)


def _collect_mp4_under(folder: Path) -> list[Path]:
    if not folder.is_dir():
        return []
    direct = sorted(folder.glob("*.mp4"))
    if direct:
        return [p for p in direct if p.is_file()]
    return sorted(p for p in folder.rglob("*.mp4") if p.is_file())


def _random_long_short_videos(
    dataset_root: Path,
    n_long: int,
    n_short: int,
    seed: int | None,
) -> list[Path]:
    root = Path(dataset_root)
    long_dir = root / "videos" / "long_range"
    short_dir = root / "videos" / "short_clips"
    long_paths = _collect_mp4_under(long_dir)
    short_paths = _collect_mp4_under(short_dir)
    rng = random.Random(seed)
    out: list[Path] = []
    if n_long > 0 and long_paths:
        k = min(n_long, len(long_paths))
        out.extend(rng.sample(long_paths, k))
    if n_short > 0 and short_paths:
        k = min(n_short, len(short_paths))
        out.extend(rng.sample(short_paths, k))
    return out


def _bbox_area_from_kpts(kp_px: np.ndarray, conf: np.ndarray, vis_thr: float = 0.3) -> float:
    m = conf > vis_thr
    if not np.any(m):
        return float(max(kp_px.shape[0], 1) * 100.0)
    pts = kp_px[m]
    x0, y0 = pts.min(axis=0)
    x1, y1 = pts.max(axis=0)
    w = max(x1 - x0, 1.0)
    h = max(y1 - y0, 1.0)
    return float(w * h)


def compute_pck(
    pred_px: np.ndarray,
    gt_px: np.ndarray,
    threshold: float,
    h: int,
    w: int,
    mask: np.ndarray | None = None,
) -> float:
    """Fraction of joints with pixel L2 distance below threshold * max(h, w)."""
    thr_px = threshold * float(max(h, w))
    d = np.linalg.norm(pred_px - gt_px, axis=-1)
    if mask is None:
        mask = np.ones(pred_px.shape[0], dtype=bool)
    if not np.any(mask):
        return 0.0
    return float(np.mean(d[mask] < thr_px))


def compute_mpjpe(
    pred_px: np.ndarray,
    gt_px: np.ndarray,
    mask: np.ndarray,
) -> float:
    d = np.linalg.norm(pred_px - gt_px, axis=-1)
    if not np.any(mask):
        return float("nan")
    return float(np.mean(d[mask]))


def compute_oks(
    pred_px: np.ndarray,
    gt_px: np.ndarray,
    confidences: np.ndarray,
    area: float,
    sigmas: np.ndarray = COCO_SIGMAS,
) -> float:
    """Mean OKS over visible joints (COCO-style, single person)."""
    d2 = np.sum((pred_px - gt_px) ** 2, axis=-1)
    var = 2 * (sigmas ** 2) * area + 1e-9
    e = np.exp(-d2 / var)
    vis = confidences > 0.3
    if not np.any(vis):
        return float("nan")
    return float(np.mean(e[vis]))


def _norm_to_px(kp_norm: np.ndarray, w: int, h: int) -> np.ndarray:
    out = kp_norm.astype(np.float64).copy()
    out[:, 0] *= w
    out[:, 1] *= h
    return out


def evaluate_video(
    video_path: str,
    detectors: dict[str, Any],
    ref_key: str,
    max_frames: int,
    pck_thr: float,
    vis_thr: float,
) -> dict[str, Any]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {"error": f"cannot open {video_path}"}
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_read = 0

    per_model: dict[str, dict[str, list]] = {
        k: {
            "pck": [],
            "mpjpe": [],
            "oks": [],
            "inference_ms": [],
        }
        for k in detectors
    }

    ref_det = detectors[ref_key]

    while n_read < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        n_read += 1

        t0 = time.perf_counter()
        gt_norm, gt_conf = ref_det.detect(frame)
        t_ref_ms = (time.perf_counter() - t0) * 1000.0
        if gt_norm is None:
            continue
        gt_px = _norm_to_px(gt_norm, w, h)
        area = _bbox_area_from_kpts(gt_px, gt_conf, vis_thr)
        ref_mask = gt_conf > vis_thr

        for key, det in detectors.items():
            if key == ref_key:
                kp_norm, conf = gt_norm, gt_conf
                inf_ms = t_ref_ms
            else:
                t1 = time.perf_counter()
                try:
                    kp_norm, conf = det.detect(frame)
                except Exception:
                    kp_norm, conf = None, None
                inf_ms = (time.perf_counter() - t1) * 1000.0
            per_model[key]["inference_ms"].append(inf_ms)

            if kp_norm is None:
                continue
            pred_px = _norm_to_px(kp_norm, w, h)
            mask = ref_mask & (conf > vis_thr)

            if key == ref_key:
                per_model[key]["pck"].append(1.0)
                per_model[key]["mpjpe"].append(0.0)
                per_model[key]["oks"].append(1.0)
                continue

            pck = compute_pck(pred_px, gt_px, pck_thr, h, w, mask=mask)
            mpj = compute_mpjpe(pred_px, gt_px, mask)
            ok = compute_oks(pred_px, gt_px, conf, area)
            if not np.isnan(mpj):
                per_model[key]["mpjpe"].append(mpj)
            per_model[key]["pck"].append(pck)
            if not np.isnan(ok):
                per_model[key]["oks"].append(ok)

    cap.release()

    out: dict[str, Any] = {"video": str(video_path), "frames_used": n_read, "h": h, "w": w}
    agg: dict[str, Any] = {}
    for key, buckets in per_model.items():
        ims = buckets["inference_ms"]
        total_time_s = sum(ims) / 1000.0 if ims else 0.0
        nf = len(ims)
        fps = (nf / total_time_s) if total_time_s > 0 else 0.0
        sec_per_frame = (total_time_s / nf) if nf > 0 else float("nan")
        agg[key] = {
            "fps": float(fps),
            "sec_per_frame": float(sec_per_frame),
            "num_frames": nf,
            "pck_0.05": float(np.mean(buckets["pck"])) if buckets["pck"] else float("nan"),
            "mpjpe": float(np.mean(buckets["mpjpe"])) if buckets["mpjpe"] else float("nan"),
            "oks": float(np.mean(buckets["oks"])) if buckets["oks"] else float("nan"),
            "num_keypoints": 17,
        }
    out["per_model"] = agg
    return out


def main() -> int:
    p = argparse.ArgumentParser(description="Pose metrics vs ViTPose pseudo-GT on videos.")
    p.add_argument("--videos", nargs="*", default=[], help="Video paths")
    p.add_argument("--dataset-root", default="./qevd-fit-coach-data")
    p.add_argument("--random-split", nargs=2, type=int, metavar=("LONG", "SHORT"), default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-frames", type=int, default=100, help="Max frames per video")
    p.add_argument("--reference", default="vitpose", help="Registry key for pseudo-GT (default: vitpose)")
    p.add_argument("--pck-threshold", type=float, default=0.05, help="PCK threshold as frac of max(h,w)")
    p.add_argument("--output-json", default="", help="Optional path to save aggregated JSON")
    args = p.parse_args()

    paths = [Path(v) for v in args.videos]
    if args.random_split is not None:
        nl, ns = args.random_split
        paths.extend(
            _random_long_short_videos(Path(args.dataset_root), nl, ns, args.seed)
        )
    if not paths:
        print("Provide --videos and/or --random-split LONG SHORT")
        return 1

    detectors: dict[str, Any] = {}
    labels: dict[str, str] = {}
    for key, disp, det in iter_default_comparison_detectors():
        detectors[key] = det
        labels[key] = disp

    ref_key = args.reference
    if ref_key not in detectors:
        print(f"✗ Reference model '{ref_key}' not available. Options: {list(detectors)}")
        return 1

    print(f"Pseudo–ground truth: {labels[ref_key]} ({ref_key})")
    print(f"PCK threshold: {args.pck_threshold} * max(h,w) pixels")
    print(f"Videos: {len(paths)}, max_frames={args.max_frames}\n")

    all_rows: list[dict[str, Any]] = []
    for vp in paths:
        if not vp.is_file():
            print(f"✗ Skip (not a file): {vp}")
            continue
        print(f"Processing {vp} ...")
        row = evaluate_video(
            str(vp),
            detectors,
            ref_key=ref_key,
            max_frames=args.max_frames,
            pck_thr=args.pck_threshold,
            vis_thr=0.3,
        )
        if "error" in row:
            print(f"  ✗ {row['error']}")
            continue
        all_rows.append(row)
        for mk, m in row["per_model"].items():
            if mk == ref_key:
                continue
            print(
                f"  {labels.get(mk, mk):22s}  "
                f"fps={m['fps']:6.2f}  "
                f"PCK@0.05={m['pck_0.05']:.3f}  "
                f"MPJPE={m['mpjpe']:.2f}px  "
                f"OKS={m['oks']:.3f}"
            )

    if not all_rows:
        return 1

    # Mean over videos (macro)
    keys = [k for k in all_rows[0]["per_model"] if k != ref_key]
    summary: dict[str, Any] = {ref_key: {"note": "reference (pseudo-GT)"}}
    for mk in keys:
        summary[mk] = {
            "fps": float(np.nanmean([r["per_model"][mk]["fps"] for r in all_rows])),
            "sec_per_frame": float(
                np.nanmean([r["per_model"][mk]["sec_per_frame"] for r in all_rows])
            ),
            "pck_0.05": float(np.nanmean([r["per_model"][mk]["pck_0.05"] for r in all_rows])),
            "mpjpe": float(np.nanmean([r["per_model"][mk]["mpjpe"] for r in all_rows])),
            "oks": float(np.nanmean([r["per_model"][mk]["oks"] for r in all_rows])),
            "num_keypoints": 17,
        }

    print("\n" + "=" * 72)
    print("MACRO AVERAGE (over videos) — metrics vs", labels[ref_key])
    print("=" * 72)
    for mk in keys:
        s = summary[mk]
        print(
            f"{labels.get(mk, mk):22s}  "
            f"fps={s['fps']:6.2f}  "
            f"sec/frame={s['sec_per_frame']:.4f}  "
            f"PCK@0.05={s['pck_0.05']:.3f}  "
            f"MPJPE={s['mpjpe']:.2f}px  "
            f"OKS={s['oks']:.3f}  "
            f"K={s['num_keypoints']}"
        )

    payload = {
        "reference": ref_key,
        "reference_label": labels[ref_key],
        "pck_threshold_scale": args.pck_threshold,
        "note": "PCK/MPJPE/OKS are vs reference model, not human labels.",
        "per_video": all_rows,
        "macro_average": summary,
    }
    if args.output_json:
        outp = Path(args.output_json)
        outp.parent.mkdir(parents=True, exist_ok=True)
        with open(outp, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"\nWrote {outp.resolve()}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
