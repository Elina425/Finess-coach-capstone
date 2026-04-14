#!/usr/bin/env python3
"""
Compare two Riccio preprocessing runs on the **same** merged keypoint / angle NPZs.

Use when you ran ``riccio_kaggle_video_pipeline.py`` twice (e.g. baseline vs ``--rich-preprocess``)
with the same ``--output-stem`` / video order so row-aligned ``(T, …)`` tensors match.

Metrics (keypoints, normalized torso space after step 3):
  - MAE / RMSE per joint and mean (same units as stored keypoints).
  - NME: RMSE divided by a **torso scale** (median shoulder width) so error is scale-free.
  - Pearson r: per joint (x and y time series averaged) for temporal agreement.
  - Mean absolute **velocity** difference (optional sync / smoothing proxy; not full DTW).

Angles (``*_biomechanics.npz``): MAE, RMSE, Pearson r per angle dimension.

**Not** computed here (need different data or extra deps):
  - PCK vs human GT (use ``evaluate_pose_metrics.py`` vs a reference detector on video).
  - MAE/RMSE vs true occluded joints (needs ground truth).
  - Full DTW (use ``dtaidistance`` or subsample + ``scipy``); ICC across subjects (needs repeated measures design).

Usage:
  ./venv/bin/python compare_preprocessing_npz.py \\
    --baseline-kp results/run_a/riccio_realtime_exercise_recognition_keypoints.npz \\
    --alternate-kp results/run_b/riccio_realtime_exercise_recognition_keypoints.npz \\
    --baseline-ang results/run_a/riccio_realtime_exercise_recognition_biomechanics.npz \\
    --alternate-ang results/run_b/riccio_realtime_exercise_recognition_biomechanics.npz
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def _load_kp(path: Path) -> np.ndarray:
    d = np.load(path)
    if "keypoints" not in d:
        raise KeyError(f"{path} missing 'keypoints' array")
    return np.asarray(d["keypoints"], dtype=np.float64)


def _load_angles(path: Path) -> np.ndarray:
    d = np.load(path)
    if "angles" not in d:
        raise KeyError(f"{path} missing 'angles' array")
    return np.asarray(d["angles"], dtype=np.float64)


def _torso_scale_shoulder(kp: np.ndarray) -> float:
    """Median shoulder width (COCO 5–6)."""
    if kp.shape[1] < 7:
        return 1.0
    d = np.linalg.norm(kp[:, 5] - kp[:, 6], axis=-1)
    s = float(np.nanmedian(d))
    return s if s > 1e-8 else 1.0


def _pearson_per_joint(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """(J,) correlation per joint over time (x,y combined as mean of r_x, r_y)."""
    j = a.shape[1]
    out = np.zeros(j, dtype=np.float64)
    for ji in range(j):
        ax, ay = a[:, ji, 0], a[:, ji, 1]
        bx, by = b[:, ji, 0], b[:, ji, 1]
        def _r(x, y):
            if np.std(x) < 1e-12 or np.std(y) < 1e-12:
                return np.nan
            return float(np.corrcoef(x, y)[0, 1])

        rx = _r(ax, bx)
        ry = _r(ay, by)
        out[ji] = float(np.nanmean([rx, ry]))
    return out


def compare_keypoints(k_a: np.ndarray, k_b: np.ndarray) -> dict:
    t = min(len(k_a), len(k_b))
    a = k_a[:t]
    b = k_b[:t]
    diff = a - b
    mae = float(np.mean(np.abs(diff)))
    rmse = float(np.sqrt(np.mean(diff**2)))
    scale = _torso_scale_shoulder(a)
    nme = rmse / scale
    pj = _pearson_per_joint(a, b)
    vel_a = np.diff(a, axis=0)
    vel_b = np.diff(b, axis=0)
    vel_diff = float(np.mean(np.linalg.norm(vel_a - vel_b, axis=-1)))

    # Per-joint L2 RMSE over time: (17,)
    pj_rmse = np.sqrt(np.mean(np.sum(diff**2, axis=-1), axis=0))

    return {
        "frames_compared": int(t),
        "mae_mean": mae,
        "rmse_mean": rmse,
        "nme_torso_norm_rmse": nme,
        "torso_scale_shoulder_median": scale,
        "mean_pearson_joint": float(np.nanmean(pj)),
        "pearson_per_joint": pj.tolist(),
        "rmse_per_joint_l2": pj_rmse.tolist(),
        "mean_abs_velocity_diff": vel_diff,
    }


def compare_angles(a: np.ndarray, b: np.ndarray) -> dict:
    """
    Angle streams often contain NaN when joint geometry is degenerate (undefined angle).
    Metrics use only pairs of finite values per element; rows with any NaN in a or b
    are skipped for global MAE/RMSE (element-wise mask).
    """
    t = min(len(a), len(b))
    aa = a[:t].astype(np.float64)
    bb = b[:t].astype(np.float64)
    both = np.isfinite(aa) & np.isfinite(bb)
    diff = aa - bb
    abs_err = np.abs(diff)
    mae = float(np.nanmean(np.where(both, abs_err, np.nan)))
    rmse = float(np.sqrt(np.nanmean(np.where(both, diff**2, np.nan))))

    per_dim = []
    for j in range(aa.shape[1]):
        m = both[:, j]
        if not np.any(m):
            per_dim.append(float("nan"))
        else:
            d = aa[m, j] - bb[m, j]
            per_dim.append(float(np.sqrt(np.mean(d**2))))

    n_pair = int(np.sum(both))
    n_frames_any_nan = int(t - np.sum(np.all(both, axis=1)))

    return {
        "frames_compared": int(t),
        "finite_angle_pairs": n_pair,
        "frames_with_any_nan_either_run": n_frames_any_nan,
        "mae": mae,
        "rmse": rmse,
        "per_dim_rmse": per_dim,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Compare two Riccio NPZ preprocessing outputs.")
    ap.add_argument("--baseline-kp", type=Path, required=True)
    ap.add_argument("--alternate-kp", type=Path, required=True)
    ap.add_argument("--baseline-ang", type=Path, default=None)
    ap.add_argument("--alternate-ang", type=Path, default=None)
    ap.add_argument("--json-out", type=Path, default=None, help="Write metrics JSON here")
    args = ap.parse_args()

    k_a = _load_kp(args.baseline_kp)
    k_b = _load_kp(args.alternate_kp)
    if k_a.ndim != 3 or k_b.ndim != 3:
        print("Expected keypoints (T,17,2)", file=sys.stderr)
        return 1

    kp_metrics = compare_keypoints(k_a, k_b)
    out: dict = {"keypoints": kp_metrics}

    if args.baseline_ang and args.alternate_ang:
        a_a = _load_angles(args.baseline_ang)
        a_b = _load_angles(args.alternate_ang)
        out["angles"] = compare_angles(a_a, a_b)

    print(json.dumps(out, indent=2))
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        with open(args.json_out, "w") as f:
            json.dump(out, f, indent=2)
        print(f"Wrote {args.json_out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
