#!/usr/bin/env python3
"""
Biomechanical features from 2D pose keypoints (COCO-17).

**Expect preprocessed keypoints** (e.g. after ``apply_keypoint_preprocessing_pipeline``:
torso normalization, imputation, FPS sync; optionally bone proportion or graph-Laplacian
spatial imputation). See ``docs/CAPSTONE_PIPELINE.md`` §3–4.

Joint angles are computed in the image plane from three points (vertex at the joint).
They are invariant to **translation** and **uniform scaling** of the skeleton, which is
why they are preferred over raw (x, y) for coaching and movement analysis from monocular
video — the person can stand at different depths or lateral positions while limb geometry
stays comparable (up to perspective / out-of-plane rotation limits).

Typical references:
  • Grood & Suntay (1983) — joint coordinate systems (3D); 2D projections use planar angles.
  • Pose-estimation fitness pipelines (e.g., MediaPipe / BlazePose + angle thresholds).
  • Winter, "Biomechanics and Motor Control of Human Movement" — segment angles from landmarks.

This module uses **planar angles** between 2D vectors (a−b) and (c−b) at vertex b.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

# COCO-17 indices (same as pose_estimation_core.COCO_KEYPOINTS.NAMES)
NOSE, L_EYE, R_EYE, L_EAR, R_EAR = 0, 1, 2, 3, 4
L_SH, R_SH, L_EL, R_EL, L_WR, R_WR = 5, 6, 7, 8, 9, 10
L_HIP, R_HIP, L_KN, R_KN, L_ANK, R_ANK = 11, 12, 13, 14, 15, 16


def angle_degrees_2d(
    a: np.ndarray,
    b: np.ndarray,
    c: np.ndarray,
) -> float:
    """
    Interior angle at vertex b (degrees, 0–180) for points a, b, c in 2D.
    Vectors: (a−b) and (c−b).
    """
    a = np.asarray(a, dtype=np.float64).reshape(2)
    b = np.asarray(b, dtype=np.float64).reshape(2)
    c = np.asarray(c, dtype=np.float64).reshape(2)
    ba = a - b
    bc = c - b
    n1 = float(np.linalg.norm(ba))
    n2 = float(np.linalg.norm(bc))
    if n1 < 1e-10 or n2 < 1e-10:
        return float("nan")
    cosang = float(np.clip(np.dot(ba, bc) / (n1 * n2), -1.0, 1.0))
    return float(np.degrees(np.arccos(cosang)))


def _valid_point(p: np.ndarray) -> bool:
    p = np.asarray(p, dtype=np.float64).reshape(-1)
    if p.size < 2:
        return False
    if not np.all(np.isfinite(p[:2])):
        return False
    # Treat (0,0) as missing after imputation conventions
    if abs(p[0]) < 1e-12 and abs(p[1]) < 1e-12:
        return False
    return True


def compute_frame_angles(kp_17x2: np.ndarray) -> Dict[str, float]:
    """
    Compute named joint angles for one frame.

    Angles (vertex in the middle):
      - left_elbow, right_elbow: shoulder–elbow–wrist
      - left_knee, right_knee: hip–knee–ankle
      - left_hip, right_hip: shoulder–hip–knee (hip flexion / opening in the sagittal-like 2D view)
      - left_shoulder, right_shoulder: elbow–shoulder–hip (arm elevation relative to trunk)
    """
    kp = np.asarray(kp_17x2, dtype=np.float64)
    if kp.shape != (17, 2):
        raise ValueError(f"Expected keypoints shape (17, 2), got {kp.shape}")

    def triplet(
        ia: int,
        iv: int,
        ic: int,
    ) -> float:
        a, b, c = kp[ia], kp[iv], kp[ic]
        if not (_valid_point(a) and _valid_point(b) and _valid_point(c)):
            return float("nan")
        return angle_degrees_2d(a, b, c)

    return {
        "left_elbow": triplet(L_SH, L_EL, L_WR),
        "right_elbow": triplet(R_SH, R_EL, R_WR),
        "left_knee": triplet(L_HIP, L_KN, L_ANK),
        "right_knee": triplet(R_HIP, R_KN, R_ANK),
        "left_hip": triplet(L_SH, L_HIP, L_KN),
        "right_hip": triplet(R_SH, R_HIP, R_KN),
        "left_shoulder": triplet(L_EL, L_SH, L_HIP),
        "right_shoulder": triplet(R_EL, R_SH, R_HIP),
    }


ANGLE_FEATURE_NAMES: Tuple[str, ...] = (
    "left_elbow",
    "right_elbow",
    "left_knee",
    "right_knee",
    "left_hip",
    "right_hip",
    "left_shoulder",
    "right_shoulder",
)


def _normalize_skeleton_xy_coco17(keypoints: np.ndarray) -> np.ndarray:
    """
    Pelvis-centered, shoulder-width–scaled 2D coords (T, 17, 2).
    Reduces sensitivity to camera distance / position vs raw pixels (cf. mixed features in
    Riccio, "Real-Time Fitness Exercise Classification…", arXiv:2411.11548).
    """
    kp = np.asarray(keypoints, dtype=np.float64)
    if kp.ndim != 3 or kp.shape[1:] != (17, 2):
        raise ValueError(f"Expected (T, 17, 2), got {kp.shape}")
    mid = (kp[:, L_HIP, :] + kp[:, R_HIP, :]) / 2.0
    sh = np.linalg.norm(kp[:, L_SH] - kp[:, R_SH], axis=1)
    scale = float(np.nanmedian(np.where(sh > 1e-6, sh, np.nan)))
    if not np.isfinite(scale) or scale < 1e-6:
        scale = 1.0
    out = (kp - mid[:, None, :]) / scale
    return out.astype(np.float32)


def compute_coords_only_sequence_features(
    keypoints: np.ndarray,
    *,
    coords_already_normalized: bool = False,
) -> np.ndarray:
    """
    Pelvis-centered, scale-normalized (x,y) per frame, flattened to (T, 34).
    Ablation: coordinates without joint angles.

    Set ``coords_already_normalized=True`` when keypoints already went through
    ``apply_keypoint_preprocessing_pipeline`` (torso / hip-centered normalization in
    ``PosePreprocessor.skeleton_based_normalize``) so coordinates are not scaled twice.
    """
    kp = np.asarray(keypoints, dtype=np.float64)
    if coords_already_normalized:
        nk = kp.astype(np.float32)
    else:
        nk = _normalize_skeleton_xy_coco17(kp)
    return nk.reshape(nk.shape[0], -1).astype(np.float32)


def compute_mixed_sequence_features(
    keypoints: np.ndarray,
    *,
    coords_already_normalized: bool = False,
) -> Tuple[np.ndarray, Tuple[str, ...]]:
    """
    Concatenate joint angles (T, 8) with normalized (x, y) for all 17 COCO joints (T, 34)
    → (T, 42). Mirrors the paper's "angles + coordinates" mixed input (78-dim there; we use
    COCO-17 + 8 angles = 42).

    Use ``coords_already_normalized=True`` for pipeline-preprocessed ``*_keypoints.npz`` files
    (skeleton-based normalization already applied before feature extraction).
    """
    angles, angle_names = compute_sequence_angles(keypoints)
    kp = np.asarray(keypoints, dtype=np.float64)
    if coords_already_normalized:
        nk = kp.astype(np.float32)
    else:
        nk = _normalize_skeleton_xy_coco17(kp)
    T = nk.shape[0]
    flat = nk.reshape(T, -1)
    ang = np.nan_to_num(angles.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    mixed = np.concatenate([ang, flat], axis=1)
    extra = tuple(f"norm_xy_{i}" for i in range(flat.shape[1]))
    return mixed, angle_names + extra


def compute_sequence_angles(keypoints: np.ndarray) -> Tuple[np.ndarray, Tuple[str, ...]]:
    """
    keypoints: (T, 17, 2) — pixel or normalized coordinates (angles are scale-invariant in 2D).

    Returns:
      angles: (T, 8) with columns in ANGLE_FEATURE_NAMES order; NaN if joint triplets invalid.
    """
    kp = np.asarray(keypoints, dtype=np.float64)
    if kp.ndim != 3 or kp.shape[1:] != (17, 2):
        raise ValueError(f"Expected (T, 17, 2), got {kp.shape}")
    T = kp.shape[0]
    out = np.full((T, len(ANGLE_FEATURE_NAMES)), np.nan, dtype=np.float64)
    for t in range(T):
        d = compute_frame_angles(kp[t])
        for i, name in enumerate(ANGLE_FEATURE_NAMES):
            out[t, i] = d[name]
    return out, ANGLE_FEATURE_NAMES


def angles_to_dict_per_frame(
    angles: np.ndarray,
    names: Sequence[str] = ANGLE_FEATURE_NAMES,
) -> List[Dict[str, float]]:
    """Convert (T, K) array to list of dicts for JSON export."""
    rows: List[Dict[str, float]] = []
    for t in range(angles.shape[0]):
        rows.append({names[i]: float(angles[t, i]) for i in range(len(names))})
    return rows


def keypoints_npz_coords_already_normalized(data: object) -> bool:
    """
    True when ``*_keypoints.npz`` was saved after ``apply_keypoint_preprocessing_pipeline``
    with skeleton-based torso normalization (step 3 before angle / mixed features).
    """
    try:
        if "techniques_json" not in data:
            return False
        raw = data["techniques_json"]
        if isinstance(raw, np.ndarray):
            raw = raw.item()
        if isinstance(raw, bytes):
            raw = raw.decode()
        tj = json.loads(raw) if isinstance(raw, str) else {}
        return tj.get("normalization") == "skeleton-based-torso"
    except Exception:
        return False


def load_keypoints_npz(path: Union[str, Path]) -> np.ndarray:
    data = np.load(path, allow_pickle=True)
    if "keypoints" in data:
        return np.asarray(data["keypoints"], dtype=np.float64)
    raise KeyError(f"No 'keypoints' array in {path}")


def process_npz_file(
    npz_in: Path,
    npz_out: Optional[Path] = None,
    json_out: Optional[Path] = None,
) -> Dict[str, Union[str, int, List[str]]]:
    """
    Load *_keypoints.npz, compute angles, save *_biomechanics.npz (+ optional summary JSON).
    """
    kp = load_keypoints_npz(npz_in)
    angles, names = compute_sequence_angles(kp)
    stem = npz_in.stem.replace("_keypoints", "")
    out_dir = npz_in.parent
    if npz_out is None:
        npz_out = out_dir / f"{stem}_biomechanics.npz"
    if json_out is None:
        json_out = out_dir / f"{stem}_biomechanics_summary.json"

    np.savez_compressed(
        npz_out,
        angles=angles,
        angle_names=np.array(names, dtype=object),
        num_frames=np.array([angles.shape[0]]),
        source_npz=str(npz_in.name),
    )

    valid = np.isfinite(angles)
    summary = {
        "source_npz": str(npz_in.name),
        "output_npz": str(npz_out.name),
        "num_frames": int(angles.shape[0]),
        "angle_names": list(names),
        "per_angle_mean_deg": {
            names[i]: float(np.nanmean(angles[:, i])) for i in range(len(names))
        },
        "per_angle_std_deg": {
            names[i]: float(np.nanstd(angles[:, i])) for i in range(len(names))
        },
        "fraction_valid": float(np.mean(valid)),
    }
    with open(json_out, "w") as f:
        json.dump(summary, f, indent=2)

    return summary


def batch_process_directory(
    input_dir: Union[str, Path],
    pattern: str = "*_keypoints.npz",
) -> List[Dict]:
    """Process all keypoint NPZ files in a directory."""
    d = Path(input_dir)
    results = []
    for p in sorted(d.glob(pattern)):
        if not p.is_file():
            continue
        try:
            results.append(process_npz_file(p))
        except Exception as e:
            results.append({"source_npz": str(p.name), "error": str(e)})
    return results


def main() -> int:
    import argparse
    import sys

    ap = argparse.ArgumentParser(
        description="Compute joint-angle biomechanical features from processed keypoint NPZ files."
    )
    ap.add_argument(
        "--input-dir",
        default="./results/processed_keypoints_mediapipe",
        help="Directory containing *_keypoints.npz",
    )
    ap.add_argument(
        "--npz",
        default="",
        help="Single *_keypoints.npz file (if set, --input-dir batch is skipped)",
    )
    args = ap.parse_args()

    if args.npz:
        p = Path(args.npz)
        if not p.is_file():
            print(f"✗ Not found: {p}", file=sys.stderr)
            return 1
        s = process_npz_file(p)
        print(json.dumps(s, indent=2))
        return 0

    results = batch_process_directory(args.input_dir)
    print(f"Processed {len(results)} file(s) under {Path(args.input_dir).resolve()}")
    for r in results:
        if "error" in r:
            print(f"  ✗ {r.get('source_npz')}: {r['error']}")
        else:
            print(f"  ✓ {r['output_npz']}  ({r['num_frames']} frames)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
