#!/usr/bin/env python3
"""
For each row in exercise_training_index.csv with an existing short_clips video file,
run MediaPipe → (optional) keypoint preprocessing → joint angles → save results/exercise_angles/{stem}_biomechanics.npz

**Preprocessing (default on)** matches capstone step 3 / ``apply_keypoint_preprocessing_pipeline``:
torso-based normalization (reduces camera distance / scale), spatial + temporal imputation for
low-confidence joints, and FPS resampling for consistent timing across clips — aligned with the
spirit of joint reliability and temporal consistency discussed in Jiang et al. (MM'22, D-MAE)
for robust skeletal sequences (see also ``pose_estimation_core.apply_keypoint_preprocessing_pipeline``).

Use --max-videos to cap work for development.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import List, Optional

import numpy as np

from biomechanical_features import compute_mixed_sequence_features, compute_sequence_angles
from pose_estimation_core import (
    MediaPipeDetector,
    VideoProcessor,
    apply_keypoint_preprocessing_pipeline,
)


def angles_from_video(
    video_path: Path,
    max_frames: int | None,
    *,
    preprocess: bool = True,
    preprocessing_techniques: Optional[List[str]] = None,
    source_fps: Optional[float] = None,
    target_fps: float = 30.0,
    savgol_window_length: int = 7,
    savgol_polyorder: int = 2,
    kalman_process_noise: float = 1e-4,
    kalman_measurement_noise: float = 1e-2,
) -> np.ndarray | None:
    out = angles_and_keypoints_from_video(
        video_path,
        max_frames,
        preprocess=preprocess,
        preprocessing_techniques=preprocessing_techniques,
        source_fps=source_fps,
        target_fps=target_fps,
        savgol_window_length=savgol_window_length,
        savgol_polyorder=savgol_polyorder,
        kalman_process_noise=kalman_process_noise,
        kalman_measurement_noise=kalman_measurement_noise,
    )
    return None if out is None else out[0]


def angles_and_keypoints_from_video(
    video_path: Path,
    max_frames: int | None,
    *,
    preprocess: bool = True,
    preprocessing_techniques: Optional[List[str]] = None,
    source_fps: Optional[float] = None,
    target_fps: float = 30.0,
    savgol_window_length: int = 7,
    savgol_polyorder: int = 2,
    kalman_process_noise: float = 1e-4,
    kalman_measurement_noise: float = 1e-2,
) -> tuple[np.ndarray, np.ndarray] | None:
    """
    One MediaPipe pass → (angles (T,8), keypoints (T,17,2)).

    With ``preprocess=True`` (default), runs normalization + imputation + FPS sync before angles
    (same order as ``apply_keypoint_preprocessing_pipeline``). Keypoints are then torso-normalized
    coordinates suitable for ST-GCN / saved NPZs. With ``preprocess=False``, returns raw pixel xy.
    """
    det = MediaPipeDetector()
    if not getattr(det, "available", True):
        print("MediaPipe unavailable", file=sys.stderr)
        return None
    vp = VideoProcessor(str(video_path))
    pose_results = vp.process_with_detector(det, max_frames=max_frames)
    measured_fps = float(vp.fps) if vp.fps and vp.fps > 1e-3 else 30.0
    vp.close()
    if not pose_results:
        return None
    arr = np.stack([r.keypoints for r in pose_results], axis=0)

    if not preprocess:
        angles, _ = compute_sequence_angles(arr)
        return angles.astype(np.float32), arr.astype(np.float32)

    kp_seq = [arr[i].astype(np.float32).copy() for i in range(arr.shape[0])]
    conf_seq = [np.asarray(r.confidence, dtype=np.float32).copy() for r in pose_results]
    techniques = preprocessing_techniques
    if techniques is None:
        techniques = ["normalization", "imputation", "fps_sync"]
    src = float(source_fps) if source_fps is not None and source_fps > 1e-6 else measured_fps
    processed = apply_keypoint_preprocessing_pipeline(
        kp_seq,
        conf_seq,
        preprocessing_techniques=list(techniques),
        target_fps=float(target_fps),
        source_fps=src,
        original_frames=len(kp_seq),
        savgol_window_length=int(savgol_window_length),
        savgol_polyorder=int(savgol_polyorder),
        kalman_process_noise=float(kalman_process_noise),
        kalman_measurement_noise=float(kalman_measurement_noise),
    )
    fk = np.stack(processed["final_keypoints"], axis=0)
    angles, _ = compute_sequence_angles(fk)
    return angles.astype(np.float32), fk.astype(np.float32)


def mixed_features_from_video(video_path: Path, max_frames: int | None) -> np.ndarray | None:
    """Angles + pelvis-centered normalized coords per frame (Riccio-style mixed features)."""
    det = MediaPipeDetector()
    if not getattr(det, "available", True):
        print("MediaPipe unavailable", file=sys.stderr)
        return None
    vp = VideoProcessor(str(video_path))
    pose_results = vp.process_with_detector(det, max_frames=max_frames)
    vp.close()
    if not pose_results:
        return None
    arr = np.stack([r.keypoints for r in pose_results], axis=0)
    mixed, _ = compute_mixed_sequence_features(arr)
    return mixed


def _resolve_video_path(row: dict, dataset_root: Path) -> Path | None:
    """Prefer CSV video_path; if empty (index built with --no-require-file), try short_clips/{stem}.mp4."""
    vp = (row.get("video_path") or "").strip()
    stem = (row.get("video_stem") or "").strip()
    if vp:
        p = Path(vp)
        if p.is_file():
            return p
    if stem:
        p = (dataset_root / "videos" / "short_clips" / f"{stem}.mp4").resolve()
        if p.is_file():
            return p
    return None


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--index-csv", default="./results/exercise_training_index.csv")
    p.add_argument("--dataset-root", default="./qevd-fit-coach-data", help="Used if video_path column is empty")
    p.add_argument("--output-dir", default="./results/exercise_angles")
    p.add_argument("--max-videos", type=int, default=0, help="Cap processed videos (0=all)")
    p.add_argument("--max-frames", type=int, default=0, help="Cap frames per video (0=all)")
    args = p.parse_args()

    inp = Path(args.index_csv)
    if not inp.is_file():
        print(f"Missing {inp}", file=sys.stderr)
        return 1
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dataset_root = Path(args.dataset_root)

    mf = args.max_frames if args.max_frames > 0 else None
    done = 0
    skipped = 0
    with open(inp, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            stem = row.get("video_stem", "")
            vid = _resolve_video_path(row, dataset_root)
            if vid is None:
                skipped += 1
                if skipped <= 3:
                    print(
                        f"  (skip) no readable file for stem={stem!r} — "
                        "symlink missing or path wrong",
                        file=sys.stderr,
                    )
                continue
            outp = out_dir / f"{stem}_biomechanics.npz"
            if outp.is_file():
                continue
            ang = angles_from_video(vid, mf)
            if ang is None:
                continue
            np.savez_compressed(
                outp,
                angles=ang,
                source_video=str(vid),
            )
            done += 1
            print(f"✓ {stem}  T={ang.shape[0]}")
            if args.max_videos and done >= args.max_videos:
                break

    print(f"Saved {done} angle files under {out_dir.resolve()}")
    if done == 0 and skipped > 0:
        print(
            f"No videos could be opened ({skipped} rows skipped).\n"
            "  • short_clips must be real .mp4 files (not broken symlinks).\n"
            "  • Or use long_range videos: build a separate index / point --dataset-root at data with files.",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
