#!/usr/bin/env python3
"""
For each row in exercise_training_index.csv with an existing short_clips video file,
run MediaPipe → pixel keypoints → joint angles → save results/exercise_angles/{stem}_biomechanics.npz

Use --max-videos to cap work for development.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

from biomechanical_features import compute_mixed_sequence_features, compute_sequence_angles
from pose_estimation_core import MediaPipeDetector, VideoProcessor


def angles_from_video(video_path: Path, max_frames: int | None) -> np.ndarray | None:
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
    angles, _ = compute_sequence_angles(arr)
    return angles


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
