#!/usr/bin/env python3
"""
Preprocess pose keypoints after extracting landmarks with MediaPipe (best model on your benchmark).

**Order (capstone step 3 — before angles / BiLSTM features):** run this pipeline first; the same
logic lives in ``pose_estimation_core.apply_keypoint_preprocessing_pipeline`` for reuse and tests.

1. **Spatial imputation** — low-confidence joints filled from COCO neighbors in the same frame.
2. **Skeleton-based normalization** — scale by torso length, center on hips (reduces camera distance).
3. **Temporal imputation** — linear interpolation along time for flickering joints.
4. **FPS resampling** — uniform timeline at `--target-fps` (default 30 Hz) using measured video FPS.

Optional: `--dwt` adds wavelet-based normalization (requires PyWavelets).
Optional: `--bone-proportion` applies BioPose-inspired limb/torso proportion scaling (arXiv:2501.07800).
Optional: `--laplacian-spatial` uses graph Laplacian harmonic imputation for low-confidence joints (arXiv:2204.10312).
Optional: `--biomechanics` runs `biomechanical_features.py` to save joint angles (knee, hip, elbow, shoulder) per frame.

This script is **keypoint** preprocessing only (RGB / ImageNet frame features are out of scope here).

Usage:
  ./venv/bin/python keypoint_preprocessing_pipeline.py --video-ids 0000 0001 \\
      --dataset-root ./qevd-fit-coach-data --output-dir ./results/processed_keypoints_mediapipe

  ./venv/bin/python keypoint_preprocessing_pipeline.py \\
      --video-path ./qevd-fit-coach-data/videos/long_range/0000.mp4 --target-fps 30

  # Tasks 3+4 for many videos (every *.mp4 under dataset ``videos/``, cap at 20):
  ./venv/bin/python keypoint_preprocessing_pipeline.py --all-videos --max-videos 20 \\
      --dataset-root ./qevd-fit-coach-data --output-dir ./results/processed_keypoints_mediapipe --biomechanics

  # Tasks 3+4 for 20 short_clips that appear in fine_grained_labels.json (BiLSTM training alignment):
  ./venv/bin/python keypoint_preprocessing_pipeline.py --from-fine-grained-labels --max-videos 20 \\
      --dataset-root ./qevd-fit-coach-data --output-dir ./results/processed_keypoints_short_clips --biomechanics
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from pose_estimation_core import MediaPipeDetector
from qevd_dataset_integration import DatasetPreprocessor, QEVDDatasetLoader


def _process_short_clip_stems(
    pre: DatasetPreprocessor,
    stems: list,
    det,
    techniques: list,
    target_fps: float,
    dataset_root: str,
    short_clips_dir: str | None = None,
) -> dict:
    """Run MediaPipe + preprocessing on each stem’s resolved short-clip ``.mp4`` path."""
    from build_exercise_training_index import resolve_short_clip_mp4_for_stem

    out: dict = {}
    root = Path(dataset_root)
    sc_opt = Path(short_clips_dir).resolve() if short_clips_dir else None
    for stem in stems:
        sc = resolve_short_clip_mp4_for_stem(root, stem, sc_opt)
        if sc is None:
            print(f"⚠ skip (no readable file for stem {stem!r})")
            continue
        r = pre.preprocess_video_file(
            str(sc),
            det,
            preprocessing_techniques=techniques,
            target_fps=target_fps,
        )
        if r:
            out[stem] = r
    return out


def main() -> int:
    p = argparse.ArgumentParser(
        description="MediaPipe landmarks → normalize, impute, FPS sync (for downstream features)."
    )
    p.add_argument("--dataset-root", default="./qevd-fit-coach-data")
    p.add_argument(
        "--video-ids",
        nargs="*",
        default=None,
        help="Dataset video stems (e.g. 0000 0001). Default: 0000 0001 0002 (ignored if --all-videos)",
    )
    p.add_argument(
        "--all-videos",
        action="store_true",
        help="Process every *.mp4 under dataset-root/videos/ (stems deduplicated, sorted)",
    )
    p.add_argument(
        "--max-videos",
        type=int,
        default=0,
        help="Cap count: with --all-videos / --short-clips-only, or with --from-fine-grained-labels (default 20 if unset)",
    )
    p.add_argument(
        "--short-clips-only",
        action="store_true",
        help="Only videos under videos/short_clips/ (use with --max-videos or --all-videos)",
    )
    p.add_argument(
        "--from-fine-grained-labels",
        action="store_true",
        help="Stream fine_grained_labels.json; take first N unique stems with readable short_clips/*.mp4 (BiLSTM alignment)",
    )
    p.add_argument(
        "--fine-labels",
        default="./qevd-fit-coach-data/annotations/labels/fine_grained_labels.json",
        help="Used with --from-fine-grained-labels",
    )
    p.add_argument(
        "--short-clips-dir",
        default="",
        help="If MP4s live outside dataset layout, set to the folder containing {stem}.mp4 (e.g. external drive)",
    )
    p.add_argument("--video-path", default="", help="Process one file by path instead of dataset ids")
    p.add_argument(
        "--output-dir",
        default="./results/processed_keypoints_mediapipe",
        help="NPZ + JSON output per video",
    )
    p.add_argument("--target-fps", type=float, default=30.0)
    p.add_argument(
        "--no-fps-sync",
        action="store_true",
        help="Skip temporal resampling (keep native frame rate)",
    )
    p.add_argument("--dwt", action="store_true", help="Append DWT normalization (PyWavelets)")
    p.add_argument(
        "--bone-proportion",
        action="store_true",
        help="After torso normalization: limb lengths → anthropometric ratios × torso (BioPose-style)",
    )
    p.add_argument(
        "--laplacian-spatial",
        action="store_true",
        help="Spatial imputation via skeleton graph Laplacian (L=D−A), harmonic relaxation (arXiv:2204.10312)",
    )
    p.add_argument(
        "--biomechanics",
        action="store_true",
        help="After saving keypoints, compute joint-angle features (*_biomechanics.npz)",
    )

    args = p.parse_args()

    techniques = ["normalization", "imputation", "fps_sync"]
    if args.no_fps_sync:
        techniques = [t for t in techniques if t != "fps_sync"]
    if args.bone_proportion:
        if "normalization" in techniques:
            techniques.insert(
                techniques.index("normalization") + 1,
                "bone_proportion",
            )
        else:
            techniques.insert(0, "bone_proportion")
    if args.laplacian_spatial:
        techniques.append("laplacian_spatial")
    if args.dwt:
        techniques.append("dwt")

    det = MediaPipeDetector()
    if not getattr(det, "available", True):
        print("✗ MediaPipe detector not available")
        return 1

    loader = QEVDDatasetLoader(args.dataset_root)
    pre = DatasetPreprocessor(loader)
    out: dict = {}

    if args.video_path:
        r = pre.preprocess_video_file(
            args.video_path,
            det,
            preprocessing_techniques=techniques,
            target_fps=args.target_fps,
        )
        if r:
            out[Path(args.video_path).stem] = r
    elif args.from_fine_grained_labels:
        cap = args.max_videos if args.max_videos > 0 else 20
        try:
            from build_exercise_training_index import (
                print_diagnostic_first_label_paths,
                stream_unique_stems_from_fine_grained,
            )
        except ImportError:
            print("✗ Install ijson: pip install ijson", file=sys.stderr)
            return 1
        fl = Path(args.fine_labels)
        if not fl.is_file():
            print(f"✗ Not found: {fl}", file=sys.stderr)
            return 1
        sc_override = Path(args.short_clips_dir).resolve() if args.short_clips_dir else None
        stems, scanned, skipped = stream_unique_stems_from_fine_grained(
            fl,
            Path(args.dataset_root),
            cap,
            max_scan_items=2_000_000,
            short_clips_dir=sc_override,
        )
        print(
            f"fine_grained_labels: {len(stems)} unique short_clip stem(s) "
            f"(scanned {scanned} JSON items, {skipped} entries skipped — no readable local mp4)."
        )
        if not stems:
            print(
                "✗ No stems with a **readable** .mp4 (followed symlinks must point to real files).\n"
                "  Labels use paths like videos/short_clips/00000000.mp4 under --dataset-root.\n"
                "  If you see .mp4 names in short_clips/ but the pipeline skips them, they are often "
                "**broken symlinks** — fix the target path or copy the video bytes into that folder.",
                file=sys.stderr,
            )
            print_diagnostic_first_label_paths(
                fl, Path(args.dataset_root), sc_override
            )
            return 1
        out = _process_short_clip_stems(
            pre,
            stems,
            det,
            techniques,
            args.target_fps,
            args.dataset_root,
            short_clips_dir=args.short_clips_dir or None,
        )
    elif args.short_clips_only:
        ids = loader.list_short_clip_videos()
        if not ids:
            print(
                f"✗ No readable .mp4 under {Path(args.dataset_root).resolve() / 'videos' / 'short_clips'}"
            )
            return 1
        if args.max_videos and args.max_videos > 0:
            ids = ids[: args.max_videos]
        print(f"Short clips only: {len(ids)} stem(s) to process.")
        out = _process_short_clip_stems(
            pre,
            ids,
            det,
            techniques,
            args.target_fps,
            args.dataset_root,
            short_clips_dir=args.short_clips_dir or None,
        )
    else:
        if args.all_videos:
            ids = sorted(set(loader.list_videos()))
            if not ids:
                print(
                    f"✗ No videos found under {Path(args.dataset_root).resolve() / 'videos'}"
                )
                return 1
            if args.max_videos and args.max_videos > 0:
                ids = ids[: args.max_videos]
            print(f"Discovered {len(ids)} video stem(s) to process.")
        else:
            ids = args.video_ids if args.video_ids is not None else ["0000", "0001", "0002"]
        out = pre.preprocess_batch(
            ids,
            det,
            preprocessing_techniques=techniques,
            target_fps=args.target_fps,
        )

    if not out:
        print("✗ No videos processed")
        return 1

    pre.save_processed_data(out, args.output_dir)
    out_dir = Path(args.output_dir).resolve()
    print(f"\n✓ Done. Keypoints under {out_dir}")

    if args.biomechanics:
        from biomechanical_features import batch_process_directory

        batch_process_directory(out_dir)
        print("✓ Biomechanics angles written next to keypoints (*_biomechanics.npz)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
