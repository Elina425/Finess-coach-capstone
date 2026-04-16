#!/usr/bin/env python3
"""
Build riccio_realtime_*_biomechanics.npz + *_keypoints.npz + *_labels.npz from the **video-folder**
layout of `riccardoriccio/real-time-exercise-recognition-dataset` (no landmarks.csv).

Each frame is labeled with the parent folder name (e.g. squat, push-up) for BiLSTM (angles) and
ST-GCN (COCO-17 xy keypoints). ``*_labels.npz`` also stores ``video_id`` (int per frame) so
``build_kaggle_angle_datasets`` can split **train / val / test by source video** (stratified by class)
for BiLSTM without leakage across clips. Before angles/features, keypoints follow capstone step 3
(``apply_keypoint_preprocessing_pipeline``): **normalization** (torso-based scale, mitigates
camera distance), **imputation** (spatial + temporal for occluded / low-confidence joints), and
**FPS sync** to a common rate — consistent with robust skeletal-sequence practice discussed in
Jiang et al., *A Dual-Masked Auto-Encoder for Robust Motion Capture* (MM'22, DOI 10.1145/3503161.3547796)
regarding reliable joints and temporal consistency. Use ``--raw-keypoints`` to skip this (legacy).

Example (full preprocessing stack for capstone §3 — torso norm + impute + FPS + Laplacian + bone + DWT + Savitzky–Golay)::

  ./venv/bin/python riccio_kaggle_video_pipeline.py --download --max-videos 50 --rich-preprocess

**Speed:** MediaPipe in Python here is **CPU-bound** (not accelerated by NVIDIA GPU the way PyTorch is).
Use ``--workers 0`` (**default**) to auto-pick a process count from ``os.cpu_count()`` (capped by
``RICCIO_MP_MAX_WORKERS``, default 8), or set ``--workers N`` explicitly. Parallel workers process
**different videos** at once and are usually much faster than a single process until you saturate
CPU or RAM. A free **Colab GPU** runtime does **not** speed this step; it helps **BiLSTM training**
instead. For BiLSTM-only training add ``--skip-keypoints`` to skip large ``*_keypoints.npz`` files.
Use ``--max-frames N`` for quick dry runs.

Or with an explicit KaggleHub path::

  ./venv/bin/python riccio_kaggle_video_pipeline.py \\
    --dataset-root ~/.cache/kagglehub/datasets/riccardoriccio/real-time-exercise-recognition-dataset/versions/3 \\
    --max-videos 50 --laplacian-spatial --bone-proportion --dwt --savgol
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

from fitness_coach.pipelines.batch_compute_angles_for_index import riccio_parallel_video_job


def resolve_riccio_worker_count(cli_workers: int) -> int:
    """
    Map CLI ``--workers`` to a pool size.

    - ``N > 0`` → exactly ``N`` processes.
    - ``N == 0`` → ``RICCIO_MP_WORKERS`` if set to a positive int, else
      ``min(RICCIO_MP_MAX_WORKERS, max(1, os.cpu_count() or 2))``.
    """
    nw = int(cli_workers)
    if nw > 0:
        return nw
    env_fixed = os.environ.get("RICCIO_MP_WORKERS", "").strip()
    if env_fixed.isdigit():
        v = int(env_fixed)
        if v > 0:
            return v
    cpu = os.cpu_count() or 2
    cap_raw = os.environ.get("RICCIO_MP_MAX_WORKERS", "8").strip()
    try:
        cap = max(1, int(cap_raw))
    except ValueError:
        cap = 8
    return min(cap, max(1, cpu))


def build_riccio_preprocessing_techniques(
    *,
    no_fps_sync: bool,
    bone_proportion: bool,
    laplacian_spatial: bool,
    dwt: bool,
    savgol: bool = False,
    kalman: bool = False,
) -> list[str]:
    """Same technique list order as kaggle_exercise_recognition_pipeline.build_preprocessing_techniques."""
    if savgol and kalman:
        raise ValueError("Use only one of savgol=True or kalman=True for temporal smoothing.")
    techniques = ["normalization", "imputation", "fps_sync"]
    if no_fps_sync:
        techniques = [t for t in techniques if t != "fps_sync"]
    if bone_proportion:
        if "normalization" in techniques:
            techniques.insert(techniques.index("normalization") + 1, "bone_proportion")
        else:
            techniques.insert(0, "bone_proportion")
    if laplacian_spatial:
        techniques.append("laplacian_spatial")
    if savgol:
        techniques.append("savgol")
    elif kalman:
        techniques.append("kalman")
    if dwt:
        techniques.append("dwt")
    return techniques

VIDEO_EXT = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}

DEFAULT_SUBSETS = (
    "similar_dataset",
    "final_kaggle_with_additional_video",
    "synthetic_dataset",
    "my_test_video_1",
)


def is_riccio_kaggle_video_layout(root: Path) -> bool:
    return any((root / s).is_dir() for s in DEFAULT_SUBSETS)


def hub_versions_root(slug: str) -> Path:
    """~/.cache/kagglehub/datasets/<owner>/<name>/versions"""
    parts = slug.strip().split("/", 1)
    if len(parts) != 2:
        raise ValueError(f"Expected owner/name Kaggle slug, got: {slug!r}")
    owner, name = parts
    return Path.home() / ".cache/kagglehub/datasets" / owner / name / "versions"


def resolve_riccio_dataset_root(cli_path: str, slug: str) -> Path:
    """Explicit --dataset-root, ``EXERCISE_RECOGNITION_ROOT``, or newest kagglehub Riccio layout."""
    if cli_path.strip():
        return Path(cli_path).expanduser().resolve()
    env = os.environ.get("EXERCISE_RECOGNITION_ROOT", "").strip()
    if env:
        return Path(env).expanduser().resolve()
    hub = hub_versions_root(slug)
    if hub.is_dir():
        versions = sorted(
            [p for p in hub.glob("*") if p.is_dir()],
            key=lambda p: p.name,
            reverse=True,
        )
        for v in versions:
            if is_riccio_kaggle_video_layout(v):
                print(f"Using kagglehub cache (Riccio video layout): {v}")
                return v.resolve()
    raise FileNotFoundError(
        "No Riccio dataset folder found. Options:\n"
        "  • Run with --download (requires: pip install kagglehub)\n"
        "  • Or: ./venv/bin/python download_riccio_kaggle_dataset.py  then pass --dataset-root PATH\n"
        "  • Or set EXERCISE_RECOGNITION_ROOT to the extracted folder\n"
        f"  • Expected under: {hub}/<version>/ with folders like similar_dataset/"
    )


def download_riccio_dataset(slug: str) -> Path:
    try:
        import kagglehub
    except ImportError as e:
        raise SystemExit("pip install kagglehub") from e
    p = kagglehub.dataset_download(slug)
    print("Dataset path:", p)
    return Path(p).resolve()


def iter_riccio_videos(
    dataset_root: Path, subsets: Sequence[str]
) -> List[Tuple[Path, str]]:
    """(path, exercise_class) — class is the immediate parent folder name."""
    out: List[Tuple[Path, str]] = []
    for sub in subsets:
        d = dataset_root / sub
        if not d.is_dir():
            continue
        for p in d.rglob("*"):
            if not p.is_file():
                continue
            if p.suffix.lower() not in VIDEO_EXT:
                continue
            cls = p.parent.name.strip()
            if cls:
                out.append((p, cls))
    out.sort(key=lambda x: str(x[0]))
    return out


def run_riccio_video_to_npz(
    dataset_root: Path,
    out_dir: Path,
    *,
    output_stem: str,
    kaggle_slug: str,
    subsets: Sequence[str],
    max_videos: int,
    max_frames: int | None,
    skip_keypoints: bool = False,
    raw_keypoints: bool = False,
    preprocessing_techniques: list[str] | None = None,
    source_fps: float | None = None,
    target_fps: float = 30.0,
    savgol_window_length: int = 7,
    savgol_polyorder: int = 2,
    kalman_process_noise: float = 1e-4,
    kalman_measurement_noise: float = 1e-2,
    num_workers: int = 0,
    mediapipe_model_complexity: int = 1,
    mediapipe_smooth_landmarks: bool = True,
    mediapipe_quiet: bool = True,
    detection_stride: int = 1,
    detection_max_long_edge: int = 0,
) -> Dict[str, Any]:
    vids = iter_riccio_videos(dataset_root, subsets)
    if not vids:
        raise FileNotFoundError(
            f"No videos under {dataset_root} (subsets={list(subsets)}). "
            "Expected folders like similar_dataset/, synthetic_dataset/, …"
        )
    if max_videos > 0:
        vids = vids[:max_videos]

    angle_chunks: List[np.ndarray] = []
    kp_chunks: List[np.ndarray] = []
    label_chunks: List[np.ndarray] = []
    video_id_chunks: List[np.ndarray] = []
    sources: List[str] = []
    skipped = 0

    jobs: List[Dict[str, Any]] = []
    for i, (vp, cls) in enumerate(vids):
        jobs.append(
            {
                "index": i,
                "video_path": str(vp.resolve()),
                "exercise_class": cls,
                "max_frames": max_frames,
                "skip_keypoints": skip_keypoints,
                "raw_keypoints": raw_keypoints,
                "preprocessing_techniques": preprocessing_techniques,
                "source_fps": source_fps,
                "target_fps": float(target_fps),
                "savgol_window_length": int(savgol_window_length),
                "savgol_polyorder": int(savgol_polyorder),
                "kalman_process_noise": float(kalman_process_noise),
                "kalman_measurement_noise": float(kalman_measurement_noise),
                "mediapipe_model_complexity": int(mediapipe_model_complexity),
                "mediapipe_smooth_landmarks": bool(mediapipe_smooth_landmarks),
                "mediapipe_quiet": bool(mediapipe_quiet),
                "detection_stride": int(detection_stride),
                "detection_max_long_edge": int(detection_max_long_edge),
            }
        )

    nw = resolve_riccio_worker_count(int(num_workers))
    if nw <= 1:
        results = []
        for j in jobs:
            print(
                f"[{j['index'] + 1}/{len(jobs)}] {Path(j['video_path']).name}  class={j['exercise_class']!r}",
                flush=True,
            )
            results.append(riccio_parallel_video_job(j))
    else:
        print(
            f"Processing {len(jobs)} videos with {nw} parallel workers (CPU / MediaPipe; order preserved).",
            flush=True,
        )
        with ProcessPoolExecutor(max_workers=nw) as ex:
            results = list(ex.map(riccio_parallel_video_job, jobs))

    for r in results:
        if not r.get("ok"):
            skipped += 1
            continue
        ang = r["angles"]
        kp = r["keypoints"]
        cls = r["exercise_class"]
        if ang.shape[0] == 0:
            skipped += 1
            continue
        if not skip_keypoints and (kp is None or kp.shape[0] != ang.shape[0]):
            skipped += 1
            continue
        t = ang.shape[0]
        angle_chunks.append(ang.astype(np.float32))
        if not skip_keypoints and kp is not None:
            kp_chunks.append(kp.astype(np.float32))
        label_chunks.append(np.array([cls] * t, dtype=object))
        video_id_chunks.append(np.full(t, len(sources), dtype=np.int32))
        sources.append(str(r["source"]))

    if not angle_chunks:
        raise RuntimeError(
            "No angle sequences produced (MediaPipe failed or no readable frames). "
            "Check MediaPipe and OpenCV video codecs."
        )

    angles_all = np.vstack(angle_chunks)
    pose_all = np.concatenate(label_chunks)
    video_id_all = np.concatenate(video_id_chunks)
    kp_all: np.ndarray | None = None
    if not skip_keypoints and kp_chunks:
        kp_all = np.vstack(kp_chunks)
        if kp_all.shape[0] != angles_all.shape[0]:
            raise RuntimeError("Internal error: keypoints/angles frame count mismatch")

    out_dir.mkdir(parents=True, exist_ok=True)
    bio_path = out_dir / f"{output_stem}_biomechanics.npz"
    lab_path = out_dir / f"{output_stem}_labels.npz"
    np.savez_compressed(bio_path, angles=angles_all)
    np.savez_compressed(lab_path, pose=pose_all, video_id=video_id_all)
    kp_path: Path | None = None
    if kp_all is not None:
        kp_path = out_dir / f"{output_stem}_keypoints.npz"
        np.savez_compressed(kp_path, keypoints=kp_all)

    summary: Dict[str, Any] = {
        "dataset_root": str(dataset_root.resolve()),
        "kaggle_slug": kaggle_slug,
        "pipeline": "riccio_kaggle_video_pipeline",
        "num_videos_requested": len(vids),
        "num_videos_used": len(sources),
        "videos_skipped": skipped,
        "video_paths": sources,
        "total_frames": int(angles_all.shape[0]),
        "angle_dim": int(angles_all.shape[1]),
        "biomechanics_npz": bio_path.name,
        "labels_npz": lab_path.name,
        "keypoints_npz": kp_path.name if kp_path else None,
        "classes_present": sorted({str(x) for x in pose_all}),
        "num_distinct_videos": int(np.max(video_id_all) + 1) if video_id_all.size else 0,
        "labels_include_video_id": True,
        "num_workers": int(nw),
        "mediapipe_model_complexity": int(mediapipe_model_complexity),
        "mediapipe_smooth_landmarks": bool(mediapipe_smooth_landmarks),
        "detection_stride": int(detection_stride),
        "detection_max_long_edge": int(detection_max_long_edge),
        "preprocessing": (
            "raw_pixel_xy"
            if raw_keypoints
            else (preprocessing_techniques or ["normalization", "imputation", "fps_sync"])
        ),
        "target_fps": target_fps,
        "source_fps_override": source_fps,
    }
    with open(out_dir / f"{output_stem}_pipeline_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    return summary


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Riccio Kaggle video folders → single *_biomechanics.npz + *_labels.npz for train_exercise_bilstm.py"
    )
    ap.add_argument(
        "--dataset-root",
        default="",
        help="KaggleHub extract (e.g. …/versions/3). If omitted, uses EXERCISE_RECOGNITION_ROOT or the "
        "newest ~/.cache/kagglehub/.../versions/* that looks like Riccio (similar_dataset/, …).",
    )
    ap.add_argument(
        "--download",
        action="store_true",
        help="Download dataset via kagglehub (same slug as --kaggle-slug), then process. Requires kagglehub.",
    )
    ap.add_argument(
        "--output-dir",
        default="./results/riccio_realtime_exercise_recognition",
    )
    ap.add_argument(
        "--output-stem",
        default="riccio_realtime_exercise_recognition",
    )
    ap.add_argument(
        "--kaggle-slug",
        default="riccardoriccio/real-time-exercise-recognition-dataset",
    )
    ap.add_argument(
        "--subsets",
        default=",".join(DEFAULT_SUBSETS),
        help="Comma-separated top-level folders under dataset root to scan",
    )
    ap.add_argument("--max-videos", type=int, default=0, help="Cap videos (0=all)")
    ap.add_argument("--max-frames", type=int, default=0, help="Cap frames per video (0=all)")
    ap.add_argument(
        "--workers",
        type=int,
        default=0,
        help="Parallel processes per video (MediaPipe is CPU-bound). 0=auto from CPU count (cap "
        "RICCIO_MP_MAX_WORKERS, default 8) or RICCIO_MP_WORKERS; 1=sequential; N>0 fixed pool.",
    )
    ap.add_argument(
        "--skip-keypoints",
        action="store_true",
        help="Only write biomechanics + labels (faster disk); skip ST-GCN *_keypoints.npz",
    )
    ap.add_argument(
        "--raw-keypoints",
        action="store_true",
        help="Skip normalize/impute/FPS (legacy raw MediaPipe pixels). Default: full preprocessing.",
    )
    ap.add_argument("--target-fps", type=float, default=30.0, help="FPS resampling target (default 30)")
    ap.add_argument(
        "--source-fps",
        type=float,
        default=0.0,
        help="Override native video FPS for sync (0 = use OpenCV-reported FPS)",
    )
    ap.add_argument(
        "--no-fps-sync",
        action="store_true",
        help="Keep native frame timing (no resample to --target-fps)",
    )
    ap.add_argument(
        "--bone-proportion",
        action="store_true",
        help="After torso normalization: BioPose-style limb ratios (arXiv:2501.07800)",
    )
    ap.add_argument(
        "--laplacian-spatial",
        action="store_true",
        help="Spatial imputation via graph Laplacian (arXiv:2204.10312)",
    )
    ap.add_argument("--dwt", action="store_true", help="Append DWT normalization (PyWavelets)")
    ap.add_argument(
        "--savgol",
        action="store_true",
        help="Savitzky–Golay temporal smoothing (after FPS sync)",
    )
    ap.add_argument(
        "--kalman",
        action="store_true",
        help="Kalman temporal smoothing (pick one of --savgol / --kalman)",
    )
    ap.add_argument("--savgol-window", type=int, default=7, help="SG window length (odd)")
    ap.add_argument("--savgol-poly", type=int, default=2, help="SG polynomial order")
    ap.add_argument("--kalman-q", type=float, default=1e-4, help="Kalman process noise")
    ap.add_argument("--kalman-r", type=float, default=1e-2, help="Kalman measurement noise")
    ap.add_argument(
        "--rich-preprocess",
        action="store_true",
        help="Shortcut: enable --laplacian-spatial, --bone-proportion, --dwt, and --savgol (full §3 extras).",
    )
    ap.add_argument(
        "--mediapipe-model-complexity",
        type=int,
        choices=(0, 1, 2),
        default=1,
        help="BlazePose solutions API only: 0=fastest, 1=default balance, 2=heaviest (ignored by tasks API).",
    )
    ap.add_argument(
        "--mediapipe-no-smooth-landmarks",
        action="store_true",
        help="Disable temporal landmark smoothing (solutions API only; slightly faster).",
    )
    ap.add_argument(
        "--mediapipe-fast",
        action="store_true",
        help="Shorthand: --mediapipe-model-complexity 0, --mediapipe-no-smooth-landmarks, and "
        "--detection-max-long-edge 480 if not already set (big Colab speedup).",
    )
    ap.add_argument(
        "--detection-stride",
        type=int,
        default=1,
        help="Run pose on every Nth frame (e.g. 2 ≈ 2× fewer detections; FPS preprocessing uses effective rate).",
    )
    ap.add_argument(
        "--detection-max-long-edge",
        type=int,
        default=0,
        help="If >0, downscale frames so max(h,w)<=this before MediaPipe (faster); landmarks mapped to full resolution.",
    )
    args = ap.parse_args()

    if args.mediapipe_fast:
        args.mediapipe_model_complexity = 0
        args.mediapipe_no_smooth_landmarks = True
        if int(args.detection_max_long_edge) <= 0:
            args.detection_max_long_edge = 480

    if args.rich_preprocess:
        args.laplacian_spatial = True
        args.bone_proportion = True
        args.dwt = True
        args.savgol = True

    if args.download and args.dataset_root.strip():
        print("Use only one of --download or --dataset-root", file=sys.stderr)
        return 1

    try:
        if args.download:
            root = download_riccio_dataset(args.kaggle_slug)
        else:
            root = resolve_riccio_dataset_root(args.dataset_root, args.kaggle_slug)
    except FileNotFoundError as e:
        print(f"✗ {e}", file=sys.stderr)
        return 1

    if not root.is_dir():
        dr = args.dataset_root.strip()
        print(f"Not a directory: {root}", file=sys.stderr)
        if dr in ("DATASET_ROOT", "PATH/TO/DATASET", "..."):
            print(
                "Replace the placeholder with a real path, or omit --dataset-root to auto-detect "
                "from ~/.cache/kagglehub/, or use --download.",
                file=sys.stderr,
            )
        return 1
    subsets = tuple(s.strip() for s in args.subsets.split(",") if s.strip())
    mf = args.max_frames if args.max_frames > 0 else None
    techniques = None
    if not args.raw_keypoints:
        techniques = build_riccio_preprocessing_techniques(
            no_fps_sync=args.no_fps_sync,
            bone_proportion=args.bone_proportion,
            laplacian_spatial=args.laplacian_spatial,
            dwt=args.dwt,
            savgol=args.savgol,
            kalman=args.kalman,
        )
    src_fps = args.source_fps if args.source_fps > 1e-6 else None
    try:
        summary = run_riccio_video_to_npz(
            root,
            Path(args.output_dir).resolve(),
            output_stem=args.output_stem,
            kaggle_slug=args.kaggle_slug,
            subsets=subsets,
            max_videos=args.max_videos,
            max_frames=mf,
            skip_keypoints=args.skip_keypoints,
            raw_keypoints=args.raw_keypoints,
            preprocessing_techniques=techniques,
            source_fps=src_fps,
            target_fps=float(args.target_fps),
            savgol_window_length=args.savgol_window,
            savgol_polyorder=args.savgol_poly,
            kalman_process_noise=args.kalman_q,
            kalman_measurement_noise=args.kalman_r,
            num_workers=args.workers,
            mediapipe_model_complexity=int(args.mediapipe_model_complexity),
            mediapipe_smooth_landmarks=not bool(args.mediapipe_no_smooth_landmarks),
            mediapipe_quiet=True,
            detection_stride=max(1, int(args.detection_stride)),
            detection_max_long_edge=max(0, int(args.detection_max_long_edge)),
        )
    except (FileNotFoundError, RuntimeError) as e:
        print(f"✗ {e}", file=sys.stderr)
        return 1

    print(f"\n✓ Wrote {summary['biomechanics_npz']}  T={summary['total_frames']}")
    if summary.get("keypoints_npz"):
        print(f"  {summary['keypoints_npz']} (ST-GCN)")
    print(f"  Classes: {summary['classes_present']}")
    print("Train BiLSTM:")
    print(
        f"  ./venv/bin/python train_exercise_bilstm.py --preset riccio --standardize --eval-test \\\n"
        f"    --kaggle-angles-dir {args.output_dir} \\\n"
        f"    --kaggle-stem {args.output_stem}"
    )
    if summary.get("keypoints_npz"):
        print("Train ST-GCN:")
        print(
            f"  ./venv/bin/python train_exercise_stgcn.py --standardize --eval-test \\\n"
            f"    --kaggle-keypoints-dir {args.output_dir} \\\n"
            f"    --kaggle-stem {args.output_stem}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
