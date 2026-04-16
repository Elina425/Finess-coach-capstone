#!/usr/bin/env python3
"""
For each row in exercise_training_index.csv with an existing short_clips video file,
run MediaPipe → (optional) keypoint preprocessing → joint angles → save results/exercise_angles/{stem}_biomechanics.npz

**EgoExo-Fitness (frame folders):** if a row has ``frame_root``, ``frame_start``, ``frame_end`` (from ``build_egoexo_fitness_index.py``) and there is no usable ``video_path``, pass ``--dataset-root`` to the
dataset root (e.g. ``data/EgoExo-Fitness``) and ``--egoexo-view`` (default ``ego_l``) so frames are read
from ``{dataset_root}/{frame_root}/{view}/{padded}.jpg``.

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
import re
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

from fitness_coach.core.biomechanical_features import compute_mixed_sequence_features, compute_sequence_angles
from fitness_coach.core.pose_estimation_core import (
    MediaPipeDetector,
    PoseEstimationResult,
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
    return_keypoints: bool = True,
    mediapipe_model_complexity: int = 1,
    mediapipe_smooth_landmarks: bool = True,
    mediapipe_quiet: bool = False,
    detection_stride: int = 1,
    detection_max_long_edge: int = 0,
) -> tuple[np.ndarray, np.ndarray | None] | None:
    """
    One MediaPipe pass → (angles (T,8), keypoints (T,17,2)).

    With ``preprocess=True`` (default), runs normalization + imputation + FPS sync before angles
    (same order as ``apply_keypoint_preprocessing_pipeline``). Keypoints are then torso-normalized
    coordinates suitable for ST-GCN / saved NPZs. With ``preprocess=False``, returns raw pixel xy.

    If ``return_keypoints=False``, the second return value is ``None`` (saves memory; use with
    BiLSTM-only pipelines and ``--skip-keypoints``).
    """
    det = MediaPipeDetector(
        model_complexity=int(mediapipe_model_complexity),
        smooth_landmarks=bool(mediapipe_smooth_landmarks),
        quiet=bool(mediapipe_quiet),
    )
    if not getattr(det, "available", True):
        print("MediaPipe unavailable", file=sys.stderr)
        return None
    vp = VideoProcessor(str(video_path))
    stride = max(1, int(detection_stride))
    pose_results = vp.process_with_detector(
        det,
        max_frames=max_frames,
        detection_stride=stride,
        detection_max_long_edge=int(detection_max_long_edge),
    )
    measured_fps = float(vp.fps) if vp.fps and vp.fps > 1e-3 else 30.0
    if stride > 1:
        measured_fps = measured_fps / float(stride)
    vp.close()
    if not pose_results:
        return None
    arr = np.stack([r.keypoints for r in pose_results], axis=0)

    if not preprocess:
        angles, _ = compute_sequence_angles(arr)
        ang_out = angles.astype(np.float32)
        if return_keypoints:
            return ang_out, arr.astype(np.float32)
        return ang_out, None

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
    ang_out = angles.astype(np.float32)
    if return_keypoints:
        return ang_out, fk.astype(np.float32)
    return ang_out, None


def riccio_parallel_video_job(job: dict) -> dict:
    """
    Picklable worker for ``ProcessPoolExecutor`` (one video per job). ``job`` is built by
    ``riccio_kaggle_video_pipeline.run_riccio_video_to_npz``.
    """
    from pathlib import Path

    idx = int(job["index"])
    vp = Path(job["video_path"])
    cls = str(job["exercise_class"])
    skip_keypoints = bool(job["skip_keypoints"])
    both = angles_and_keypoints_from_video(
        vp,
        job["max_frames"],
        preprocess=not bool(job["raw_keypoints"]),
        preprocessing_techniques=job.get("preprocessing_techniques"),
        source_fps=job.get("source_fps"),
        target_fps=float(job["target_fps"]),
        savgol_window_length=int(job["savgol_window_length"]),
        savgol_polyorder=int(job["savgol_polyorder"]),
        kalman_process_noise=float(job["kalman_process_noise"]),
        kalman_measurement_noise=float(job["kalman_measurement_noise"]),
        return_keypoints=not skip_keypoints,
        mediapipe_model_complexity=int(job.get("mediapipe_model_complexity", 1)),
        mediapipe_smooth_landmarks=bool(job.get("mediapipe_smooth_landmarks", True)),
        mediapipe_quiet=bool(job.get("mediapipe_quiet", True)),
        detection_stride=int(job.get("detection_stride", 1)),
        detection_max_long_edge=int(job.get("detection_max_long_edge", 0)),
    )
    if both is None:
        return {"index": idx, "ok": False, "exercise_class": cls, "source": str(vp)}
    ang, kp = both
    if ang.shape[0] == 0:
        return {"index": idx, "ok": False, "exercise_class": cls, "source": str(vp)}
    if not skip_keypoints:
        if kp is None or kp.shape[0] != ang.shape[0]:
            return {"index": idx, "ok": False, "exercise_class": cls, "source": str(vp)}
    return {
        "index": idx,
        "ok": True,
        "angles": ang,
        "keypoints": kp,
        "exercise_class": cls,
        "source": str(vp),
    }


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


def resolve_egoexo_frame_dir(dataset_root: Path, frame_root: str, egoexo_view: str) -> Path | None:
    """
    ``frame_root`` is e.g. ``frames_open/ThEnUZ``; released frames live in ``.../ego_l``, etc.
    """
    root = (dataset_root / str(frame_root).strip().lstrip("/")).resolve()
    if not root.is_dir():
        return None
    v = (egoexo_view or "").strip()
    if v:
        cand = root / v
        if cand.is_dir():
            return cand
    subs = sorted([p for p in root.iterdir() if p.is_dir()])
    if len(subs) == 1:
        return subs[0]
    return None


def _find_frame_file(frame_dir: Path, frame_idx: int) -> Path | None:
    """Match common numeric image naming (EgoExo-style release is typically zero-padded jpg)."""
    for ext in (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"):
        for width in (6, 5, 8, 4):
            p = frame_dir / f"{frame_idx:0{width}d}{ext}"
            if p.is_file():
                return p
        p2 = frame_dir / f"{frame_idx}{ext}"
        if p2.is_file():
            return p2
    for ext in (".jpg", ".jpeg", ".png"):
        hits = list(frame_dir.glob(f"*{frame_idx}*{ext}"))
        if len(hits) == 1:
            return hits[0]
    return None


def _infer_frame_pattern(frame_dir: Path) -> Optional[Tuple[int, str]]:
    """From first ``*.jpg``/``*.png``, infer (zero_pad_width, extension) if name is all digits."""
    for pat in ("*.jpg", "*.jpeg", "*.png", "*.JPG"):
        for p in sorted(frame_dir.glob(pat))[:50]:
            stem = p.stem
            if re.fullmatch(r"\d+", stem):
                return len(stem), p.suffix
    return None


def _find_frame_file_cached(
    frame_dir: Path,
    frame_idx: int,
    cache: List[Optional[Tuple[int, str]]],
) -> Path | None:
    if cache[0] is None:
        cache[0] = _infer_frame_pattern(frame_dir)
    pat = cache[0]
    if pat is not None:
        w, ext = pat
        p = frame_dir / f"{frame_idx:0{w}d}{ext}"
        if p.is_file():
            return p
    return _find_frame_file(frame_dir, frame_idx)


def angles_and_keypoints_from_frame_directory(
    frame_dir: Path,
    frame_start: int,
    frame_end: int,
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
    EgoExo-style frame folder → (angles (T,8), keypoints (T,17,2)) after the same preprocessing
    stack as ``angles_and_keypoints_from_video`` when ``preprocess=True`` (torso-normalized xy for ST-GCN).
    """
    if frame_end < frame_start:
        frame_start, frame_end = frame_end, frame_start
    det = MediaPipeDetector()
    if not getattr(det, "available", True):
        print("MediaPipe unavailable", file=sys.stderr)
        return None

    indices = list(range(frame_start, frame_end + 1))
    if max_frames and len(indices) > max_frames:
        indices = indices[: int(max_frames)]

    pattern_cache: List[Optional[Tuple[int, str]]] = [None]
    ref_w = ref_h = 0
    for idx in indices:
        path = _find_frame_file_cached(frame_dir, idx, pattern_cache)
        if path is None:
            continue
        frame = cv2.imread(str(path))
        if frame is not None:
            ref_h, ref_w = frame.shape[:2]
            break
    if ref_w <= 0:
        return None

    pose_results: List[PoseEstimationResult] = []
    n_ok = 0
    last_w, last_h = ref_w, ref_h
    for fi, idx in enumerate(indices):
        path = _find_frame_file_cached(frame_dir, idx, pattern_cache)
        frame = cv2.imread(str(path)) if path is not None else None
        if frame is not None:
            last_h, last_w = frame.shape[:2]
        keypoints, confidence = det.detect(frame) if frame is not None else (None, None)
        w, h = last_w, last_h
        if keypoints is not None:
            z = keypoints.astype(np.float32).copy()
            z[:, 0] *= float(w)
            z[:, 1] *= float(h)
            c = (
                np.asarray(confidence, dtype=np.float32).copy()
                if confidence is not None
                else np.ones(17, dtype=np.float32)
            )
            n_ok += 1
        else:
            z = np.zeros((17, 2), dtype=np.float32)
            c = np.zeros(17, dtype=np.float32)
        pose_results.append(
            PoseEstimationResult(
                model_name=det.__class__.__name__,
                frame_idx=fi,
                keypoints=z,
                confidence=c,
                inference_time=0.0,
            )
        )

    if n_ok < max(3, int(0.5 * len(indices))):
        print(
            f"  (warn) only {n_ok}/{len(indices)} frames had pose in {frame_dir}",
            file=sys.stderr,
        )

    arr = np.stack([r.keypoints for r in pose_results], axis=0)

    if not preprocess:
        angles, _ = compute_sequence_angles(arr)
        return angles.astype(np.float32), arr.astype(np.float32)

    kp_seq = [arr[i].astype(np.float32).copy() for i in range(arr.shape[0])]
    conf_seq = [
        np.asarray(r.confidence, dtype=np.float32).copy() for r in pose_results
    ]
    techniques = preprocessing_techniques
    if techniques is None:
        techniques = ["normalization", "imputation", "fps_sync"]
    src = float(source_fps) if source_fps is not None and source_fps > 1e-6 else 30.0
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


def angles_from_frame_directory(
    frame_dir: Path,
    frame_start: int,
    frame_end: int,
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
    """
    Run MediaPipe on each image in ``[frame_start, frame_end]`` (inclusive); same preprocessing
    as ``angles_from_video`` when ``preprocess=True``. Missing images get zero keypoints / confidence.
    """
    out = angles_and_keypoints_from_frame_directory(
        frame_dir,
        frame_start,
        frame_end,
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
    p.add_argument(
        "--dataset-root",
        default="./data",
        help="Root for short_clips/ or EgoExo frame folders (Riccio users: prefer step_03_04_riccio_videos_to_angles.sh)",
    )
    p.add_argument("--output-dir", default="./results/exercise_angles")
    p.add_argument("--max-videos", type=int, default=0, help="Cap processed videos (0=all)")
    p.add_argument("--max-frames", type=int, default=0, help="Cap frames per video (0=all)")
    p.add_argument(
        "--egoexo-view",
        default="ego_l",
        help="View subfolder under CSV frame_root for EgoExo (ego_l, exo_m, …).",
    )
    p.add_argument(
        "--save-keypoints",
        action="store_true",
        help="Also write {stem}_keypoints.npz (COCO-17, T,17,2) next to biomechanics — "
        "same preprocessing as video path; use with train_exercise_stgcn.py --index-csv … --keypoints-dir ….",
    )
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
    warned_missing_frame_tree = False
    with open(inp, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            stem = (row.get("video_stem") or "").strip()
            if not stem:
                continue
            outp = out_dir / f"{stem}_biomechanics.npz"
            kp_out = out_dir / f"{stem}_keypoints.npz"
            need_kp = bool(args.save_keypoints)
            if outp.is_file() and (not need_kp or kp_out.is_file()):
                continue

            vid = _resolve_video_path(row, dataset_root)
            ang = None
            kp = None
            source_meta = ""

            if vid is not None:
                if need_kp:
                    both = angles_and_keypoints_from_video(vid, mf)
                    if both is not None:
                        ang, kp = both
                        if ang is not None:
                            source_meta = str(vid)
                else:
                    ang = angles_from_video(vid, mf)
                    if ang is not None:
                        source_meta = str(vid)

            if ang is None:
                fr = (row.get("frame_root") or "").strip()
                fs_raw = (row.get("frame_start") or "").strip()
                fe_raw = (row.get("frame_end") or "").strip()
                if fr and fs_raw and fe_raw:
                    try:
                        fs_i = int(fs_raw)
                        fe_i = int(fe_raw)
                    except ValueError:
                        fs_i = fe_i = -1
                    if fs_i >= 0 and fe_i >= 0:
                        fdir = resolve_egoexo_frame_dir(dataset_root, fr, args.egoexo_view)
                        if fdir is not None:
                            if need_kp:
                                both = angles_and_keypoints_from_frame_directory(
                                    fdir, fs_i, fe_i, mf
                                )
                                if both is not None:
                                    ang, kp = both
                                    if ang is not None:
                                        source_meta = f"{fdir}#{fs_i}-{fe_i}"
                            else:
                                ang = angles_from_frame_directory(fdir, fs_i, fe_i, mf)
                                if ang is not None:
                                    source_meta = f"{fdir}#{fs_i}-{fe_i}"
                        else:
                            top = fr.split("/")[0].strip() if fr else ""
                            if top and not (dataset_root / top).is_dir():
                                if not warned_missing_frame_tree:
                                    print(
                                        f"  (skip) missing {dataset_root / top!s} — "
                                        "annotations-only downloads have no frames. "
                                        "Run download_egoexo_fitness_dataset.py without --annotations-only "
                                        "(or --allow-patterns 'frames_open/<record_id>/**').",
                                        file=sys.stderr,
                                    )
                                    warned_missing_frame_tree = True
                            elif skipped < 3:
                                print(
                                    f"  (skip) no frame dir for stem={stem!r} "
                                    f"frame_root={fr!r} under {dataset_root} "
                                    f"(try --egoexo-view)",
                                    file=sys.stderr,
                                )

            if ang is None:
                skipped += 1
                if skipped <= 3 and not (row.get("frame_root") or "").strip():
                    print(
                        f"  (skip) no video or frame folder for stem={stem!r}",
                        file=sys.stderr,
                    )
                continue

            np.savez_compressed(
                outp,
                angles=ang,
                source_video=source_meta,
            )
            if need_kp:
                if kp is not None and kp.shape[0] == ang.shape[0]:
                    np.savez_compressed(kp_out, keypoints=kp.astype(np.float32))
                else:
                    print(
                        f"  (warn) --save-keypoints but no keypoints for stem={stem!r} — "
                        "only biomechanics written",
                        file=sys.stderr,
                    )
            done += 1
            print(f"ok {stem}  T={ang.shape[0]}")
            if args.max_videos and done >= args.max_videos:
                break

    print(f"Saved {done} angle files under {out_dir.resolve()}")
    if done == 0 and skipped > 0:
        print(
            f"No angle sequences saved ({skipped} rows skipped).\n"
            "  • Video: short_clips .mp4 under --dataset-root, or set video_path in the CSV.\n"
            "  • EgoExo: download frame folders, --dataset-root data/EgoExo-Fitness, "
            "and frame_root / frame_start / frame_end in the index.",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
