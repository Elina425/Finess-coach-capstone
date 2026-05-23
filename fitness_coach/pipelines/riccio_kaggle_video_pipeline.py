#!/usr/bin/env python3
"""
Build riccio_realtime_*_biomechanics.npz + *_keypoints.npz + *_labels.npz from the **video-folder**
layout of `riccardoriccio/real-time-exercise-recognition-dataset` (no landmarks.csv).

Pose estimation: **MediaPipe** (default), **Ultralytics YOLO26 COCO-17** (``--pose-backend yolo26``),
**ViTPose / RTMPose** via **rtmlib** (``vitpose`` or ``rtmpose_x``), or **``yolo26_then_vitpose``**
(YOLO26 per frame for detection and optional bilateral gating, then **ViTPose** on the same frame;
stages 2–6 use **ViTPose** keypoints + confidences). Install: ``pip install rtmlib onnxruntime`` (optional
``onnxruntime-gpu`` + ``--rtmlib-device cuda`` if supported).
Each frame is labeled with the parent folder name (e.g. squat, push-up) for BiLSTM (angles) and
ST-GCN (COCO-17 xy keypoints). ``*_labels.npz`` also stores ``video_id`` (int per frame) so
``build_kaggle_angle_datasets`` can split **train / val / test by source video** (stratified by class)
for BiLSTM without leakage across clips. Before angles/features, keypoints follow capstone step 3
(``apply_keypoint_preprocessing_pipeline``): **normalization** (torso-based scale, mitigates
camera distance), **imputation** (spatial + temporal for occluded / low-confidence joints), and
**FPS sync** to a common rate — consistent with robust skeletal-sequence practice discussed in
Jiang et al., *A Dual-Masked Auto-Encoder for Robust Motion Capture* (MM'22, DOI 10.1145/3503161.3547796)
regarding reliable joints and temporal consistency. Use ``--raw-keypoints`` to skip this (legacy).

Optional **bilateral frame filter** (``--bilateral-filter``): drop frames where both sides of a limb
pair fall below τ (default 0.3). By default **ankles are excluded** from that check (often cropped);
use ``--bilateral-include-ankles`` to require ankle confidence as well.

``*_keypoints.npz`` includes ``techniques_json`` when preprocessing runs so mixed-feature loaders
use ``coords_already_normalized=True`` (no double scaling).

Example (YOLO26 + PosePulse-style bilateral + raw CSV sidecar + NPZs)::

  ./venv/bin/python riccio_kaggle_video_pipeline.py \\
      --dataset-root ~/.cache/kagglehub/.../versions/3 \\
      --pose-backend yolo26 --bilateral-filter \\
      --export-csv-dir results/riccio_posepulse/raw_csv \\
      --output-dir results/riccio_posepulse --output-stem riccio_posepulse

Example (YOLO26 bilateral gate + ViTPose keypoints → same NPZ layout as pure YOLO26)::

  ./venv/bin/python riccio_kaggle_video_pipeline.py \\
      --dataset-root ~/.cache/kagglehub/.../versions/3 \\
      --pose-backend yolo26_then_vitpose --bilateral-filter \\
      --output-dir results/riccio_yolo_vitpose --output-stem riccio_yolo_vitpose

Example (ViTPose / RTMPose via rtmlib — same COCO-17 → preprocessing → ``(T,42)`` path as YOLO)::

  ./venv/bin/python riccio_kaggle_video_pipeline.py \\
      --dataset-root ~/.cache/kagglehub/.../versions/3 \\
      --pose-backend vitpose --rtmlib-device cpu \\
      --output-dir results/riccio_vitpose --output-stem riccio_vitpose

**Speed:** MediaPipe / YOLO / rtmlib pose in Python is **mostly CPU-bound** (Ultralytics can use GPU if
configured; rtmlib uses ONNX Runtime — use GPU builds + ``--rtmlib-device cuda`` when available).
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

# Running this file as ``python fitness_coach/pipelines/riccio_kaggle_video_pipeline.py`` puts only
# ``…/pipelines`` on sys.path; add repo root so ``import fitness_coach`` works without ``pip install -e .``.
_REPO_ROOT = str(Path(__file__).resolve().parents[2])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

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
    """Build technique tags for ``apply_keypoint_preprocessing_pipeline``.

    PosePulse Row 1 order inside the pipeline is always spatial → normalization → temporal when those
    tags are enabled; ``bone_proportion`` runs **after** temporal (optional extension). List order here
    is only used for membership + logging compatibility with ``kaggle_exercise_recognition_pipeline``.
    """
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
    pose_backend: str = "mediapipe",
    yolo_pose_model: str = "yolo26n-pose.pt",
    bilateral_filter: bool = False,
    bilateral_conf_tau: float = 0.3,
    bilateral_include_ankles: bool = False,
    export_csv_dir: str | None = None,
    rtmlib_device: str = "cpu",
    rtmlib_mode: str = "balanced",
    representation: str = "angles",
    vit_feature_encoder: str = "paper",
    vitpose_checkpoint: str | None = None,
    vit_model_name: str = "vit_small_patch16_224",
    vit_device: str = "cpu",
    bbox_margin: float = 0.12,
    resnet_variant: str = "resnet50",
    resnet_device: str = "cpu",
    strict_pipeline_crops: bool = True,
    save_conceptual_cleaned_keypoints: bool = True,
) -> Dict[str, Any]:
    vids = iter_riccio_videos(dataset_root, subsets)
    if not vids:
        raise FileNotFoundError(
            f"No videos under {dataset_root} (subsets={list(subsets)}). "
            "Expected folders like similar_dataset/, synthetic_dataset/, …"
        )
    if max_videos > 0:
        vids = vids[:max_videos]

    rep = str(representation or "angles").strip().lower()
    is_frame_emb = rep in ("vit_backbone", "resnet_backbone")

    angle_chunks: List[np.ndarray] = []
    feature_chunks: List[np.ndarray] = []
    conceptual_chunks: List[np.ndarray] = []
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
                "skip_keypoints": True if is_frame_emb else skip_keypoints,
                "raw_keypoints": raw_keypoints,
                "representation": rep,
                "vit_feature_encoder": str(vit_feature_encoder),
                "vitpose_checkpoint": str(Path(vitpose_checkpoint).resolve())
                if vitpose_checkpoint
                else None,
                "vit_model_name": str(vit_model_name),
                "vit_device": str(vit_device),
                "bbox_margin": float(bbox_margin),
                "resnet_variant": str(resnet_variant),
                "resnet_device": str(resnet_device),
                "strict_pipeline_crops": bool(strict_pipeline_crops),
                "save_conceptual_cleaned_keypoints": bool(save_conceptual_cleaned_keypoints),
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
                "pose_backend": "yolo26" if is_frame_emb else str(pose_backend),
                "yolo_pose_model": str(yolo_pose_model),
                "bilateral_filter": bool(bilateral_filter),
                "bilateral_conf_tau": float(bilateral_conf_tau),
                "bilateral_include_ankles": bool(bilateral_include_ankles),
                "export_csv_root": str(Path(export_csv_dir).resolve()) if export_csv_dir else None,
                "rtmlib_device": str(rtmlib_device),
                "rtmlib_mode": str(rtmlib_mode),
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
            f"Processing {len(jobs)} videos with {nw} parallel workers (CPU / pose; order preserved).",
            flush=True,
        )
        with ProcessPoolExecutor(max_workers=nw) as ex:
            results = list(ex.map(riccio_parallel_video_job, jobs))

    techniques_json_global: str | None = None
    for r in results:
        if r.get("ok") and r.get("techniques_json"):
            techniques_json_global = str(r["techniques_json"])
            break

    for r in results:
        if not r.get("ok"):
            skipped += 1
            continue
        cls = r["exercise_class"]
        if is_frame_emb:
            fe = r.get("frame_features")
            if fe is None or np.asarray(fe).size == 0:
                skipped += 1
                continue
            fe = np.asarray(fe, dtype=np.float32)
            if fe.ndim != 2 or fe.shape[0] == 0:
                skipped += 1
                continue
            t = int(fe.shape[0])
            feature_chunks.append(fe.astype(np.float32))
            if save_conceptual_cleaned_keypoints:
                ck = r.get("conceptual_cleaned_keypoints")
                if ck is None:
                    raise RuntimeError(
                        "Frame-embedding job missing conceptual_cleaned_keypoints — "
                        "use a current vit_frame_features / resnet_frame_features build or pass "
                        "--omit-conceptual-keypoints-from-npz to skip stacking Row-1 skeletons."
                    )
                ck = np.asarray(ck, dtype=np.float32)
                if ck.ndim != 3 or ck.shape[0] != t or ck.shape[1:] != (17, 2):
                    raise RuntimeError(
                        f"Bad conceptual_cleaned_keypoints shape {ck.shape} vs frame_features T={t}"
                    )
                conceptual_chunks.append(ck)
        else:
            ang = r.get("angles")
            kp = r.get("keypoints")
            if ang is None or np.asarray(ang).size == 0:
                skipped += 1
                continue
            ang = np.asarray(ang, dtype=np.float32)
            if ang.shape[0] == 0:
                skipped += 1
                continue
            if not skip_keypoints and (kp is None or np.asarray(kp).shape[0] != ang.shape[0]):
                skipped += 1
                continue
            t = int(ang.shape[0])
            angle_chunks.append(ang.astype(np.float32))
            if not skip_keypoints and kp is not None:
                kp_chunks.append(np.asarray(kp, dtype=np.float32))
        label_chunks.append(np.array([cls] * t, dtype=object))
        video_id_chunks.append(np.full(t, len(sources), dtype=np.int32))
        sources.append(str(r["source"]))

    if is_frame_emb:
        if not feature_chunks:
            raise RuntimeError(
                "No frame-feature sequences produced (YOLO26 crop encoder failed or no frames). "
                "For vit_backbone install ultralytics + ViTPose weights; "
                "for resnet_backbone install torchvision."
            )
        features_all = np.vstack(feature_chunks)
        save_ck = bool(save_conceptual_cleaned_keypoints)
        if save_ck:
            if len(conceptual_chunks) != len(feature_chunks):
                raise RuntimeError(
                    "Internal error: conceptual_cleaned_keypoints chunks != feature chunks"
                )
            conceptual_all = np.vstack(conceptual_chunks)
            if conceptual_all.shape[0] != features_all.shape[0]:
                raise RuntimeError(
                    "Internal error: conceptual_cleaned_keypoints T != frame_features T "
                    f"({conceptual_all.shape[0]} vs {features_all.shape[0]})"
                )
    else:
        if not angle_chunks:
            raise RuntimeError(
                "No angle sequences produced (pose estimation failed or no readable frames). "
                "Check OpenCV codecs, --pose-backend, ultralytics (for yolo26), or rtmlib (for vitpose / rtmpose_x)."
            )
        angles_all = np.vstack(angle_chunks)

    pose_all = np.concatenate(label_chunks)
    video_id_all = np.concatenate(video_id_chunks)
    kp_all: np.ndarray | None = None
    if not is_frame_emb and not skip_keypoints and kp_chunks:
        kp_all = np.vstack(kp_chunks)
        if kp_all.shape[0] != angles_all.shape[0]:
            raise RuntimeError("Internal error: keypoints/angles frame count mismatch")

    out_dir.mkdir(parents=True, exist_ok=True)
    bio_path = out_dir / f"{output_stem}_biomechanics.npz"
    lab_path = out_dir / f"{output_stem}_labels.npz"
    if is_frame_emb:
        bio_kw: Dict[str, Any] = {
            "frame_features": features_all,
            "feat_dim": np.int32(int(features_all.shape[1])),
        }
        if save_conceptual_cleaned_keypoints:
            bio_kw["conceptual_cleaned_keypoints"] = conceptual_all
        np.savez_compressed(bio_path, **bio_kw)
    else:
        np.savez_compressed(bio_path, angles=angles_all)
    np.savez_compressed(lab_path, pose=pose_all, video_id=video_id_all)
    kp_path: Path | None = None
    if kp_all is not None:
        kp_path = out_dir / f"{output_stem}_keypoints.npz"
        kp_kw: Dict[str, Any] = {"keypoints": kp_all}
        if techniques_json_global:
            kp_kw["techniques_json"] = np.array(techniques_json_global)
        np.savez_compressed(kp_path, **kp_kw)

    series = features_all if is_frame_emb else angles_all
    summary: Dict[str, Any] = {
        "dataset_root": str(dataset_root.resolve()),
        "kaggle_slug": kaggle_slug,
        "pipeline": "riccio_kaggle_video_pipeline",
        "num_videos_requested": len(vids),
        "num_videos_used": len(sources),
        "videos_skipped": skipped,
        "video_paths": sources,
        "total_frames": int(series.shape[0]),
        "angle_dim": int(series.shape[1]),
        "representation": rep,
        "vit_model_name": str(vit_model_name) if rep == "vit_backbone" else None,
        "vit_device": str(vit_device) if rep == "vit_backbone" else None,
        "resnet_variant": str(resnet_variant) if rep == "resnet_backbone" else None,
        "resnet_device": str(resnet_device) if rep == "resnet_backbone" else None,
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
        "pose_backend": str(pose_backend),
        "yolo_pose_model": str(yolo_pose_model),
        "bilateral_filter": bool(bilateral_filter),
        "bilateral_conf_tau": float(bilateral_conf_tau),
        "bilateral_include_ankles": bool(bilateral_include_ankles),
        "export_csv_dir": export_csv_dir,
        "rtmlib_device": str(rtmlib_device),
        "rtmlib_mode": str(rtmlib_mode),
        "preprocessing": (
            "raw_pixel_xy"
            if raw_keypoints
            else (preprocessing_techniques or ["normalization", "imputation", "fps_sync"])
        ),
        "target_fps": target_fps,
        "source_fps_override": source_fps,
        "strict_pipeline_crops": bool(strict_pipeline_crops) if is_frame_emb else None,
        "conceptual_cleaned_keypoints_in_biomechanics": (
            bool(save_conceptual_cleaned_keypoints) if is_frame_emb else None
        ),
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
    ap.add_argument(
        "--pose-backend",
        choices=("mediapipe", "yolo26", "vitpose", "rtmpose_x", "yolo26_then_vitpose"),
        default="mediapipe",
        help="Pose: MediaPipe; YOLO26 COCO-17; ViTPose (rtmlib); RTMPose-X large; or yolo26_then_vitpose "
        "(YOLO26 + bilateral gate, then ViTPose keypoints for preprocessing / 42-D features).",
    )
    ap.add_argument(
        "--yolo-pose-model",
        default="yolo26n-pose.pt",
        help="Ultralytics checkpoint when --pose-backend yolo26 (e.g. yolo26n-pose.pt, yolo26m-pose.pt).",
    )
    ap.add_argument(
        "--rtmlib-device",
        default="cpu",
        help="ONNX Runtime device for --pose-backend vitpose or rtmpose_x (e.g. cpu, cuda).",
    )
    ap.add_argument(
        "--rtmlib-mode",
        choices=("balanced", "lightweight", "performance"),
        default="balanced",
        help="rtmlib Body mode when --pose-backend vitpose (ignored for rtmpose_x, which uses performance).",
    )
    ap.add_argument(
        "--representation",
        choices=("angles", "vit_backbone", "resnet_backbone"),
        default="angles",
        help="angles: YOLO/MediaPipe → (T,8) biomechanics. vit_backbone: YOLO26 person crop → frozen ViTPose-S "
        "(256-D, default) or timm ViT (--vit-encoder timm). resnet_backbone: same crops → torchvision ResNet "
        "(512-D for ResNet-18/34, 2048-D for ResNet-50/101; --resnet-model).",
    )
    ap.add_argument(
        "--vit-encoder",
        choices=("paper", "timm"),
        default="paper",
        help="vit_backbone frame encoder: paper=ViTPose-S COCO (256-D, MAE ViT-S backbone); timm=ImageNet ViT (--vit-model).",
    )
    ap.add_argument(
        "--vitpose-checkpoint",
        default="",
        help="Optional path to td-hm_ViTPose-small_...pth; default downloads OpenMMLab weights to ~/.cache/fitness_coach/vitpose/.",
    )
    ap.add_argument(
        "--vit-model",
        default="vit_small_patch16_224",
        help="timm ViT model name when --representation vit_backbone and --vit-encoder timm (pip install timm).",
    )
    ap.add_argument(
        "--vit-device",
        default="cpu",
        help="Torch device for ViT when --representation vit_backbone (cpu, cuda, mps).",
    )
    ap.add_argument(
        "--resnet-model",
        choices=("resnet18", "resnet34", "resnet50", "resnet101"),
        default="resnet50",
        help="torchvision ResNet variant when --representation resnet_backbone.",
    )
    ap.add_argument(
        "--resnet-device",
        default="cpu",
        help="Torch device for ResNet when --representation resnet_backbone (cpu, cuda, mps).",
    )
    ap.add_argument(
        "--bbox-margin",
        type=float,
        default=0.12,
        help="Expand tight COCO-17 YOLO bbox for person crop before ViT / ResNet frame encoders.",
    )
    ap.add_argument(
        "--allow-detector-crop-fallback",
        action="store_true",
        help="For vit_backbone/resnet_backbone: if denormed pipeline skeleton has no visible joints, "
        "fall back to YOLO person box or raw keypoints for the crop (default: strict Row-1 geometry only; "
        "zero embedding when empty).",
    )
    ap.add_argument(
        "--omit-conceptual-keypoints-from-npz",
        action="store_true",
        help="Do not save conceptual_cleaned_keypoints (T,17,2) next to frame_features in *_biomechanics.npz.",
    )
    ap.add_argument(
        "--bilateral-filter",
        action="store_true",
        help="Drop frames where both sides of a limb pair have conf < --bilateral-conf-tau (PosePulse-style).",
    )
    ap.add_argument(
        "--bilateral-conf-tau",
        type=float,
        default=0.3,
        help="Confidence threshold τ for bilateral limb-pair filter (default 0.3).",
    )
    ap.add_argument(
        "--bilateral-include-ankles",
        action="store_true",
        help="Also require left/right ankle conf ≥ τ (default: ankles ignored — often cropped).",
    )
    ap.add_argument(
        "--export-csv-dir",
        default="",
        help="If set, write per-video CSVs of raw pixel keypoints (post bilateral) under <dir>/<exercise_class>/.",
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
            pose_backend=str(args.pose_backend),
            yolo_pose_model=str(args.yolo_pose_model),
            bilateral_filter=bool(args.bilateral_filter),
            bilateral_conf_tau=float(args.bilateral_conf_tau),
            bilateral_include_ankles=bool(args.bilateral_include_ankles),
            export_csv_dir=(args.export_csv_dir.strip() or None),
            rtmlib_device=str(args.rtmlib_device),
            rtmlib_mode=str(args.rtmlib_mode),
            representation=str(args.representation),
            vit_feature_encoder=str(args.vit_encoder),
            vitpose_checkpoint=(args.vitpose_checkpoint.strip() or None),
            vit_model_name=str(args.vit_model),
            vit_device=str(args.vit_device),
            bbox_margin=float(args.bbox_margin),
            resnet_variant=str(args.resnet_model),
            resnet_device=str(args.resnet_device),
            strict_pipeline_crops=not bool(args.allow_detector_crop_fallback),
            save_conceptual_cleaned_keypoints=not bool(args.omit_conceptual_keypoints_from_npz),
        )
    except (FileNotFoundError, RuntimeError) as e:
        print(f"✗ {e}", file=sys.stderr)
        return 1

    print(f"\n✓ Wrote {summary['biomechanics_npz']}  T={summary['total_frames']}")
    if summary.get("keypoints_npz"):
        print(f"  {summary['keypoints_npz']} (ST-GCN)")
    print(f"  Classes: {summary['classes_present']}")
    print("Train BiLSTM:")
    if str(args.representation) == "vit_backbone":
        print(
            f"  ./venv/bin/python train_exercise_bilstm.py --preset riccio --standardize --eval-test \\\n"
            f"    --kaggle-angles-dir {args.output_dir} \\\n"
            f"    --kaggle-stem {args.output_stem} \\\n"
            f"    --feature-mode vit_backbone"
        )
    elif str(args.representation) == "resnet_backbone":
        print(
            f"  ./venv/bin/python train_exercise_bilstm.py --preset paper_posepulse_resnet --standardize --eval-test \\\n"
            f"    --kaggle-angles-dir {args.output_dir} \\\n"
            f"    --kaggle-stem {args.output_stem}"
        )
        print(
            "  (equivalent: --feature-mode resnet_backbone with your preferred --preset / architecture)"
        )
    else:
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
