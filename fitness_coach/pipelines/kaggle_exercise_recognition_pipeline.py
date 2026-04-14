#!/usr/bin/env python3
"""
Kaggle: `muhannadtuameh/exercise-recognition` — **integrated steps 3 & 4** (same as QEVD capstone).

**Step 3 — preprocess keypoints:** ``apply_keypoint_preprocessing_pipeline`` with the same options as
``keypoint_preprocessing_pipeline.py``: skeleton (torso) normalization, spatial + temporal
imputation, optional graph-Laplacian spatial imputation, optional bone-proportion scaling,
FPS resampling to ``--target-fps``, optional DWT.

**Step 4 — biomechanical features:** eight planar joint angles via ``biomechanical_features``
(``process_npz_file`` + summary JSON), aligned with ``docs/CAPSTONE_PIPELINE.md`` §3–4.

Inputs are **pre-extracted MediaPipe** rows in ``landmarks.csv`` (33× x,y,z) + ``labels.csv``;
no video re-encoding.

Examples::

  pip install kagglehub
  ./venv/bin/python kaggle_exercise_recognition_pipeline.py --download

  # Riccardo Riccio dataset (video folders; runs MediaPipe — slow on full data):
  ./venv/bin/python kaggle_exercise_recognition_pipeline.py --download --riccio --riccio-max-videos 30

  ./venv/bin/python riccio_kaggle_video_pipeline.py --dataset-root ~/.cache/kagglehub/datasets/riccardoriccio/real-time-exercise-recognition-dataset/versions/3

  ./venv/bin/python kaggle_exercise_recognition_pipeline.py \\
      --dataset-root ~/.cache/kagglehub/datasets/muhannadtuameh/exercise-recognition/versions/5

  # Same preprocessing options as keypoint_preprocessing_pipeline.py:
  ./venv/bin/python kaggle_exercise_recognition_pipeline.py --download \\
      --laplacian-spatial --bone-proportion --source-fps 30 --target-fps 30
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

from fitness_coach.core.biomechanical_features import process_npz_file
from fitness_coach.core.pose_estimation_core import MEDIAPIPE_TO_COCO, apply_keypoint_preprocessing_pipeline

SLUG_MUHANNAD = "muhannadtuameh/exercise-recognition"
SLUG_RICCIO = "riccardoriccio/real-time-exercise-recognition-dataset"
DEFAULT_OUTPUT_STEM = "kaggle_exercise_recognition"
STEM_RICCIO = "riccio_realtime_exercise_recognition"
DEFAULT_OUTPUT_DIR_KAGGLE = "./results/kaggle_exercise_recognition"
DEFAULT_OUTPUT_DIR_RICCIO = "./results/riccio_realtime_exercise_recognition"


def hub_versions_root(slug: str) -> Path:
    """~/.cache/kagglehub/datasets/<owner>/<name>/versions"""
    parts = slug.strip().split("/", 1)
    if len(parts) != 2:
        raise ValueError(f"Expected owner/name Kaggle slug, got: {slug!r}")
    owner, name = parts
    return Path.home() / ".cache/kagglehub/datasets" / owner / name / "versions"


def maybe_download(slug: str) -> Path:
    try:
        import kagglehub
    except ImportError as e:
        raise SystemExit(
            "pip install kagglehub\n"
            "Or pass --dataset-root to an extracted exercise-recognition folder."
        ) from e
    p = kagglehub.dataset_download(slug)
    print("Dataset path:", p)
    return Path(p)


def resolve_dataset_root(cli_path: str, slug: str) -> Path:
    """CLI path, EXERCISE_RECOGNITION_ROOT, or newest kagglehub cache (landmarks.csv or Riccio video folders)."""
    from fitness_coach.pipelines.riccio_kaggle_video_pipeline import is_riccio_kaggle_video_layout

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
            if (v / "landmarks.csv").is_file():
                print(f"Using kagglehub cache: {v}")
                return v.resolve()
        for v in versions:
            if is_riccio_kaggle_video_layout(v):
                print(f"Using kagglehub cache (Riccio video layout, no landmarks.csv): {v}")
                return v.resolve()
    raise FileNotFoundError(
        "No dataset folder found. Pass --dataset-root, set EXERCISE_RECOGNITION_ROOT, "
        "or use --download (requires kagglehub)."
    )


def build_preprocessing_techniques(
    *,
    no_fps_sync: bool,
    bone_proportion: bool,
    laplacian_spatial: bool,
    dwt: bool,
    savgol: bool = False,
    kalman: bool = False,
) -> List[str]:
    """Match ``keypoint_preprocessing_pipeline.py`` technique list order."""
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


def load_mediapipe33_from_landmarks_csv(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(path)
    if "pose_id" not in df.columns:
        raise ValueError("Expected column pose_id in landmarks.csv")
    cols = [c for c in df.columns if c != "pose_id"]
    if len(cols) != 99:
        raise ValueError(f"Expected 99 xyz columns (33 joints), got {len(cols)}")
    arr = df[cols].to_numpy(dtype=np.float64)
    if np.any(~np.isfinite(arr)):
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    kp33 = arr.reshape(len(df), 33, 3)
    pose_ids = df["pose_id"].to_numpy()
    return kp33, pose_ids


def mediapipe33_to_coco17_xy(kp33: np.ndarray) -> np.ndarray:
    if kp33.ndim != 3 or kp33.shape[1] != 33:
        raise ValueError(f"Expected (T, 33, D), got {kp33.shape}")
    t = kp33.shape[0]
    out = np.zeros((t, 17, 2), dtype=np.float64)
    for coco_i, mp_i in enumerate(MEDIAPIPE_TO_COCO):
        out[:, coco_i, :] = kp33[:, mp_i, :2]
    return out


def run_pipeline(
    root: Path,
    out_dir: Path,
    *,
    output_stem: str,
    kaggle_slug: str,
    source_fps: float,
    target_fps: float,
    techniques: Sequence[str],
    skip_biomechanics: bool = False,
    savgol_window_length: int = 7,
    savgol_polyorder: int = 2,
    kalman_process_noise: float = 1e-4,
    kalman_measurement_noise: float = 1e-2,
) -> Dict[str, Any]:
    """
    Core integration: CSV → COCO-17 → apply_keypoint_preprocessing_pipeline → NPZ + angles.
    Returns a summary dict.
    """
    lm = root / "landmarks.csv"
    lab = root / "labels.csv"
    if not lm.is_file():
        raise FileNotFoundError(f"Missing {lm}")

    kp33, pose_ids = load_mediapipe33_from_landmarks_csv(lm)
    kp17_xy = mediapipe33_to_coco17_xy(kp33)
    t = kp17_xy.shape[0]
    conf = np.ones((t, 17), dtype=np.float32)

    labels_list: List[str] = []
    if lab.is_file():
        df_lab = pd.read_csv(lab)
        if "pose" in df_lab.columns and len(df_lab) == t:
            labels_list = [str(x) for x in df_lab["pose"].tolist()]

    keypoint_sequence = [kp17_xy[i].astype(np.float32) for i in range(t)]
    confidence_sequence = [conf[i] for i in range(t)]

    processed = apply_keypoint_preprocessing_pipeline(
        keypoint_sequence,
        confidence_sequence,
        preprocessing_techniques=list(techniques),
        target_fps=float(target_fps),
        source_fps=float(source_fps),
        original_frames=t,
        savgol_window_length=int(savgol_window_length),
        savgol_polyorder=int(savgol_polyorder),
        kalman_process_noise=float(kalman_process_noise),
        kalman_measurement_noise=float(kalman_measurement_noise),
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    fk = np.stack(processed["final_keypoints"], axis=0)
    conf_out = processed["confidence_sequence"]
    conf_arr = np.stack([c for c in conf_out], axis=0)

    save_kw: Dict[str, Any] = {
        "keypoints": fk.astype(np.float32),
        "confidence": conf_arr.astype(np.float32),
        "original_frames": np.array([t]),
        "techniques_json": json.dumps(processed.get("techniques_applied", {})),
        "source_fps": np.array([float(processed.get("source_fps", source_fps))]),
        "target_fps": np.array([float(processed.get("target_fps", target_fps))]),
        "dataset": str(root.resolve()),
        "kaggle_slug": kaggle_slug,
        "pipeline": "kaggle_exercise_recognition_pipeline",
    }
    kp_path = out_dir / f"{output_stem}_keypoints.npz"
    np.savez_compressed(kp_path, **save_kw)

    if labels_list and len(labels_list) == fk.shape[0]:
        np.savez_compressed(
            out_dir / f"{output_stem}_labels.npz",
            pose=np.array(labels_list, dtype=object),
            pose_id=pose_ids.astype(np.int64),
        )

    bio_summary: Dict[str, Any] = {}
    if not skip_biomechanics:
        # Step 4: same as keypoint script + biomechanical_features.batch_process_directory
        bio_summary = process_npz_file(kp_path)

    summary: Dict[str, Any] = {
        "dataset_root": str(root.resolve()),
        "input_frames": t,
        "output_frames": int(fk.shape[0]),
        "source_fps": float(source_fps),
        "target_fps": float(target_fps),
        "techniques_applied": processed.get("techniques_applied", {}),
        "technique_list": list(techniques),
        "keypoints_npz": kp_path.name,
        "biomechanics_npz": f"{output_stem}_biomechanics.npz",
        "biomechanics_summary_json": f"{output_stem}_biomechanics_summary.json",
        "biomechanics_process_npz": bio_summary,
    }
    with open(out_dir / f"{output_stem}_pipeline_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    return summary


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Integrate Kaggle exercise-recognition CSVs: step 3 preprocessing + step 4 joint angles."
    )
    ap.add_argument(
        "--download",
        action="store_true",
        help="Fetch dataset via kagglehub (cached under ~/.cache/kagglehub/...)",
    )
    ap.add_argument(
        "--dataset-root",
        default="",
        help="Folder with landmarks.csv (+ labels.csv). If omitted, uses EXERCISE_RECOGNITION_ROOT or newest kagglehub cache.",
    )
    ap.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR_KAGGLE,
        help="Output directory for NPZ + JSON (same layout as processed_keypoints_* )",
    )
    ap.add_argument(
        "--source-fps",
        type=float,
        default=30.0,
        help="Assumed native sampling rate of rows in landmarks.csv (used for FPS sync). Set to match recording if known.",
    )
    ap.add_argument("--target-fps", type=float, default=30.0)
    ap.add_argument(
        "--no-fps-sync",
        action="store_true",
        help="Skip temporal resampling to target-fps",
    )
    ap.add_argument(
        "--bone-proportion",
        action="store_true",
        help="After torso normalization: BioPose-style limb/torso ratios (arXiv:2501.07800)",
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
        help="Savitzky–Golay temporal smoothing (after FPS sync; preserves motion peaks vs box blur)",
    )
    ap.add_argument(
        "--kalman",
        action="store_true",
        help="1D constant-velocity Kalman smoothing per joint (alternative to --savgol; pick one)",
    )
    ap.add_argument("--savgol-window", type=int, default=7, help="SG filter window length (odd; capped by T)")
    ap.add_argument("--savgol-poly", type=int, default=2, help="SG polynomial order")
    ap.add_argument("--kalman-q", type=float, default=1e-4, help="Kalman process noise scale")
    ap.add_argument("--kalman-r", type=float, default=1e-2, help="Kalman measurement noise")
    ap.add_argument(
        "--skip-biomechanics",
        action="store_true",
        help="Only write keypoints NPZ (skip angle NPZ and process_npz_file summaries)",
    )
    ap.add_argument(
        "--riccio",
        action="store_true",
        help=f"Use Kaggle dataset {SLUG_RICCIO} (MediaPipe CSVs) and write outputs as {STEM_RICCIO}_*.npz.",
    )
    ap.add_argument(
        "--kaggle-slug",
        default="",
        help="Kaggle dataset id (owner/name). Default: muhannadtuameh/exercise-recognition; "
        f"--riccio sets {SLUG_RICCIO}.",
    )
    ap.add_argument(
        "--output-stem",
        default="",
        help="Prefix for *_keypoints.npz, *_labels.npz, *_biomechanics.npz. "
        f"Default: {DEFAULT_OUTPUT_STEM}, or {STEM_RICCIO} with --riccio.",
    )
    ap.add_argument(
        "--riccio-max-videos",
        type=int,
        default=0,
        help="Riccio video layout only: cap videos (0=all). Use e.g. 20 while testing.",
    )
    ap.add_argument(
        "--riccio-max-frames",
        type=int,
        default=0,
        help="Riccio video layout only: max frames per clip (0=full video).",
    )
    ap.add_argument(
        "--riccio-subsets",
        default="similar_dataset,final_kaggle_with_additional_video,synthetic_dataset,my_test_video_1",
        help="Comma-separated top-level folders to scan for Riccio video layout.",
    )
    ap.add_argument(
        "--riccio-skip-keypoints",
        action="store_true",
        help="Riccio video layout: only angles + labels (omit *_keypoints.npz for ST-GCN).",
    )
    ap.add_argument(
        "--riccio-raw-keypoints",
        action="store_true",
        help="Riccio video layout: skip normalize/impute/FPS (raw MediaPipe pixels). Default: full preprocessing.",
    )

    args = ap.parse_args()

    if args.riccio and args.kaggle_slug.strip():
        print("Use only one of --riccio or --kaggle-slug", file=sys.stderr)
        return 1

    if args.riccio:
        slug = SLUG_RICCIO
        output_stem = args.output_stem.strip() or STEM_RICCIO
    elif args.kaggle_slug.strip():
        slug = args.kaggle_slug.strip()
        output_stem = args.output_stem.strip() or DEFAULT_OUTPUT_STEM
    else:
        slug = SLUG_MUHANNAD
        output_stem = args.output_stem.strip() or DEFAULT_OUTPUT_STEM

    try:
        if args.download:
            root = maybe_download(slug)
        else:
            root = resolve_dataset_root(args.dataset_root, slug)
    except FileNotFoundError as e:
        print(f"✗ {e}", file=sys.stderr)
        return 1

    out_dir = Path(args.output_dir)
    if args.riccio and out_dir.resolve() == Path(DEFAULT_OUTPUT_DIR_KAGGLE).resolve():
        out_dir = Path(DEFAULT_OUTPUT_DIR_RICCIO)
    out_dir = out_dir.resolve()

    from fitness_coach.pipelines.riccio_kaggle_video_pipeline import is_riccio_kaggle_video_layout, run_riccio_video_to_npz

    if (root / "landmarks.csv").is_file():
        techniques = build_preprocessing_techniques(
            no_fps_sync=args.no_fps_sync,
            bone_proportion=args.bone_proportion,
            laplacian_spatial=args.laplacian_spatial,
            dwt=args.dwt,
            savgol=args.savgol,
            kalman=args.kalman,
        )

        print("Kaggle slug:", slug)
        print("Output stem:", output_stem)
        print("Preprocessing techniques:", techniques)
        print(f"FPS: source={args.source_fps} → target={args.target_fps}")

        try:
            summary = run_pipeline(
                root,
                out_dir,
                output_stem=output_stem,
                kaggle_slug=slug,
                source_fps=args.source_fps,
                target_fps=args.target_fps,
                techniques=techniques,
                skip_biomechanics=args.skip_biomechanics,
                savgol_window_length=args.savgol_window,
                savgol_polyorder=args.savgol_poly,
                kalman_process_noise=args.kalman_q,
                kalman_measurement_noise=args.kalman_r,
            )
        except Exception as e:
            print(f"✗ {e}", file=sys.stderr)
            raise

        print(f"\n✓ Integrated pipeline → {out_dir}/")
        print(f"  {output_stem}_keypoints.npz  frames={summary['output_frames']}")
        if not args.skip_biomechanics:
            bio = summary.get("biomechanics_process_npz") or {}
            print(
                f"  {output_stem}_biomechanics.npz  angle frames={bio.get('num_frames')}"
            )
            print(f"  {output_stem}_biomechanics_summary.json")
        print(f"  {output_stem}_pipeline_summary.json")
        return 0

    if is_riccio_kaggle_video_layout(root):
        print("Kaggle slug:", slug)
        print("Output stem:", output_stem)
        riccio_techniques = None
        if not args.riccio_raw_keypoints:
            riccio_techniques = build_preprocessing_techniques(
                no_fps_sync=args.no_fps_sync,
                bone_proportion=args.bone_proportion,
                laplacian_spatial=args.laplacian_spatial,
                dwt=args.dwt,
                savgol=args.savgol,
                kalman=args.kalman,
            )
            print("Keypoint preprocessing (before angles/features):", riccio_techniques)
            print(f"FPS: source={args.source_fps} → target={args.target_fps}")
        else:
            print("Riccio: --riccio-raw-keypoints (no normalize / impute / FPS sync)")
        print(
            "Riccio dataset: video folders (no landmarks.csv). MediaPipe → optional preprocessing → angles "
            "(use --riccio-max-videos N for a quick test)."
        )
        subsets = tuple(s.strip() for s in args.riccio_subsets.split(",") if s.strip())
        mf = args.riccio_max_frames if args.riccio_max_frames > 0 else None
        src_override = float(args.source_fps) if args.source_fps > 1e-6 else None
        try:
            summary = run_riccio_video_to_npz(
                root,
                out_dir,
                output_stem=output_stem,
                kaggle_slug=slug,
                subsets=subsets,
                max_videos=args.riccio_max_videos,
                max_frames=mf,
                skip_keypoints=args.riccio_skip_keypoints,
                raw_keypoints=args.riccio_raw_keypoints,
                preprocessing_techniques=riccio_techniques,
                source_fps=src_override,
                target_fps=float(args.target_fps),
                savgol_window_length=args.savgol_window,
                savgol_polyorder=args.savgol_poly,
                kalman_process_noise=args.kalman_q,
                kalman_measurement_noise=args.kalman_r,
            )
        except (FileNotFoundError, RuntimeError) as e:
            print(f"✗ {e}", file=sys.stderr)
            return 1

        print(f"\n✓ Riccio video pipeline → {out_dir}/")
        print(f"  {output_stem}_biomechanics.npz  T={summary['total_frames']}")
        if summary.get("keypoints_npz"):
            print(f"  {summary['keypoints_npz']} (ST-GCN)")
        print(f"  {output_stem}_labels.npz")
        print(f"  {output_stem}_pipeline_summary.json")
        print("Train BiLSTM:")
        print(
            f"  ./venv/bin/python train_exercise_bilstm.py --preset riccio --standardize --eval-test \\\n"
            f"    --kaggle-angles-dir {out_dir} \\\n"
            f"    --kaggle-stem {output_stem}"
        )
        if summary.get("keypoints_npz"):
            print("Train ST-GCN:")
            print(
                f"  ./venv/bin/python train_exercise_stgcn.py --standardize --eval-test \\\n"
                f"    --kaggle-keypoints-dir {out_dir} \\\n"
                f"    --kaggle-stem {output_stem}"
            )
        return 0

    print(
        f"✗ No landmarks.csv and not a Riccio video-folder layout: {root}\n"
        "  Expected either landmarks.csv (+ labels.csv) or folders like similar_dataset/, synthetic_dataset/.",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
