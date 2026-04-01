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

from biomechanical_features import process_npz_file
from pose_estimation_core import MEDIAPIPE_TO_COCO, apply_keypoint_preprocessing_pipeline

OUTPUT_STEM = "kaggle_exercise_recognition"
KAGGLE_SLUG = "muhannadtuameh/exercise-recognition"


def maybe_download() -> Path:
    try:
        import kagglehub
    except ImportError as e:
        raise SystemExit(
            "pip install kagglehub\n"
            "Or pass --dataset-root to an extracted exercise-recognition folder."
        ) from e
    p = kagglehub.dataset_download(KAGGLE_SLUG)
    print("Dataset path:", p)
    return Path(p)


def resolve_dataset_root(cli_path: str) -> Path:
    """CLI path, EXERCISE_RECOGNITION_ROOT, or newest kagglehub cache version with landmarks.csv."""
    if cli_path.strip():
        return Path(cli_path).expanduser().resolve()
    env = os.environ.get("EXERCISE_RECOGNITION_ROOT", "").strip()
    if env:
        return Path(env).expanduser().resolve()
    hub = Path.home() / ".cache/kagglehub/datasets/muhannadtuameh/exercise-recognition"
    if hub.is_dir():
        for v in sorted(hub.glob("versions/*"), key=lambda p: p.name, reverse=True):
            if v.is_dir() and (v / "landmarks.csv").is_file():
                print(f"Using kagglehub cache: {v}")
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
) -> List[str]:
    """Match ``keypoint_preprocessing_pipeline.py`` technique list order."""
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
    source_fps: float,
    target_fps: float,
    techniques: Sequence[str],
    skip_biomechanics: bool = False,
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
        "kaggle_slug": KAGGLE_SLUG,
        "pipeline": "kaggle_exercise_recognition_pipeline",
    }
    kp_path = out_dir / f"{OUTPUT_STEM}_keypoints.npz"
    np.savez_compressed(kp_path, **save_kw)

    if labels_list and len(labels_list) == fk.shape[0]:
        np.savez_compressed(
            out_dir / f"{OUTPUT_STEM}_labels.npz",
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
        "biomechanics_npz": f"{OUTPUT_STEM}_biomechanics.npz",
        "biomechanics_summary_json": f"{OUTPUT_STEM}_biomechanics_summary.json",
        "biomechanics_process_npz": bio_summary,
    }
    with open(out_dir / f"{OUTPUT_STEM}_pipeline_summary.json", "w") as f:
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
        default="./results/kaggle_exercise_recognition",
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
        "--skip-biomechanics",
        action="store_true",
        help="Only write keypoints NPZ (skip angle NPZ and process_npz_file summaries)",
    )

    args = ap.parse_args()

    try:
        if args.download:
            root = maybe_download()
        else:
            root = resolve_dataset_root(args.dataset_root)
    except FileNotFoundError as e:
        print(f"✗ {e}", file=sys.stderr)
        return 1

    if not (root / "landmarks.csv").is_file():
        print(f"✗ Not found: {root / 'landmarks.csv'}", file=sys.stderr)
        return 1

    techniques = build_preprocessing_techniques(
        no_fps_sync=args.no_fps_sync,
        bone_proportion=args.bone_proportion,
        laplacian_spatial=args.laplacian_spatial,
        dwt=args.dwt,
    )

    print("Preprocessing techniques:", techniques)
    print(f"FPS: source={args.source_fps} → target={args.target_fps}")

    out_dir = Path(args.output_dir)
    try:
        summary = run_pipeline(
            root,
            out_dir,
            source_fps=args.source_fps,
            target_fps=args.target_fps,
            techniques=techniques,
            skip_biomechanics=args.skip_biomechanics,
        )
    except Exception as e:
        print(f"✗ {e}", file=sys.stderr)
        raise

    print(f"\n✓ Integrated pipeline → {out_dir.resolve()}/")
    print(f"  {OUTPUT_STEM}_keypoints.npz  frames={summary['output_frames']}")
    if not args.skip_biomechanics:
        bio = summary.get("biomechanics_process_npz") or {}
        print(
            f"  {OUTPUT_STEM}_biomechanics.npz  angle frames={bio.get('num_frames')}"
        )
        print(f"  {OUTPUT_STEM}_biomechanics_summary.json")
    print(f"  {OUTPUT_STEM}_pipeline_summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
