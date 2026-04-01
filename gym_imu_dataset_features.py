#!/usr/bin/env python3
"""
Gym Workout IMU (Kaggle: shakthisairam123/gym-workout-imu-dataset) — preprocessing & features.

**Not the same as this repo’s video pipeline.** ``pose_estimation_core`` + ``biomechanical_features``
expect **RGB video → 2D body keypoints → joint angles**. The Kaggle set is **wrist IMU**
time series (~100 Hz CSVs) with ``Activity`` / ``ActivityEncoded`` labels — there are no
pixel coordinates, so knee/elbow angles from COCO triplets do **not** apply.

This script provides a parallel workflow:
  **resample / normalize sensor streams → windowed IMU features** (mean, std, magnitude
  per axis per window) for exercise classification or as inputs to a sequence model.

Example after ``kagglehub.dataset_download("shakthisairam123/gym-workout-imu-dataset")``::

  ./venv/bin/python gym_imu_dataset_features.py --dataset-root /path/from/kagglehub --inspect
  ./venv/bin/python gym_imu_dataset_features.py --dataset-root /path --output-dir ./results/gym_imu_features

Optional: ``pip install kagglehub`` and use ``--download`` to fetch the dataset (same slug).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd


def maybe_download_kaggle_dataset() -> Path:
    try:
        import kagglehub
    except ImportError as e:
        raise SystemExit(
            "Install kagglehub: pip install kagglehub\n"
            "Or pass --dataset-root pointing to an already-downloaded folder."
        ) from e
    p = kagglehub.dataset_download("shakthisairam123/gym-workout-imu-dataset")
    print("Downloaded to:", p)
    return Path(p)


def find_csv_files(root: Path) -> List[Path]:
    return sorted(root.rglob("*.csv"))


def _numeric_feature_columns(df: pd.DataFrame) -> List[str]:
    skip = {
        "activity",
        "activityencoded",
        "label",
        "time",
        "timestamp",
        "timestamps",
    }
    cols: List[str] = []
    for c in df.columns:
        cl = c.strip().lower()
        if cl in skip:
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            cols.append(c)
    return cols


def inspect_csv(path: Path, nrows: int = 8) -> None:
    df = pd.read_csv(path, nrows=nrows)
    print(f"\n=== {path} ===")
    print("columns:", list(df.columns))
    print(df.head(nrows).to_string())
    num = _numeric_feature_columns(df)
    print("numeric columns (candidate):", num)


def sliding_window_features(
    x: np.ndarray,
    window: int,
    hop: int,
) -> Tuple[np.ndarray, int]:
    """
    x: (T, F) sensor features. Returns (num_windows, F*2) with mean and std per window per axis.
    """
    t, f = x.shape
    if t < window:
        return np.zeros((0, f * 2), dtype=np.float64), 0
    rows: List[np.ndarray] = []
    for start in range(0, t - window + 1, hop):
        w = x[start : start + window]
        m = np.mean(w, axis=0)
        s = np.std(w, axis=0)
        rows.append(np.concatenate([m, s]))
    return np.stack(rows, axis=0), len(rows)


def process_one_csv(
    csv_path: Path,
    window: int,
    hop: int,
    label_col: str,
) -> dict | None:
    df = pd.read_csv(csv_path)
    num_cols = _numeric_feature_columns(df)
    if not num_cols:
        return None
    x = df[num_cols].to_numpy(dtype=np.float64)
    if np.any(~np.isfinite(x)):
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    # Per-file normalization (remove scale differences across sessions)
    x = (x - np.mean(x, axis=0)) / (np.std(x, axis=0) + 1e-8)
    feats, nwin = sliding_window_features(x, window, hop)
    label = None
    if label_col in df.columns and len(df) > 0:
        label = str(df[label_col].iloc[0])
    return {
        "file": str(csv_path),
        "label": label,
        "window_features": feats,
        "feature_dim": int(feats.shape[1]) if feats.size else 0,
        "num_windows": nwin,
        "numeric_columns": num_cols,
    }


def main() -> int:
    ap = argparse.ArgumentParser(
        description="IMU CSV preprocessing + window features (Gym Workout IMU Kaggle dataset)."
    )
    ap.add_argument(
        "--dataset-root",
        default="",
        help="Folder containing CSVs (from kagglehub or manual download)",
    )
    ap.add_argument("--download", action="store_true", help="Download via kagglehub to cache")
    ap.add_argument("--inspect", action="store_true", help="Print schema of first few CSVs")
    ap.add_argument("--max-inspect", type=int, default=3, help="Number of CSVs to show in --inspect")
    ap.add_argument("--output-dir", default="./results/gym_imu_features", help="NPZ summaries per file")
    ap.add_argument("--window", type=int, default=100, help="Window length in samples (~1 s at 100 Hz)")
    ap.add_argument("--hop", type=int, default=50, help="Window hop")
    ap.add_argument(
        "--label-col",
        default="Activity",
        help="Column name for exercise label (if present)",
    )
    args = ap.parse_args()

    root: Path
    if args.download:
        root = maybe_download_kaggle_dataset()
    elif args.dataset_root:
        root = Path(args.dataset_root).resolve()
    else:
        print("Set --dataset-root or use --download", file=sys.stderr)
        return 1

    if not root.is_dir():
        print(f"Not a directory: {root}", file=sys.stderr)
        return 1

    csvs = find_csv_files(root)
    if not csvs:
        print(f"No CSV files under {root}", file=sys.stderr)
        return 1

    print(f"Found {len(csvs)} CSV file(s) under {root}")

    if args.inspect:
        for p in csvs[: max(1, args.max_inspect)]:
            inspect_csv(p)
        return 0

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    meta: List[dict] = []
    ok = 0
    for i, p in enumerate(csvs):
        r = process_one_csv(p, window=args.window, hop=args.hop, label_col=args.label_col)
        if r is None or r["num_windows"] == 0:
            continue
        stem = p.stem
        outp = out_dir / f"{stem}_imu_windows.npz"
        np.savez_compressed(
            outp,
            window_features=r["window_features"],
            label=np.array([r["label"] or ""], dtype=object),
            source_csv=str(p),
            numeric_columns=np.array(r["numeric_columns"], dtype=object),
        )
        meta.append(
            {
                "stem": stem,
                "npz": str(outp.name),
                "label": r["label"],
                "num_windows": r["num_windows"],
                "feature_dim": r["feature_dim"],
            }
        )
        ok += 1
        if ok <= 5 or (i + 1) % 100 == 0:
            print(f"  ✓ {stem}  windows={r['num_windows']}  dim={r['feature_dim']}")

    with open(out_dir / "imu_processing_summary.json", "w") as f:
        json.dump({"dataset_root": str(root), "files_written": ok, "items": meta}, f, indent=2)
    print(f"\n✓ Wrote {ok} NPZ file(s) under {out_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
