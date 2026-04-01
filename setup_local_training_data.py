#!/usr/bin/env python3
"""
Copy precomputed *_biomechanics.npz (e.g. from processed_keypoints_mediapipe/) into
results/exercise_angles/ and build exercise_training_index_long_range.csv for train_exercise_bilstm.py.

This avoids re-running MediaPipe — optimal when angles already exist.

Long-range metadata.csv does not list specific exercise names (mostly "fitness").
Options:
  --pseudo-buckets N  → exercise_class = pseudo_{stem_int % N} for runnable multi-class smoke tests.
  --pseudo-buckets 0  → exercise_class from metadata exercise_type column (often single "fitness").

For real exercise labels, merge fine_grained_labels or edit the CSV manually.

Usage:
  ./venv/bin/python setup_local_training_data.py \\
    --source-dir results/processed_keypoints_mediapipe \\
    --dest-dir results/exercise_angles \\
    --output-csv results/exercise_training_index_long_range.csv \\
    --pseudo-buckets 8
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd


def _stem_from_biomech(name: str) -> str:
    return name.replace("_biomechanics.npz", "").replace(".npz", "")


def main() -> int:
    p = argparse.ArgumentParser(description="Copy local biomechanics NPZ + build training CSV")
    p.add_argument(
        "--source-dir",
        default="./results/processed_keypoints_mediapipe",
        help="Folder containing *_biomechanics.npz",
    )
    p.add_argument("--dest-dir", default="./results/exercise_angles")
    p.add_argument(
        "--output-csv",
        default="./results/exercise_training_index_long_range.csv",
    )
    p.add_argument(
        "--dataset-root",
        default="./qevd-fit-coach-data",
        help="Used to resolve videos/long_range/{stem}.mp4 for video_path",
    )
    p.add_argument(
        "--metadata",
        default="./qevd-fit-coach-data/metadata.csv",
        help="For exercise_type when --pseudo-buckets 0",
    )
    p.add_argument(
        "--pseudo-buckets",
        type=int,
        default=8,
        help="If >0, assign pseudo_{i%%N} classes. If 0, use metadata exercise_type.",
    )
    p.add_argument("--val-fraction", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    src = Path(args.source_dir)
    dst = Path(args.dest_dir)
    root = Path(args.dataset_root)
    dst.mkdir(parents=True, exist_ok=True)

    npz_files = sorted(src.glob("*_biomechanics.npz"))
    if not npz_files:
        print(f"No *_biomechanics.npz under {src.resolve()}", file=sys.stderr)
        return 1

    meta: Optional[pd.DataFrame] = None
    meta_path = Path(args.metadata)
    if args.pseudo_buckets == 0 and meta_path.is_file():
        meta = pd.read_csv(meta_path)
        # CSV may parse 0000 as int 0; normalize for stem lookup
        meta["_vid"] = meta["video_id"].map(
            lambda x: str(x).zfill(4) if str(x).strip().isdigit() else str(x).strip()
        )
        meta = meta.set_index("_vid", drop=False)

    rows: List[Dict[str, Any]] = []
    copied = 0
    rng = __import__("random").Random(args.seed)

    for f in npz_files:
        stem = _stem_from_biomech(f.name)
        out = dst / f"{stem}_biomechanics.npz"
        shutil.copy2(f, out)
        copied += 1

        vid_path = root / "videos" / "long_range" / f"{stem}.mp4"
        video_path_str = str(vid_path.resolve()) if vid_path.is_file() else ""

        if args.pseudo_buckets and args.pseudo_buckets > 0:
            # Stable bucket from stem (handles zero-padded ids)
            h = int(hashlib.md5(stem.encode()).hexdigest(), 16)
            b = h % args.pseudo_buckets
            ex = f"pseudo_{b}"
        else:
            if meta is not None and stem in meta.index:
                ex = str(meta.loc[stem, "exercise_type"]).lower().strip()
            else:
                ex = "fitness"

        split = "val" if rng.random() < args.val_fraction else "train"
        rows.append(
            {
                "video_stem": stem,
                "video_path": video_path_str,
                "video_key": f"videos/long_range/{stem}.mp4",
                "exercise_class": ex,
                "quality": round(0.65 + 0.3 * rng.random(), 4),
                "split": split,
                "num_labels": 1,
            }
        )

    outp = Path(args.output_csv)
    outp.parent.mkdir(parents=True, exist_ok=True)
    keys = list(rows[0].keys())
    with open(outp, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)

    print(f"Copied {copied} files → {dst.resolve()}")
    print(f"Wrote index → {outp.resolve()}  ({len(rows)} rows)")
    if args.pseudo_buckets and args.pseudo_buckets > 0:
        print(
            f"Using {args.pseudo_buckets} pseudo exercise buckets (not real exercise names — for pipeline test only)."
        )
    else:
        print(
            "exercise_class from metadata (often a single 'fitness' label). "
            "Use --pseudo-buckets 8 for multi-class smoke training, or edit CSV."
        )
    print("\nTrain with:")
    print(
        f"  ./venv/bin/python train_exercise_bilstm.py \\\n"
        f"    --index-csv {outp} \\\n"
        f"    --angles-dir {dst}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
