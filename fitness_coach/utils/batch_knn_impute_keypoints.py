#!/usr/bin/env python3
"""
Batch KMeans + KNN skeleton imputation (OPSTL-style, arXiv:2309.12029 Sec. II-D).

Run **after** standard per-video preprocessing (or on any folder of ``*_keypoints.npz``).
Requires **at least two** sequences. Writes ``*_keypoints_knn.npz`` + ``knn_imputation_meta.json``.

Example:
  ./venv/bin/python batch_knn_impute_keypoints.py \\
    --input-dir results/processed_keypoints_mediapipe \\
    --output-dir results/processed_keypoints_knn \\
    --n-clusters 8 --k-neighbors 5 --max-files 50
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from fitness_coach.core.knn_skeleton_imputation import impute_directory


def main() -> int:
    p = argparse.ArgumentParser(description="KMeans+KNN imputation for COCO-17 keypoint NPZ batches (OPSTL-style).")
    p.add_argument("--input-dir", required=True, help="Folder containing *_keypoints.npz")
    p.add_argument("--output-dir", required=True, help="Where to write *_keypoints_knn.npz")
    p.add_argument("--pattern", default="*_keypoints.npz", help="Glob under input-dir")
    p.add_argument("--target-t", type=int, default=32, help="Time resample length for clustering features")
    p.add_argument("--n-clusters", type=int, default=8, help="KMeans clusters (capped by #sequences)")
    p.add_argument("--k-neighbors", type=int, default=5, help="KNN neighbors inside cluster")
    p.add_argument("--seed", type=int, default=42, help="KMeans / sampling seed")
    p.add_argument("--max-files", type=int, default=0, help="Optional cap on NPZ files (0=all)")
    args = p.parse_args()

    try:
        n = impute_directory(
            Path(args.input_dir),
            Path(args.output_dir),
            pattern=args.pattern,
            target_t=args.target_t,
            n_clusters=args.n_clusters,
            k_neighbors=args.k_neighbors,
            seed=args.seed,
            max_files=args.max_files,
        )
    except ValueError as e:
        print(f"✗ {e}", file=sys.stderr)
        return 1

    print(f"✓ Wrote {n} files under {args.output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
