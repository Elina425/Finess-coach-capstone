#!/usr/bin/env python3
"""
Build multi-model skeleton grid PNGs (same style as ``run_full_comparison``) for Riccio clips
whose path contains a folder segment matching ``--exercise`` (e.g. ``squat``, ``shoulder press``).

Uses ``discover_mp4_paths``, random-samples ``--num-videos`` clips, loads pose detectors once, then
``visualize_model_comparison`` per clip.

Examples::

  PYTHONPATH=. ./venv/bin/python generate_riccio_squat_skeleton_grids.py \\
    --dataset-root \"$HOME/.cache/kagglehub/.../versions/3\" \\
    --exercise squat --output-dir results/squat_skeleton_grids --num-videos 30 --seed 42

  PYTHONPATH=. ./venv/bin/python generate_riccio_squat_skeleton_grids.py \\
    --dataset-root \"$HOME/.cache/kagglehub/.../versions/3\" \\
    --exercise \"shoulder press\" --output-dir results/shoulder_press_skeleton_grids
"""

from __future__ import annotations

import argparse
import random
import re
import sys
from pathlib import Path

from fitness_coach.core.pose_estimation_core import iter_default_comparison_detectors
from fitness_coach.pipelines.run_full_comparison import discover_mp4_paths, visualize_model_comparison


def _exercise_slug(name: str) -> str:
    s = name.strip().lower().replace(" ", "_")
    return re.sub(r"[^a-z0-9_-]+", "_", s).strip("_") or "exercise"


def _clips_for_exercise(all_paths: list[Path], exercise: str) -> list[Path]:
    e = exercise.strip().casefold()
    return [p for p in all_paths if any(part.casefold() == e for part in p.parts)]


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Skeleton grid PNGs for Riccio videos matching a coarse exercise folder name"
    )
    ap.add_argument("--dataset-root", type=Path, required=True)
    ap.add_argument(
        "--exercise",
        type=str,
        default="squat",
        help='Folder name to match in paths (e.g. "squat", "shoulder press", "push-up")',
    )
    ap.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Where to save PNGs (default: results/<exercise_slug>_skeleton_grids)",
    )
    ap.add_argument("--num-videos", type=int, default=30, help="Random clips (capped by pool size)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--num-sample-frames", type=int, default=3, help="Columns in each grid")
    args = ap.parse_args()

    exercise = str(args.exercise).strip()
    if not exercise:
        print("ERROR: --exercise must be non-empty", file=sys.stderr)
        return 1

    slug = _exercise_slug(exercise)
    out_dir = args.output_dir
    if out_dir is None:
        out_dir = Path("results") / f"{slug}_skeleton_grids"
    out_dir = out_dir.expanduser().resolve()

    root = args.dataset_root.expanduser().resolve()
    if not root.is_dir():
        print(f"ERROR: not a directory: {root}", file=sys.stderr)
        return 1

    all_paths, scan = discover_mp4_paths(str(root))
    pool = _clips_for_exercise(all_paths, exercise)
    if not pool:
        print(f"ERROR: no '{exercise}' .mp4 under {root} ({scan})", file=sys.stderr)
        return 1

    rng = random.Random(int(args.seed))
    k = min(int(args.num_videos), len(pool))
    picked = rng.sample(sorted(set(pool)), k)

    out_dir.mkdir(parents=True, exist_ok=True)

    detectors = list(iter_default_comparison_detectors())
    if not detectors:
        print("ERROR: no pose detectors available", file=sys.stderr)
        return 1

    print(f"{exercise!r} pool: {len(pool)} clips → sampling {len(picked)} (seed={args.seed})")
    print(f"Output: {out_dir}\n")

    ok = 0
    for i, p in enumerate(picked):
        stem_tag = f"{p.parent.name}_{p.stem}"
        print(f"[{i + 1}/{len(picked)}] {stem_tag}")
        try:
            r = visualize_model_comparison(
                str(p),
                output_dir=str(out_dir),
                output_stem=stem_tag,
                num_sample_frames=int(args.num_sample_frames),
                detectors=detectors,
            )
            if r:
                ok += 1
        except Exception as e:
            print(f"  ✗ {e}", file=sys.stderr)

    print(f"\nDone: {ok}/{len(picked)} PNGs in {out_dir}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
