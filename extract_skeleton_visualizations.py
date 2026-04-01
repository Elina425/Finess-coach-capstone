#!/usr/bin/env python3
"""
Generate side-by-side skeleton overlays so you can judge which model fits the person best.

Rows = each loaded pose model. Columns = evenly spaced frames across the video (default 3).

Usage:
  ./venv/bin/python extract_skeleton_visualizations.py --videos path/a.mp4 path/b.mp4
  ./venv/bin/python extract_skeleton_visualizations.py --dataset-first 5
  ./venv/bin/python extract_skeleton_visualizations.py --video path/to/clip.mp4 --cols 4
  ./venv/bin/python extract_skeleton_visualizations.py --random-split 5 5 --seed 42
  # --video can be repeated; output: results/skeleton_comparison/skeleton_detection_<stem>.png
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path
from run_full_comparison import visualize_model_comparison


def _collect_mp4_under(folder: Path) -> list[Path]:
    if not folder.is_dir():
        return []
    # Prefer shallow *.mp4 (fast for flat long_range/ and short_clips/); fall back to rglob.
    # Only include paths that open as regular files (skip broken symlinks).
    direct = sorted(folder.glob("*.mp4"))
    if direct:
        return [p for p in direct if p.is_file()]
    return sorted(p for p in folder.rglob("*.mp4") if p.is_file())


def _random_long_short_videos(
    dataset_root: Path,
    n_long: int,
    n_short: int,
    seed: int | None,
) -> list[tuple[Path, str]]:
    """
    Return (path, output_stem) for PNG filenames: long_<stem> / short_<stem>.
    """
    root = Path(dataset_root)
    long_dir = root / "videos" / "long_range"
    short_dir = root / "videos" / "short_clips"
    long_paths = _collect_mp4_under(long_dir)
    short_paths = _collect_mp4_under(short_dir)
    if n_short > 0 and not short_paths:
        print(
            "⚠ videos/short_clips/: no readable .mp4 files (entries may be broken symlinks). "
            "Place the QEVD-FIT-COACH assets or copy real MP4s into that folder."
        )
    if n_long > 0 and not long_paths:
        print("⚠ videos/long_range/: no readable .mp4 files found.")
    rng = random.Random(seed)
    out: list[tuple[Path, str]] = []
    if n_long > 0 and long_paths:
        k = min(n_long, len(long_paths))
        for p in rng.sample(long_paths, k):
            out.append((p, f"long_{p.stem}"))
    if n_short > 0 and short_paths:
        k = min(n_short, len(short_paths))
        for p in rng.sample(short_paths, k):
            out.append((p, f"short_{p.stem}"))
    return out


def _resolve_dataset_videos(n: int, dataset_root: Path) -> list[Path]:
    try:
        from qevd_dataset_integration import QEVDDatasetLoader
    except ImportError as e:
        print(f"✗ Cannot load dataset: {e}")
        return []

    root = Path(dataset_root)
    if not root.is_dir():
        print(f"✗ Dataset not found: {root}")
        return []

    loader = QEVDDatasetLoader(str(root))
    ids = loader.list_videos()[:n]
    out: list[Path] = []
    for vid in ids:
        p = loader.get_video_path(vid)
        if p:
            out.append(Path(p))
    return out


def main() -> int:
    p = argparse.ArgumentParser(
        description="Export multi-model skeleton comparison PNGs for manual quality review."
    )
    p.add_argument(
        "--videos",
        nargs="*",
        default=[],
        help="One or more video file paths",
    )
    p.add_argument(
        "--video",
        action="append",
        default=[],
        metavar="PATH",
        help="Single video (repeatable); same as adding to --videos",
    )
    p.add_argument(
        "--dataset-first",
        type=int,
        metavar="N",
        default=0,
        help="Take first N videos from --dataset-root (default: skip)",
    )
    p.add_argument(
        "--dataset-root",
        default="./qevd-fit-coach-data",
        help="Root folder for --dataset-first and --random-split",
    )
    p.add_argument(
        "--random-split",
        nargs=2,
        type=int,
        metavar=("LONG", "SHORT"),
        default=None,
        help="Sample LONG random videos from videos/long_range/ and SHORT from "
        "videos/short_clips/ (e.g. 5 5 for five from each). Uses --seed.",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="RNG seed for --random-split (default: 42)",
    )
    p.add_argument(
        "--output",
        "-o",
        default="./results/skeleton_comparison",
        help="Output directory for PNG grids",
    )
    p.add_argument(
        "--cols",
        type=int,
        default=3,
        metavar="K",
        help="Number of sample frames (evenly spaced along the video timeline)",
    )
    args = p.parse_args()

    paths: list[Path] = [Path(v) for v in args.videos]
    paths.extend(Path(v) for v in args.video)
    if args.dataset_first > 0:
        paths.extend(_resolve_dataset_videos(args.dataset_first, Path(args.dataset_root)))

    jobs: list[tuple[Path, str | None]] = [(p, None) for p in paths]
    if args.random_split is not None:
        n_long, n_short = args.random_split
        if n_long < 0 or n_short < 0:
            print("✗ --random-split counts must be non-negative")
            return 1
        extra = _random_long_short_videos(
            Path(args.dataset_root),
            n_long,
            n_short,
            args.seed,
        )
        jobs.extend(extra)
        print(
            f"Random split (seed={args.seed}): {n_long} from long_range, "
            f"{n_short} from short_clips → {len(extra)} paths resolved"
        )

    if not jobs:
        if args.random_split is not None:
            print(
                "✗ --random-split resolved no usable .mp4 files (empty pool or broken symlinks)."
            )
        else:
            print("No videos specified. Examples:")
            print("  python extract_skeleton_visualizations.py --videos ./data/a.mp4")
            print("  python extract_skeleton_visualizations.py --dataset-first 3")
            print(
                "  python extract_skeleton_visualizations.py --random-split 5 5 "
                "--dataset-root ./qevd-fit-coach-data"
            )
        return 1

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Models load when each grid is built (first run may download weights).")
    print(f"Output directory: {out_dir.resolve()}\n")

    ok = 0
    for vp, out_stem in jobs:
        if not vp.is_file():
            if vp.is_symlink() and not vp.exists():
                print(
                    f"✗ Skip (broken symlink — target missing): {vp} → {vp.readlink()}"
                )
            else:
                print(f"✗ Skip (not a file): {vp}")
            continue
        print(f"Processing: {vp}")
        try:
            r = visualize_model_comparison(
                str(vp),
                output_dir=str(out_dir),
                frame_indices=None,
                num_sample_frames=max(1, args.cols),
                output_stem=out_stem,
            )
            if r:
                print(f"  Models in grid: {', '.join(r['models'])}")
                ok += 1
        except Exception as e:
            print(f"  ✗ Error: {e}")

    print(f"\nDone: {ok}/{len(jobs)} visualizations written under {out_dir}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
