#!/usr/bin/env python3
"""
Download the gated [EgoExo-Fitness](https://huggingface.co/datasets/Lymann/EgoExo-Fitness) snapshot
to a local folder (requires Hugging Face account + accepting dataset terms + token).

The **full** dataset is very large (tens of GB, mostly ``frames_open`` / ``features_open`` / ``img``).
If the download fails with **No space left on device**, either:

  • Use ``--annotations-only`` to fetch only ``raw_annotations`` (small; enough to build the quality CSV),
    then supply videos from another source or download frames to an **external drive**; or
  • Set ``--local-dir`` to a volume with **≥ ~40 GB** free (full snapshot).

  # Auth: accept the dataset terms on the Hub, then either log in once:
  #   huggingface-cli login
  # or set a **read** token for this shell only (never commit the value):
  #   export HF_TOKEN="hf_..."
  ./venv/bin/python download_egoexo_fitness_dataset.py --local-dir data/EgoExo-Fitness

  # Small download (labels / quality JSON only):
  ./venv/bin/python download_egoexo_fitness_dataset.py --annotations-only --local-dir data/EgoExo-Fitness

Then point ``build_egoexo_fitness_index.py --annotations-json`` at ``raw_annotations/*.json`` under
that folder.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path


def _disk_free_gb(path: Path) -> float:
    """
    Free space (GB) on the volume that would hold ``path``. Does not create directories:
    creating ``/mnt/...`` on macOS hits a read-only ``/mnt`` and used to crash here.
    """
    p = path.expanduser()
    try:
        p = p.resolve()
    except OSError:
        p = path.expanduser()
    cur: Path = p if p.is_dir() else p.parent
    while not cur.exists():
        parent = cur.parent
        if parent == cur:
            cur = Path.cwd()
            break
        cur = parent
    return shutil.disk_usage(cur).free / (1024.0**3)


def main() -> int:
    ap = argparse.ArgumentParser(description="Download Lymann/EgoExo-Fitness from Hugging Face Hub")
    ap.add_argument(
        "--repo-id",
        default="Lymann/EgoExo-Fitness",
        help="Dataset repo id on the Hub",
    )
    ap.add_argument(
        "--local-dir",
        type=str,
        default="data/EgoExo-Fitness",
        help="Directory to snapshot files into",
    )
    ap.add_argument(
        "--token",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Use Hub authentication (default: True). Use --no-token only for public assets.",
    )
    ap.add_argument(
        "--hf-token",
        default=None,
        metavar="STRING",
        help="Optional token string. Prefer env HF_TOKEN (avoids shell history). Never commit tokens.",
    )
    ap.add_argument(
        "--annotations-only",
        action="store_true",
        help="Skip large folders (frames_open, features_open, img); keep raw_annotations + small files.",
    )
    ap.add_argument(
        "--ignore-patterns",
        action="append",
        default=[],
        metavar="PATTERN",
        help="Extra gitignore-style patterns to skip (repeatable). Example: --ignore-patterns 'frames_open/**'",
    )
    ap.add_argument(
        "--allow-patterns",
        action="append",
        default=[],
        metavar="PATTERN",
        help="If set, only download paths matching these patterns (repeatable).",
    )
    ap.add_argument(
        "--min-free-gb",
        type=float,
        default=None,
        help="Abort if destination disk has less than this many GB free. "
        "Default: 1.0 with --annotations-only, else 40.0. Set 0 to disable.",
    )
    args = ap.parse_args()

    dest = Path(args.local_dir)
    if args.min_free_gb is None:
        min_gb = 1.0 if args.annotations_only else 40.0
    else:
        min_gb = float(args.min_free_gb)

    if min_gb > 0:
        free = _disk_free_gb(dest)
        if free < min_gb:
            print(
                f"Not enough free disk space on this volume: {free:.1f} GB available (destination "
                f"{dest}), need at least {min_gb:.1f} GB for a full snapshot. "
                "This is not about RAM: PyTorch DataLoader / batching only helps at train time; "
                "Hub download still needs room to store files. "
                "Options: --local-dir on a larger or external drive, --annotations-only, "
                "--allow-patterns for a subset, or --min-free-gb 0 to skip this check (if you accept the risk).",
                file=sys.stderr,
            )
            return 1

    ignore: list[str] = list(args.ignore_patterns or [])
    if args.annotations_only:
        for pat in ("frames_open/**", "features_open/**", "img/**"):
            if pat not in ignore:
                ignore.append(pat)

    allow = list(args.allow_patterns) if args.allow_patterns else None

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print(
            "Install into the same venv as python:  ./venv/bin/python -m pip install huggingface_hub",
            file=sys.stderr,
        )
        return 1
    try:
        from huggingface_hub.utils import HfHubHTTPError as _HubHttpErr
    except ImportError:  # pragma: no cover
        _HubHttpErr = None

    if args.token:
        if args.hf_token and str(args.hf_token).strip():
            hub_token: str | bool = str(args.hf_token).strip()
        elif (env_t := os.environ.get("HF_TOKEN", "").strip()):
            hub_token = env_t
        else:
            hub_token = True
    else:
        hub_token = False

    kw: dict = {
        "repo_id": args.repo_id,
        "repo_type": "dataset",
        "local_dir": args.local_dir,
        "token": hub_token,
    }
    if ignore:
        kw["ignore_patterns"] = ignore
    if allow:
        kw["allow_patterns"] = allow

    try:
        path = snapshot_download(**kw)
    except OSError as e:
        if getattr(e, "errno", None) == 28 or "No space left" in str(e):
            print(
                f"{e}\n"
                "Full EgoExo-Fitness needs tens of GB. Use a larger disk, --annotations-only, "
                "or --allow-patterns for a subset.",
                file=sys.stderr,
            )
            return 1
        raise
    except Exception as e:
        if _HubHttpErr is not None and isinstance(e, _HubHttpErr):
            print(
                f"Hub request failed ({e}). For gated datasets: accept terms on the dataset page, "
                "then set HF_TOKEN to a read token or run `huggingface-cli login`. "
                "Do not commit tokens to git.",
                file=sys.stderr,
            )
            return 1
        low = str(e).lower()
        if any(x in low for x in ("401", "403", "gated", "repository not found", "not found")):
            print(
                f"Download failed ({e}). For gated datasets: accept terms on the Hub, set HF_TOKEN, "
                "or run `huggingface-cli login`. Do not commit tokens to git.",
                file=sys.stderr,
            )
            return 1
        raise
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
