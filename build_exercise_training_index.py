#!/usr/bin/env python3
"""
Stream fine_grained_labels.json + feedbacks_short_clips.json (ijson) and write a CSV index
for training: video_stem, exercise_class, quality_target, split, label_path_key.

Optionally filter to videos that exist on disk under dataset_root/videos/...
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from exercise_label_utils import (
    exercise_class_from_labels,
    heuristic_quality_score,
    normalize_class_name,
)


def _load_feedbacks_map(feedbacks_json: Path) -> Dict[str, List[str]]:
    """Path key -> feedback list (video_path as in JSON)."""
    import ijson

    out: Dict[str, List[str]] = {}
    with open(feedbacks_json, "rb") as f:
        for item in ijson.items(f, "item"):
            vp = item.get("video_path", "")
            fb = item.get("feedbacks") or []
            out[str(vp)] = list(fb)
    return out


def _resolved_mp4_path(dataset_root: Path, stem: str) -> Path:
    """Path under short_clips/; resolve() so valid symlinks count as present."""
    return (dataset_root / "videos" / "short_clips" / f"{stem}.mp4").resolve()


def _video_readable(rel: Path) -> bool:
    """True if file exists and is readable (follows symlinks to real files)."""
    try:
        return rel.is_file()
    except OSError:
        return False


def describe_mp4_path_status(p: Path) -> str:
    """
    Explain whether a short-clip path is usable. Broken symlinks are common: the .mp4
    exists in the folder listing but ``is_file()`` is False until the link target exists.
    """
    try:
        if p.is_file():
            return "readable"
    except OSError as e:
        return f"os error: {e}"
    if p.is_symlink():
        try:
            tgt = p.readlink()
            return f"broken symlink → {tgt}"
        except OSError as e:
            return f"symlink: {e}"
    return "missing (not a file)"


def stream_fine_grained(
    path: Path,
    feedbacks_map: Optional[Dict[str, List[str]]],
    dataset_root: Path,
    require_file: bool,
    max_rows: Optional[int],
    max_scan_items: Optional[int],
) -> tuple[List[Dict[str, Any]], int, int]:
    """
    Returns (rows, items_scanned, skipped_missing_file).
    """
    import ijson

    rows: List[Dict[str, Any]] = []
    items_scanned = 0
    skipped_missing = 0
    with open(path, "rb") as f:
        for item in ijson.items(f, "item"):
            items_scanned += 1
            if max_scan_items is not None and items_scanned > max_scan_items:
                break
            vp = item.get("video_path", "")
            labels = item.get("labels") or []
            split = item.get("split", "train")
            if not labels or not vp:
                continue
            stem = Path(vp).stem
            try:
                rel = _resolved_mp4_path(dataset_root, stem)
            except OSError:
                rel = dataset_root / "videos" / "short_clips" / f"{stem}.mp4"
            if require_file and not _video_readable(rel):
                skipped_missing += 1
                continue
            ex = exercise_class_from_labels(labels)
            ex_key = normalize_class_name(ex)
            fbs = None
            if feedbacks_map is not None:
                fbs = feedbacks_map.get(str(vp))
            q = heuristic_quality_score(labels, fbs)
            rows.append(
                {
                    "video_stem": stem,
                    "video_path": str(rel) if _video_readable(rel) else "",
                    "video_key": str(vp),
                    "exercise_class": ex_key,
                    "quality": q,
                    "split": split,
                    "num_labels": len(labels),
                }
            )
            if max_rows and len(rows) >= max_rows:
                break
    return rows, items_scanned, skipped_missing


def _first_readable_short_clip_mp4(
    dataset_root: Path,
    video_path: str,
    stem: str,
    short_clips_dir: Optional[Path],
) -> Optional[Path]:
    """
    Match JSON ``video_path`` (often ``videos/short_clips/NNNNNNNN.mp4``) to a local file.
    Tries: ``dataset_root / video_path``, ``short_clips_dir / {stem}.mp4`` if given,
    then ``dataset_root / videos / short_clips / {stem}.mp4``.
    """
    candidates: List[Path] = []
    vp = (video_path or "").strip().replace("\\", "/")
    if vp and not vp.startswith(".."):
        try:
            candidates.append((dataset_root / vp).resolve())
        except OSError:
            pass
    if short_clips_dir is not None:
        try:
            candidates.append((short_clips_dir / f"{stem}.mp4").resolve())
        except OSError:
            pass
    try:
        candidates.append(_resolved_mp4_path(dataset_root, stem))
    except OSError:
        candidates.append((dataset_root / "videos" / "short_clips" / f"{stem}.mp4").resolve())
    seen: set[str] = set()
    for p in candidates:
        key = str(p)
        if key in seen:
            continue
        seen.add(key)
        if _video_readable(p):
            return p
    return None


def resolve_short_clip_mp4_for_stem(
    dataset_root: Path,
    stem: str,
    short_clips_dir: Optional[Path] = None,
) -> Optional[Path]:
    """Resolve ``{stem}.mp4`` the same way as label streaming (relative JSON layout)."""
    vp = f"videos/short_clips/{stem}.mp4"
    return _first_readable_short_clip_mp4(dataset_root, vp, stem, short_clips_dir)


def print_diagnostic_first_label_paths(
    path: Path,
    dataset_root: Path,
    short_clips_dir: Optional[Path],
) -> None:
    """When no MP4s resolve, print how the first labelled row maps to filesystem paths."""
    import ijson

    with open(path, "rb") as f:
        for item in ijson.items(f, "item"):
            vp = item.get("video_path", "")
            labels = item.get("labels") or []
            if not labels or not vp:
                continue
            stem = Path(vp).stem
            print("  Diagnostic (first labelled row in JSON):", file=sys.stderr)
            print(f"    video_path: {vp!r}", file=sys.stderr)
            p_raw = dataset_root / vp
            try:
                p_res = p_raw.resolve()
            except OSError:
                p_res = p_raw
            print(
                f"    dataset_root / video_path → {p_res}\n"
                f"      status: {describe_mp4_path_status(p_raw)}",
                file=sys.stderr,
            )
            p_def_raw = dataset_root / "videos" / "short_clips" / f"{stem}.mp4"
            try:
                p_def = p_def_raw.resolve()
            except OSError:
                p_def = p_def_raw
            print(
                f"    default short_clips path → {p_def}\n"
                f"      status: {describe_mp4_path_status(p_def_raw)}",
                file=sys.stderr,
            )
            if short_clips_dir is not None:
                p_alt_raw = short_clips_dir / f"{stem}.mp4"
                try:
                    p_alt = p_alt_raw.resolve()
                except OSError:
                    p_alt = p_alt_raw
                print(
                    f"    --short-clips-dir → {p_alt}\n"
                    f"      status: {describe_mp4_path_status(p_alt_raw)}",
                    file=sys.stderr,
                )
            sc_dir = dataset_root / "videos" / "short_clips"
            if sc_dir.is_dir():
                print(
                    "  Sample entries in dataset_root/videos/short_clips/ (first 5 .mp4):",
                    file=sys.stderr,
                )
                shown = 0
                for f in sorted(sc_dir.glob("*.mp4"))[:5]:
                    print(f"    {f.name}: {describe_mp4_path_status(f)}", file=sys.stderr)
                    shown += 1
                if shown == 0:
                    print("    (no *.mp4 in this directory)", file=sys.stderr)
            print(
                "  Hint: listing a file in the folder is not enough — Python needs a "
                "**readable** file. Fix **broken symlinks** (mount/copy the target) or "
                "replace with real MP4s.",
                file=sys.stderr,
            )
            return


def stream_unique_stems_from_fine_grained(
    path: Path,
    dataset_root: Path,
    max_stems: int,
    max_scan_items: Optional[int] = None,
    short_clips_dir: Optional[Path] = None,
) -> tuple[List[str], int, int]:
    """
    Unique short-clip stems from ``fine_grained_labels.json`` in order of first appearance,
    requiring a readable local ``.mp4`` (see ``_first_readable_short_clip_mp4``).
    Stops once ``max_stems`` stems are collected (for keypoint preprocessing aligned with BiLSTM labels).

    Returns (stems, items_scanned, skipped_missing_file).
    """
    import ijson

    seen: set[str] = set()
    stems: List[str] = []
    items_scanned = 0
    skipped_missing = 0
    with open(path, "rb") as f:
        for item in ijson.items(f, "item"):
            items_scanned += 1
            if max_scan_items is not None and items_scanned > max_scan_items:
                break
            vp = item.get("video_path", "")
            labels = item.get("labels") or []
            if not labels or not vp:
                continue
            stem = Path(vp).stem
            if stem in seen:
                continue
            found = _first_readable_short_clip_mp4(dataset_root, vp, stem, short_clips_dir)
            if found is None:
                skipped_missing += 1
                continue
            seen.add(stem)
            stems.append(stem)
            if len(stems) >= max_stems:
                break
    return stems, items_scanned, skipped_missing


def main() -> int:
    p = argparse.ArgumentParser(description="Build CSV index for exercise BiLSTM training.")
    p.add_argument(
        "--fine-labels",
        default="./qevd-fit-coach-data/annotations/labels/fine_grained_labels.json",
    )
    p.add_argument(
        "--feedbacks",
        default="./qevd-fit-coach-data/annotations/labels/feedbacks_short_clips.json",
    )
    p.add_argument("--dataset-root", default="./qevd-fit-coach-data")
    p.add_argument("--output", default="./results/exercise_training_index.csv")
    p.add_argument(
        "--no-require-file",
        action="store_true",
        help="Include rows even if short_clips mp4 is missing (for syllabus-only)",
    )
    p.add_argument("--max-rows", type=int, default=0, help="Cap rows (0 = no cap)")
    p.add_argument(
        "--max-scan",
        type=int,
        default=0,
        help="Max JSON array items to read (0 = auto: 2M if require_file, else unlimited). "
        "Stops early if your short_clips are mostly broken symlinks.",
    )
    p.add_argument("--no-feedbacks", action="store_true")
    args = p.parse_args()

    try:
        import ijson  # noqa: F401
    except ImportError:
        print("Install ijson: pip install ijson", file=sys.stderr)
        return 1

    fb_map = None
    if not args.no_feedbacks and Path(args.feedbacks).is_file():
        print("Loading feedbacks map...")
        fb_map = _load_feedbacks_map(Path(args.feedbacks))

    max_rows = args.max_rows if args.max_rows > 0 else None
    require_file = not args.no_require_file
    if args.max_scan > 0:
        max_scan = args.max_scan
    elif require_file:
        max_scan = 2_000_000
    else:
        max_scan = None

    rows, scanned, skipped = stream_fine_grained(
        Path(args.fine_labels),
        fb_map,
        Path(args.dataset_root),
        require_file=require_file,
        max_rows=max_rows,
        max_scan_items=max_scan,
    )
    outp = Path(args.output)
    outp.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        print(
            f"No rows written (scanned {scanned} label entries, skipped {skipped} missing videos).\n"
            "  • short_clips are often broken symlinks — the .mp4 is not readable on this machine.\n"
            "  • Build the CSV anyway (labels only; video_path may be empty until you add MP4s):\n"
            "      ./venv/bin/python build_exercise_training_index.py --no-require-file\n"
            "    Optional: cap size for a quick test:\n"
            "      ./venv/bin/python build_exercise_training_index.py --no-require-file --max-rows 5000\n"
            "  • Or restore/copy real .mp4 files into qevd-fit-coach-data/videos/short_clips/\n"
            "    so require-file mode can resolve paths.",
            file=sys.stderr,
        )
        return 1

    keys = list(rows[0].keys())
    with open(outp, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)

    classes = sorted({r["exercise_class"] for r in rows})
    meta = {"num_rows": len(rows), "num_classes": len(classes), "classes": classes}
    with open(outp.with_suffix(".meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"Wrote {len(rows)} rows → {outp.resolve()}")
    print(f"Unique exercise classes: {len(classes)}")
    print(f"(scanned {scanned} entries, skipped {skipped} without a local video file)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
