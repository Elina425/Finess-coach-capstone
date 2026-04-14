#!/usr/bin/env python3
"""
Build ``exercise_training_index``-style CSV from
`EgoExo-Fitness <https://huggingface.co/datasets/Lymann/EgoExo-Fitness>`__
using the layout described in the dataset
`README <https://huggingface.co/datasets/Lymann/EgoExo-Fitness/blob/main/README.md>`__.

**Released data (per README):** synchronized egocentric/exocentric **30 fps frame folders**
(``frames_open/...``) and **CLIP-B features**; **raw MP4 is not distributed** (contact authors if
needed). Quality supervision comes from ``raw_annotations/interpretable_action_judgement.json``:
each clip key (e.g. ``ThEnUZ_action_1``) has ``frame_root``, ``st_ed_frame``, and
``annotations[]`` with ``action_quality_score`` (paper example uses integers like 1–5), ``action_name``,
``comment``, ``action_guidance``.

This script:

 - **``--format interpretable``** (or **auto** when the JSON matches): one CSV row per annotator
    entry; columns ``judgement_key``, ``frame_root``, ``frame_start``, ``frame_end`` reference frames
    for a future frame-sequence → angles step. ``video_path`` is left empty (no MP4 in release).
  - **``--format generic``**: legacy deep-search for flat dicts with quality + name + optional video keys.

``action_quality_score`` is written to CSV column ``quality``. By default (**``--quality-scale raw``**) the
numeric score is kept as-is (e.g. paper README uses integer 1–5). Use ``1-5``, ``auto``, etc. only if you
want targets rescaled to **[0, 1]**.

Citation: Li et al., ECCV 2024 (*EgoExo-Fitness: Towards Egocentric and Exocentric Full-Body Action Understanding*).

Example (official judgement file)::

  ./venv/bin/python build_egoexo_fitness_index.py \\
    --annotations-json data/EgoExo-Fitness/raw_annotations/interpretable_action_judgement.json \\
    --dataset-root data/EgoExo-Fitness \\
    --format interpretable \\
    --output results/egoexo_fitness_index.csv

  ./venv/bin/python split_exercise_index.py \\
    --input results/egoexo_fitness_index.csv \\
    --output results/egoexo_fitness_index_split.csv

  ./venv/bin/python train_exercise_bilstm.py \\
    --index-csv results/egoexo_fitness_index_split.csv \\
    --angles-dir results/egoexo_exercise_angles \\
    --standardize

  # No frames: annotation-sequence BiLSTM (verification_json + comment/guidance)
  ./venv/bin/python train_exercise_annotation_bilstm.py \\
    --index-csv results/egoexo_fitness_index_split.csv \\
    --standardize --output-dir results/exercise_annotation_bilstm_egoexo

**Angles:** run ``batch_compute_angles_for_index.py`` with ``--dataset-root`` pointing at the
EgoExo-Fitness tree (downloaded frames). It reads ``frame_root`` / ``frame_start`` / ``frame_end`` and
``--egoexo-view`` (e.g. ``ego_l``) under each record's ``frames_open/...`` folder.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from fitness_coach.core.exercise_label_utils import normalize_class_name, parse_base_exercise

QUALITY_KEYS: Sequence[str] = (
    "action_quality_score",
    "quality_score",
    "score",
    "form_score",
    "performance_score",
)
NAME_KEYS: Sequence[str] = (
    "action_name",
    "exercise",
    "exercise_name",
    "label",
    "action",
)
VIDEO_KEYS: Sequence[str] = (
    "video_path",
    "video_file",
    "filepath",
    "path",
    "clip_path",
    "rgb_video",
    "video",
)


def _pick(d: Dict[str, Any], keys: Sequence[str]) -> Any:
    for k in keys:
        if k not in d:
            continue
        v = d[k]
        if v is None:
            continue
        s = str(v).strip()
        if s == "":
            continue
        return v
    return None


def _iter_nested_dicts(node: Any) -> Iterable[Dict[str, Any]]:
    if isinstance(node, dict):
        yield node
        for v in node.values():
            yield from _iter_nested_dicts(v)
    elif isinstance(node, list):
        for v in node:
            yield from _iter_nested_dicts(v)


def _is_annotation_leaf(d: Dict[str, Any]) -> bool:
    q = _pick(d, QUALITY_KEYS)
    name = _pick(d, NAME_KEYS)
    return q is not None and name is not None


def _normalize_quality(raw: Any, scale: str) -> float:
    if scale == "raw":
        x = float(raw)
        if not math.isfinite(x):
            raise ValueError(f"non-finite quality: {raw!r}")
        return x
    x = float(raw)
    if scale == "0-1":
        return float(max(0.0, min(1.0, x)))
    if scale == "1-5":
        return float(max(0.0, min(1.0, (x - 1.0) / 4.0)))
    if scale == "1-10":
        return float(max(0.0, min(1.0, (x - 1.0) / 9.0)))
    if scale == "percent":
        return float(max(0.0, min(1.0, x / 100.0)))
    if scale == "auto":
        if 0.0 <= x <= 1.0:
            return float(x)
        if 1.0 <= x <= 5.0:
            return float(max(0.0, min(1.0, (x - 1.0) / 4.0)))
        if 1.0 <= x <= 10.0:
            return float(max(0.0, min(1.0, (x - 1.0) / 9.0)))
        if x > 1.0 and x <= 100.0:
            return float(max(0.0, min(1.0, x / 100.0)))
        raise ValueError(f"Cannot auto-scale quality value: {x!r}")
    raise ValueError(f"Unknown --quality-scale: {scale}")


def _resolve_video_path(
    raw: Any,
    dataset_root: Path,
    video_search_roots: Sequence[Path],
) -> Tuple[Optional[Path], str]:
    """Return (absolute path if file exists, string form for CSV)."""
    s = str(raw).strip()
    if not s:
        return None, ""
    p = Path(s)
    if p.is_file():
        return p.resolve(), str(p.resolve())
    rel = dataset_root / s
    if rel.is_file():
        return rel.resolve(), str(rel.resolve())
    for root in video_search_roots:
        cand = (root / s).resolve()
        if cand.is_file():
            return cand, str(cand)
        cand2 = root / Path(s).name
        if cand2.is_file():
            return cand2.resolve(), str(cand2.resolve())
    return None, s


def load_annotation_records(path: Path) -> List[Dict[str, Any]]:
    with open(path, encoding="utf-8", errors="replace") as f:
        data = json.load(f)
    out: List[Dict[str, Any]] = []
    for d in _iter_nested_dicts(data):
        if _is_annotation_leaf(d):
            out.append(d)
    return out


def _is_interpretable_judgement_root(data: Any) -> bool:
    """Detect ``interpretable_action_judgement.json`` (dataset card README)."""
    if not isinstance(data, dict) or not data:
        return False
    for v in data.values():
        if not isinstance(v, dict):
            continue
        anns = v.get("annotations")
        if (
            "frame_root" in v
            and isinstance(anns, list)
            and anns
            and isinstance(anns[0], dict)
            and "action_quality_score" in anns[0]
            and "action_name" in anns[0]
        ):
            return True
    return False


def build_rows_interpretable(
    data: Dict[str, Any],
    *,
    quality_scale: str,
    default_split: str,
) -> Tuple[List[Dict[str, str]], int]:
    """One row per (clip_key, annotation index) from official interpretable_action_judgement.json."""
    rows: List[Dict[str, str]] = []
    skipped_bad_q = 0
    for clip_key, block in data.items():
        if not isinstance(block, dict):
            continue
        anns = block.get("annotations")
        if not isinstance(anns, list):
            continue
        frame_root = str(block.get("frame_root") or "").strip()
        st_ed = block.get("st_ed_frame")
        f0, f1 = "", ""
        if isinstance(st_ed, list) and len(st_ed) >= 2:
            f0, f1 = str(int(st_ed[0])), str(int(st_ed[1]))

        for i, ann in enumerate(anns):
            if not isinstance(ann, dict):
                continue
            raw_q = ann.get("action_quality_score")
            raw_name = ann.get("action_name")
            if raw_q is None or raw_name is None:
                continue
            try:
                q = _normalize_quality(raw_q, quality_scale)
            except (TypeError, ValueError):
                skipped_bad_q += 1
                continue
            name = parse_base_exercise(str(raw_name))
            ex_key = normalize_class_name(name)
            safe_key = re.sub(r"[^\w\-.]+", "_", str(clip_key))[:160]
            stem = f"{safe_key}__ann{i}"
            comment = ann.get("comment")
            guidance = ann.get("action_guidance")
            annotator = ann.get("annotator")
            ver_list: List[Dict[str, object]] = []
            kp = ann.get("key_point_verification")
            if isinstance(kp, list):
                for item in kp:
                    if isinstance(item, (list, tuple)) and len(item) >= 2:
                        t_raw, v_raw = item[0], item[1]
                        ok = str(v_raw).strip().lower() in (
                            "true",
                            "1",
                            "yes",
                            "t",
                        )
                        ver_list.append(
                            {"text": str(t_raw).strip(), "ok": ok}
                        )
            verification_json = json.dumps(ver_list, ensure_ascii=False)
            rows.append(
                {
                    "video_stem": stem,
                    "video_path": "",
                    "exercise_class": ex_key,
                    "quality": f"{q:.6f}",
                    "split": default_split,
                    "action_name_raw": str(raw_name).strip(),
                    "comment": "" if comment is None else str(comment).strip(),
                    "action_guidance": "" if guidance is None else str(guidance).strip(),
                    "judgement_key": str(clip_key),
                    "frame_root": frame_root,
                    "frame_start": f0,
                    "frame_end": f1,
                    "annotator": "" if annotator is None else str(annotator).strip(),
                    "verification_json": verification_json,
                }
            )
    return rows, skipped_bad_q


def build_rows(
    records: Sequence[Dict[str, Any]],
    *,
    dataset_root: Path,
    video_search_roots: Sequence[Path],
    quality_scale: str,
    default_split: str,
) -> Tuple[List[Dict[str, str]], int]:
    rows: List[Dict[str, str]] = []
    skipped_bad_q = 0
    for d in records:
        raw_q = _pick(d, QUALITY_KEYS)
        raw_name = _pick(d, NAME_KEYS)
        if raw_q is None or raw_name is None:
            continue
        try:
            q = _normalize_quality(raw_q, quality_scale)
        except (TypeError, ValueError):
            skipped_bad_q += 1
            continue
        name = parse_base_exercise(str(raw_name))
        ex_key = normalize_class_name(name)

        vp_raw = _pick(d, VIDEO_KEYS)
        video_path_str = ""
        stem = ""
        if vp_raw is not None:
            p_resolved, video_path_str = _resolve_video_path(
                vp_raw, dataset_root, video_search_roots
            )
            if p_resolved is not None:
                stem = p_resolved.stem
            else:
                stem = Path(str(vp_raw)).stem
        else:
            for key in ("clip_id", "take_id", "clip_uid", "video_id", "uid"):
                if d.get(key) is not None and str(d[key]).strip():
                    stem = re.sub(r"[^\w\-]+", "_", str(d[key]).strip())[:200]
                    break
        if not stem:
            stem = f"row_{len(rows)}"

        comment = _pick(d, ("comment", "comments", "notes"))
        guidance = _pick(d, ("action_guidance", "guidance", "coaching"))

        split = str(d.get("split", default_split) or default_split).strip() or default_split

        rows.append(
            {
                "video_stem": stem,
                "video_path": video_path_str,
                "exercise_class": ex_key,
                "quality": f"{q:.6f}",
                "split": split,
                "action_name_raw": str(raw_name).strip(),
                "comment": str(comment).strip() if comment is not None else "",
                "action_guidance": str(guidance).strip() if guidance is not None else "",
                "judgement_key": "",
                "frame_root": "",
                "frame_start": "",
                "frame_end": "",
                "annotator": "",
                "verification_json": "",
            }
        )
    return rows, skipped_bad_q


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Build training CSV from EgoExo-Fitness-style JSON (action_quality_score + action_name)."
    )
    ap.add_argument(
        "--annotations-json",
        type=Path,
        required=True,
        help="Path to annotations JSON (e.g. raw_annotations/*.json under the HF dataset folder)",
    )
    ap.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("."),
        help="Root used to resolve relative video paths (HF snapshot folder)",
    )
    ap.add_argument(
        "--video-search-dir",
        action="append",
        default=[],
        metavar="DIR",
        help="Extra folders to try when resolving relative video paths (repeatable). "
        "Default adds <dataset-root>/frames_open and <dataset-root>/features_open if present.",
    )
    ap.add_argument("--output", type=Path, default=Path("./results/egoexo_fitness_index.csv"))
    ap.add_argument(
        "--quality-scale",
        choices=("raw", "auto", "0-1", "1-5", "1-10", "percent"),
        default="raw",
        help="raw: use annotation score as float (no rescaling); other choices map to [0,1] for regression",
    )
    ap.add_argument(
        "--default-split",
        default="train",
        help="If JSON rows have no 'split' field, use this (then run split_exercise_index.py)",
    )
    ap.add_argument(
        "--no-require-file",
        action="store_true",
        help="Include rows even when video file is not found on disk (for path fix-up + angle batch later)",
    )
    ap.add_argument(
        "--format",
        choices=("auto", "interpretable", "generic"),
        default="auto",
        help="auto: detect interpretable_action_judgement layout; interpretable: official README schema; "
        "generic: deep search for quality+name dicts.",
    )
    args = ap.parse_args()

    if not args.annotations_json.is_file():
        print(f"Missing {args.annotations_json}", file=sys.stderr)
        return 1

    dataset_root = args.dataset_root.resolve()
    extra = [Path(p).resolve() for p in (args.video_search_dir or [])]
    for name in ("frames_open", "features_open", "img"):
        cand = dataset_root / name
        if cand.is_dir() and cand not in extra:
            extra.append(cand)

    try:
        with open(args.annotations_json, encoding="utf-8", errors="replace") as f:
            raw_data = json.load(f)
    except json.JSONDecodeError as e:
        print(f"Invalid JSON: {e}", file=sys.stderr)
        return 1

    fmt = args.format
    if fmt == "auto":
        fmt = "interpretable" if _is_interpretable_judgement_root(raw_data) else "generic"

    n_src = 0
    if fmt == "interpretable":
        if not isinstance(raw_data, dict):
            print("interpretable format expects a JSON object at top level.", file=sys.stderr)
            return 1
        rows, skip_q = build_rows_interpretable(
            raw_data,
            quality_scale=args.quality_scale,
            default_split=args.default_split,
        )
        n_src = len(raw_data)
        if not rows:
            print("No rows from interpretable_action_judgement structure.", file=sys.stderr)
            return 1
    else:
        records = []
        for d in _iter_nested_dicts(raw_data):
            if _is_annotation_leaf(d):
                records.append(d)
        n_src = len(records)
        if not records:
            print(
                "No annotation leaves found (need dicts with a quality key + name key). "
                "For official EgoExo use interpretable_action_judgement.json with --format interpretable "
                "or --format auto.",
                file=sys.stderr,
            )
            return 1
        rows, skip_q = build_rows(
            records,
            dataset_root=dataset_root,
            video_search_roots=tuple(extra),
            quality_scale=args.quality_scale,
            default_split=args.default_split,
        )

    skip_vid = 0
    path_filter = not args.no_require_file and fmt != "interpretable"
    if path_filter:
        before = len(rows)
        filtered_nv: List[Dict[str, str]] = []
        for r in rows:
            vp = (r.get("video_path") or "").strip()
            if vp and Path(vp).is_file():
                filtered_nv.append(r)
        rows = filtered_nv
        skip_vid = before - len(rows)

    if not rows:
        print(
            f"No rows after filtering (skipped_missing_file={skip_vid}, bad_quality={skip_q}). "
            "Try --no-require-file or fix --dataset-root / --video-search-dir.",
            file=sys.stderr,
        )
        return 1

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "video_stem",
        "video_path",
        "exercise_class",
        "quality",
        "split",
        "action_name_raw",
        "comment",
        "action_guidance",
        "judgement_key",
        "frame_root",
        "frame_start",
        "frame_end",
        "annotator",
        "verification_json",
    ]
    with open(args.output, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

    print(f"Wrote {len(rows)} rows → {args.output.resolve()}  (format={fmt}, source_units={n_src})")
    print(f"skipped_bad_quality: {skip_q}")
    if fmt == "interpretable":
        print(
            "Note: HF release often has frame folders under frame_root (see dataset README); "
            "use batch_compute_angles_for_index.py with --dataset-root and --egoexo-view."
        )
    print(
        "Next: split_exercise_index.py → (optional frames) batch_compute_angles_for_index.py → "
        "train_exercise_bilstm.py  |  annotations-only: train_exercise_annotation_bilstm.py"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
