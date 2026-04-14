#!/usr/bin/env python3
"""
Assign train / val / test splits on unique video_stem rows (stratified by exercise_class when possible).

Use this after build_exercise_training_index.py or setup_local_training_data.py so that:
  • Training uses split=train, early stopping uses split=val
  • Final evaluation uses split=test (evaluate_exercise_models.py) — no leakage

Typical fractions: 70% train, 15% val, 15% test of unique stems.

Example:
  ./venv/bin/python split_exercise_index.py \\
    --input results/exercise_training_index.csv \\
    --output results/exercise_training_index_split.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple


def _read_rows(path: Path) -> Tuple[List[str], List[Dict[str, Any]]]:
    with open(path, newline="") as f:
        r = csv.DictReader(f)
        fieldnames = r.fieldnames
        if not fieldnames:
            raise ValueError("Empty CSV")
        rows = list(r)
    return list(fieldnames), rows


def _stem_majority(rows: List[Dict[str, Any]]) -> Tuple[Dict[str, str], Dict[str, Dict[str, Any]]]:
    """One representative row per stem (first wins); stem -> exercise_class."""
    stem_class: Dict[str, str] = {}
    stem_row0: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        stem = (row.get("video_stem") or "").strip()
        if not stem:
            continue
        if stem not in stem_row0:
            stem_row0[stem] = dict(row)
            stem_class[stem] = (row.get("exercise_class") or "unknown").strip()
    return stem_class, stem_row0


def _split_stems(
    stems: List[str],
    labels: List[str],
    test_ratio: float,
    val_ratio: float,
    seed: int,
) -> Tuple[set, set, set]:
    """Return (train_stems, val_stems, test_stems) as sets."""
    try:
        from sklearn.model_selection import train_test_split
    except ImportError:
        print("Install scikit-learn: pip install scikit-learn", file=sys.stderr)
        raise

    n = len(stems)
    if n == 0:
        return set(), set(), set()
    if n == 1:
        return {stems[0]}, set(), set()
    if n == 2:
        rng = __import__("random").Random(seed)
        order = stems[:]
        rng.shuffle(order)
        return {order[0]}, set(), {order[1]}

    # n >= 3: three-way split
    strat = None
    if len(set(labels)) > 1:
        try:
            from collections import Counter

            c = Counter(labels)
            if min(c.values()) >= 2:
                strat = labels
        except Exception:
            pass

    stems_a = stems
    labels_a = labels
    try:
        tv_idx, test_idx = train_test_split(
            range(n),
            test_size=test_ratio,
            stratify=strat,
            random_state=seed,
        )
    except ValueError:
        tv_idx, test_idx = train_test_split(
            range(n),
            test_size=test_ratio,
            random_state=seed,
        )

    tv_stems = [stems_a[i] for i in tv_idx]
    tv_labels = [labels_a[i] for i in tv_idx]
    test_stems = {stems_a[i] for i in test_idx}

    n_tv = len(tv_stems)
    val_rel = val_ratio / (1.0 - test_ratio) if (1.0 - test_ratio) > 1e-6 else 0.0
    val_rel = min(max(val_rel, 0.0), 0.99)

    strat2 = None
    if len(set(tv_labels)) > 1:
        try:
            c2 = __import__("collections").Counter(tv_labels)
            if min(c2.values()) >= 2:
                strat2 = tv_labels
        except Exception:
            pass

    if n_tv < 2:
        train_stems = set(tv_stems)
        val_stems = set()
    else:
        try:
            tr_idx, va_idx = train_test_split(
                range(n_tv),
                test_size=val_rel,
                stratify=strat2,
                random_state=seed + 1,
            )
        except ValueError:
            tr_idx, va_idx = train_test_split(
                range(n_tv),
                test_size=val_rel,
                random_state=seed + 1,
            )
        train_stems = {tv_stems[i] for i in tr_idx}
        val_stems = {tv_stems[i] for i in va_idx}

    return train_stems, val_stems, test_stems


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Stratified train/val/test split by video_stem",
        epilog="Run with no arguments: reads results/exercise_training_index.csv → "
        "results/exercise_training_index_split.csv",
    )
    ap.add_argument(
        "--input",
        default="./results/exercise_training_index.csv",
        help="Input index CSV (default: results/exercise_training_index.csv)",
    )
    ap.add_argument(
        "--output",
        default="./results/exercise_training_index_split.csv",
        help="Output CSV (default: *_split.csv next to input; use same path as --input to overwrite with .bak)",
    )
    ap.add_argument("--test-ratio", type=float, default=0.15, help="Fraction of stems for test")
    ap.add_argument("--val-ratio", type=float, default=0.15, help="Fraction of stems for val")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dry-run", action="store_true", help="Print counts only")
    args = ap.parse_args()

    inp = Path(args.input)
    if not inp.is_file():
        print(
            f"Missing {inp}\n"
            "  Build the index first, e.g.:\n"
            "    ./venv/bin/python build_exercise_training_index.py --no-require-file",
            file=sys.stderr,
        )
        return 1

    fieldnames, rows = _read_rows(inp)
    stem_class, _ = _stem_majority(rows)
    stems = sorted(stem_class.keys())
    labels = [stem_class[s] for s in stems]

    train_s, val_s, test_s = _split_stems(stems, labels, args.test_ratio, args.val_ratio, args.seed)

    print(
        f"Unique stems: {len(stems)}  →  train={len(train_s)}  val={len(val_s)}  test={len(test_s)}"
    )
    if args.dry_run:
        return 0

    out_path = Path(args.output) if args.output else inp
    if out_path.resolve() == inp.resolve():
        bak = inp.with_suffix(inp.suffix + ".bak")
        bak.write_text(inp.read_text())
        print(f"Backed up original → {bak}")

    stem_to_split = {}
    for s in train_s:
        stem_to_split[s] = "train"
    for s in val_s:
        stem_to_split[s] = "val"
    for s in test_s:
        stem_to_split[s] = "test"

    out_rows = []
    for row in rows:
        stem = (row.get("video_stem") or "").strip()
        if stem in stem_to_split:
            row = dict(row)
            row["split"] = stem_to_split[stem]
        out_rows.append(row)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(out_rows)

    meta = {
        "input": str(inp.resolve()),
        "output": str(out_path.resolve()),
        "test_ratio": args.test_ratio,
        "val_ratio": args.val_ratio,
        "seed": args.seed,
        "counts": {"train": len(train_s), "val": len(val_s), "test": len(test_s)},
    }
    with open(out_path.with_suffix(".split_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"Wrote {len(out_rows)} rows → {out_path.resolve()}")
    print("Use split=test only in evaluate_exercise_models.py for unbiased metrics.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
