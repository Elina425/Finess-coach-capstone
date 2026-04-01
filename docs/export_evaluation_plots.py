#!/usr/bin/env python3
"""
Load JSON from evaluate_exercise_models.py --json-out and save simple bar charts.

Usage:
  ./venv/bin/python docs/export_evaluation_plots.py --json results/eval_test.json --out-dir docs/figures
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, default=Path("docs/figures"))
    args = ap.parse_args()

    if not args.json.is_file():
        print(f"Missing {args.json}", file=sys.stderr)
        return 1

    with open(args.json) as f:
        data = json.load(f)

    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("Install matplotlib: pip install matplotlib", file=sys.stderr)
        return 1

    args.out_dir.mkdir(parents=True, exist_ok=True)

    tags = [d.get("tag", f"model_{i}") for i, d in enumerate(data)]
    acc = [d["classification"]["accuracy"] for d in data]
    f1m = [d["classification"]["f1_macro"] for d in data]
    mae = [d["quality_regression"]["mae"] for d in data]
    r2 = []
    for d in data:
        v = d["quality_regression"]["r2"]
        r2.append(v if isinstance(v, (int, float)) and np.isfinite(v) else 0.0)

    x = np.arange(len(tags))
    w = 0.35

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(x - w / 2, acc, w, label="Accuracy")
    ax.bar(x + w / 2, f1m, w, label="F1 macro")
    ax.set_xticks(x)
    ax.set_xticklabels(tags, rotation=15, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Exercise classification (window-level)")
    ax.legend()
    fig.tight_layout()
    p1 = args.out_dir / "eval_classification.png"
    fig.savefig(p1, dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(x - w / 2, mae, w, label="MAE (quality)")
    ax.bar(x + w / 2, np.clip(r2, 0, 1), w, label="R² (clipped 0–1)")
    ax.set_xticks(x)
    ax.set_xticklabels(tags, rotation=15, ha="right")
    ax.set_ylabel("Error / R²")
    ax.set_title("Quality regression (window-level)")
    ax.legend()
    fig.tight_layout()
    p2 = args.out_dir / "eval_quality.png"
    fig.savefig(p2, dpi=150)
    plt.close(fig)

    print(f"Wrote {p1}\nWrote {p2}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
