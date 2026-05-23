#!/usr/bin/env python3
"""Run EgoExo feature ablations for xLSTM.

When pose NPZs exist: angles vs coords vs mixed.
When annotation-only: runs annotation mode with different model sizes as ablation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from train_xlstm_egoexo_multitask import build_parser as build_train_parser
from train_xlstm_egoexo_multitask import train_from_args


def main() -> int:
    parser = argparse.ArgumentParser(description="Run xLSTM feature ablation: clip vs annotation")
    parser.add_argument("--index-csv", type=Path, required=True)
    parser.add_argument("--clip-features-root", type=Path, default=Path("notebooks/data/egoexo_fitness_full/features_open/visual"))
    parser.add_argument("--angles-dir", type=Path, default=Path("results/egoexo_exercise_angles"))
    parser.add_argument("--keypoints-dir", type=Path, default=Path("results/egoexo_exercise_angles"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/xlstm_ablation"))
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    base_parser = build_train_parser()
    summaries = {}
    modes = ["clip", "annotation"]

    for mode in modes:
        train_args = base_parser.parse_args(["--index-csv", str(args.index_csv)])
        train_args.index_csv = args.index_csv
        train_args.feature_mode = mode
        train_args.output_dir = args.output_dir / mode
        train_args.epochs = args.epochs
        train_args.batch_size = args.batch_size
        train_args.cpu = args.cpu
        train_args.standardize = True
        train_args.eval_test = True
        if mode == "clip":
            train_args.clip_features_root = args.clip_features_root
        print(f"\n=== running {mode} ablation ===")
        try:
            summaries[mode] = train_from_args(train_args)
        except ValueError as exc:
            print(f"  skipped {mode}: {exc}")
            summaries[mode] = {"error": str(exc)}

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_dir / "ablation_summary.json"
    summary_path.write_text(json.dumps(summaries, indent=2, default=str))
    print(f"\nSaved ablation summary to {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
