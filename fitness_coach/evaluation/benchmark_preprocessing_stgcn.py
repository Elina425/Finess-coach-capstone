#!/usr/bin/env python3
"""
Train ST-GCN on two Riccio NPZ folders (e.g. run_default vs run_rich_no_dwt) and write
comparison JSON + Markdown report.

Requires **keypoints** NPZs: ``{stem}_keypoints.npz`` + ``{stem}_labels.npz`` in each folder.
If you only ran the video pipeline with ``--skip-keypoints``, regenerate without it for ST-GCN.

Uses ``train_exercise_stgcn.py`` with ``--standardize --eval-test``. Extra CLI tokens are
forwarded to both runs (do not pass ``--kaggle-keypoints-dir`` or ``--output-dir`` there).

Example:
  ./venv/bin/python benchmark_preprocessing_stgcn.py \\
    --default-keypoints-dir results/run_default \\
    --rich-keypoints-dir results/run_rich_no_dwt \\
    --output-root results/preprocess_stgcn_benchmark \\
    --epochs 30 --kaggle-seed 42
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _repo_root() -> Path:
    cwd = Path.cwd()
    if (cwd / "train_exercise_stgcn.py").is_file():
        return cwd
    return Path(__file__).resolve().parents[2]


def _load_test_metrics(out_dir: Path) -> dict[str, Any] | None:
    p = out_dir / "test_classification_metrics.json"
    if not p.is_file():
        return None
    with open(p) as f:
        return json.load(f)


def _run_stgcn(
    repo: Path,
    keypoints_dir: Path,
    out_dir: Path,
    *,
    kaggle_stem: str,
    kaggle_seed: int,
    epochs: int,
    extra: list[str],
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        str(repo / "train_exercise_stgcn.py"),
        "--standardize",
        "--eval-test",
        "--kaggle-keypoints-dir",
        str(keypoints_dir.resolve()),
        "--kaggle-stem",
        kaggle_stem,
        "--output-dir",
        str(out_dir.resolve()),
        "--kaggle-seed",
        str(kaggle_seed),
        "--epochs",
        str(epochs),
    ] + list(extra)
    print("\n" + " ".join(cmd) + "\n", flush=True)
    subprocess.run(cmd, cwd=str(repo), check=True)


def _delta(a: float | None, b: float | None) -> float | None:
    if a is None or b is None:
        return None
    return float(b) - float(a)


def _write_report(
    path: Path,
    *,
    default_dir: Path,
    rich_dir: Path,
    default_out: Path,
    rich_out: Path,
    stem: str,
    kaggle_seed: int,
    epochs: int,
    extra: list[str],
    m_def: dict[str, Any] | None,
    m_rich: dict[str, Any] | None,
) -> None:
    lines: list[str] = []
    lines.append("# Preprocessing benchmark — ST-GCN (Riccio keypoints)")
    lines.append("")
    lines.append(f"Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
    lines.append("")
    lines.append("## Training configuration")
    lines.append("")
    lines.append("| Setting | Value |")
    lines.append("|--------|--------|")
    lines.append("| Model | ExerciseSTGCN (COCO-17, xy) |")
    lines.append("| Standardize | yes |")
    lines.append("| Eval test | yes |")
    lines.append(f"| Epochs | {epochs} |")
    lines.append(f"| Kaggle split seed | {kaggle_seed} |")
    lines.append(f"| NPZ stem | `{stem}` |")
    lines.append(f"| Default keypoints dir | `{default_dir}` |")
    lines.append(f"| Rich keypoints dir | `{rich_dir}` |")
    if extra:
        lines.append(f"| Extra train args | `{' '.join(extra)}` |")
    lines.append("")
    lines.append("## Test-set metrics (held-out video split)")
    lines.append("")
    if m_def is None or m_rich is None:
        lines.append("**Missing** `test_classification_metrics.json` for one or both runs.")
        lines.append("")
        path.write_text("\n".join(lines), encoding="utf-8")
        return

    def _row(name: str, key: str) -> tuple[str, str, str, str]:
        v0 = m_def.get(key)
        v1 = m_rich.get(key)
        if v0 is None or v1 is None:
            return name, "—", "—", "—"
        d = float(v1) - float(v0)
        return name, f"{float(v0):.4f}", f"{float(v1):.4f}", f"{d:+.4f}"

    lines.append("| Metric | Default preprocess | Rich preprocess | Δ (rich − default) |")
    lines.append("|--------|-------------------:|----------------:|-------------------:|")
    for name, key in [
        ("Accuracy", "accuracy"),
        ("F1 macro", "f1_macro"),
        ("F1 weighted", "f1_weighted"),
        ("Recall macro", "recall_macro"),
        ("Precision macro", "precision_macro"),
    ]:
        n, a, b, d = _row(name, key)
        lines.append(f"| {n} | {a} | {b} | {d} |")
    lines.append("")
    lines.append("### Per-class F1")
    lines.append("")
    f0 = m_def.get("f1_per_class") or {}
    f1 = m_rich.get("f1_per_class") or {}
    classes = sorted(set(f0) | set(f1))
    lines.append("| Class | Default F1 | Rich F1 | Δ |")
    lines.append("|-------|-----------:|--------:|--:|")
    for c in classes:
        a = f0.get(c)
        b = f1.get(c)
        if a is None or b is None:
            lines.append(f"| {c} | — | — | — |")
        else:
            d = float(b) - float(a)
            lines.append(f"| {c} | {float(a):.4f} | {float(b):.4f} | {d:+.4f} |")
    lines.append("")
    lines.append("## Artifacts")
    lines.append("")
    lines.append(f"- Default checkpoint: `{default_out / 'exercise_stgcn_best.pt'}`")
    lines.append(f"- Rich checkpoint: `{rich_out / 'exercise_stgcn_best.pt'}`")
    lines.append(f"- Default test metrics: `{default_out / 'test_classification_metrics.json'}`")
    lines.append(f"- Rich test metrics: `{rich_out / 'test_classification_metrics.json'}`")
    lines.append("")
    lines.append("## Note")
    lines.append("")
    lines.append(
        "ST-GCN uses **2D keypoint** windows, not angle features. Preprocessing affects "
        "`*_keypoints.npz` only if the video pipeline was run **without** `--skip-keypoints`."
    )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Train ST-GCN twice (default vs rich keypoint NPZs) and write comparison JSON + report."
    )
    ap.add_argument("--default-keypoints-dir", type=Path, required=True)
    ap.add_argument("--rich-keypoints-dir", type=Path, required=True)
    ap.add_argument(
        "--output-root",
        type=Path,
        default=Path("results/preprocess_stgcn_benchmark"),
        help="Parent folder for default/ and rich/ ST-GCN training outputs + comparison files",
    )
    ap.add_argument("--kaggle-stem", default="riccio_realtime_exercise_recognition")
    ap.add_argument("--kaggle-seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=30)
    args, train_extra = ap.parse_known_args()

    repo = _repo_root()
    if not (repo / "train_exercise_stgcn.py").is_file():
        print(f"Cannot find train_exercise_stgcn.py under {repo}", file=sys.stderr)
        return 1

    root = args.output_root.resolve()
    out_default = root / "default"
    out_rich = root / "rich"
    stem = args.kaggle_stem

    for label, kdir in (
        ("default", args.default_keypoints_dir),
        ("rich", args.rich_keypoints_dir),
    ):
        kp = kdir / f"{stem}_keypoints.npz"
        lab = kdir / f"{stem}_labels.npz"
        if not kp.is_file():
            print(
                f"ERROR [{label}]: missing {kp.name} under {kdir}. "
                "Regenerate with riccio_kaggle_video_pipeline.py **without** --skip-keypoints.",
                file=sys.stderr,
            )
            return 1
        if not lab.is_file():
            print(f"ERROR [{label}]: missing {lab.name} under {kdir}", file=sys.stderr)
            return 1

    print(f"Repo root: {repo}", flush=True)
    print(f"ST-GCN default → {out_default}", flush=True)
    _run_stgcn(
        repo,
        args.default_keypoints_dir,
        out_default,
        kaggle_stem=stem,
        kaggle_seed=args.kaggle_seed,
        epochs=args.epochs,
        extra=train_extra,
    )
    print(f"ST-GCN rich → {out_rich}", flush=True)
    _run_stgcn(
        repo,
        args.rich_keypoints_dir,
        out_rich,
        kaggle_stem=stem,
        kaggle_seed=args.kaggle_seed,
        epochs=args.epochs,
        extra=train_extra,
    )

    m_def = _load_test_metrics(out_default)
    m_rich = _load_test_metrics(out_rich)

    table: dict[str, Any] = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "repo_root": str(repo),
        "model": "stgcn",
        "train": {
            "standardize": True,
            "eval_test": True,
            "epochs": args.epochs,
            "kaggle_seed": args.kaggle_seed,
            "kaggle_stem": stem,
            "extra_args": train_extra,
        },
        "default": {
            "kaggle_keypoints_dir": str(args.default_keypoints_dir.resolve()),
            "output_dir": str(out_default),
            "test_metrics": m_def,
        },
        "rich": {
            "kaggle_keypoints_dir": str(args.rich_keypoints_dir.resolve()),
            "output_dir": str(out_rich),
            "test_metrics": m_rich,
        },
        "delta_rich_minus_default": {},
    }
    if m_def and m_rich:
        for key in (
            "accuracy",
            "f1_macro",
            "f1_weighted",
            "recall_macro",
            "precision_macro",
        ):
            table["delta_rich_minus_default"][key] = _delta(m_def.get(key), m_rich.get(key))

    json_path = root / "preprocessing_stgcn_comparison.json"
    with open(json_path, "w") as f:
        json.dump(table, f, indent=2, default=str)

    report_path = root / "PREPROCESSING_STGCN_REPORT.md"
    _write_report(
        report_path,
        default_dir=args.default_keypoints_dir,
        rich_dir=args.rich_keypoints_dir,
        default_out=out_default,
        rich_out=out_rich,
        stem=stem,
        kaggle_seed=args.kaggle_seed,
        epochs=args.epochs,
        extra=train_extra,
        m_def=m_def,
        m_rich=m_rich,
    )

    print(f"\nWrote {json_path}", flush=True)
    print(f"Wrote {report_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
