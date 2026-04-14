#!/usr/bin/env python3
"""
Train the same Riccio BiLSTM setup on two NPZ folders (e.g. run_default vs run_rich),
then write a JSON metrics table and a Markdown comparison report.

Uses the same defaults as ``pipeline/step_05_train_bilstm_riccio.sh`` (preset riccio,
standardize, eval-test, cnn_attn, classification-only). Extra CLI tokens are forwarded
to ``train_exercise_bilstm.py`` for both runs (do not pass ``--kaggle-angles-dir`` or
``--output-dir`` there; this wrapper sets those per run).

Example:
  ./venv/bin/python benchmark_preprocessing_training.py \\
    --default-angles-dir results/run_default \\
    --rich-angles-dir results/run_rich \\
    --output-root results/preprocess_train_benchmark \\
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
    if (cwd / "train_exercise_bilstm.py").is_file():
        return cwd
    return Path(__file__).resolve().parents[2]


def _load_test_metrics(out_dir: Path) -> dict[str, Any] | None:
    p = out_dir / "test_classification_metrics.json"
    if not p.is_file():
        return None
    with open(p) as f:
        return json.load(f)


def _run_train(
    repo: Path,
    angles_dir: Path,
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
        str(repo / "train_exercise_bilstm.py"),
        "--preset",
        "riccio",
        "--standardize",
        "--eval-test",
        "--architecture",
        "cnn_attn",
        "--classification-only",
        "--kaggle-angles-dir",
        str(angles_dir.resolve()),
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
    lines.append("# Preprocessing benchmark — BiLSTM (Riccio)")
    lines.append("")
    lines.append(f"Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
    lines.append("")
    lines.append("## Training configuration")
    lines.append("")
    lines.append("| Setting | Value |")
    lines.append("|--------|--------|")
    lines.append(f"| Preset | riccio |")
    lines.append(f"| Architecture | cnn_attn |")
    lines.append(f"| Classification only | yes |")
    lines.append(f"| Standardize | yes |")
    lines.append(f"| Eval test | yes |")
    lines.append(f"| Epochs | {epochs} |")
    lines.append(f"| Kaggle split seed | {kaggle_seed} |")
    lines.append(f"| NPZ stem | `{stem}` |")
    lines.append(f"| Default angles dir | `{default_dir}` |")
    lines.append(f"| Rich angles dir | `{rich_dir}` |")
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

    def _row(name: str, key: str, fmt: str = ".4f") -> tuple[str, str, str, str]:
        v0 = m_def.get(key)
        v1 = m_rich.get(key)
        if v0 is None or v1 is None:
            return name, "—", "—", "—"
        d = float(v1) - float(v0)
        return name, f"{float(v0):{fmt}}", f"{float(v1):{fmt}}", f"{d:+.4f}"

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
    lines.append(f"- Default checkpoint: `{default_out / 'exercise_bilstm_best.pt'}`")
    lines.append(f"- Rich checkpoint: `{rich_out / 'exercise_bilstm_best.pt'}`")
    lines.append(f"- Default test metrics JSON: `{default_out / 'test_classification_metrics.json'}`")
    lines.append(f"- Rich test metrics JSON: `{rich_out / 'test_classification_metrics.json'}`")
    lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    lines.append(
        "Higher test **accuracy** / **F1 macro** on the **same split seed** supports the claim that "
        "the corresponding preprocessing is better **for this classifier**, subject to training noise. "
        "If differences are small, repeat with a different `--kaggle-seed` or longer `--epochs`."
    )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Train BiLSTM twice (default vs rich NPZs) and write comparison JSON + Markdown report."
    )
    ap.add_argument("--default-angles-dir", type=Path, required=True)
    ap.add_argument("--rich-angles-dir", type=Path, required=True)
    ap.add_argument(
        "--output-root",
        type=Path,
        default=Path("results/preprocess_train_benchmark"),
        help="Parent folder for default/ and rich/ training outputs + comparison files",
    )
    ap.add_argument("--kaggle-stem", default="riccio_realtime_exercise_recognition")
    ap.add_argument("--kaggle-seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=30)
    args, train_extra = ap.parse_known_args()

    repo = _repo_root()
    if not (repo / "train_exercise_bilstm.py").is_file():
        print(f"Cannot find train_exercise_bilstm.py under {repo}", file=sys.stderr)
        return 1

    root = args.output_root.resolve()
    out_default = root / "default"
    out_rich = root / "rich"

    for label, adir, odir in (
        ("default", args.default_angles_dir, out_default),
        ("rich", args.rich_angles_dir, out_rich),
    ):
        bio = adir / f"{args.kaggle_stem}_biomechanics.npz"
        lab = adir / f"{args.kaggle_stem}_labels.npz"
        if not bio.is_file() or not lab.is_file():
            print(f"ERROR [{label}]: missing NPZs under {adir} (need {bio.name} + {lab.name})", file=sys.stderr)
            return 1

    print(f"Repo root: {repo}", flush=True)
    print(f"Training default → {out_default}", flush=True)
    _run_train(
        repo,
        args.default_angles_dir,
        out_default,
        kaggle_stem=args.kaggle_stem,
        kaggle_seed=args.kaggle_seed,
        epochs=args.epochs,
        extra=train_extra,
    )
    print(f"Training rich → {out_rich}", flush=True)
    _run_train(
        repo,
        args.rich_angles_dir,
        out_rich,
        kaggle_stem=args.kaggle_stem,
        kaggle_seed=args.kaggle_seed,
        epochs=args.epochs,
        extra=train_extra,
    )

    m_def = _load_test_metrics(out_default)
    m_rich = _load_test_metrics(out_rich)

    table: dict[str, Any] = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "repo_root": str(repo),
        "train": {
            "preset": "riccio",
            "architecture": "cnn_attn",
            "classification_only": True,
            "standardize": True,
            "eval_test": True,
            "epochs": args.epochs,
            "kaggle_seed": args.kaggle_seed,
            "kaggle_stem": args.kaggle_stem,
            "extra_args": train_extra,
        },
        "default": {
            "kaggle_angles_dir": str(args.default_angles_dir.resolve()),
            "output_dir": str(out_default),
            "test_metrics": m_def,
        },
        "rich": {
            "kaggle_angles_dir": str(args.rich_angles_dir.resolve()),
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

    json_path = root / "preprocessing_train_comparison.json"
    with open(json_path, "w") as f:
        json.dump(table, f, indent=2, default=str)

    report_path = root / "PREPROCESSING_TRAINING_REPORT.md"
    _write_report(
        report_path,
        default_dir=args.default_angles_dir,
        rich_dir=args.rich_angles_dir,
        default_out=out_default,
        rich_out=out_rich,
        stem=args.kaggle_stem,
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
