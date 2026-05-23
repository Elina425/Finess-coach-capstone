#!/usr/bin/env python3
"""Side‑by‑side training curves from multitask XLSTM JSON history (matches paper figure layout).

Left: composite + task losses on log‑scale y‑axis (total loss dominates early; cls/reg stay visible).

Right: validation accuracy (solid) and macro‑F1 (dashed); optional quality‑bucket metrics when present.

Example:
  python scripts/plot_multitask_training_curves.py \\
    --history results/xlstm_egoexo_multitask_allviews/training_history.json \\
    --out docs/figures/fig_multitask_training_curves.png
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _load_history(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("training_history.json must be a list of epoch records")
    return data


def plot_curves(history: list[dict], *, title: str, out_path: Path, dpi: int = 120) -> None:
    ep = np.array([float(r["epoch"]) for r in history], dtype=np.float64)
    train_total = np.array([max(float(r["train_loss"]), 1e-16) for r in history])
    train_cls = np.array([max(float(r["train_cls"]), 1e-16) for r in history])
    train_reg = np.array([max(float(r["train_reg"]), 1e-16) for r in history])
    val_acc = np.array([float(r["val_accuracy"]) for r in history])
    val_f1 = np.array([float(r["val_f1_macro"]) for r in history])
    q_acc = []
    q_f1 = []
    has_q = False
    for r in history:
        if "val_quality_accuracy" in r:
            qa = float(r["val_quality_accuracy"])
            qf = float(r.get("val_quality_f1_macro", float("nan")))
            q_acc.append(qa if np.isfinite(qa) else np.nan)
            q_f1.append(qf if np.isfinite(qf) else np.nan)
            has_q = True
        else:
            q_acc.append(np.nan)
            q_f1.append(np.nan)

    fig, ax = plt.subplots(1, 2, figsize=(13, 4))

    ax0 = ax[0]
    ax0.semilogy(ep, train_total, "-", linewidth=2.2, color="black", label="train loss (total)")
    ax0.semilogy(ep, train_cls, "--", linewidth=1.35, color="C0", label="train cls")
    ax0.semilogy(ep, train_reg, ":", linewidth=1.65, color="C3", label="train reg")
    ax0.set_title(title + "\nTraining losses")
    ax0.set_xlabel("Epoch")
    ax0.set_ylabel("Loss (log scale)")
    ax0.legend(loc="upper right", fontsize=9)
    ax0.grid(True, alpha=0.35, which="both")

    ax1 = ax[1]
    ax1.plot(ep, val_acc, "-", linewidth=2.0, color="C0", label="val accuracy")
    ax1.plot(ep, val_f1, "--", linewidth=2.0, color="C1", label="val macro-F1")
    if has_q and np.nanmax(np.array(q_acc, dtype=float)) >= 0.0:
        q_acc_a = np.array(q_acc, dtype=float)
        q_f1_a = np.array(q_f1, dtype=float)
        if np.any(np.isfinite(q_acc_a)):
            ax1.plot(ep, q_acc_a, "-.", linewidth=1.65, color="C2", label="val quality acc")
        if np.any(np.isfinite(q_f1_a)):
            ax1.plot(ep, q_f1_a, ":", linewidth=1.65, color="C4", label="val quality F1 macro")
    lo = float(np.nanmin(np.concatenate([val_acc, val_f1])))
    hi = float(np.nanmax(np.concatenate([val_acc, val_f1])))
    if has_q:
        qa = np.array(q_acc, dtype=float)
        qf = np.array(q_f1, dtype=float)
        lo = float(min(lo, np.nanmin(qa), np.nanmin(qf)))
        hi = float(max(hi, np.nanmax(qa), np.nanmax(qf)))
    pad = max(0.02, (hi - lo) * 0.06)
    ax1.set_ylim(max(0.0, lo - pad), min(1.0 + 5e-3, hi + pad))
    ax1.set_title("Validation metrics — accuracy (solid) / F1 macro (dashed)")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Score")
    ax1.legend(loc="lower right", fontsize=9)
    ax1.grid(True, alpha=0.35)

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close()
    print(f"saved → {out_path}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    root = Path(__file__).resolve().parents[1]
    p.add_argument(
        "--history",
        type=Path,
        default=root / "results/xlstm_egoexo_multitask_allviews/training_history.json",
        help="Path to training_history.json",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=root / "docs/figures/fig_multitask_training_curves.png",
        help="Output PNG path",
    )
    p.add_argument("--title", type=str, default="EgoExo multitask xLSTM (allviews)", help="Subtitle for loss panel")
    p.add_argument("--dpi", type=int, default=120)
    args = p.parse_args()
    history = _load_history(args.history)
    if not history:
        raise SystemExit("empty history")
    plot_curves(history, title=args.title, out_path=args.out, dpi=args.dpi)


if __name__ == "__main__":
    main()
