#!/usr/bin/env python3
"""
Generate histograms for xlstm_egoexo_multitask_allviews from:
  - training_history.json (per-epoch train / val metrics)
  - metrics.json (best_val snapshot: per-class scores, confusion row counts)

metrics.json does not include per-epoch training; use training_history.json for those.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_json(path: Path):
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def plot_epoch_histograms(history: list, out_path: Path) -> None:
    train_loss = [r["train_loss"] for r in history]
    train_cls = [r["train_cls"] for r in history]
    train_reg = [r["train_reg"] for r in history]
    val_acc = [r["val_accuracy"] for r in history]
    val_f1 = [r["val_f1_macro"] for r in history]
    val_mae = [r["val_mae"] for r in history]

    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    fig.suptitle("Per-epoch metric distributions (histogram over training epochs)", fontsize=12)

    def hist(ax, data, title, color):
        ax.hist(data, bins=min(20, max(8, len(data) // 2)), color=color, alpha=0.85, edgecolor="white")
        ax.set_title(title)
        ax.set_ylabel("Epochs")
        ax.grid(True, alpha=0.3)

    hist(axes[0, 0], train_loss, "train_loss", "C0")
    hist(axes[0, 1], train_cls, "train_cls", "C0")
    hist(axes[0, 2], train_reg, "train_reg", "C0")
    hist(axes[1, 0], val_acc, "val_accuracy", "C1")
    hist(axes[1, 1], val_f1, "val_f1_macro", "C1")
    hist(axes[1, 2], val_mae, "val_mae", "C1")

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def confusion_row_counts(cm: list) -> np.ndarray:
    return np.array(cm, dtype=float).sum(axis=1)


def plot_metrics_histograms(metrics: dict, out_path: Path) -> None:
    bv = metrics["best_val"]
    f1_vals = list(bv["f1_per_class"].values())
    labels = list(bv["f1_per_class"].keys())
    counts = confusion_row_counts(bv["confusion_matrix"])

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    fig.suptitle("metrics.json — best validation snapshot", fontsize=12)

    axes[0].hist(f1_vals, bins=10, color="steelblue", edgecolor="white", alpha=0.9)
    axes[0].axvline(np.mean(f1_vals), color="crimson", linestyle="--", label=f"mean={np.mean(f1_vals):.3f}")
    axes[0].set_title("F1 per class (12 values)")
    axes[0].set_xlabel("F1")
    axes[0].set_ylabel("Classes (count in bin)")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    idx = np.arange(len(labels))
    axes[1].bar(idx, f1_vals, color="steelblue", edgecolor="white")
    axes[1].set_xticks(idx)
    axes[1].set_xticklabels(labels, rotation=75, ha="right", fontsize=7)
    axes[1].set_ylim(0, 1.05)
    axes[1].set_title("F1 by class (best val)")
    axes[1].grid(True, axis="y", alpha=0.3)

    axes[2].bar(idx, counts, color="seagreen", edgecolor="white")
    axes[2].set_xticks(idx)
    axes[2].set_xticklabels(labels, rotation=75, ha="right", fontsize=7)
    axes[2].set_title("Samples per class (val, true labels)")
    axes[2].set_ylabel("count")
    axes[2].grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results/xlstm_egoexo_multitask_allviews"),
        help="Directory containing metrics.json and training_history.json",
    )
    args = p.parse_args()
    results_dir: Path = args.results_dir
    metrics_path = results_dir / "metrics.json"
    hist_path_file = results_dir / "training_history.json"
    plots = results_dir / "plots"
    plots.mkdir(parents=True, exist_ok=True)

    metrics = load_json(metrics_path)
    history = load_json(hist_path_file)

    plot_epoch_histograms(history, plots / "epoch_histograms_train_and_val.png")
    plot_metrics_histograms(metrics, plots / "metrics_best_val_class_histograms.png")

    print(f"Wrote: {plots / 'epoch_histograms_train_and_val.png'}")
    print(f"Wrote: {plots / 'metrics_best_val_class_histograms.png'}")


if __name__ == "__main__":
    main()
