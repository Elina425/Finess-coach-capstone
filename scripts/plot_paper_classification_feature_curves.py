#!/usr/bin/env python3
"""
Paper-style curves: BiLSTM-CNN vs xLSTM[7:1], grouped by backbone **feature source**
(hand-crafted / ResNet crops / ViTPose-S frame embeddings).

Reads ``history.json`` files written by ``train_paper_classification.py``.

* **Accuracy** — validation accuracy (solid) and train accuracy (dashed), like ``fig_training_curves.png``.
* **Train loss vs epoch** — training CE vs epoch (solid, one curve per feature).
* **Validation vs train loss — trajectory (default)** — Cartesian paths with **training CE loss on x**
  and **validation accuracy on y** (points coloured by epoch). This matches the usual shorthand
  "val vs loss" when ``val_loss`` is not logged.
* **Train vs validation accuracy — trajectory** — **train_acc** (x) vs **val_acc** (y); always in
  ``history.json``. Optional diagonal shows y = x (no generalization gap on that metric).
* **Train vs validation CE loss — trajectory** — same as **training CE (x) vs validation CE (y)**
  (**``--out-val-ce-vs-train-ce``**); only if histories contain ``val_loss``.
* **Optional** — **``val_loss`` vs epoch** next to ``train_loss`` (**``--plot-epochs-train-val-loss``**),
  legacy style; older runs omit ``val_loss``.

Re-run ``train_paper_classification.py`` to log ``val_loss`` for CE-on-CE trajectory and epoch CE overlays.

Example::

  python3 scripts/plot_paper_classification_feature_curves.py
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, MutableMapping, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize


def aggregate_by_epoch(hist: Sequence[Dict[str, Any]]) -> Dict[float, Dict[str, float]]:
    """Merge overlapping keys onto one dict per epoch (float epoch index)."""

    buckets: MutableMapping[float, Dict[str, float]] = defaultdict(dict)
    for row in hist:
        if not isinstance(row, dict) or "epoch" not in row:
            continue
        ep = float(row["epoch"])
        slot = buckets[ep]
        for key in ("train_loss", "val_loss", "val_acc", "train_acc"):
            if key not in row:
                continue
            v = row[key]
            if isinstance(v, (int, float)) and v == v:
                slot[key] = float(v)
    return dict(buckets)


def traj_series_xyz(
    history: Sequence[Dict[str, Any]], x_key: str, y_key: str
) -> Tuple[List[float], List[float], List[float]]:
    agg = aggregate_by_epoch(history)

    xs, ys, es = [], [], []
    for ep in sorted(agg.keys()):
        pt = agg[ep]
        if x_key not in pt or y_key not in pt:
            continue
        xs.append(pt[x_key])
        ys.append(pt[y_key])

        es.append(ep)
    return xs, ys, es


def load_history(path: Path) -> List[Dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"{path}: expected JSON list")
    return data


def epochs_and(
    hist: Sequence[Dict[str, Any]],
    key: str,
) -> Tuple[List[float], List[float]]:
    xs: List[float] = []
    ys: List[float] = []
    for row in hist:
        if key not in row or "epoch" not in row:
            continue
        v = row[key]
        if isinstance(v, (int, float)) and v == v:
            xs.append(float(row["epoch"]))
            ys.append(float(v))
    return xs, ys


def _history_samples_val_loss(history: Sequence[Dict[str, Any]]) -> bool:
    for row in history:
        if isinstance(row.get("val_loss"), (int, float)) and row["val_loss"] == row["val_loss"]:
            return True
    return False


def manifest_val_loss_warnings(labels: List[str], m_paths: Dict[str, Path]) -> bool:
    """Print gaps to stderr; return True if every history has usable val_loss."""
    all_ok = True
    missing: List[str] = []
    for lb in labels:
        for side in ("bilstm", "xlstm"):
            hp = _pick_path(m_paths, lb, side)
            h = load_history(hp)
            if _history_samples_val_loss(h):
                continue
            missing.append(str(hp))
            all_ok = False
    if missing:
        print(
            "[plot_paper_classification_feature_curves] Some histories omit val_loss "
            "(not logged in older trainings). Train vs val curves will show train loss only "
            "for those files. To fill val_loss, re-run:",
            "\n    python train_paper_classification.py  # same CLI as original run\n\n"
            "Missing val_loss:",
            *[f"    - {p}" for p in missing],
            sep="\n",
            file=sys.stderr,
        )
    return all_ok


def load_manifest(repo: Path, manifest_path: Path) -> Tuple[List[str], Dict[str, Path]]:
    raw = json.loads(manifest_path.read_text(encoding="utf-8"))

    feats = raw.get("features")
    if not isinstance(feats, list):
        raise ValueError("manifest.features must be a list")

    out: Dict[str, Path] = {}
    labels: List[str] = []
    seen_labels: Dict[str, int] = {}

    for i, item in enumerate(feats):
        if not isinstance(item, dict):
            raise ValueError(f"manifest.features[{i}] must be an object")

        lb = str(item.get("label", f"feat_{i}")).strip()
        if not lb:
            lb = f"feat_{i}"
        if lb in seen_labels:
            seen_labels[lb] += 1
            lb = f"{lb} ({seen_labels[lb]})"
        else:
            seen_labels[lb] = 0
        labels.append(lb)

        for side in ("bilstm_history", "xlstm_history"):
            p = Path(str(item.get(side, ""))).expanduser()
            if not p.is_absolute():
                p = (repo / p).resolve()
            if not p.is_file():
                raise FileNotFoundError(f"manifest {lb} · missing {side}: {p}")
            key = f"{lb}::{side.removesuffix('_history')}"
            out[key] = p

    return labels, out


def _pick_path(m_paths: Dict[str, Path], label: str, side: str) -> Path:
    k = f"{label}::{side}"
    if k not in m_paths:
        keys = sorted(m_paths.keys())
        raise KeyError(f"no manifest entry for {k!r} (have {keys[:6]}…) ")
    return m_paths[k]


def plot_two_panel_xy_trajectories(
    repo: Path,
    labels: List[str],
    m_paths: Dict[str, Path],
    out_path: Path,
    *,
    x_key: str,
    y_key: str,
    xlab: str,
    ylab: str,
    suptitle: str,
    dpi: int,
    log_x: bool,
    log_y: bool,
    diagonal_0_1: bool = False,
) -> int:
    """Return number of plotted feature-series (combined both panels)."""

    cmap_feat = plt.get_cmap("tab10")
    cmap_epoch = plt.get_cmap("viridis")
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 4.75))
    total_series = 0

    for ax, title_prefix, side in zip(
        axes, ("BiLSTM-CNN", "xLSTM[7:1]"), ("bilstm", "xlstm"), strict=True
    ):
        traces: List[Tuple[List[float], List[float], List[float], str, int]] = []
        for i, lb in enumerate(labels):
            hp = _pick_path(m_paths, lb, side)
            h = load_history(hp)
            xs, ys, es = traj_series_xyz(h, x_key, y_key)

            if not xs:
                continue
            traces.append((xs, ys, es, lb, i))

        if not traces:
            ax.set_title(f"{title_prefix} · no data for ({x_key} vs {y_key})")

            ax.set_xlabel(xlab)
            ax.set_ylabel(ylab)



            ax.grid(True, alpha=0.33)
            continue



        epochs_here = [e for t in traces for e in t[2]]

        vmin, vmax = min(epochs_here), max(epochs_here)
        norm = Normalize(vmin=vmin, vmax=vmax if vmax > vmin else vmin + 1.0)

        if diagonal_0_1:
            ax.plot([0.0, 1.0], [0.0, 1.0], "k--", alpha=0.28, linewidth=1.0, zorder=1)

        for xs, ys, es, lbl, colour_idx in traces:
            total_series += 1
            clr = cmap_feat(colour_idx % 10)
            lx = [max(v, 1e-16) if log_x else float(v) for v in xs]
            ly = [max(v, 1e-16) if log_y else float(v) for v in ys]


            ax.plot(lx, ly, "-", color=clr, linewidth=1.55, alpha=0.82, label=lbl)



            ax.scatter(
                lx,
                ly,

                c=es,
                cmap=cmap_epoch,

                norm=norm,
                s=34,
                linewidths=0.55,
                edgecolors=clr,

                alpha=0.94,
                zorder=5,

            )





        sm_bar = ScalarMappable(norm=norm, cmap=cmap_epoch)
        sm_bar.set_array(epochs_here)
        plt.colorbar(sm_bar, ax=ax, fraction=0.058, pad=0.02, label="Epoch")

        if log_x:
            ax.set_xscale("log")
        if log_y:
            ax.set_yscale("log")

        ax.set_title(title_prefix)
        ax.set_xlabel(xlab)
        ax.set_ylabel(ylab)



        ax.grid(True, alpha=0.33, which="both" if (log_x or log_y) else "major")

        handles, leg_labels = ax.get_legend_handles_labels()
        if leg_labels:






            ax.legend(handles, leg_labels, loc="best", fontsize=8)

    fig.suptitle(suptitle, fontsize=11, y=1.015)
    plt.tight_layout()

    if total_series == 0:
        plt.close()
        return 0


    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close()
    rel = out_path.relative_to(repo) if out_path.is_relative_to(repo) else out_path
    print(f"saved trajectory {x_key}->{y_key} ({total_series} series) → {rel}")

    return total_series


def plot_accuracy_figure(
    repo: Path,
    labels: List[str],
    m_paths: Dict[str, Path],
    out_path: Path,
    *,
    dpi: int,
    max_epoch_bilstm: Optional[int] = None,
    max_epoch_xlstm: Optional[int] = None,
) -> None:
    cmap = plt.get_cmap("tab10")
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 4.2))

    side_max_epoch = {"bilstm": max_epoch_bilstm, "xlstm": max_epoch_xlstm}

    def _trim(ep: List[float], y: List[float], max_ep: Optional[int]) -> Tuple[List[float], List[float]]:
        if max_ep is None or not ep:
            return ep, y
        keep = [(e, v) for e, v in zip(ep, y) if e <= max_ep]
        if not keep:
            return ep, y
        ex, yx = zip(*keep)
        return list(ex), list(yx)

    for ax, title_prefix, side in zip(
        axes, ("BiLSTM-CNN", "xLSTM[7:1]"), ("bilstm", "xlstm"), strict=True
    ):
        y_min, y_max = 1.0, 0.0
        max_ep = side_max_epoch[side]
        for i, lb in enumerate(labels):
            hp = _pick_path(m_paths, lb, side)
            h = load_history(hp)
            ep_v, ya_v = epochs_and(h, "val_acc")
            ep_t, ya_t = epochs_and(h, "train_acc")
            ep_v, ya_v = _trim(ep_v, ya_v, max_ep)
            ep_t, ya_t = _trim(ep_t, ya_t, max_ep)
            color = cmap(i % 10)
            ax.plot(ep_v, ya_v, "-", color=color, linewidth=2.0, label=lb)
            ax.plot(ep_t, ya_t, "--", color=color, linewidth=1.45, alpha=0.9)
            if ya_v:
                y_min = min(y_min, min(ya_v))
                y_max = max(y_max, max(ya_v))
            if ya_t:
                y_min = min(y_min, min(ya_t))
                y_max = max(y_max, max(ya_t))

        pad = max(0.02, (y_max - y_min) * 0.06)
        lo = max(0.0, y_min - pad)
        hi = min(1.0 + pad * 0.35, y_max + pad)
        ax.set_ylim(lo, hi)
        if max_ep is not None:
            ax.set_xlim(0, max_ep)
        ax.set_title(f"{title_prefix} — Validation (solid) / Train (dashed)")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Accuracy")
        ax.grid(True, alpha=0.33)
        ax.legend(loc="lower right", fontsize=9)

    fig.suptitle("PosePulse paper classification — per-feature backends", fontsize=11, y=1.02)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close()
    print(f"saved accuracy → {out_path.relative_to(repo) if out_path.is_relative_to(repo) else out_path}")


def plot_train_val_loss_figure(
    repo: Path,
    labels: List[str],
    m_paths: Dict[str, Path],
    out_path: Path,
    *,
    dpi: int,
    log_scale: bool,
) -> None:
    cmap = plt.get_cmap("tab10")
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 4.2))

    for ax, title_prefix, side in zip(
        axes, ("BiLSTM-CNN", "xLSTM[7:1]"), ("bilstm", "xlstm"), strict=True
    ):
        if log_scale:
            ax.set_yscale("log")

        panel_has_val = False
        for i, lb in enumerate(labels):
            hp = _pick_path(m_paths, lb, side)
            h = load_history(hp)
            color = cmap(i % 10)
            ep_va, ya_va = epochs_and(h, "val_loss")
            ep_tr, ya_tr = epochs_and(h, "train_loss")
            if ya_va:
                panel_has_val = True
                y_va = [max(v, 1e-16) if log_scale else max(v, 0.0) for v in ya_va]
                ax.plot(ep_va, y_va, "-", color=color, linewidth=2.0, label=lb)
            if ya_tr:
                y_tr = [max(v, 1e-16) if log_scale else max(v, 0.0) for v in ya_tr]
                tr_kw: Dict[str, Any] = {} if ya_va else {"label": lb}
                ax.plot(
                    ep_tr,
                    y_tr,
                    "--",
                    color=color,
                    linewidth=1.45,
                    alpha=0.9,
                    **tr_kw,
                )

        ylab = (
            "CE loss (same objective as train & val passes — log)" if log_scale else "CE loss (linear y)"
        )
        ax.set_ylabel(ylab)
        ax.set_xlabel("Epoch")
        ttl = (
            f"{title_prefix} — Validation (solid) / Train (dashed)"
            if panel_has_val
            else f"{title_prefix} — train loss only (add val_loss to history → re-run training)"
        )
        ax.set_title(ttl)
        ax.grid(True, alpha=0.33, which="both" if log_scale else "major")
        handles, leg_labels = ax.get_legend_handles_labels()
        if leg_labels:
            ax.legend(handles, leg_labels, loc="upper right", fontsize=9)

    fig.suptitle(
        "PosePulse paper classification — validation vs train CE loss per feature",
        fontsize=11,
        y=1.02,
    )
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close()
    print(
        "saved train-vs-val-loss → "
        f"{out_path.relative_to(repo) if out_path.is_relative_to(repo) else out_path}"
    )


def plot_loss_figure(
    repo: Path,
    labels: List[str],
    m_paths: Dict[str, Path],
    out_path: Path,
    *,
    dpi: int,
    log_scale: bool,
) -> None:
    cmap = plt.get_cmap("tab10")
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 4.2))

    for ax, title_prefix, side in zip(
        axes, ("BiLSTM-CNN", "xLSTM[7:1]"), ("bilstm", "xlstm"), strict=True
    ):
        if log_scale:
            ax.set_yscale("log")
        for i, lb in enumerate(labels):
            hp = _pick_path(m_paths, lb, side)
            h = load_history(hp)
            ep, yl = epochs_and(h, "train_loss")
            color = cmap(i % 10)
            ys = [max(v, 1e-16) if log_scale else max(v, 0.0) for v in yl]
            ax.plot(ep, ys, "-", color=color, linewidth=2.0, label=lb)

        ax.set_title(f"{title_prefix} — Train loss ({'log scale' if log_scale else 'linear'} y)")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Train CE loss")
        ax.grid(True, alpha=0.33, which="both" if log_scale else "major")
        ax.legend(loc="upper right", fontsize=9)

    fig.suptitle("PosePulse paper classification — train loss per feature", fontsize=11, y=1.02)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close()
    print(f"saved loss     → {out_path.relative_to(repo) if out_path.is_relative_to(repo) else out_path}")


def main() -> None:
    repo = Path(__file__).resolve().parents[1]
    default_manifest = repo / "scripts/paper_classification_feature_manifest.default.json"

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--manifest", type=Path, default=default_manifest, help="feature manifest JSON")
    ap.add_argument(
        "--out-acc",
        type=Path,
        default=repo / "docs/figures/fig_paper_bilstm_xlstm_accuracy_by_feature.png",
    )
    ap.add_argument(
        "--out-loss",
        type=Path,
        default=repo / "docs/figures/fig_paper_bilstm_xlstm_loss_by_feature.png",
    )
    ap.add_argument(
        "--out-val-vs-loss-trajectory",
        type=Path,
        default=repo / "docs/figures/fig_paper_bilstm_xlstm_val_vs_train_loss_trajectory.png",
        help="Trajectory: validation accuracy vs train CE loss (default 'val vs loss' figure).",
    )
    ap.add_argument(
        "--out-val-ce-vs-train-ce",
        type=Path,
        default=repo / "docs/figures/fig_paper_bilstm_xlstm_val_ce_vs_train_ce_trajectory.png",
        help="Trajectory: train CE loss (x) vs validation CE loss (y); needs val_loss in histories.",
    )
    ap.add_argument(
        "--out-train-vs-val-acc-trajectory",
        type=Path,
        default=repo / "docs/figures/fig_paper_bilstm_xlstm_train_vs_val_accuracy_trajectory.png",
        help="Trajectory: train accuracy (x) vs validation accuracy (y).",
    )
    ap.add_argument(
        "--out-epochs-train-val-loss",
        type=Path,
        default=repo / "docs/figures/fig_paper_bilstm_xlstm_train_val_loss_by_feature.png",
        help="Epoch x-axis overlay of train/val CE (see --plot-epochs-train-val-loss).",
    )
    ap.add_argument("--dpi", type=int, default=120)
    ap.add_argument("--accuracy-only", action="store_true", help="Skip all loss PNGs")
    ap.add_argument("--loss-only", action="store_true", help="Skip accuracy PNG")
    ap.add_argument(
        "--skip-train-only-loss",
        action="store_true",
        help="Skip fig_paper_bilstm_xlstm_loss_by_feature.png (train loss only)",
    )
    ap.add_argument(
        "--skip-val-vs-loss-trajectory",
        action="store_true",
        help="Skip trajectory PNG (validation accuracy vs train CE loss).",
    )
    ap.add_argument(
        "--skip-val-ce-trajectory",
        action="store_true",
        help="Skip trajectory PNG (train CE vs validation CE loss).",
    )
    ap.add_argument(
        "--skip-train-vs-val-acc-trajectory",
        action="store_true",
        help="Skip trajectory PNG (train vs validation accuracy).",
    )
    ap.add_argument(
        "--plot-epochs-train-val-loss",
        action="store_true",
        help=(
            "Write epoch plot (--out-epochs-train-val-loss) of val/train CE overlay. "
            "Requires val_loss in history.json."
        ),
    )
    ap.add_argument(
        "--linear-trajectory-x",
        action="store_true",
        help="Linear horizontal axis on trajectory plots (default: log x when x is train CE).",
    )
    ap.add_argument(
        "--linear-loss",
        action="store_true",
        help="Use linear y for epoch-based loss PNGs (default: log y for readability)",
    )
    ap.add_argument(
        "--max-epoch-bilstm",
        type=int,
        default=None,
        help="Trim BiLSTM panels at this epoch (default: full history). Useful when the "
             "curves saturate early — set to the epoch beyond which nothing meaningful happens.",
    )
    ap.add_argument(
        "--max-epoch-xlstm",
        type=int,
        default=None,
        help="Trim xLSTM panels at this epoch (default: full history). xLSTM saturates much "
             "earlier than BiLSTM on these features, so a smaller number is usually appropriate.",
    )
    args = ap.parse_args()

    manifest = args.manifest if args.manifest.is_absolute() else (repo / args.manifest).resolve()
    labels, m_paths = load_manifest(repo, manifest)

    try:
        log_y = not args.linear_loss
        traj_log_x = not args.linear_trajectory_x
        ce_trajectory_log = not args.linear_loss
        if not args.loss_only:
            plot_accuracy_figure(
                repo, labels, m_paths, args.out_acc, dpi=args.dpi,
                max_epoch_bilstm=args.max_epoch_bilstm,
                max_epoch_xlstm=args.max_epoch_xlstm,
            )
        if not args.accuracy_only:
            if not args.skip_train_only_loss:
                plot_loss_figure(
                    repo, labels, m_paths, args.out_loss, dpi=args.dpi, log_scale=log_y
                )



            xlab_traj = (
                "Training CE loss (log-scale x)" if traj_log_x else "Training CE loss (linear x)"
            )
            if not args.skip_val_vs_loss_trajectory:
                plot_two_panel_xy_trajectories(
                    repo,
                    labels,
                    m_paths,
                    args.out_val_vs_loss_trajectory,
                    x_key="train_loss",
                    y_key="val_acc",
                    xlab=xlab_traj,
                    ylab="Validation accuracy",
                    suptitle=(
                        "PosePulse paper classification — validation accuracy vs train loss trajectory "
                        "(colour = epoch along path)"
                    ),
                    dpi=args.dpi,
                    log_x=traj_log_x,
                    log_y=False,
                )
            if not args.skip_val_ce_trajectory:
                ylab_ce = (
                    "Validation CE loss (log-scale y)"
                    if ce_trajectory_log
                    else "Validation CE loss (linear y)"
                )
                plot_two_panel_xy_trajectories(
                    repo,
                    labels,
                    m_paths,
                    args.out_val_ce_vs_train_ce,
                    x_key="train_loss",
                    y_key="val_loss",
                    xlab=xlab_traj,
                    ylab=ylab_ce,
                    suptitle=(
                        "PosePulse paper classification — train vs validation CE loss trajectory "
                        "(colour = epoch along path)"
                    ),
                    dpi=args.dpi,
                    log_x=traj_log_x,
                    log_y=ce_trajectory_log,
                )
            if not args.skip_train_vs_val_acc_trajectory:
                plot_two_panel_xy_trajectories(
                    repo,
                    labels,
                    m_paths,
                    args.out_train_vs_val_acc_trajectory,
                    x_key="train_acc",
                    y_key="val_acc",
                    xlab="Training accuracy",
                    ylab="Validation accuracy",
                    suptitle=(
                        "PosePulse paper classification — train vs validation accuracy trajectory "
                        "(dashed: y = x; colour = epoch)"
                    ),
                    dpi=args.dpi,
                    log_x=False,
                    log_y=False,
                    diagonal_0_1=True,
                )
            if args.plot_epochs_train_val_loss:
                manifest_val_loss_warnings(labels, m_paths)
                plot_train_val_loss_figure(
                    repo,
                    labels,
                    m_paths,
                    args.out_epochs_train_val_loss,
                    dpi=args.dpi,
                    log_scale=log_y,
                )
    except FileNotFoundError as e:
        raise SystemExit(f"{e}") from e


if __name__ == "__main__":
    main()
