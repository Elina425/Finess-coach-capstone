#!/usr/bin/env python3
"""Build BiLSTM (hidden=64) vs xLSTM 7:1 comparison figure from saved test metrics."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

DEFAULT_BILSTM_JSON = REPO / "results/paper_bilstm_hidden64/bilstm_cnn/test_classification_metrics.json"
DEFAULT_XLSTM_JSON = REPO / "results/paper_xlstm_vit256/xlstm_7_1/test_classification_metrics.json"
DEFAULT_LATENCY_JSON = REPO / "results/paper_model_video_benchmark.json"
DEFAULT_OUT_PNG = REPO / "docs/diagrams/paper_bilstm64_vs_xlstm71_riccio.png"
DEFAULT_META = REPO / "docs/diagrams/paper_bilstm64_vs_xlstm71_riccio_meta.json"


def _load_metrics(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text())


def _short_class_labels(names: List[str]) -> List[str]:
    out = []
    for n in names:
        n = str(n).strip()
        if "biceps" in n.lower() or "curl" in n.lower() and "hammer" not in n.lower():
            out.append("Biceps curl")
        elif "push" in n.lower():
            out.append("Push-up")
        elif "shoulder" in n.lower():
            out.append("Shoulder press")
        elif "squat" in n.lower():
            out.append("Squat")
        else:
            out.append(n[:18] + ("…" if len(n) > 18 else ""))
    return out


def _pick_latency_ms(
    bench_path: Optional[Path],
) -> Tuple[Optional[float], Optional[float], str]:
    if not bench_path or not bench_path.is_file():
        return None, None, "no benchmark JSON found"
    raw = json.loads(bench_path.read_text())
    syn = raw.get("synthetic_forward_only")
    if syn and isinstance(syn, dict):
        b = syn.get("bilstm_per_window_ms") or {}
        x = syn.get("xlstm_per_window_ms") or {}
        mb, mx = b.get("median_ms"), x.get("median_ms")
        if mb is not None and mx is not None:
            return float(mb), float(mx), str(raw.get("device", "unknown device"))
    vids = raw.get("videos") or []
    meds_b, meds_x = [], []
    for e in vids:
        if not isinstance(e, dict) or e.get("error"):
            continue
        bb = (e.get("bilstm_per_window_ms") or {}).get("median_ms")
        xx = (e.get("xlstm_per_window_ms") or {}).get("median_ms")
        if bb is not None and xx is not None:
            meds_b.append(float(bb))
            meds_x.append(float(xx))
    if meds_b:
        return (
            float(np.median(meds_b)),
            float(np.median(meds_x)),
            str(raw.get("device", "unknown device")),
        )
    return None, None, "no timing block in benchmark JSON"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bilstm-metrics", type=Path, default=DEFAULT_BILSTM_JSON)
    ap.add_argument("--xlstm-metrics", type=Path, default=DEFAULT_XLSTM_JSON)
    ap.add_argument("--latency-json", type=Path, default=DEFAULT_LATENCY_JSON)
    ap.add_argument("--no-latency", action="store_true", help="Omit latency rows from table")
    ap.add_argument("--out-png", type=Path, default=DEFAULT_OUT_PNG)
    ap.add_argument("--out-meta", type=Path, default=DEFAULT_META)
    args = ap.parse_args()

    mb = _load_metrics(args.bilstm_metrics)
    mx = _load_metrics(args.xlstm_metrics)
    names = list(mb["class_names"])
    if list(mx["class_names"]) != names:
        raise SystemExit("class_names order mismatch between metric files")

    # Per-class recall (TP/support) matches "accuracy on examples of that class"
    r_b = [100.0 * float(mb["recall_per_class"][c]) for c in names]
    r_x = [100.0 * float(mx["recall_per_class"][c]) for c in names]
    labels = _short_class_labels(names)

    acc_b = 100.0 * float(mb["accuracy"])
    acc_x = 100.0 * float(mx["accuracy"])
    f1_b = float(mb["f1_macro"])
    f1_x = float(mx["f1_macro"])

    import torch

    from fitness_coach.models.exercise_bilstm_model import ExerciseBiLSTMCNN

    dropout = 0.3
    bilstm_params = sum(
        p.numel()
        for p in ExerciseBiLSTMCNN(
            input_dim=256,
            num_classes=len(names),
            bilstm_hidden=64,
            seq_len=30,
            dropout=dropout,
        ).parameters()
    )
    xlstm_params: Optional[int] = None
    x_ckpt = REPO / "results/paper_xlstm_vit256/xlstm_7_1/xlstm_last_best.pt"
    if x_ckpt.is_file():
        ck = torch.load(x_ckpt, map_location="cpu", weights_only=False)
        xlstm_params = sum(t.numel() for t in ck["model"].values())

    lat_b, lat_x, lat_dev = _pick_latency_ms(None if args.no_latency else args.latency_json)

    meta = {
        "bilstm_metrics": str(args.bilstm_metrics),
        "xlstm_metrics": str(args.xlstm_metrics),
        "dataset_note": "Riccio · ViT-256 patches · 4 classes · same test split",
        "accuracy_bilstm_pct": round(acc_b, 2),
        "accuracy_xlstm_pct": round(acc_x, 2),
        "f1_macro_bilstm": round(f1_b, 4),
        "f1_macro_xlstm": round(f1_x, 4),
        "parameters_bilstm": int(bilstm_params),
        "parameters_xlstm": int(xlstm_params) if xlstm_params is not None else None,
        "per_class_recall_pct": {
            "bilstm": {labels[i]: r_b[i] for i in range(len(labels))},
            "xlstm": {labels[i]: r_x[i] for i in range(len(labels))},
        },
        "forward_median_ms_cpu": (
            {"bilstm": lat_b, "xlstm": lat_x, "device_note": lat_dev}
            if lat_b is not None
            else None
        ),
    }
    args.out_meta.parent.mkdir(parents=True, exist_ok=True)
    args.out_meta.write_text(json.dumps(meta, indent=2))

    # --- figure
    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.titlesize": 13,
            "figure.facecolor": "white",
        }
    )

    fig = plt.figure(figsize=(11.5, 6.2), dpi=150)
    fig.suptitle(
        "Classification performance — BiLSTM-CNN vs xLSTM [7:1]",
        fontsize=15,
        fontweight="600",
        y=0.98,
    )
    fig.text(
        0.5,
        0.93,
        "Test set · Riccio (ViT-256 frame features) · four exercise classes",
        ha="center",
        fontsize=11,
        color="#444",
    )

    ax1 = fig.add_axes([0.08, 0.12, 0.52, 0.72])
    x = np.arange(len(labels))
    w = 0.36
    c_b, c_x = "#2563eb", "#ea580c"
    ax1.bar(x - w / 2, r_b, width=w, label="BiLSTM-CNN (hidden 64)", color=c_b, edgecolor="white", linewidth=0.6)
    ax1.bar(x + w / 2, r_x, width=w, label="xLSTM [7:1]", color=c_x, edgecolor="white", linewidth=0.6)
    ax1.set_ylabel("Recall (%) on true class")
    ax1.set_ylim(0, 105)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=15, ha="right")
    ax1.set_title("Per-class recall", fontweight="600", pad=8)
    ax1.legend(loc="lower right", frameon=True, fancybox=False, edgecolor="#ccc")
    ax1.grid(axis="y", linestyle="--", alpha=0.35)
    ax1.axhline(100, color="#ddd", linewidth=0.8, zorder=0)

    # Table panel
    ax2 = fig.add_axes([0.64, 0.12, 0.33, 0.72])
    ax2.axis("off")

    rows = [
        ("Accuracy (BiLSTM)", f"{acc_b:.1f}%"),
        ("Accuracy (xLSTM)", f"{acc_x:.1f}%"),
        ("F1-macro (BiLSTM)", f"{f1_b:.3f}"),
        ("F1-macro (xLSTM)", f"{f1_x:.3f}"),
        ("Parameters (BiLSTM)", f"{bilstm_params / 1e6:.2f}M"),
        ("Parameters (xLSTM)", f"{(xlstm_params or 0) / 1e6:.2f}M" if xlstm_params else "—"),
    ]
    if lat_b is not None and lat_x is not None:
        rows.append(("Median forward (BiLSTM)", f"{lat_b:.1f} ms"))
        rows.append(("Median forward (xLSTM)", f"{lat_x:.1f} ms"))

    table = ax2.table(
        cellText=[[a, b] for a, b in rows],
        colLabels=["Metric", "Value"],
        loc="center",
        cellLoc="left",
        colLoc="left",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.05, 1.35)
    for (r, c), cell in table.get_celld().items():
        cell.set_edgecolor("#ccc")
        if r == 0:
            cell.set_facecolor("#f3f4f6")
            cell.get_text().set_fontweight("600")
        else:
            cell.set_facecolor("#fafafa" if r % 2 == 0 else "white")
    ax2.text(0.05, 0.98, "Summary", transform=ax2.transAxes, fontweight="600", fontsize=12, va="top")

    foot = (
        "Per-class recall: fraction of clips with that ground-truth label predicted correctly. "
        "Architecture parameter counts from model definitions/checkpoint. "
    )
    if lat_b is not None:
        foot += (
            f"Median forward timings from backbone micro-benchmark ({lat_dev}); "
            "see results/paper_model_video_benchmark.json — use CUDA/MPS for deployment-relevant numbers."
        )
    else:
        foot += "Forward timing: run `scripts/benchmark_paper_models_on_videos.py` with your `--videos` to fill in."
    fig.text(0.5, 0.02, foot, ha="center", fontsize=8.5, color="#555", wrap=True)

    args.out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out_png, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {args.out_png}")
    print(f"Wrote {args.out_meta}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
