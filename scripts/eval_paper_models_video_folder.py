#!/usr/bin/env python3
"""Evaluate paper BiLSTM-CNN / xLSTM checkpoints on a folder tree of labeled MP4s.

Ground-truth coarse class is inferred from the first directory under ``--dataset-root``
(e.g. ``.../Original Datasets/Bicep Curl/clips/foo.mp4`` → ``barbell biceps curl``).

Videos are fed through ``vit_frame_features_from_yolo_video`` (YOLO + ViTPose crops → T×256),
standardized with the **same train-only mean/std** as ``train_paper_classification.py`` builds
from Riccio ``vit_backbone`` NPZs (hammer curl excluded), then sliding windows → **majority-vote**
per-video predicted class among the checkpoint's labels.

OOD folder names (no mapping to paper classes) are still scored in a secondary report whose
truth labels retain the folder name."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from fitness_coach.datasets.exercise_bilstm_dataset import build_kaggle_frame_feature_datasets, make_windows

_CLASS_NAMES_PAPER = ("barbell biceps curl", "push-up", "shoulder press", "squat")


def _load_benchmark_module():
    spec = importlib.util.spec_from_file_location(
        "benchmark_paper_models_on_videos",
        REPO_ROOT / "scripts/benchmark_paper_models_on_videos.py",
    )
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def _folder_truth(folder_name: str) -> Tuple[Optional[str], str]:
    """Returns (canonical_paper_label or None if OOD, raw_folder_display_name)."""
    n = folder_name.strip().lower()
    raw = folder_name.strip()
    if ("bicep" in n or "biceps" in n) and "curl" in n:
        return "barbell biceps curl", raw
    if "shoulder" in n and "press" in n:
        return "shoulder press", raw
    if "squat" in n:
        return "squat", raw
    if "push" in n and "up" in n:
        return "push-up", raw
    if "hammer" in n and "curl" in n:
        return None, raw  # excluded from paper 4-way
    return None, raw


def _discover_videos(root: Path) -> List[Tuple[Path, str, Optional[str]]]:
    out: List[Tuple[Path, str, Optional[str]]] = []
    root = root.resolve()
    for mp4 in sorted(root.rglob("*.mp4")):
        try:
            rel = mp4.relative_to(root)
        except ValueError:
            continue
        if not rel.parts:
            continue
        top = rel.parts[0]
        canonical, disp = _folder_truth(top)
        out.append((mp4, disp, canonical))
    return out


def _scalar_stats(
    yt: Sequence[str], yp: Sequence[str], labels: Sequence[str]
) -> Dict[str, Any]:
    from sklearn.metrics import accuracy_score, f1_score, classification_report

    acc = float(accuracy_score(yt, yp))
    fi = float(f1_score(yt, yp, labels=list(labels), average="macro", zero_division=0))
    rep = classification_report(list(yt), list(yp), labels=list(labels), zero_division=0, digits=4)
    rep_d = classification_report(
        list(yt), list(yp), labels=list(labels), zero_division=0, output_dict=True, digits=6
    )
    return {"accuracy": acc, "f1_macro": fi, "classification_report_text": rep, "classification_report": rep_d}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-root", type=Path, required=True)
    ap.add_argument(
        "--bilstm-ckpt",
        type=Path,
        default=REPO_ROOT / "results/paper_classification_vit256/bilstm_cnn/best.pt",
    )
    ap.add_argument(
        "--xlstm-ckpt",
        type=Path,
        default=REPO_ROOT / "results/paper_xlstm_vit256/xlstm_7_1/xlstm_last_best.pt",
    )
    ap.add_argument(
        "--riccio-features-dir",
        type=Path,
        default=REPO_ROOT / "results/riccio_vit256_features",
        help="Used only to reconstruct train split mean/std (same recipe as train_paper_classification).",
    )
    ap.add_argument("--riccio-stem", default="riccio_realtime_exercise_recognition")
    ap.add_argument("--exclude-hammer-curl", action="store_true", default=True)
    ap.add_argument(
        "--max-frames",
        type=int,
        default=300,
        help="Max frames passed to ViT pipeline per video (0 = entire clip; large clips can be very slow).",
    )
    ap.add_argument(
        "--max-videos",
        type=int,
        default=0,
        help="Process at most this many MP4s (0 = all). Useful for smoke tests.",
    )
    ap.add_argument("--device", default="auto")
    ap.add_argument("--vit-device", default=None)
    ap.add_argument(
        "--output-json",
        type=Path,
        default=REPO_ROOT / "results/original_datasets_paper_eval/classification_report.json",
    )
    args = ap.parse_args()

    print("[eval] loading checkpoint helpers (YOLO/ViT code)…", flush=True)
    bm = _load_benchmark_module()

    device = bm._pick_device(args.device)
    vit_dev = args.vit_device or str(device)
    exclude = ["hammer curl"] if args.exclude_hammer_curl else []

    print(
        f"[eval] building train-split standardizer from {args.riccio_features_dir}…",
        flush=True,
    )
    rows = build_kaggle_frame_feature_datasets(
        args.riccio_features_dir.expanduser(),
        stem=args.riccio_stem,
        window=int(30),
        stride=int(15),
        seed=42,
        standardize=True,
        window_label="first",
        exclude_coarse_classes=exclude or None,
    )
    _, _, _, _, _, scale_mean, scale_std = rows
    if scale_mean is None or scale_std is None:
        print("Could not derive standardizer from Riccio NPZ — check paths.", file=sys.stderr)
        return 1

    b_ckpt = bm._load_ckpt(args.bilstm_ckpt.expanduser(), torch.device("cpu"))
    x_ckpt = bm._load_ckpt(args.xlstm_ckpt.expanduser(), torch.device("cpu"))
    num_classes_b = int(b_ckpt["args"].get("num_classes", len(_CLASS_NAMES_PAPER)))
    idx_to_class = {i: _CLASS_NAMES_PAPER[i] for i in range(min(num_classes_b, len(_CLASS_NAMES_PAPER)))}
    if num_classes_b != len(idx_to_class):
        print(f"[warn] checkpoint num_classes={num_classes_b}; expected {_CLASS_NAMES_PAPER}", flush=True)

    b_model = bm.build_bilstm_from_paper_ckpt(b_ckpt, device)
    x_model = bm.build_xlstm_from_paper_ckpt(x_ckpt, device)
    print("[eval] checkpoints on device; starting per-video inference…", flush=True)
    seq_len = int(b_ckpt["args"].get("seq_len", 30))
    stride = int(b_ckpt["args"].get("stride", 15))

    vids = _discover_videos(args.dataset_root.expanduser())
    if args.max_videos and args.max_videos > 0:
        vids = vids[: int(args.max_videos)]
    if not vids:
        print(f"No .mp4 under {args.dataset_root}", file=sys.stderr)
        return 1
    print(f"[eval] {len(vids)} videos  device={device} vit_device={vit_dev}  max_frames={args.max_frames}", flush=True)

    per_video: List[Dict[str, Any]] = []

    def _pred_majority(model: torch.nn.Module, forward_fn: Callable, wins: List[np.ndarray]) -> int:
        votes: Counter[int] = Counter()
        with torch.no_grad():
            for arr in wins:
                x = torch.from_numpy(arr).float().unsqueeze(0).to(device)
                logits = forward_fn(model, x)
                votes[int(logits.argmax(1).item())] += 1
        return votes.most_common(1)[0][0]

    for ni, (vp, fold_disp, canonical) in enumerate(vids, start=1):
        print(f"  [{ni}/{len(vids)}] {vp.name}  folder={fold_disp!r}", flush=True)
        feats, elapsed, meta = bm.extract_vit_features(
            vp, max_frames=args.max_frames if args.max_frames > 0 else None, vit_device=str(vit_dev)
        )
        rec: Dict[str, Any] = {
            "path": str(vp),
            "folder_label": fold_disp,
            "canonical_truth": canonical,
            "vit_extract_s": elapsed,
            "vit_meta": meta,
            "error": None,
            "predictions": {},
        }
        if feats is None or feats.shape[0] < seq_len:
            rec["error"] = "short_or_no_features"
            per_video.append(rec)
            continue
        z = ((feats.astype(np.float32) - scale_mean) / scale_std).astype(np.float32)
        wins = list(make_windows(z, seq_len, stride))
        if not wins:
            rec["error"] = "no_windows"
            per_video.append(rec)
            continue
        bi = _pred_majority(b_model, bm._forward_bilstm, wins)
        xi = _pred_majority(x_model, bm._forward_xlstm_cls, wins)
        rec["predictions"] = {
            "bilstm_cnn": idx_to_class.get(int(bi), str(bi)),
            "xlstm": idx_to_class.get(int(xi), str(xi)),
            "n_windows": len(wins),
        }
        per_video.append(rec)

    labels_paper = [idx_to_class[i] for i in range(num_classes_b)]

    def build_lists(model_key: str) -> Tuple[List[str], List[str], List[int]]:
        yt, yp, ok_idx = [], [], []
        for i, row in enumerate(per_video):
            if row.get("error"):
                continue
            pred = row["predictions"][model_key]
            can = row["canonical_truth"]
            if can is None:
                continue
            yt.append(can)
            yp.append(pred)
            ok_idx.append(i)
        return yt, yp, ok_idx

    reports: Dict[str, Any] = {
        "dataset_root": str(args.dataset_root.resolve()),
        "n_videos_seen": len(vids),
        "checkpoints": {
            "bilstm": str(args.bilstm_ckpt),
            "xlstm": str(args.xlstm_ckpt),
        },
        "standardizer": {
            "source": str(args.riccio_features_dir),
            "stem": args.riccio_stem,
            "exclude_hammer_curl": bool(args.exclude_hammer_curl),
        },
        "per_video": per_video,
        "subset_in_distribution": {},
        "subset_all_mapped_folders_macro": {},
    }

    for mkey in ("bilstm_cnn", "xlstm"):
        yt_id, yp_id, _ = build_lists(mkey)
        if yt_id:
            reports["subset_in_distribution"][mkey] = _scalar_stats(yt_id, yp_id, labels=list(dict.fromkeys(yt_id)))

    # Macro over every video whose folder mapped to some paper label (truth in labels_paper).
    def build_full_paper_truth(model_key: str) -> Tuple[List[str], List[str]]:
        yt, yp = [], []
        for row in per_video:
            if row.get("error"):
                continue
            can = row["canonical_truth"]
            if can is None or can not in labels_paper:
                continue
            yt.append(can)
            yp.append(row["predictions"][model_key])
        return yt, yp

    for mkey in ("bilstm_cnn", "xlstm"):
        ytf, ypf = build_full_paper_truth(mkey)
        if ytf:
            reports["subset_all_mapped_folders_macro"][mkey] = _scalar_stats(
                ytf, ypf, labels=list(labels_paper)
            )

    # Full-dataset report: use canonical paper string as truth when known; OOD folders
    # (e.g. Front Raise) use ``OOD:<folder>`` so strings align with 4-way predictions.
    for mkey in ("bilstm_cnn", "xlstm"):
        rows_ok = [r for r in per_video if not r.get("error")]
        y_all: List[Tuple[str, str]] = []
        for r in rows_ok:
            c = r.get("canonical_truth")
            yt = str(c) if c is not None else f"OOD:{r['folder_label']}"
            y_all.append((yt, r["predictions"][mkey]))
        if not y_all:
            continue
        yt2 = [a for a, _ in y_all]
        yp2 = [b for _, b in y_all]
        uniq = sorted(set(yt2) | set(yp2))
        reports[f"subset_all_videos_canonical_or_ood_{mkey}"] = _scalar_stats(yt2, yp2, labels=uniq)

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(reports, indent=2, default=str))

    print(f"Wrote {args.output_json}")

    print("\n" + "=" * 72)
    print("IN-DISTRIBUTION ONLY (videos whose folder mapped to paper names: biceps, shoulder, etc.)")
    print("Excluded: unmapped folders (e.g. Front Raise) and failed extracts.")
    print("=" * 72)
    for mkey in ("bilstm_cnn", "xlstm"):
        block = reports.get("subset_in_distribution", {}).get(mkey)
        if not block:
            print(f"\n[{mkey}] — no such clips.")
            continue
        print(f"\n--- {mkey} ---")
        print(block["classification_report_text"])
        print(f"accuracy={block['accuracy']:.4f}  macro-F1={block['f1_macro']:.4f}")

    print("\n" + "=" * 72)
    print(
        "ALL VIDEOS — truth = canonical name, or OOD:<folder> (e.g. OOD:Front Raise); "
        "pred ∈ {4 paper classes}."
    )
    print("=" * 72)
    for mkey, label in (("bilstm_cnn", "bilstm_cnn"), ("xlstm", "xlstm")):
        block = reports.get(f"subset_all_videos_canonical_or_ood_{mkey}")
        if block:
            print(f"\n--- {label} ---")
            print(block["classification_report_text"])
            print(f"accuracy={block['accuracy']:.4f}  macro-F1={block['f1_macro']:.4f}")
    print(
        "\nNote: Trained on Riccio; domain shift on this camera explains confusions.\n"
        "Full JSON: subset_all_videos_canonical_or_ood_* and per-video predictions.",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
