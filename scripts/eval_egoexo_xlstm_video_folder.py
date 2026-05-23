#!/usr/bin/env python3
"""Run an EgoExo-trained ``xlstm_egoexo_multitask_best.pt`` checkpoint on a folder tree of MP4s.

This uses **CLIP ViT-B/32 × 512-D** frame features (same modality as EgoExo training), *not*
the ViTPose-256 pipeline used by ``eval_paper_models_video_folder.py``.

Ground-truth **folder names** (e.g. ``Bicep Curl``) are **not** EgoExo class names; the model
predicts one of the 12 EgoExo actions from the checkpoint. Reports are grouped by source folder
with confusion-style heatmaps (folder × predicted EgoExo class).

Example:

  ./venv/bin/python scripts/eval_egoexo_xlstm_video_folder.py \\
      --dataset-root "/path/to/Original Datasets" \\
      --checkpoint results/xlstm_egoexo_multitask_branched_m8/xlstm_egoexo_multitask_best.pt \\
      --out-dir results/original_datasets_egoexo_branched_m8
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from fitness_coach.models.xlstm_model import xLSTMExerciseClassifier  # noqa: E402
from fitness_coach.preprocessing.clip_frame_features import (  # noqa: E402
    CLIPViTB32Extractor,
    clip_vit_b32_frames_from_video,
)


def _discover_videos(root: Path) -> List[Tuple[Path, str]]:
    out: List[Tuple[Path, str]] = []
    root = root.resolve()
    for mp4 in sorted(root.rglob("*.mp4")):
        try:
            rel = mp4.relative_to(root)
        except ValueError:
            continue
        if not rel.parts:
            continue
        top = rel.parts[0]
        out.append((mp4, top))
    return out


def _normalize_ckpt_dict_keys(d: Dict) -> Dict[int, Any]:
    return {int(k): v for k, v in d.items()}


def _forward_cls(model: xLSTMExerciseClassifier, xb: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    pooled = model.encode(xb)
    a, b, _z = model.fuse(pooled)
    logits = model.class_head(a)
    b_q = model.quality_branch_feat(b, logits, None)
    qh = model.quality_head(b_q)
    if getattr(model, "quality_is_classification", False):
        quality = qh  # logits; callers can softmax or compare argmax elsewhere
    else:
        quality = torch.sigmoid(qh) * model.quality_scale + model.quality_output_low
    return logits, quality


def load_egoexo_multitask_model(ckpt_path: Path, device: torch.device) -> Tuple[xLSTMExerciseClassifier, Dict[str, Any]]:
    blob = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = blob["model"]
    n_err_keys = sum(1 for k in sd if "error_head" in k)
    num_error_tags = 0 if n_err_keys == 0 else len(blob.get("error_tags") or [])

    classes: List[str] = list(blob["classes"])
    model = xLSTMExerciseClassifier(
        input_size=int(blob["input_size"]),
        hidden_size=int(blob["hidden"]),
        num_layers=int(blob["layers"]),
        num_classes=len(classes),
        dropout=float(blob["dropout"]),
        num_heads=int(blob["num_heads"]),
        conv_kernel_size=int(blob["conv_kernel_size"]),
        projection_factor=float(blob["projection_factor"]),
        num_error_tags=num_error_tags,
        quality_scale=float(blob.get("quality_scale") or 1.0),
        quality_output_low=float(blob.get("quality_output_low", 0.0)),
        num_quality_classes=int(blob.get("num_quality_classes", 1)),
        block_pattern=str(blob["block_pattern"]),
        use_attention_pool=bool(blob["use_attention_pool"]),
        use_fusion=bool(blob["use_fusion"]),
        fusion_dim=int(blob["fusion_dim"]),
        quality_class_conditioning=bool(blob.get("teacher_force_quality", False)),
    )
    model.load_state_dict(sd)
    g = _normalize_ckpt_dict_keys(blob["guidance_table"])
    ic = _normalize_ckpt_dict_keys(blob["idx_to_class"])
    model.set_guidance_table(g, ic)
    ct_raw = blob.get("comment_table") or {}
    comment_table = {}
    for key, text in ct_raw.items():
        c, bk = key.split("|")
        comment_table[(int(c), int(bk))] = str(text)
    edges = tuple(float(e) for e in (blob.get("comment_quality_bucket_edges") or (0.4, 0.7)))
    q_lo = float(blob.get("quality_domain_lo", 0.0))
    q_hi = float(blob.get("quality_domain_hi", 1.0))
    model.set_comment_table(comment_table, edges, domain_lo=q_lo, domain_hi=q_hi)
    model.to(device)
    model.eval()
    meta = {
        "classes": classes,
        "mean": np.asarray(blob["mean"], dtype=np.float32),
        "std": np.asarray(blob["std"], dtype=np.float32),
    }
    if meta["mean"].shape[0] != int(blob["input_size"]) or meta["std"].shape[0] != int(blob["input_size"]):
        raise ValueError("Checkpoint mean/std length does not match input_size")
    return model, meta


def _pick_device(explicit: str) -> torch.device:
    e = (explicit or "auto").strip().lower()
    if e == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        m = getattr(torch.backends, "mps", None)
        if m is not None and m.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(e)


def _clip_device_arg(model_dev: torch.device, explicit: str) -> str:
    """HF CLIP is most reliable on CUDA/CPU; MPS can fail on some ops."""
    ex = (explicit or "auto").strip().lower()
    if ex != "auto":
        return ex
    if model_dev.type == "mps":
        return "cpu"
    return str(model_dev)


def _plot_heatmap(
    matrix: np.ndarray,
    row_labels: List[str],
    col_labels: List[str],
    title: str,
    out_png: Path,
) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(max(8, len(col_labels) * 0.35), max(4, len(row_labels) * 0.45)))
    im = ax.imshow(matrix, aspect="auto", interpolation="nearest", cmap="Blues")
    ax.set_xticks(np.arange(len(col_labels)))
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_xticklabels(col_labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(row_labels, fontsize=9)
    ax.set_title(title)
    ax.set_xlabel("Predicted EgoExo class")
    ax.set_ylabel("Source folder")
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            v = matrix[i, j]
            if v > 0:
                ax.text(j, i, str(int(v)), ha="center", va="center", color="black", fontsize=7)
    fig.colorbar(im, ax=ax, fraction=0.02, pad=0.02)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=160)
    plt.close(fig)


def _plot_folder_bars(counts_by_folder: Dict[str, Dict[str, int]], classes: List[str], out_png: Path) -> None:
    import matplotlib.pyplot as plt

    folders = sorted(counts_by_folder.keys())
    x = np.arange(len(classes))
    width = 0.8 / max(1, len(folders))
    fig, ax = plt.subplots(figsize=(max(10, len(classes) * 0.45), 5))
    for i, folder in enumerate(folders):
        block = counts_by_folder[folder]
        vals = [block.get(c, 0) for c in classes]
        ax.bar(x + (i - (len(folders) - 1) / 2) * width, vals, width, label=folder)
    ax.set_xticks(x)
    ax.set_xticklabels(classes, rotation=55, ha="right", fontsize=8)
    ax.set_ylabel("Videos")
    ax.legend(fontsize=8)
    ax.set_title("Predicted EgoExo class counts by source folder")
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=160)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dataset-root", type=Path, required=True)
    ap.add_argument(
        "--checkpoint",
        type=Path,
        default=REPO_ROOT / "results/xlstm_egoexo_multitask_branched_m8/xlstm_egoexo_multitask_best.pt",
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=REPO_ROOT / "results/original_datasets_egoexo_branched_m8",
    )
    ap.add_argument("--device", type=str, default="auto", help="xLSTM device (auto picks cuda > mps > cpu).")
    ap.add_argument(
        "--clip-device",
        type=str,
        default="auto",
        help="CLIP backbone device. Default: same as --device except MPS → CPU for stability.",
    )
    ap.add_argument("--max-frames", type=int, default=300)
    ap.add_argument("--clip-stride", type=int, default=3)
    ap.add_argument("--clip-batch-size", type=int, default=16)
    ap.add_argument("--max-videos", type=int, default=0, help="0 = all")
    args = ap.parse_args()

    device = _pick_device(args.device)
    clip_dev = _clip_device_arg(device, args.clip_device)
    clip_extractor = CLIPViTB32Extractor(device=clip_dev)
    model, meta = load_egoexo_multitask_model(args.checkpoint.resolve(), device)
    classes: List[str] = meta["classes"]
    mean = meta["mean"]
    std = meta["std"]

    vids = _discover_videos(args.dataset_root.resolve())
    if args.max_videos and args.max_videos > 0:
        vids = vids[: args.max_videos]

    results: List[Dict[str, Any]] = []
    counts: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for mp4, folder in vids:
        extracted = clip_vit_b32_frames_from_video(
            mp4,
            max_frames=args.max_frames,
            subsample_stride=args.clip_stride,
            device=clip_dev,
            batch_size=args.clip_batch_size,
            extractor=clip_extractor,
        )
        if extracted is None:
            results.append(
                {
                    "video": str(mp4),
                    "folder": folder,
                    "error": "clip_extraction_failed",
                }
            )
            continue
        feats, clip_meta = extracted
        x = (feats.astype(np.float32) - mean) / std
        xb = torch.from_numpy(x[None, ...]).to(device)
        with torch.no_grad():
            logits, q = _forward_cls(model, xb)
        prob = torch.softmax(logits, dim=-1)[0].cpu().numpy()
        pred_idx = int(logits.argmax(dim=-1).item())
        pred_name = classes[pred_idx]
        if getattr(model, "quality_is_classification", False):
            qp = torch.softmax(q, dim=-1)[0].cpu().numpy()
            centres = np.asarray(model._quality_bucket_centres, dtype=np.float64)
            q_scalar = float((qp.reshape(-1) * centres.reshape(-1)).sum())
            q_disp = qp.tolist()
        else:
            q_scalar = float(q.squeeze().cpu().item())
            q_disp = q_scalar
        counts[folder][pred_name] += 1
        results.append(
            {
                "video": str(mp4),
                "folder": folder,
                "predicted_class": pred_name,
                "predicted_index": pred_idx,
                "quality_pred": q_scalar,
                "quality_detail": q_disp,
                "confidence": float(prob[pred_idx]),
                "probabilities": {classes[i]: float(prob[i]) for i in range(len(classes))},
                "clip_meta": clip_meta,
            }
        )

    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / "classification_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "checkpoint": str(args.checkpoint.resolve()),
                "dataset_root": str(args.dataset_root.resolve()),
                "xlstm_device": str(device),
                "clip_device": clip_dev,
                "n_videos": len(vids),
                "egoexo_classes": classes,
                "per_video": results,
            },
            f,
            indent=2,
        )

    folder_labels = sorted(counts.keys())
    mat = np.zeros((len(folder_labels), len(classes)), dtype=np.float32)
    for i, fd in enumerate(folder_labels):
        for j, cname in enumerate(classes):
            mat[i, j] = float(counts[fd].get(cname, 0))

    _plot_heatmap(
        mat,
        folder_labels,
        classes,
        "EgoExo multitask xLSTM: source folder vs predicted class (counts)",
        out_dir / "folder_vs_prediction_heatmap.png",
    )

    cb_flat: Dict[str, Dict[str, int]] = {fd: dict(counts[fd]) for fd in counts}
    _plot_folder_bars(cb_flat, classes, out_dir / "predictions_by_folder_bars.png")

    print(f"Wrote {report_path}")
    print(f"Wrote {out_dir / 'folder_vs_prediction_heatmap.png'}")
    print(f"Wrote {out_dir / 'predictions_by_folder_bars.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
