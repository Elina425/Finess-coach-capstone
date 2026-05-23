#!/usr/bin/env python3
"""
Benchmark **paper** BiLSTM-CNN vs xLSTM on identical ViT-256 frame sequences.

Two layers of timing (both matter for “real-time”):

1. **Feature extraction** — YOLO pose + ViTPose-S crops → (T, 256). Dominates wall
   clock if you run the full pipeline every time.
2. **Classifier forward** — same sliding windows fed to each checkpoint; answers
   “which backbone is faster *given* features”.

``--videos`` is required (one or more mp4 files).

Example (your last trained paper checkpoints)::

  ./venv/bin/python scripts/benchmark_paper_models_on_videos.py \\
    --bilstm-ckpt results/paper_classification_vit256/bilstm_cnn/best.pt \\
    --xlstm-ckpt  results/paper_xlstm_vit256/xlstm_7_1/xlstm_last_best.pt \\
    --videos ~/data/fitness_aqa/squat.mp4 ~/data/fitness_aqa/shoulder_press.mp4
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from fitness_coach.datasets.exercise_bilstm_dataset import make_windows
from fitness_coach.models.exercise_bilstm_model import ExerciseBiLSTMCNN
from fitness_coach.models.xlstm_model import xLSTMExerciseClassifier


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def _pick_device(name: str) -> torch.device:
    n = (name or "auto").lower()
    if n == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(n)


def _load_ckpt(path: Path, device: torch.device) -> Dict[str, Any]:
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def _a(args: Dict[str, Any], key: str, default: Any) -> Any:
    v = args.get(key, default)
    return default if v is None else v


def build_bilstm_from_paper_ckpt(ckpt: Dict[str, Any], device: torch.device) -> ExerciseBiLSTMCNN:
    ap = ckpt["args"]
    m = ExerciseBiLSTMCNN(
        input_dim=int(_a(ap, "feature_dim", 256)),
        num_classes=int(_a(ap, "num_classes", 4)),
        bilstm_hidden=int(_a(ap, "bilstm_hidden", 4)),
        seq_len=int(_a(ap, "seq_len", 30)),
        dropout=float(_a(ap, "dropout", 0.0)),
    ).to(device)
    m.load_state_dict(ckpt["model"], strict=True)
    m.eval()
    return m


def _infer_xlstm_head_options(sd: Dict[str, torch.Tensor], ap: Dict[str, Any]) -> Dict[str, Any]:
    use_fusion = any(k.startswith("fusion.") for k in sd)
    use_attn_pool = any("attn_pool." in k for k in sd)
    linear_cls = ("class_head.weight" in sd) and ("class_head.0.weight" not in sd)
    layer_ids = []
    for k in sd:
        parts = k.split(".")
        if len(parts) >= 3 and parts[0] == "error_head" and parts[1].isdigit():
            layer_ids.append(int(parts[1]))
    num_error_tags = 0
    if layer_ids:
        last = max(layer_ids)
        bkey = f"error_head.{last}.bias"
        if bkey in sd:
            num_error_tags = int(sd[bkey].shape[0])
    temporal = str(_a(ap, "xlstm_pool", "mean")).strip().lower()
    if use_attn_pool:
        temporal = "attention"
    if temporal not in ("mean", "last", "attention"):
        temporal = "mean"
    return {
        "use_fusion": use_fusion,
        "use_attention_pool": bool(use_attn_pool),
        "temporal_pool": temporal,
        "linear_classifier": bool(linear_cls),
        "num_error_tags": int(num_error_tags),
    }


def build_xlstm_from_paper_ckpt(ckpt: Dict[str, Any], device: torch.device) -> xLSTMExerciseClassifier:
    ap = ckpt["args"]
    sd = ckpt["model"]
    h = _infer_xlstm_head_options(sd, ap)
    num_err = int(h.pop("num_error_tags"))
    m = xLSTMExerciseClassifier(
        input_size=int(_a(ap, "feature_dim", 256)),
        hidden_size=int(_a(ap, "xlstm_hidden", 256)),
        num_classes=int(_a(ap, "num_classes", 4)),
        dropout=float(_a(ap, "dropout", 0.0)),
        num_heads=int(_a(ap, "xlstm_num_heads", 4)),
        conv_kernel_size=int(_a(ap, "xlstm_conv_kernel_size", 4)),
        projection_factor=float(_a(ap, "xlstm_projection_factor", 4.0 / 3.0)),
        block_pattern=str(_a(ap, "xlstm_block_pattern", "mmmmmmms")),
        num_error_tags=num_err,
        fusion_dim=128,
        input_dropout=float(_a(ap, "xlstm_input_dropout", 0.0) or 0.0),
        **h,
    ).to(device)
    m.load_state_dict(sd, strict=True)
    m.eval()
    return m


def _forward_bilstm(m: ExerciseBiLSTMCNN, x: torch.Tensor) -> torch.Tensor:
    out = m(x)
    return out[0] if isinstance(out, (tuple, list)) else out


def _forward_xlstm_cls(m: xLSTMExerciseClassifier, x: torch.Tensor) -> torch.Tensor:
    out = m(x)
    return out[0] if isinstance(out, (tuple, list)) else out


def _time_fn(fn: Callable[[], Any], *, warmup: int, rounds: int, device: torch.device) -> Dict[str, float]:
    for _ in range(max(0, warmup)):
        fn()
    _sync(device)
    samples: List[float] = []
    for _ in range(max(1, rounds)):
        _sync(device)
        t0 = time.perf_counter()
        fn()
        _sync(device)
        samples.append((time.perf_counter() - t0) * 1000.0)
    arr = np.asarray(samples, dtype=np.float64)
    return {
        "n_rounds": int(rounds),
        "mean_ms": float(arr.mean()),
        "median_ms": float(np.median(arr)),
        "p95_ms": float(np.percentile(arr, 95)),
    }


def extract_vit_features(
    video: Path,
    *,
    max_frames: int,
    vit_device: str,
) -> Tuple[Optional[np.ndarray], float, str]:
    t0 = time.perf_counter()
    err = ""
    try:
        from fitness_coach.preprocessing.vit_frame_features import vit_frame_features_from_yolo_video
    except ImportError as e:
        return None, time.perf_counter() - t0, f"import vit_frame_features: {e}"
    out = vit_frame_features_from_yolo_video(
        video,
        max_frames if max_frames > 0 else None,
        vit_device=str(vit_device),
    )
    elapsed = time.perf_counter() - t0
    if out is None:
        return None, elapsed, "vit_frame_features_from_yolo_video returned None"
    feats, meta = out
    return feats.astype(np.float32, copy=False), elapsed, json.dumps(meta, default=str)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument(
        "--bilstm-ckpt",
        type=Path,
        default=REPO_ROOT / "results/paper_classification_vit256/bilstm_cnn/best.pt",
    )
    p.add_argument(
        "--xlstm-ckpt",
        type=Path,
        default=REPO_ROOT / "results/paper_xlstm_vit256/xlstm_7_1/xlstm_last_best.pt",
    )
    p.add_argument(
        "--videos",
        nargs="+",
        type=Path,
        required=True,
        help="One or more mp4 files (Fitness AQA, Riccio, etc.).",
    )
    p.add_argument("--device", default="auto", help="cpu | cuda | mps | auto")
    p.add_argument("--vit-device", default=None,
                   help="Device for ViTPose/YOLO front-end (defaults to --device).")
    p.add_argument("--seq-len", type=int, default=0,
                   help="Window length (0 = read from BiLSTM checkpoint args).")
    p.add_argument("--stride", type=int, default=0,
                   help="Window stride (0 = read from BiLSTM checkpoint args).")
    p.add_argument("--max-frames", type=int, default=300, help="Cap frames for ViT extraction (0=all).")
    p.add_argument("--warmup", type=int, default=15)
    p.add_argument("--rounds", type=int, default=80, help="Timed forward passes per window tensor.")
    p.add_argument("--output", type=Path, default=REPO_ROOT / "results/paper_model_video_benchmark.json")
    args = p.parse_args()

    device = _pick_device(args.device)
    vit_dev = args.vit_device or str(device)

    b_ckpt = _load_ckpt(args.bilstm_ckpt.expanduser().resolve(), torch.device("cpu"))
    x_ckpt = _load_ckpt(args.xlstm_ckpt.expanduser().resolve(), torch.device("cpu"))
    seq_len = int(args.seq_len or _a(b_ckpt["args"], "seq_len", 30))
    stride = int(args.stride or _a(b_ckpt["args"], "stride", 15))

    b_model = build_bilstm_from_paper_ckpt(b_ckpt, device)
    x_model = build_xlstm_from_paper_ckpt(x_ckpt, device)
    fdim = int(_a(b_ckpt["args"], "feature_dim", 256))

    print("=" * 72)
    print(" Paper checkpoints")
    print(f"   BiLSTM : {args.bilstm_ckpt}")
    print(f"   xLSTM  : {args.xlstm_ckpt}")
    print(f"   device : {device}   window {seq_len} stride {stride} feat_dim {fdim}")
    print("=" * 72)

    results: Dict[str, Any] = {
        "device": str(device),
        "vit_device": vit_dev,
        "seq_len": seq_len,
        "stride": stride,
        "feature_dim": fdim,
        "videos": [],
    }

    def bench_windows(windows: List[np.ndarray], label: str) -> Dict[str, Any]:
        if not windows:
            return {"error": "no windows"}
        # Use first window for micro-bench (identical input to both models).
        w0 = torch.from_numpy(windows[0]).float().unsqueeze(0).to(device)
        b_stats = _time_fn(
            lambda: _forward_bilstm(b_model, w0),
            warmup=args.warmup,
            rounds=args.rounds,
            device=device,
        )
        x_stats = _time_fn(
            lambda: _forward_xlstm_cls(x_model, w0),
            warmup=args.warmup,
            rounds=args.rounds,
            device=device,
        )
        ratio = b_stats["median_ms"] / max(1e-9, x_stats["median_ms"])
        if ratio < 1.0:
            faster_tag = "bilstm"
        elif ratio > 1.0:
            faster_tag = "xlstm"
        else:
            faster_tag = "tie"
        out = {
            "label": label,
            "n_windows": len(windows),
            "bilstm_per_window_ms": b_stats,
            "xlstm_per_window_ms": x_stats,
            "median_ms_ratio_bilstm_over_xlstm": float(ratio),
            "faster_backbone_forward": faster_tag,
        }
        print(f"\n[{label}] windows={len(windows)}")
        print(f"  BiLSTM forward  median {b_stats['median_ms']:.3f} ms  (p95 {b_stats['p95_ms']:.3f})")
        print(f"  xLSTM  forward  median {x_stats['median_ms']:.3f} ms  (p95 {x_stats['p95_ms']:.3f})")
        print(f"  Ratio BiLSTM/xLSTM median latency = {ratio:.2f}×  → faster (forward-only): {faster_tag}")
        # Optional per-video prediction (first window only)
        with torch.no_grad():
            bi = _forward_bilstm(b_model, w0).argmax(1).item()
            xi = _forward_xlstm_cls(x_model, w0).argmax(1).item()
        out["argmax_bilstm_idx"] = int(bi)
        out["argmax_xlstm_idx"] = int(xi)
        return out

    for vp in args.videos:
        vp = vp.expanduser().resolve()
        entry: Dict[str, Any] = {"path": str(vp)}
        if not vp.is_file():
            entry["error"] = "file not found"
            results["videos"].append(entry)
            print(f"\n[skip] missing {vp}")
            continue

        feats, vit_s, vit_note = extract_vit_features(
            vp, max_frames=int(args.max_frames), vit_device=vit_dev
        )
        entry["vit_extract_seconds"] = float(vit_s)
        entry["vit_meta"] = vit_note
        if feats is None or feats.shape[0] < seq_len:
            entry["error"] = "features missing or shorter than seq_len"
            results["videos"].append(entry)
            print(f"\n[skip] {vp.name}: {entry['error']}")
            continue
        wins = list(make_windows(feats, seq_len, stride)) or [feats[-seq_len:].copy()]
        entry["T_frames"] = int(feats.shape[0])
        entry["n_windows"] = len(wins)
        entry["throughput_frontend_fps"] = float(feats.shape[0] / vit_s) if vit_s > 1e-6 else None

        fwd = bench_windows(wins, label=vp.name)
        entry.update(fwd)
        median_b = fwd.get("bilstm_per_window_ms", {}).get("median_ms", 0.0)
        median_x = fwd.get("xlstm_per_window_ms", {}).get("median_ms", 0.0)
        nwin = max(1, int(fwd.get("n_windows", 1)))
        entry["wall_estimate_s_bilstm"] = float(vit_s + nwin * median_b / 1000.0)
        entry["wall_estimate_s_xlstm"] = float(vit_s + nwin * median_x / 1000.0)

        print(f"  ViT+YOLO extract: {vit_s:.2f}s for T={feats.shape[0]} (~{entry['throughput_frontend_fps']:.1f} eff. fps)")
        print(f"  Rough E2E (extract + {nwin}×window): BiLSTM ~{entry['wall_estimate_s_bilstm']:.2f}s | "
              f"xLSTM ~{entry['wall_estimate_s_xlstm']:.2f}s")

        results["videos"].append(entry)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2, default=str))
    print(f"\nWrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
