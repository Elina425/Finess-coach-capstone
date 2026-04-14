#!/usr/bin/env python3
"""
Complete Pose Estimation Pipeline with Model Comparison and Visualization
Processes sample videos and generates skeleton detection visualizations
"""

import argparse
import json
import random
import sys
import cv2
import numpy as np
from pathlib import Path
from typing import Any, List, Optional, Sequence, Tuple
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from fitness_coach.core.pose_estimation_core import iter_default_comparison_detectors


def _model_display_name(key: str) -> str:
    names = {
        "mediapipe": "MediaPipe",
        "yolo": "YOLOv8-Pose",
        "rtmpose_x": "RTMPose-X",
        "vitpose": "ViTPose",
        "detrpose": "DETRPose",
        "openpose": "OpenPose",
    }
    return names.get(key, key)


def aggregate_benchmark_over_videos(
    video_paths: List[str],
    max_frames: int = 100,
) -> dict:
    """
    Run model_comparison.ModelComparison on each video and mean-aggregate metrics.
    Returns dict[model_key] -> ComparisonMetrics (from model_comparison).
    """
    from collections import defaultdict

    from fitness_coach.pipelines.model_comparison import ComparisonMetrics, ModelComparison

    mc = ModelComparison()
    mc.initialize_models()
    if not mc.detectors:
        return {}

    buckets: dict = defaultdict(lambda: defaultdict(list))
    for vp in video_paths:
        p = Path(vp)
        if not p.is_file():
            continue
        mc.benchmark_video(str(p), max_frames)
        for name, m in mc.metrics.items():
            buckets[name]["fps"].append(m.avg_fps)
            buckets[name]["inf_ms"].append(m.avg_inference_time_ms)
            buckets[name]["conf"].append(m.avg_confidence)
            buckets[name]["det"].append(m.detection_rate)
            buckets[name]["mem"].append(m.memory_peak_mb)
            buckets[name]["frames"].append(m.total_frames_processed)

    out = {}
    for name, b in buckets.items():
        if not b["fps"]:
            continue
        out[name] = ComparisonMetrics(
            model_name=name,
            avg_fps=float(np.mean(b["fps"])),
            avg_inference_time_ms=float(np.mean(b["inf_ms"])),
            total_frames_processed=int(np.sum(b["frames"])),
            avg_confidence=float(np.mean(b["conf"])),
            detection_rate=float(np.mean(b["det"])),
            memory_peak_mb=float(np.mean(b["mem"])),
        )
    return out


def pick_best_models(metrics_by_key: dict) -> dict:
    """
    From aggregated ComparisonMetrics, pick best per criterion and a balanced winner.
    All metrics assumed higher-is-better except inference time (lower is better).
    """
    if not metrics_by_key:
        return {}
    items = list(metrics_by_key.items())
    by_fps = max(items, key=lambda x: x[1].avg_fps)
    by_conf = max(items, key=lambda x: x[1].avg_confidence)
    by_det = max(items, key=lambda x: x[1].detection_rate)
    by_speed = min(items, key=lambda x: x[1].avg_inference_time_ms)

    keys = [k for k, _ in items]
    fps = np.array([metrics_by_key[k].avg_fps for k in keys])
    conf = np.array([metrics_by_key[k].avg_confidence for k in keys])
    det = np.array([metrics_by_key[k].detection_rate for k in keys])

    def _norm(a: np.ndarray) -> np.ndarray:
        lo, hi = float(a.min()), float(a.max())
        if hi - lo < 1e-12:
            return np.ones_like(a)
        return (a - lo) / (hi - lo)

    score = _norm(fps) + _norm(conf) + _norm(det)
    bi = int(np.argmax(score))
    balanced = keys[bi]

    return {
        "fastest_fps": by_fps[0],
        "lowest_latency_ms": by_speed[0],
        "best_confidence": by_conf[0],
        "best_detection_rate": by_det[0],
        "balanced_score": balanced,
    }


def generate_comparison_summary_png(
    metrics_by_key: dict,
    best: dict,
    video_count: int,
    out_path: Path,
    json_path: Optional[Path] = None,
) -> None:
    """Build comparison_summary.png from measured benchmark metrics."""
    if not metrics_by_key:
        return

    keys = sorted(metrics_by_key.keys(), key=lambda k: metrics_by_key[k].avg_fps, reverse=True)
    labels = [_model_display_name(k) for k in keys]
    fps = [metrics_by_key[k].avg_fps for k in keys]
    inf_ms = [metrics_by_key[k].avg_inference_time_ms for k in keys]
    conf = [metrics_by_key[k].avg_confidence * 100.0 for k in keys]
    det = [metrics_by_key[k].detection_rate for k in keys]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        "Pose model comparison (measured on sample videos — same clips as skeleton PNGs)\n"
        "FPS / latency / confidence / detection rate from ModelComparison.benchmark_video",
        fontsize=12,
        fontweight="bold",
    )

    colors = plt.cm.Set2(np.linspace(0, 1, max(len(keys), 3)))

    ax1 = axes[0, 0]
    bars = ax1.bar(labels, fps, color=colors[: len(keys)], edgecolor="black", linewidth=1)
    ax1.set_ylabel("FPS", fontweight="bold")
    ax1.set_title("Throughput (higher is better)", fontweight="bold")
    ax1.grid(axis="y", alpha=0.3)
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=25, ha="right")
    for bar, val in zip(bars, fps):
        ax1.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height(),
            f"{val:.1f}",
            ha="center",
            va="bottom",
            fontsize=8,
            fontweight="bold",
        )

    ax2 = axes[0, 1]
    bars2 = ax2.bar(labels, inf_ms, color=colors[: len(keys)], edgecolor="black", linewidth=1)
    ax2.set_ylabel("ms / frame (mean)", fontweight="bold")
    ax2.set_title("Latency (lower is better)", fontweight="bold")
    ax2.grid(axis="y", alpha=0.3)
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=25, ha="right")
    for bar, val in zip(bars2, inf_ms):
        ax2.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height(),
            f"{val:.1f}",
            ha="center",
            va="bottom",
            fontsize=8,
            fontweight="bold",
        )

    ax3 = axes[1, 0]
    x = np.arange(len(keys))
    w = 0.35
    ax3.bar(x - w / 2, conf, w, label="Mean conf. ×100", color="#4ECDC4", edgecolor="black")
    ax3.bar(x + w / 2, det, w, label="Det. rate %", color="#FF6B6B", edgecolor="black")
    ax3.set_xticks(x)
    ax3.set_xticklabels(labels, rotation=25, ha="right")
    ax3.set_ylabel("Value", fontweight="bold")
    ax3.set_title("Confidence & detection rate (higher is better)", fontweight="bold")
    ax3.legend(loc="upper right", fontsize=8)
    ax3.grid(axis="y", alpha=0.3)

    ax4 = axes[1, 1]
    ax4.axis("off")
    b = best.get("balanced_score", keys[0])
    bf = best.get("fastest_fps", b)
    bc = best.get("best_confidence", b)
    bd = best.get("best_detection_rate", b)
    bl = best.get("lowest_latency_ms", b)

    summary_text = f"""
MEASURED ON {video_count} SAMPLE VIDEO(S) (same as skeleton grids)

BEST MODEL BY METRIC
  • Highest FPS:        {_model_display_name(bf)}  ({metrics_by_key[bf].avg_fps:.1f} FPS)
  • Lowest latency:     {_model_display_name(bl)}  ({metrics_by_key[bl].avg_inference_time_ms:.1f} ms)
  • Highest confidence: {_model_display_name(bc)}  ({metrics_by_key[bc].avg_confidence:.4f})
  • Best det. rate:     {_model_display_name(bd)}  ({metrics_by_key[bd].detection_rate:.1f}%)

BALANCED PICK (normalize FPS + conf + det; tie → listed order)
  → {_model_display_name(b)}  — best overall tradeoff on this benchmark

NOTE: Skeleton PNGs are qualitative; this chart uses the same clips with
ModelComparison (timing + mean joint confidence + % frames with conf>0.3).
For keypoint error vs ViTPose, run: evaluate_pose_metrics.py
"""
    ax4.text(
        0.04,
        0.96,
        summary_text,
        transform=ax4.transAxes,
        fontfamily="monospace",
        fontsize=9,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="lightblue", alpha=0.75),
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()

    if json_path is not None:
        payload = {
            "videos_used": video_count,
            "models": {
                k: {
                    "avg_fps": float(metrics_by_key[k].avg_fps),
                    "avg_inference_time_ms": float(metrics_by_key[k].avg_inference_time_ms),
                    "avg_confidence": float(metrics_by_key[k].avg_confidence),
                    "detection_rate_pct": float(metrics_by_key[k].detection_rate),
                    "memory_peak_mb": float(metrics_by_key[k].memory_peak_mb),
                    "total_frames_aggregated": int(metrics_by_key[k].total_frames_processed),
                }
                for k in keys
            },
            "recommendations": {a: _model_display_name(best[a]) for a in best},
        }
        json_path.parent.mkdir(parents=True, exist_ok=True)
        with open(json_path, "w") as f:
            json.dump(payload, f, indent=2)


def draw_skeleton(frame: np.ndarray, keypoints: np.ndarray, confidence: np.ndarray,
                  model_name: str = "MediaPipe", color: Tuple[int, int, int] = (0, 255, 0)) -> np.ndarray:
    """
    Draw skeleton on frame given keypoints
    
    Args:
        frame: Input frame (BGR)
        keypoints: Array of shape (17, 2) with x, y coordinates
        confidence: Array of shape (17,) with confidence scores
        model_name: Name of model for label
        color: BGR color tuple
    
    Returns:
        Frame with drawn skeleton
    """
    frame_copy = frame.copy()
    h, w = frame.shape[:2]
    
    # Draw skeleton connections
    skeleton_pairs = [
        (0, 1), (0, 2), (1, 3), (2, 4),  # Head
        (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),  # Arms
        (5, 11), (6, 12), (11, 12),  # Torso
        (11, 13), (13, 15), (12, 14), (14, 16)  # Legs
    ]
    
    # Draw connections
    for start_idx, end_idx in skeleton_pairs:
        if keypoints[start_idx, 0] > 0 and keypoints[end_idx, 0] > 0:
            if confidence[start_idx] > 0.3 and confidence[end_idx] > 0.3:
                pt1 = tuple(map(int, keypoints[start_idx]))
                pt2 = tuple(map(int, keypoints[end_idx]))
                
                # Check bounds
                if (0 <= pt1[0] < w and 0 <= pt1[1] < h and 
                    0 <= pt2[0] < w and 0 <= pt2[1] < h):
                    cv2.line(frame_copy, pt1, pt2, color, 2)
    
    # Draw keypoints
    for i, (x, y) in enumerate(keypoints):
        if x > 0 and y > 0:
            conf = confidence[i]
            if conf > 0.3:
                pt = tuple(map(int, (x, y)))
                if 0 <= pt[0] < w and 0 <= pt[1] < h:
                    # Color intensity based on confidence
                    intensity = int(255 * min(conf, 1.0))
                    pt_color = (color[0], color[1], intensity)
                    cv2.circle(frame_copy, pt, 4, pt_color, -1)
                    cv2.circle(frame_copy, pt, 4, (255, 255, 255), 1)
    
    return frame_copy


def _normalized_to_pixels(keypoints: np.ndarray, w: int, h: int) -> np.ndarray:
    out = keypoints.astype(np.float32).copy()
    out[:, 0] *= float(w)
    out[:, 1] *= float(h)
    return out


def _evenly_spaced_frame_indices(total_frames: int, n: int = 3) -> List[int]:
    """Pick n frame indices spread across the clip (good for short & long videos)."""
    tf = max(int(total_frames), 1)
    if tf <= n:
        return [min(i, tf - 1) for i in range(n)]
    out = []
    for k in range(1, n + 1):
        idx = int(round(k * tf / (n + 1)))
        idx = max(0, min(idx, tf - 1))
        out.append(idx)
    return out


def visualize_model_comparison(
    video_path: str,
    output_dir: str = "./results/visualizations",
    frame_indices: Optional[List[int]] = None,
    num_sample_frames: int = 3,
    output_stem: Optional[str] = None,
    detectors: Optional[Sequence[Tuple[str, str, Any]]] = None,
) -> dict:
    """
    Visualize pose detection on sample frames for every available model
    (MediaPipe, YOLOv8, RTMPose-X, ViTPose/RTMPose, DETRPose/RTMO, OpenPose).

    Args:
        video_path: Path to a video file.
        output_dir: Where to save the PNG grid.
        frame_indices: Explicit 0-based frame indices to plot. If None, uses
            evenly spaced frames across the video (recommended).
        num_sample_frames: How many columns when frame_indices is None (default 3).
        output_stem: If set, used for the output PNG filename instead of the video stem
            (avoids collisions when the same id exists under long_range and short_clips).
        detectors: Pre-built detector list from ``iter_default_comparison_detectors()``;
            if None, loads a fresh set (slow when called many times).
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    detectors = list(detectors) if detectors is not None else list(iter_default_comparison_detectors())
    if not detectors:
        print("✗ No pose detectors available — install mediapipe, ultralytics, rtmlib, onnxruntime")
        return None

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"✗ Failed to open video: {video_path}")
        return None

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_indices is None:
        frame_indices = _evenly_spaced_frame_indices(total, num_sample_frames)
    frames_data = []
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if ret:
            frames_data.append((idx, frame.copy()))
    cap.release()

    if not frames_data:
        print("✗ Failed to extract sample frames")
        return None

    n_models = len(detectors)
    n_frames = len(frames_data)
    palette = [
        (0, 255, 0),
        (255, 128, 0),
        (0, 165, 255),
        (255, 0, 255),
        (0, 255, 255),
        (180, 105, 255),
    ]

    fig = plt.figure(figsize=(4.2 * n_frames, 3.2 * n_models))
    idx_str = ", ".join(str(i) for i, _ in frames_data)
    fig.suptitle(
        f"Skeleton comparison (COCO-17) — {Path(video_path).name}\nframes [{idx_str}]  •  rows = models, cols = time",
        fontsize=12,
        fontweight="bold",
    )

    for mi, (_key, model_label, detector) in enumerate(detectors):
        color = palette[mi % len(palette)]
        for fi, (frame_idx, frame) in enumerate(frames_data):
            h, w = frame.shape[:2]
            try:
                kp_norm, conf = detector.detect(frame)
            except Exception as ex:
                print(f"  ⚠ {model_label} frame {frame_idx}: {ex}")
                kp_norm, conf = None, None
            if kp_norm is None:
                kp_px = np.zeros((17, 2))
                conf = np.zeros(17)
            else:
                kp_px = _normalized_to_pixels(kp_norm, w, h)
            frame_draw = draw_skeleton(frame, kp_px, conf, model_label, color)
            frame_rgb = cv2.cvtColor(frame_draw, cv2.COLOR_BGR2RGB)
            ax = fig.add_subplot(n_models, n_frames, mi * n_frames + fi + 1)
            ax.imshow(frame_rgb)
            det_n = int((conf > 0.3).sum()) if conf is not None else 0
            ax.set_title(f"{model_label}\nframe {frame_idx}  ({det_n}/17)", fontsize=9)
            ax.axis("off")

    stem = output_stem if output_stem else Path(video_path).stem
    output_file = Path(output_dir) / f"skeleton_detection_{stem}.png"
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches="tight")
    print(f"✓ Visualization saved: {output_file}")
    plt.close()

    return {
        "output_file": str(output_file),
        "frames_processed": len(frames_data),
        "models": [d[1] for d in detectors],
        "video_path": video_path,
    }


def discover_mp4_paths(dataset_root: str) -> tuple[List[Path], str]:
    """
    Find .mp4 clips for comparison.

    1. **QEVD layout:** ``dataset_root/videos/**/*.mp4`` (recursive).
    2. If that yields nothing, **nested layout** (e.g. Riccio Kaggle): all ``*.mp4`` under
       ``dataset_root`` recursively, so you can pass ``--dataset-root`` to a cache folder
       without creating a ``videos/`` subfolder.
    """
    base = Path(dataset_root).expanduser().resolve()
    if not base.is_dir():
        return [], f"{base} (not a directory)"
    qevd_videos = base / "videos"
    if qevd_videos.is_dir():
        found = sorted({p.resolve() for p in qevd_videos.rglob("*.mp4") if p.is_file()})
        if found:
            return found, f"{base}/videos (QEVD layout)"
    found = sorted({p.resolve() for p in base.rglob("*.mp4") if p.is_file()})
    if found:
        return found, f"{base} (recursive .mp4 — e.g. Riccio folder layout)"
    return [], str(base)


def select_sample_video_paths(
    paths: List[Path],
    num_videos: int,
    *,
    seed: int,
    randomize: bool,
) -> List[Path]:
    if not paths:
        return []
    n = min(int(num_videos), len(paths))
    if randomize:
        rng = random.Random(seed)
        return rng.sample(paths, n)
    return paths[:n]


def run_full_pipeline(
    dataset_root: str = "./data",
    num_videos: int = 3,
    *,
    seed: int = 42,
    randomize: bool = True,
    benchmark_max_frames: int = 80,
    benchmark_num_videos: int = 3,
    skip_benchmark: bool = False,
    output_dir: str = "./results/visualizations",
) -> dict:
    """
    For each sample video: skeleton grid (all models × sample frames), then aggregate
    speed/accuracy metrics → ``comparison_summary.png`` + JSON.

    Use ``randomize=True`` with ``seed`` for reproducible random clips; ``randomize=False``
    takes the first ``num_videos`` paths in sorted order.

    **Benchmark speed:** ``benchmark_num_videos`` caps how many of the sampled clips are used
    for the FPS/latency pass (default 3). Use ``0`` to benchmark every sampled clip (slow when
    ``num_videos`` is large). ``skip_benchmark`` skips the FPS pass entirely (PNGs only).
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "="*80)
    print("QEVD-FIT-COACH POSE ESTIMATION PIPELINE - FULL EXECUTION")
    print("="*80)

    print("\n[1/4] Loading Dataset...")
    all_paths, scan_desc = discover_mp4_paths(dataset_root)
    print(f"✓ Found {len(all_paths)} video file(s) — {scan_desc}")

    sample_paths = select_sample_video_paths(
        all_paths, num_videos, seed=seed, randomize=randomize
    )
    if not sample_paths:
        print(
            "✗ No videos to process — put .mp4 files under dataset_root/videos/ (QEVD), "
            "or point --dataset-root at a folder that contains .mp4 files (e.g. Riccio Kaggle cache)."
        )
        return {
            "visualizations": [],
            "summary_file": None,
            "metrics_json": None,
            "samples_processed": 0,
        }

    mode = f"random sample (seed={seed})" if randomize else "first N (sorted paths)"
    print(f"  Selection: {mode} — {len(sample_paths)} clip(s)")
    for p in sample_paths:
        print(f"    • {p.parent.name}/{p.name}")

    manifest = {
        "dataset_root": str(Path(dataset_root).resolve()),
        "seed": seed,
        "randomize": randomize,
        "num_videos": len(sample_paths),
        "videos": [str(p) for p in sample_paths],
        "benchmark_num_videos": benchmark_num_videos,
        "skip_benchmark": skip_benchmark,
    }
    manifest_path = out_dir / "comparison_run_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"  ✓ Manifest: {manifest_path}")

    print("\n[2/4] Loading pose models once (MediaPipe, YOLO, RTMPose-X, ViTPose, DETRPose, …) …")
    detectors_list = list(iter_default_comparison_detectors())
    if not detectors_list:
        print("✗ No pose detectors available — install mediapipe, ultralytics, rtmlib, onnxruntime")
        return {
            "visualizations": [],
            "summary_file": None,
            "metrics_json": None,
            "samples_processed": 0,
        }

    print("\n[3/4] Extracting Pose Keypoints and Generating Visualizations...")
    all_results = []
    video_paths_for_benchmark: List[str] = []

    for i, video_path in enumerate(sample_paths):
        stem_tag = f"{video_path.parent.name}_{video_path.stem}"
        print(f"\n  [{i+1}/{len(sample_paths)}] {stem_tag}")
        print(f"     Path: {video_path}")

        try:
            result = visualize_model_comparison(
                str(video_path),
                output_dir=str(out_dir),
                output_stem=stem_tag,
                detectors=detectors_list,
            )
            if result:
                all_results.append(result)
                video_paths_for_benchmark.append(str(video_path))
                print("     ✓ Visualization generated")
            else:
                print("     ✗ Failed to generate visualization")
        except Exception as e:
            print(f"     ✗ Error: {e}")

    print("\n[4/4] Benchmarking models on sample videos → comparison_summary.png ...")
    summary_file = out_dir / "comparison_summary.png"
    metrics_json = out_dir / "comparison_metrics.json"
    metrics_by_key: dict = {}

    paths_for_benchmark = list(video_paths_for_benchmark)
    if not skip_benchmark and paths_for_benchmark and benchmark_num_videos > 0:
        cap = min(benchmark_num_videos, len(paths_for_benchmark))
        paths_for_benchmark = paths_for_benchmark[:cap]
        if cap < len(video_paths_for_benchmark):
            print(
                f"  (FPS benchmark: first {cap} of {len(video_paths_for_benchmark)} clips — "
                f"use --benchmark-videos 0 to benchmark all, or --skip-benchmark for PNGs only)",
                flush=True,
            )
    elif not skip_benchmark and paths_for_benchmark and benchmark_num_videos == 0:
        print(
            f"  (FPS benchmark: all {len(paths_for_benchmark)} sampled clips)",
            flush=True,
        )

    if skip_benchmark:
        print("  --skip-benchmark: skipping FPS/latency pass (no comparison_summary.png / metrics JSON).")
        metrics_by_key = {}
    elif video_paths_for_benchmark:
        metrics_by_key = aggregate_benchmark_over_videos(
            paths_for_benchmark,
            max_frames=benchmark_max_frames,
        )
        if metrics_by_key:
            best = pick_best_models(metrics_by_key)
            generate_comparison_summary_png(
                metrics_by_key,
                best,
                video_count=len(paths_for_benchmark),
                out_path=summary_file,
                json_path=metrics_json,
            )
            print(f"✓ Summary saved: {summary_file}")
            print(f"✓ Metrics JSON: {metrics_json}")
            bname = _model_display_name(best["balanced_score"])
            print(f"  Balanced best (FPS + conf + det. rate): {bname}")
        else:
            print("  ✗ No benchmark metrics collected — detectors may have failed.")
    else:
        print("  ✗ No videos benchmarked — skipping comparison_summary.png")

    print("\n" + "="*80)
    print("PIPELINE EXECUTION COMPLETE")
    print("="*80)
    print(f"""
✓ Visualizations Generated: {len(all_results)}/{len(sample_paths)} samples
✓ Output Directory: {out_dir.resolve()}/
✓ Summary Report: {summary_file}

NEXT STEPS:
  1. Read comparison_metrics.json for numeric table (same run as skeleton PNGs)
  2. For PCK/MPJPE/OKS vs ViTPose: ./venv/bin/python evaluate_pose_metrics.py --videos ...
  3. Review skeleton_detection_*.png for qualitative comparison

METRICS SOURCE: model_comparison.ModelComparison.benchmark_video
  (FPS, mean inference ms, mean joint confidence, % frames with mean conf > 0.3)
""")

    return {
        "visualizations": all_results,
        "summary_file": str(summary_file),
        "metrics_json": str(metrics_json) if metrics_by_key else None,
        "samples_processed": len(all_results),
        "manifest": str(manifest_path),
    }


def _parse_args() -> argparse.Namespace:
    import os

    _env_root = (os.environ.get("EXERCISE_RECOGNITION_ROOT") or "").strip()
    _default_root = _env_root if _env_root else "./data"

    p = argparse.ArgumentParser(
        description="Skeleton comparison + speed/accuracy benchmark across pose models (MediaPipe, YOLO, RTMPose-X, ViTPose, DETRPose, OpenPose)."
    )
    p.add_argument(
        "--dataset-root",
        default=_default_root,
        help="Folder with .mp4 (recursive ok; Riccio Kaggle layout). Default: EXERCISE_RECOGNITION_ROOT or ./data",
    )
    p.add_argument("--num-videos", type=int, default=3, help="How many clips to visualize and benchmark")
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="RNG seed when using random clip selection (default 42)",
    )
    p.add_argument(
        "--no-random",
        action="store_true",
        help="Use first N videos in sorted path order instead of random sampling",
    )
    p.add_argument(
        "--output-dir",
        default="./results/visualizations",
        help="Directory for skeleton_detection_*.png, comparison_summary.png, comparison_metrics.json",
    )
    p.add_argument(
        "--benchmark-max-frames",
        type=int,
        default=80,
        help="Max frames per video for FPS/confidence benchmark",
    )
    p.add_argument(
        "--benchmark-videos",
        type=int,
        default=3,
        help="How many of the sampled clips to use for the FPS benchmark (default 3). "
        "Use 0 to benchmark every sampled clip (slow with large --num-videos).",
    )
    p.add_argument(
        "--skip-benchmark",
        action="store_true",
        help="Only write skeleton PNGs + manifest; skip comparison_summary.png and metrics JSON.",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    result = run_full_pipeline(
        dataset_root=args.dataset_root,
        num_videos=args.num_videos,
        seed=args.seed,
        randomize=not args.no_random,
        benchmark_max_frames=args.benchmark_max_frames,
        benchmark_num_videos=args.benchmark_videos,
        skip_benchmark=args.skip_benchmark,
        output_dir=args.output_dir,
    )
    if result.get("samples_processed", 0) == 0:
        print("\n✗ No samples processed — fix --dataset-root or add .mp4 files.", file=sys.stderr)
        raise SystemExit(1)
    print("\n✓ Full pipeline execution completed successfully!")
