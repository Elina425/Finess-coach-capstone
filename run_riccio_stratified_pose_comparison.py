#!/usr/bin/env python3
"""
Stratified Riccio pose-tool comparison (same metrics style as ``results/02_step_pose_comparison``).

For each of **four** exercise classes, randomly sample ``N`` ``.mp4`` clips from under
``--dataset-root`` (Kaggle Riccio tree: any folder segment may match the class name).
Runs ``ModelComparison``-style benchmarks (FPS, latency, confidence, detection rate) per model,
aggregates **per exercise** and **globally** (all 4×N clips), then writes:

  - ``per_exercise/<slug>/comparison_summary.png`` and ``comparison_metrics.json`` — one figure
    + metrics file per exercise (same layout as the global summary; title includes ``Riccio · <exercise>``)
  - ``comparison_run_manifest.json`` — selected paths, slugs, and ``per_exercise_artifacts`` paths
  - ``per_exercise_metrics.json`` — metrics + ``recommendations`` per class (+ relative paths to plots)
  - ``comparison_metrics.json`` + ``comparison_summary.png`` — global (like 02_step)
  - ``final_decision.json`` — per-class winners + vote tally + global balanced pick

Example::

  python run_riccio_stratified_pose_comparison.py \\
    --dataset-root \"$HOME/.cache/kagglehub/datasets/riccardoriccio/real-time-exercise-recognition-dataset/versions/3\" \\
    --output-dir results/riccio_pose_tool_comparison_stratified \\
    --per-exercise 30 --seed 42 --max-frames 60
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

from fitness_coach.pipelines.run_full_comparison import (
    _model_display_name,
    aggregate_benchmark_over_videos,
    discover_mp4_paths,
    generate_comparison_summary_png,
    pick_best_models,
)


DEFAULT_EXERCISES: Tuple[str, ...] = (
    "squat",
    "push-up",
    "shoulder press",
    "barbell biceps curl",
)


def _exercise_slug(exercise: str) -> str:
    """Filesystem-safe folder name for an exercise label."""
    s = exercise.strip().lower().replace(" ", "_")
    return "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in s)


def _path_matches_exercise(path: Path, exercise: str) -> bool:
    """True if any path component equals the exercise name (case-insensitive)."""
    e = exercise.casefold()
    for part in path.parts:
        if part.casefold() == e:
            return True
    return False


def collect_videos_for_exercise(
    all_mp4: Sequence[Path],
    exercise: str,
) -> List[Path]:
    return [p for p in all_mp4 if _path_matches_exercise(p, exercise)]


def sample_videos(
    candidates: List[Path],
    n: int,
    rng: random.Random,
) -> List[Path]:
    if not candidates:
        return []
    k = min(n, len(candidates))
    return rng.sample(sorted(set(candidates)), k)


def _metrics_to_serializable(metrics_by_key: Dict[str, Any]) -> Dict[str, Any]:
    from fitness_coach.pipelines.model_comparison import ComparisonMetrics

    out: Dict[str, Any] = {}
    for k, m in metrics_by_key.items():
        if not isinstance(m, ComparisonMetrics):
            continue
        out[k] = {
            "avg_fps": float(m.avg_fps),
            "avg_inference_time_ms": float(m.avg_inference_time_ms),
            "avg_confidence": float(m.avg_confidence),
            "detection_rate_pct": float(m.detection_rate),
            "memory_peak_mb": float(m.memory_peak_mb),
            "total_frames_aggregated": int(m.total_frames_processed),
        }
    return out


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Stratified pose-tool benchmark on Riccio (4 exercises × N random videos)"
    )
    ap.add_argument(
        "--dataset-root",
        type=Path,
        required=True,
        help="Riccio dataset root (recursive .mp4 scan, e.g. KaggleHub cache path)",
    )
    ap.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/riccio_pose_tool_comparison_stratified"),
    )
    ap.add_argument(
        "--exercises",
        type=str,
        default=",".join(DEFAULT_EXERCISES),
        help="Comma-separated coarse class names (must match folder names in the tree)",
    )
    ap.add_argument("--per-exercise", type=int, default=30, metavar="N", help="Random clips per class")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-frames", type=int, default=60, help="Max frames per video per model")
    args = ap.parse_args()

    exercises = tuple(s.strip() for s in args.exercises.split(",") if s.strip())
    if len(exercises) != 4:
        print(
            "Warning: expected 4 exercises for the stratified design; got "
            f"{len(exercises)}. Proceeding anyway.",
            file=sys.stderr,
        )

    root = args.dataset_root.expanduser().resolve()
    if not root.is_dir():
        print(f"ERROR: --dataset-root is not a directory: {root}", file=sys.stderr)
        return 1

    all_paths, scan_desc = discover_mp4_paths(str(root))
    if not all_paths:
        print(f"ERROR: No .mp4 under {root} ({scan_desc})", file=sys.stderr)
        return 1

    rng = random.Random(int(args.seed))
    per_exercise_samples: Dict[str, List[Path]] = {}
    shortfall: Dict[str, int] = {}

    for ex in exercises:
        cand = collect_videos_for_exercise(all_paths, ex)
        picked = sample_videos(cand, int(args.per_exercise), rng)
        per_exercise_samples[ex] = picked
        need = int(args.per_exercise)
        if len(picked) < need:
            shortfall[ex] = need - len(picked)

    if shortfall:
        print("WARNING: Some classes have fewer than requested clips:", shortfall, file=sys.stderr)

    all_sampled: List[Path] = []
    for ex in exercises:
        all_sampled.extend(per_exercise_samples[ex])
    all_sampled = list(dict.fromkeys(all_sampled))

    out_dir = args.output_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest_base: Dict[str, Any] = {
        "dataset_root": str(root),
        "scan_description": scan_desc,
        "seed": int(args.seed),
        "per_exercise_requested": int(args.per_exercise),
        "exercises": list(exercises),
        "per_exercise_slugs": {ex: _exercise_slug(ex) for ex in exercises},
        "per_exercise_videos": {ex: [str(p) for p in per_exercise_samples[ex]] for ex in exercises},
        "all_unique_videos": [str(p) for p in all_sampled],
        "total_unique_clips": len(all_sampled),
        "max_frames_per_video": int(args.max_frames),
    }

    per_exercise_payload: Dict[str, Any] = {}
    votes_balanced: Dict[str, int] = {}
    per_exercise_artifacts: Dict[str, Dict[str, str]] = {}

    for ex in exercises:
        paths = [str(p) for p in per_exercise_samples[ex]]
        print(f"\n=== Benchmark: {ex} ({len(paths)} clips) ===")
        if not paths:
            per_exercise_payload[ex] = {"error": "no_videos", "models": {}, "recommendations": {}}
            continue
        m = aggregate_benchmark_over_videos(paths, max_frames=int(args.max_frames))
        if not m:
            per_exercise_payload[ex] = {"error": "benchmark_empty", "models": {}, "recommendations": {}}
            continue
        best = pick_best_models(m)
        slug = _exercise_slug(ex)
        ex_sub = out_dir / "per_exercise" / slug
        ex_sub.mkdir(parents=True, exist_ok=True)
        ex_summary = ex_sub / "comparison_summary.png"
        ex_metrics_p = ex_sub / "comparison_metrics.json"
        generate_comparison_summary_png(
            m,
            best,
            video_count=len(paths),
            out_path=ex_summary,
            json_path=ex_metrics_p,
            subtitle=f"Riccio · {ex}",
        )
        rel = lambda p: str(p.relative_to(out_dir))
        per_exercise_payload[ex] = {
            "videos_used": len(paths),
            "models": _metrics_to_serializable(m),
            "recommendations": {k: _model_display_name(v) for k, v in best.items()},
            "recommendations_keys": best,
            "comparison_summary_png": rel(ex_summary),
            "comparison_metrics_json": rel(ex_metrics_p),
        }
        per_exercise_artifacts[ex] = {
            "slug": slug,
            "comparison_summary_png": rel(ex_summary),
            "comparison_metrics_json": rel(ex_metrics_p),
        }
        print(f"  Wrote {rel(ex_summary)} and {rel(ex_metrics_p)}")
        w = best.get("balanced_score", "")
        if w:
            votes_balanced[w] = votes_balanced.get(w, 0) + 1

    (out_dir / "per_exercise_metrics.json").write_text(json.dumps(per_exercise_payload, indent=2))
    print("\nWrote per_exercise_metrics.json")

    print(f"\n=== Global benchmark ({len(all_sampled)} clips) ===")
    if not all_sampled:
        print("ERROR: No sampled videos to aggregate globally.", file=sys.stderr)
        return 1

    global_metrics = aggregate_benchmark_over_videos(
        [str(p) for p in all_sampled],
        max_frames=int(args.max_frames),
    )
    if not global_metrics:
        print("ERROR: Global benchmark returned no metrics.", file=sys.stderr)
        return 1

    global_best = pick_best_models(global_metrics)
    summary_png = out_dir / "comparison_summary.png"
    metrics_json = out_dir / "comparison_metrics.json"
    generate_comparison_summary_png(
        global_metrics,
        global_best,
        video_count=len(all_sampled),
        out_path=summary_png,
        json_path=metrics_json,
    )
    print(f"Wrote {metrics_json.name} and {summary_png.name}")

    overall_balanced_key = global_best.get("balanced_score", "")
    vote_winner = max(votes_balanced.items(), key=lambda kv: (kv[1], kv[0]))[0] if votes_balanced else ""

    final_decision = {
        "per_exercise_recommendations": {
            ex: per_exercise_payload.get(ex, {}).get("recommendations", {})
            for ex in exercises
        },
        "per_exercise_balanced_model_key": {
            ex: (per_exercise_payload.get(ex, {}).get("recommendations_keys") or {}).get(
                "balanced_score", ""
            )
            for ex in exercises
        },
        "votes_balanced_across_exercises": votes_balanced,
        "vote_plurality_model_key": vote_winner,
        "vote_plurality_display": _model_display_name(vote_winner) if vote_winner else "",
        "global_recommendations": {k: _model_display_name(v) for k, v in global_best.items()},
        "global_balanced_model_key": overall_balanced_key,
        "global_balanced_display": _model_display_name(overall_balanced_key)
        if overall_balanced_key
        else "",
        "final_pick_rationale": (
            "Use global_balanced_* as the primary 'best tool' on this 4×N sample; "
            "per_exercise_* shows where a different detector wins; vote_plurality_* counts "
            "how many exercise strata chose each model by balanced score."
        ),
    }
    (out_dir / "final_decision.json").write_text(json.dumps(final_decision, indent=2))

    manifest_out = {
        **manifest_base,
        "per_exercise_artifacts": per_exercise_artifacts,
        "global_comparison_summary_png": "comparison_summary.png",
        "global_comparison_metrics_json": "comparison_metrics.json",
    }
    (out_dir / "comparison_run_manifest.json").write_text(json.dumps(manifest_out, indent=2))
    print(f"\nWrote comparison_run_manifest.json ({len(all_sampled)} unique clips).")

    print("\n" + "=" * 72)
    print("SUMMARY")
    print("=" * 72)
    for ex in exercises:
        rec = per_exercise_payload.get(ex, {}).get("recommendations", {})
        bal = rec.get("balanced_score", "?")
        print(f"  {ex:22s} balanced → {bal}")
    print(f"\n  Vote (balanced, per stratum): {votes_balanced}")
    print(f"  Global balanced (all clips):   {_model_display_name(overall_balanced_key)}")
    print(f"\n  → Final decision JSON: {out_dir / 'final_decision.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
