#!/usr/bin/env python3
"""Demo: load the trained branched multi-task xLSTM and run inference on a clip.

Produces the four user-facing outputs the paper describes:
    exercise   — predicted class name
    quality    — predicted scalar score on the checkpoint quality axis (`unit` ``[0,1]``
                 or Likert-style ``[1,5]`` when trained with ``--quality-encoding likert``)
    guidance   — canonical how-to instructions (class-deterministic lookup)
    comment    — corrective critique (class × quality-bucket lookup)

Usage:
    # Demo on a random test clip
    ./venv/bin/python scripts/demo_multitask_inference.py \
        --ckpt results/xlstm_egoexo_multitask_allviews/xlstm_egoexo_multitask_best.pt

    # Demo on a specific (record_id, view) pair
    ./venv/bin/python scripts/demo_multitask_inference.py \
        --ckpt results/xlstm_egoexo_multitask_allviews/xlstm_egoexo_multitask_best.pt \
        --record-id 08ALrC --view ego_l --frame-start 0 --frame-end 240
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch

# ─── repo root on sys.path ───────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from fitness_coach.datasets.egoexo_xlstm_dataset import load_clip_segment, CLIP_SUBDIR
from fitness_coach.models.xlstm_model import xLSTMExerciseClassifier


# ─── CLI ─────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--ckpt", type=Path, required=True,
                   help="Path to a .pt checkpoint from train_xlstm_egoexo_multitask.py")
    p.add_argument("--clip-features-root", type=Path,
                   default=Path("notebooks/data/egoexo_fitness_full/features_open/visual"),
                   help="Same path used at training time.")
    p.add_argument("--index-csv", type=Path,
                   default=Path("results/egoexo_fitness_index_split.csv"),
                   help="Index CSV; used to pick a sample clip when --record-id is not given.")
    p.add_argument("--record-id", type=str, default=None,
                   help="Specific EgoExo record id (e.g. 08ALrC). If omitted, picks a random test clip.")
    p.add_argument("--view", type=str, default="ego_l",
                   help="Camera view (ego_l, ego_m, ego_r, exo_l, exo_m, exo_r).")
    p.add_argument("--frame-start", type=int, default=None)
    p.add_argument("--frame-end", type=int, default=None)
    p.add_argument("--clip-max-frames", type=int, default=300)
    p.add_argument("--clip-subsample-stride", type=int, default=3)
    p.add_argument("--n-samples", type=int, default=5,
                   help="If --record-id is omitted, run on this many random test clips.")
    p.add_argument("--device", default="cpu", choices=("cpu", "cuda", "mps"))
    return p.parse_args()


# ─── checkpoint → model + lookups ────────────────────────────────────────────

def load_model_from_checkpoint(
    ckpt_path: Path, device: torch.device,
) -> tuple[xLSTMExerciseClassifier, Dict[str, Any]]:
    """Reconstruct the trained model exactly as it was at save time, including
    the guidance and comment lookup tables. Returns `(model, checkpoint_meta)`."""
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)

    # Reconstruct the architecture from the saved hyperparameters.
    model = xLSTMExerciseClassifier(
        input_size=ck.get("input_size", 512),
        hidden_size=int(ck["hidden"]),
        num_layers=int(ck["layers"]),
        num_classes=len(ck["classes"]),
        dropout=float(ck.get("dropout", 0.15)),
        num_heads=int(ck["num_heads"]),
        conv_kernel_size=int(ck["conv_kernel_size"]),
        projection_factor=float(ck["projection_factor"]),
        num_error_tags=len(ck.get("error_tags", [])) if ck.get("error_tags") else 0,
        num_quality_classes=int(ck.get("num_quality_classes", 1)),
        quality_scale=float(ck.get("quality_scale", 1.0)),
        quality_output_low=float(ck.get("quality_output_low", 0.0)),
        block_pattern=str(ck.get("block_pattern", "mmmmmmms")),
        use_attention_pool=bool(ck.get("use_attention_pool", True)),
        use_fusion=bool(ck.get("use_fusion", True)),
        fusion_dim=int(ck.get("fusion_dim", 128)),
        quality_class_conditioning=bool(ck.get("teacher_force_quality", False)),
    ).to(device)
    model.load_state_dict(ck["model"], strict=False)
    model.eval()

    # Reinstall the lookup tables from the checkpoint sidecar.
    idx_to_class = {int(k): str(v) for k, v in ck["idx_to_class"].items()}
    guidance_table = {int(k): str(v) for k, v in ck["guidance_table"].items()}
    model.set_guidance_table(guidance_table, idx_to_class)

    comment_table_raw = ck.get("comment_table") or {}
    comment_table: Dict = {}
    for key, text in comment_table_raw.items():
        cls_idx, bucket = key.split("|")
        comment_table[(int(cls_idx), int(bucket))] = str(text)
    bucket_edges = tuple(float(e) for e in (ck.get("comment_quality_bucket_edges") or (0.4, 0.7)))
    q_lo = float(ck.get("quality_domain_lo", 0.0))
    q_hi = float(ck.get("quality_domain_hi", 1.0))
    model.set_comment_table(comment_table, bucket_edges, domain_lo=q_lo, domain_hi=q_hi)

    ck_meta = {
        "bucket_edges": bucket_edges,
        "quality_domain_lo": q_lo,
        "quality_domain_hi": q_hi,
        "quality_encoding": str(ck.get("quality_encoding", "unit")),
    }
    print(f"[demo] model restored from {ckpt_path.name}")
    print(f"[demo]   {len(ck['classes'])} classes · {len(guidance_table)} guidance entries · "
          f"{len(comment_table)} comment cells · quality axis [{q_lo:.2f}, {q_hi:.2f}] · "
          f"q-bucket edges = {bucket_edges}")
    return model, ck_meta


# ─── feature loading ─────────────────────────────────────────────────────────

def load_clip_features(
    clip_features_root: Path, record_id: str, view: str,
    frame_start: int, frame_end: int,
    max_frames: int, subsample_stride: int,
) -> Optional[torch.Tensor]:
    """Load a (T, 512) CLIP feature segment for a single clip and return as a tensor."""
    arr = load_clip_segment(
        clip_features_root, record_id, frame_start, frame_end,
        view=view, max_frames=max_frames, subsample_stride=subsample_stride,
        allow_fallback=False,
    )
    if arr is None:
        return None
    return torch.from_numpy(arr).unsqueeze(0)  # (1, T, 512)


# ─── pretty-print one inference result ───────────────────────────────────────

def print_result(
    result: Dict[str, Any],
    ground_truth: Optional[Dict[str, Any]] = None,
    *,
    q_axis_lo: float = 0.0,
    q_axis_hi: float = 1.0,
) -> None:
    print("┌" + "─" * 72)
    print(f"│  EXERCISE:  {result['exercise']}")
    ql = float(result["quality"])
    if q_axis_hi > 2.5:
        qb = result.get("quality_bucket")
        extra = f" · discrete bucket idx {int(qb)}" if isinstance(qb, int) else ""
        print(
            f"│  QUALITY:   {ql:.2f}  (checkpoint axis [{q_axis_lo:.1f}, {q_axis_hi:.1f}]; higher=better)"
            + extra,
        )
    else:
        print(f"│  QUALITY:   {ql:.3f}  ([0, 1] unit axis; readability 1–5 ~= {1.0 + 4.0 * ql:.2f})")
        if isinstance(result.get("quality_bucket"), int):
            print(f"│             discrete bucket idx {int(result['quality_bucket'])}")
    print("│")
    print(f"│  GUIDANCE:")
    for line in (result["guidance"] or "(no guidance for this class)").splitlines() or [""]:
        for chunk in [line[i:i+70] for i in range(0, max(1, len(line)), 70)]:
            print(f"│    {chunk}")
    print("│")
    print(f"│  COMMENT:")
    for line in (result["comment"] or "(no comment for this class × quality cell)").splitlines() or [""]:
        for chunk in [line[i:i+70] for i in range(0, max(1, len(line)), 70)]:
            print(f"│    {chunk}")
    if ground_truth:
        print("│")
        print("│  ─── ground truth ────────────────────────────────────")
        gt_cls = ground_truth.get("exercise", "")
        gt_q = ground_truth.get("quality", "")
        print(f"│    true exercise:  {gt_cls}  {'✓' if gt_cls.lower() == result['exercise'].lower() else '✗ (mismatch)'}")
        if gt_q != "":
            try:
                gt_q_f = float(gt_q)
                diff = abs(gt_q_f - ql)
                print(f"│    true quality :  {gt_q_f:.3f}  (|error| = {diff:.3f})")
            except Exception:
                pass
        gt_cmt = (ground_truth.get("comment") or "").strip()
        if gt_cmt:
            print(f"│    true comment :  {gt_cmt[:200]}{'…' if len(gt_cmt) > 200 else ''}")
    print("└" + "─" * 72)


# ─── main ────────────────────────────────────────────────────────────────────

def main() -> int:
    args = parse_args()
    device = torch.device(args.device)
    model, q_meta = load_model_from_checkpoint(args.ckpt, device)

    # Build the list of (record_id, view, fs, fe, gt_cls, gt_q, gt_comment) clips to demo.
    clips_to_run = []
    if args.record_id:
        fs = args.frame_start if args.frame_start is not None else 0
        fe = args.frame_end if args.frame_end is not None else 240
        clips_to_run.append({
            "record_id": args.record_id,
            "view": args.view,
            "frame_start": fs,
            "frame_end": fe,
            "exercise": "",
            "quality": "",
            "comment": "",
        })
    else:
        # Sample N random test clips from the index CSV
        import csv, random
        rows = list(csv.DictReader(open(args.index_csv)))
        test_rows = [r for r in rows if r.get("split") == "test"]
        random.seed(42)
        for r in random.sample(test_rows, min(args.n_samples, len(test_rows))):
            jk = (r.get("judgement_key") or "").strip()
            if not jk or "_action_" not in jk:
                continue
            try:
                fs = int(r.get("frame_start", 0))
                fe = int(r.get("frame_end", 0))
            except (ValueError, TypeError):
                continue
            if fe <= fs:
                continue
            clips_to_run.append({
                "record_id": jk.split("_action_")[0],
                "view": args.view,
                "frame_start": fs,
                "frame_end": fe,
                "exercise": (r.get("exercise_class") or "").strip(),
                "quality": (r.get("quality") or "").strip(),
                "comment": (r.get("comment") or "").strip(),
            })

    print(f"\n[demo] Running inference on {len(clips_to_run)} clip(s) using view '{args.view}'\n")

    n_correct = 0
    for i, clip in enumerate(clips_to_run, 1):
        x = load_clip_features(
            args.clip_features_root, clip["record_id"], clip["view"],
            clip["frame_start"], clip["frame_end"],
            args.clip_max_frames, args.clip_subsample_stride,
        )
        if x is None:
            print(f"[skip] no features for record_id={clip['record_id']} view={clip['view']}")
            continue
        x = x.to(device)
        with torch.no_grad():
            out = model.infer(x)[0]
        print(f"\nClip {i}/{len(clips_to_run)} — record {clip['record_id']} · view {clip['view']} · "
              f"frames {clip['frame_start']}-{clip['frame_end']} ({x.shape[1]} frames after subsample)")
        gt = {k: clip[k] for k in ("exercise", "quality", "comment")} if clip["exercise"] else None
        print_result(
            out,
            ground_truth=gt,
            q_axis_lo=float(q_meta["quality_domain_lo"]),
            q_axis_hi=float(q_meta["quality_domain_hi"]),
        )
        if gt and out["exercise"].lower() == gt["exercise"].lower():
            n_correct += 1

    if clips_to_run and clips_to_run[0]["exercise"]:
        print(f"\n[demo] classification matches: {n_correct}/{len(clips_to_run)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
