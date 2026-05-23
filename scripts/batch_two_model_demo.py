#!/usr/bin/env python3
"""Batch inference demo across N Riccio videos and M EgoExo clips.

Loads both trained models once and runs them on a sampled batch from each
dataset, printing a compact per-clip table and aggregate accuracy / MAE.

Usage:
    ./venv/bin/python scripts/batch_two_model_demo.py \\
        --riccio-ckpt   results/paper_xlstm_seq60_resnet/xlstm_7_1/best.pt \\
        --multitask-ckpt results/xlstm_egoexo_multitask_allviews/xlstm_egoexo_multitask_best.pt \\
        --n-riccio 15 --n-egoexo 20
"""

from __future__ import annotations

import argparse
import csv
import random
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

# Reuse helpers from the single-clip demo
from scripts.demo_two_model_inference import (
    load_multitask_model, load_riccio_model,
    load_riccio_clip_from_npz, load_egoexo_clip,
    riccio_classify, multitask_infer,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--riccio-ckpt", type=Path, required=True)
    p.add_argument("--multitask-ckpt", type=Path, required=True)
    p.add_argument("--riccio-features-dir", type=Path,
                   default=Path("results/riccio_resnet50_features"))
    p.add_argument("--riccio-features-stem", type=str,
                   default="riccio_realtime_exercise_recognition")
    p.add_argument("--egoexo-clip-root", type=Path,
                   default=Path("notebooks/data/egoexo_fitness_full/features_open/visual"))
    p.add_argument("--egoexo-index-csv", type=Path,
                   default=Path("results/egoexo_fitness_index_split.csv"))
    p.add_argument("--view", type=str, default="ego_l")
    p.add_argument("--riccio-window", type=int, default=60)
    p.add_argument("--clip-max-frames", type=int, default=300)
    p.add_argument("--clip-subsample-stride", type=int, default=3)
    p.add_argument("--n-riccio", type=int, default=15)
    p.add_argument("--n-egoexo", type=int, default=20)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cpu", choices=("cpu", "cuda", "mps"))
    p.add_argument("--show-comments", action="store_true",
                   help="Print full comment + guidance text for each EgoExo clip.")
    return p.parse_args()


def run_riccio_batch(args, model, idx_to_class, rng,
                     excluded_classes: List[str]) -> List[Dict[str, Any]]:
    print(f"\n{'═'*100}")
    print(f"  RICCIO  ·  {args.n_riccio} clips  ·  4-class xLSTM[7:1]  ·  ResNet-50 features")
    print(f"  (excluding classes the model was not trained on: {excluded_classes})")
    print(f"{'═'*100}")

    # Load NPZ once, then sample unique video_ids
    bio_path = args.riccio_features_dir / f"{args.riccio_features_stem}_biomechanics.npz"
    lab_path = args.riccio_features_dir / f"{args.riccio_features_stem}_labels.npz"
    bio = np.load(bio_path, allow_pickle=True)
    lab = np.load(lab_path, allow_pickle=True)
    frame_features = bio["frame_features"]
    poses = lab["pose"]
    video_ids_arr = lab["video_id"]

    # ─── Standardisation (required to match training-time distribution) ───
    # The training pipeline standardises features to zero mean / unit variance
    # using stats from the train split. The checkpoint does not store these
    # stats, so we approximate them from all frames in the NPZ. This is a close
    # approximation because the splits are stratified per-video.
    excluded_lower = {c.lower().strip() for c in excluded_classes}
    pose_lower = np.array([str(p).strip().lower() for p in poses])
    keep_mask = np.array([p not in excluded_lower for p in pose_lower])
    feat_for_stats = frame_features[keep_mask]
    feat_mean = feat_for_stats.mean(axis=0).astype(np.float32)
    feat_std = feat_for_stats.std(axis=0).astype(np.float32) + 1e-8
    print(f"  standardisation: mean[0]={feat_mean[0]:.4f}  std[0]={feat_std[0]:.4f}")

    # Filter video IDs whose true class is excluded
    excluded_vids = set()
    for vid in set(video_ids_arr.tolist()):
        mask = (video_ids_arr == vid)
        cls = str(poses[mask][0]).strip().lower()
        if cls in excluded_lower:
            excluded_vids.add(int(vid))
    unique_ids = sorted(v for v in set(video_ids_arr.tolist()) if int(v) not in excluded_vids)
    print(f"  available videos after exclusion: {len(unique_ids)}")

    chosen = rng.sample(unique_ids, min(args.n_riccio, len(unique_ids)))
    print(f"  {'#':>2}  {'vid':>4}  {'true class':<22s}  {'predicted':<22s}  {'conf':>5s}  match  {'#frames':>7}")
    print(f"  {'─'*2}  {'─'*4}  {'─'*22}  {'─'*22}  {'─'*5}  {'─'*5}  {'─'*7}")
    results = []
    for i, vid in enumerate(chosen, 1):
        mask = (video_ids_arr == vid)
        feats = frame_features[mask]
        pose = poses[mask]
        if feats.shape[0] < args.riccio_window:
            continue
        start = rng.randint(0, feats.shape[0] - args.riccio_window)
        win = feats[start:start + args.riccio_window].astype(np.float32)
        # Apply train-time standardisation
        win = (win - feat_mean) / feat_std
        x = torch.from_numpy(win).unsqueeze(0).to(args.device)
        out = riccio_classify(model, x, idx_to_class)
        true_cls = str(pose[0])
        match = "✓" if out["exercise"].lower() == true_cls.lower() else "✗"
        results.append({
            "video_id": int(vid),
            "true_class": true_cls,
            "pred_class": out["exercise"],
            "confidence": out["confidence"],
            "match": match == "✓",
            "n_frames_total": int(feats.shape[0]),
        })
        print(f"  {i:>2}  {vid:>4}  {true_cls:<22s}  {out['exercise']:<22s}  {out['confidence']:>5.3f}    {match}    {int(feats.shape[0]):>7d}")

    # Aggregate
    n = len(results)
    n_correct = sum(1 for r in results if r["match"])
    acc = n_correct / max(1, n)
    mean_conf = sum(r["confidence"] for r in results) / max(1, n)
    print(f"\n  ─ Aggregate over {n} Riccio clips ─")
    print(f"    accuracy           : {acc:.3f}  ({n_correct}/{n})")
    print(f"    mean confidence    : {mean_conf:.3f}")
    # Per-class breakdown
    from collections import Counter
    by_true = Counter(r["true_class"] for r in results)
    correct_by_true = Counter(r["true_class"] for r in results if r["match"])
    print(f"\n    Per-class breakdown (correct / total):")
    for cls, total in sorted(by_true.items()):
        c = correct_by_true.get(cls, 0)
        print(f"      {cls:<24s}  {c:>2d}/{total:<2d}  ({100*c/total:.0f}%)")
    return results


def run_egoexo_batch(args, model, rng, show_comments: bool,
                     feat_mean=None, feat_std=None) -> List[Dict[str, Any]]:
    print(f"\n{'═'*100}")
    print(f"  EGOEXO  ·  {args.n_egoexo} clips  ·  branched multi-task xLSTM  ·  CLIP ViT-B/32 features  ·  view={args.view}")
    print(f"{'═'*100}")

    rows = list(csv.DictReader(open(args.egoexo_index_csv)))
    test_rows = [r for r in rows if r.get("split") == "test"]
    rng.shuffle(test_rows)

    print(f"  {'#':>2}  {'record':<10s}  {'true class':<35s}  {'predicted':<35s}  {'tq':>5s}  {'pq':>5s}  {'|err|':>5s}  match")
    print(f"  {'─'*2}  {'─'*10}  {'─'*35}  {'─'*35}  {'─'*5}  {'─'*5}  {'─'*5}  {'─'*5}")
    results = []
    seen_records = set()
    for r in test_rows:
        if len(results) >= args.n_egoexo:
            break
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
        record_id = jk.split("_action_")[0]
        if record_id in seen_records:
            continue
        seen_records.add(record_id)
        from fitness_coach.datasets.egoexo_xlstm_dataset import load_clip_segment
        arr = load_clip_segment(args.egoexo_clip_root, record_id, fs, fe,
                                view=args.view, max_frames=args.clip_max_frames,
                                subsample_stride=args.clip_subsample_stride,
                                allow_fallback=False)
        if arr is None:
            continue
        # Apply train-time standardisation if the checkpoint provides stats
        if feat_mean is not None and feat_std is not None:
            arr = (arr - feat_mean) / (feat_std + 1e-8)
        x = torch.from_numpy(arr.astype(np.float32)).unsqueeze(0).to(args.device)
        out = multitask_infer(model, x)
        true_cls = (r.get("exercise_class") or "").strip()
        true_q_raw = (r.get("quality") or "").strip()
        true_cmt = (r.get("comment") or "").strip()
        try:
            true_q = float(true_q_raw)
        except Exception:
            true_q = float("nan")
        pred_q = float(out["quality"])
        err = abs(pred_q - true_q) if not (true_q != true_q) else float("nan")  # nan-safe
        match = "✓" if out["exercise"].lower() == true_cls.lower() else "✗"
        i = len(results) + 1
        print(f"  {i:>2}  {record_id:<10s}  {true_cls[:35]:<35s}  {out['exercise'][:35]:<35s}  "
              f"{true_q:>5.2f}  {pred_q:>5.2f}  {err:>5.3f}    {match}")
        results.append({
            "record_id": record_id, "true_class": true_cls, "pred_class": out["exercise"],
            "true_quality": true_q, "pred_quality": pred_q, "abs_err": err,
            "match": match == "✓",
            "guidance": out["guidance"], "comment": out["comment"], "true_comment": true_cmt,
        })

    n = len(results)
    n_correct = sum(1 for r in results if r["match"])
    acc = n_correct / max(1, n)
    valid_q = [r for r in results if not (r["abs_err"] != r["abs_err"])]
    mae = sum(r["abs_err"] for r in valid_q) / max(1, len(valid_q)) if valid_q else float("nan")
    if valid_q:
        diffs = np.array([r["pred_quality"] - r["true_quality"] for r in valid_q])
        rmse = float(np.sqrt(np.mean(diffs ** 2)))
    else:
        rmse = float("nan")

    print(f"\n  ─ Aggregate over {n} EgoExo clips ─")
    print(f"    classification acc : {acc:.3f}  ({n_correct}/{n})")
    print(f"    quality MAE        : {mae:.3f}")
    print(f"    quality RMSE       : {rmse:.3f}")

    if show_comments:
        print(f"\n{'═'*100}")
        print("  Per-clip generated comments & guidance (EgoExo)")
        print(f"{'═'*100}")
        for i, r in enumerate(results, 1):
            print(f"\n  Clip {i} · record {r['record_id']} · true={r['true_class']} pred={r['pred_class']} "
                  f"q_true={r['true_quality']:.2f} q_pred={r['pred_quality']:.2f}")
            print(f"    GUIDANCE  : {r['guidance'][:180]}{'…' if len(r['guidance']) > 180 else ''}")
            print(f"    COMMENT   : {r['comment'][:180]}{'…' if len(r['comment']) > 180 else ''}")
            if r['true_comment']:
                print(f"    GT COMMENT: {r['true_comment'][:180]}{'…' if len(r['true_comment']) > 180 else ''}")
    return results


def main() -> int:
    args = parse_args()
    device = torch.device(args.device)
    rng = random.Random(args.seed)

    multitask_model, _ = load_multitask_model(args.multitask_ckpt, device)
    riccio_model, riccio_meta = load_riccio_model(args.riccio_ckpt, device)
    args.device = device  # convenience

    # Pull the trained-time standardisation stats and excluded classes from
    # each checkpoint so inference matches the training distribution.
    mt_ck = torch.load(args.multitask_ckpt, map_location="cpu", weights_only=False)
    mt_mean = mt_ck.get("mean")
    mt_std = mt_ck.get("std")
    if mt_mean is None or mt_std is None:
        print("[warn] multi-task ckpt has no standardisation stats — predictions may degrade")
    else:
        mt_mean = np.asarray(mt_mean, dtype=np.float32)
        mt_std = np.asarray(mt_std, dtype=np.float32)
        print(f"[multitask] standardisation: mean[0]={mt_mean[0]:.4f}  std[0]={mt_std[0]:.4f}")

    rc_ck = torch.load(args.riccio_ckpt, map_location="cpu", weights_only=False)
    rc_args = rc_ck.get("args", {}) or {}
    excluded_str = str(rc_args.get("exclude_classes", "") or "")
    excluded = [c.strip() for c in excluded_str.split(",") if c.strip()]

    riccio_results = run_riccio_batch(args, riccio_model, riccio_meta["idx_to_class"], rng, excluded)
    egoexo_results = run_egoexo_batch(args, multitask_model, rng, args.show_comments,
                                      feat_mean=mt_mean, feat_std=mt_std)

    # Final summary
    print(f"\n{'═'*100}")
    print("  FINAL SUMMARY")
    print(f"{'═'*100}")
    if riccio_results:
        n = len(riccio_results)
        acc = sum(1 for r in riccio_results if r["match"]) / n
        print(f"  Riccio (n={n})   : accuracy {acc:.3f}, mean confidence {sum(r['confidence'] for r in riccio_results)/n:.3f}")
    if egoexo_results:
        n = len(egoexo_results)
        acc = sum(1 for r in egoexo_results if r["match"]) / n
        valid = [r for r in egoexo_results if not (r['abs_err'] != r['abs_err'])]
        mae = sum(r['abs_err'] for r in valid) / max(1, len(valid)) if valid else float("nan")
        print(f"  EgoExo (n={n})   : accuracy {acc:.3f}, quality MAE {mae:.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
