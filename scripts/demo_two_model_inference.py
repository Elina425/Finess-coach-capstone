#!/usr/bin/env python3
"""Two-model inference demo — Riccio classifier + EgoExo multi-task model.

The two trained models cover different datasets and use different frame features:

  1. paper_xlstm_seq60_resnet   — Riccio dataset, ResNet-50 features (2048-d),
                                  4 classes (squat, push-up, shoulder press,
                                  bicep curl).
  2. xlstm_egoexo_multitask     — EgoExo-Fitness dataset, CLIP ViT-B/32
                                  features (512-d), 12 classes + quality
                                  regression + retrieval-based comment & guidance.

There is no way to chain Riccio's 4-class output into the multi-task model
(incompatible feature spaces and label spaces). What this script does instead:

  * Picks one random clip from each dataset's test split,
  * Runs the appropriate model on each,
  * Prints both results side by side as a unified demo of what the trained
    system can produce.

Usage:
    ./venv/bin/python scripts/demo_two_model_inference.py \\
        --riccio-ckpt   results/paper_xlstm_seq60_resnet/xlstm_7_1/best.pt \\
        --multitask-ckpt results/xlstm_egoexo_multitask_allviews/xlstm_egoexo_multitask_best.pt

To pick specific clips, use --riccio-video-id and --egoexo-record-id flags.
"""

from __future__ import annotations

import argparse
import csv
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from fitness_coach.datasets.egoexo_xlstm_dataset import load_clip_segment
from fitness_coach.models.xlstm_model import xLSTMExerciseClassifier


# Loose mapping between Riccio's 4-class label space and EgoExo's 12-class label space.
# Only push-ups have a clean counterpart; sumo squat is the closest EgoExo cousin of squat.
RICCIO_TO_EGOEXO_OVERLAP: Dict[str, Tuple[str, ...]] = {
    "push-up":              ("push-ups", "kneeling pushing-ups"),
    "squat":                ("sumo squat",),
    "barbell biceps curl":  (),
    "shoulder press":       (),
}


# ─── CLI ─────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--riccio-ckpt", type=Path, required=True)
    p.add_argument("--multitask-ckpt", type=Path, required=True)
    p.add_argument("--riccio-features-dir", type=Path,
                   default=Path("results/riccio_resnet50_features"),
                   help="Directory containing the Kaggle-mode Riccio NPZ "
                        "(biomechanics + labels).")
    p.add_argument("--riccio-features-stem", type=str,
                   default="riccio_realtime_exercise_recognition",
                   help="Filename stem for the Riccio NPZ pair.")
    p.add_argument("--egoexo-clip-root", type=Path,
                   default=Path("notebooks/data/egoexo_fitness_full/features_open/visual"),
                   help="EgoExo CLIP features root.")
    p.add_argument("--egoexo-index-csv", type=Path,
                   default=Path("results/egoexo_fitness_index_split.csv"))
    p.add_argument("--riccio-video-id", type=int, default=None,
                   help="Specific Riccio video_id to sample (0-based).")
    p.add_argument("--egoexo-record-id", type=str, default=None,
                   help="Specific EgoExo record_id to sample.")
    p.add_argument("--view", type=str, default="ego_l")
    p.add_argument("--riccio-window", type=int, default=60,
                   help="Window length used by the Riccio xLSTM (default 60).")
    p.add_argument("--clip-max-frames", type=int, default=300)
    p.add_argument("--clip-subsample-stride", type=int, default=3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cpu", choices=("cpu", "cuda", "mps"))
    return p.parse_args()


# ─── checkpoint loaders ──────────────────────────────────────────────────────

def load_multitask_model(ckpt_path: Path, device: torch.device) -> Tuple[xLSTMExerciseClassifier, Dict[str, Any]]:
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
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
    idx_to_class = {int(k): str(v) for k, v in ck["idx_to_class"].items()}
    model.set_guidance_table(
        {int(k): str(v) for k, v in ck["guidance_table"].items()},
        idx_to_class,
    )
    comment_table_raw = ck.get("comment_table") or {}
    comment_table = {}
    for key, text in comment_table_raw.items():
        c, b = key.split("|")
        comment_table[(int(c), int(b))] = str(text)
    bucket_edges = tuple(float(e) for e in (ck.get("comment_quality_bucket_edges") or (0.4, 0.7)))
    q_lo = float(ck.get("quality_domain_lo", 0.0))
    q_hi = float(ck.get("quality_domain_hi", 1.0))
    model.set_comment_table(comment_table, bucket_edges, domain_lo=q_lo, domain_hi=q_hi)
    print(f"[multitask] {ckpt_path.name}: {len(ck['classes'])} classes · "
          f"comment cells = {len(comment_table)}")
    return model, {"idx_to_class": idx_to_class}


def load_riccio_model(ckpt_path: Path, device: torch.device) -> Tuple[xLSTMExerciseClassifier, Dict[str, Any]]:
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    args_dict = ck.get("args", {}) or {}

    # Riccio Kaggle pipeline ResNet-50 features are 2048-d per frame.
    feature_dim = int(args_dict.get("feature_dim", 2048))
    classes = ck.get("classes") or args_dict.get("classes") or [
        "barbell biceps curl", "push-up", "shoulder press", "squat",
    ]
    if isinstance(classes, dict):
        classes = [classes[i] for i in sorted(classes.keys())]

    model = xLSTMExerciseClassifier(
        input_size=feature_dim,
        hidden_size=int(args_dict.get("xlstm_hidden", 256)),
        num_layers=int(args_dict.get("xlstm_num_layers", 8)),
        num_classes=len(classes),
        dropout=float(args_dict.get("dropout", 0.25)),
        num_heads=int(args_dict.get("xlstm_num_heads", 4)),
        conv_kernel_size=int(args_dict.get("xlstm_conv_kernel_size", 4)),
        projection_factor=float(args_dict.get("xlstm_projection_factor", 4.0 / 3.0)),
        num_error_tags=0,
        quality_scale=1.0,
        block_pattern=str(args_dict.get("xlstm_block_pattern", "mmmmmmms")),
        use_attention_pool=False,
        use_fusion=False,
    ).to(device)
    model.load_state_dict(ck["model"], strict=False)
    model.eval()
    idx_to_class = {i: name for i, name in enumerate(classes)}
    print(f"[riccio]    {ckpt_path.name}: {len(classes)} classes · "
          f"feature_dim={feature_dim} · expected window={args_dict.get('seq_len', 60)}")
    return model, {"idx_to_class": idx_to_class, "classes": classes}


# ─── Riccio Kaggle-NPZ feature loader ────────────────────────────────────────

def load_riccio_clip_from_npz(
    features_dir: Path, stem: str, video_id: Optional[int], window: int,
    rng: random.Random,
) -> Optional[Dict[str, Any]]:
    """Pull a window of ResNet-50 features for ONE Riccio video from the
    Kaggle-mode combined NPZ. Returns dict with features tensor + label.
    """
    bio_path = features_dir / f"{stem}_biomechanics.npz"
    lab_path = features_dir / f"{stem}_labels.npz"
    if not bio_path.is_file() or not lab_path.is_file():
        print(f"[skip] Riccio NPZ not found: {bio_path}")
        return None

    bio = np.load(bio_path, allow_pickle=True)
    lab = np.load(lab_path, allow_pickle=True)
    frame_features = bio["frame_features"]              # (N_frames, 2048)
    poses = lab["pose"]                                 # (N_frames,) class strings
    video_ids = lab["video_id"]                         # (N_frames,) int

    # Pick a video_id if not specified.
    unique_ids = sorted(set(video_ids.tolist()))
    if video_id is None:
        video_id = rng.choice(unique_ids)
    if video_id not in unique_ids:
        print(f"[skip] video_id={video_id} not in NPZ (available range 0..{max(unique_ids)})")
        return None

    mask = (video_ids == video_id)
    feats = frame_features[mask]
    pose = poses[mask]
    if feats.shape[0] < window:
        print(f"[skip] video_id={video_id} has only {feats.shape[0]} frames (need {window})")
        return None

    # Pick a random window from inside this video so the demo varies.
    start = rng.randint(0, feats.shape[0] - window)
    win = feats[start:start + window].astype(np.float32)
    return {
        "features": torch.from_numpy(win).unsqueeze(0),  # (1, window, 2048)
        "video_id": int(video_id),
        "true_class": str(pose[0]),
        "window_start": int(start),
        "window_end": int(start + window),
        "n_frames_total": int(feats.shape[0]),
    }


# ─── EgoExo CLIP feature loader ──────────────────────────────────────────────

def load_egoexo_clip(
    clip_root: Path, index_csv: Path, view: str, record_id: Optional[str],
    rng: random.Random, max_frames: int, subsample_stride: int,
) -> Optional[Dict[str, Any]]:
    rows = list(csv.DictReader(open(index_csv)))
    test_rows = [r for r in rows if r.get("split") == "test"]
    if record_id:
        test_rows = [r for r in test_rows if record_id in (r.get("judgement_key") or "")]
        if not test_rows:
            print(f"[skip] EgoExo record_id={record_id!r} not found in test split")
            return None
    rng.shuffle(test_rows)
    for r in test_rows:
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
        record_id_resolved = jk.split("_action_")[0]
        arr = load_clip_segment(clip_root, record_id_resolved, fs, fe,
                                view=view, max_frames=max_frames,
                                subsample_stride=subsample_stride,
                                allow_fallback=False)
        if arr is None:
            continue
        return {
            "features": torch.from_numpy(arr).unsqueeze(0),  # (1, T, 512)
            "record_id": record_id_resolved,
            "view": view,
            "frame_start": fs, "frame_end": fe,
            "true_class": (r.get("exercise_class") or "").strip(),
            "true_quality": (r.get("quality") or "").strip(),
            "true_comment": (r.get("comment") or "").strip(),
        }
    print(f"[skip] no EgoExo clip with view={view!r} found")
    return None


# ─── inference + pretty-print ───────────────────────────────────────────────

def riccio_classify(model: xLSTMExerciseClassifier, x: torch.Tensor,
                    idx_to_class: Dict[int, str]) -> Dict[str, Any]:
    with torch.no_grad():
        out = model(x)
        logits = out[0] if isinstance(out, (tuple, list)) else out
        probs = torch.softmax(logits, dim=-1)[0].cpu().numpy()
    cls_idx = int(probs.argmax())
    return {
        "exercise": idx_to_class[cls_idx],
        "confidence": float(probs[cls_idx]),
        "all_probs": {idx_to_class[i]: float(p) for i, p in enumerate(probs)},
    }


def multitask_infer(model: xLSTMExerciseClassifier, x: torch.Tensor) -> Dict[str, Any]:
    with torch.no_grad():
        return model.infer(x)[0]


def _wrap_lines(text: str, width: int = 70, indent: str = "│      ") -> str:
    if not text:
        return f"{indent}(none)"
    out_lines = []
    for line in text.splitlines() or [""]:
        for i in range(0, max(1, len(line)), width):
            out_lines.append(f"{indent}{line[i:i+width]}")
    return "\n".join(out_lines)


def print_riccio_block(riccio_out: Dict[str, Any], ground_truth_class: str) -> None:
    print("┌" + "─" * 76)
    print("│  STAGE 1 · Riccio xLSTM[7:1] · ResNet-50 features · Riccio dataset")
    print(f"│    predicted exercise :  {riccio_out['exercise']}")
    print(f"│    confidence         :  {riccio_out['confidence']:.3f}")
    print(f"│    full distribution  :  " + ", ".join(
        f"{n}={p:.2f}" for n, p in sorted(riccio_out["all_probs"].items(),
                                          key=lambda kv: -kv[1])
    ))
    match = "✓" if riccio_out["exercise"].lower() == ground_truth_class.lower() else "✗"
    print(f"│    ground truth       :  {ground_truth_class}   [{match}]")
    print("└" + "─" * 76)


def print_multitask_block(multitask_out: Dict[str, Any], gt: Dict[str, Any]) -> None:
    print("┌" + "─" * 76)
    print("│  STAGE 2 · EgoExo branched multi-task xLSTM · CLIP features · EgoExo dataset")
    q01 = float(multitask_out["quality"])
    q15 = 1.0 + 4.0 * q01
    print(f"│    EXERCISE :  {multitask_out['exercise']}")
    print(f"│    QUALITY  :  {q01:.3f}  (≈ {q15:.2f} on the 1–5 scale)")
    print(f"│    GUIDANCE :")
    print(_wrap_lines(multitask_out["guidance"], indent="│      "))
    print(f"│    COMMENT  :")
    print(_wrap_lines(multitask_out["comment"], indent="│      "))
    print("│")
    print("│  ─── Ground truth ─────────────────────────────────────")
    print(f"│    true exercise:  {gt.get('exercise', '')}   "
          f"[{'✓' if gt.get('exercise', '').lower() == multitask_out['exercise'].lower() else '✗'}]")
    gt_q = gt.get("quality", "")
    if gt_q != "":
        try:
            gt_q_f = float(gt_q)
            print(f"│    true quality :  {gt_q_f:.3f}  (|error| = {abs(gt_q_f - q01):.3f})")
        except Exception:
            pass
    gt_cmt = (gt.get("comment") or "").strip()
    if gt_cmt:
        print(f"│    true comment :  {gt_cmt[:200]}{'…' if len(gt_cmt) > 200 else ''}")
    print("└" + "─" * 76)


def print_overlap_note(riccio_class: str, multitask_class: str) -> None:
    print()
    print("─── Cross-model relationship ────────────────────────────────────────────────")
    print("  These two predictions come from DIFFERENT physical videos taken from")
    print("  DIFFERENT datasets (Riccio gym corpus vs EgoExo-Fitness). They cannot be")
    print("  chained — Riccio's 4-class output cannot be fed into the multi-task")
    print("  model's input space, and only push-ups have a clean counterpart between")
    print("  the two label spaces.")
    rcc = riccio_class.lower()
    ego = multitask_class.lower()
    overlap = RICCIO_TO_EGOEXO_OVERLAP.get(rcc, ())
    if not overlap:
        print(f"  Riccio class '{rcc}' has no EgoExo counterpart in this label space.")
    elif any(ego == o for o in overlap):
        print(f"  ✓ Coincidentally, both models output overlapping exercises this run")
        print(f"    (Riccio '{rcc}' ≈ EgoExo '{ego}').")
    else:
        print(f"  Riccio output '{rcc}' and EgoExo output '{ego}' belong to different")
        print(f"    exercise families on this run; the two reports are independent.")


# ─── main ────────────────────────────────────────────────────────────────────

def main() -> int:
    args = parse_args()
    device = torch.device(args.device)
    rng = random.Random(args.seed)

    multitask_model, _ = load_multitask_model(args.multitask_ckpt, device)
    riccio_model, riccio_meta = load_riccio_model(args.riccio_ckpt, device)

    # === Riccio clip ===
    print("\n[demo] Loading one Riccio clip from the ResNet-50 NPZ …")
    riccio_clip = load_riccio_clip_from_npz(
        args.riccio_features_dir, args.riccio_features_stem,
        args.riccio_video_id, args.riccio_window, rng,
    )

    # === EgoExo clip ===
    print("[demo] Loading one EgoExo clip from CLIP features …\n")
    egoexo_clip = load_egoexo_clip(
        args.egoexo_clip_root, args.egoexo_index_csv, args.view,
        args.egoexo_record_id, rng, args.clip_max_frames, args.clip_subsample_stride,
    )

    # === Stage 1 ===
    riccio_out = None
    if riccio_clip is not None:
        riccio_out = riccio_classify(riccio_model, riccio_clip["features"].to(device),
                                     riccio_meta["idx_to_class"])
        print(f"Riccio clip: video_id={riccio_clip['video_id']}  "
              f"window frames {riccio_clip['window_start']}-{riccio_clip['window_end']} "
              f"(of {riccio_clip['n_frames_total']})")
        print_riccio_block(riccio_out, riccio_clip["true_class"])
    else:
        print("[note] Skipping Riccio stage — no usable NPZ clip found.")

    print()

    # === Stage 2 ===
    multitask_out = None
    if egoexo_clip is not None:
        multitask_out = multitask_infer(multitask_model, egoexo_clip["features"].to(device))
        print(f"EgoExo clip: record={egoexo_clip['record_id']}  view={egoexo_clip['view']}  "
              f"frames {egoexo_clip['frame_start']}-{egoexo_clip['frame_end']} "
              f"({egoexo_clip['features'].shape[1]} after subsample)")
        gt = {
            "exercise": egoexo_clip["true_class"],
            "quality": egoexo_clip["true_quality"],
            "comment": egoexo_clip["true_comment"],
        }
        print_multitask_block(multitask_out, gt)
    else:
        print("[note] Skipping multi-task stage — no usable EgoExo clip found.")

    if riccio_out and multitask_out:
        print_overlap_note(riccio_out["exercise"], multitask_out["exercise"])

    return 0


if __name__ == "__main__":
    sys.exit(main())
