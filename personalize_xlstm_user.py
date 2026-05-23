#!/usr/bin/env python3
"""Fine-tune a small per-user adapter on top of a frozen xLSTM checkpoint."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from fitness_coach.datasets.egoexo_xlstm_dataset import EgoExoXLSTMDataset, apply_feature_standardizer, egoexo_collate_fn
from fitness_coach.datasets.exercise_bilstm_dataset import load_index_rows
from fitness_coach.models.xlstm_model import BottleneckAdapter, xLSTMExerciseClassifier


def main() -> int:
    parser = argparse.ArgumentParser(description="Train a per-user xLSTM adapter")
    parser.add_argument("--base-checkpoint", type=Path, required=True)
    parser.add_argument("--index-csv", type=Path, required=True)
    parser.add_argument("--angles-dir", type=Path, default=Path("results/egoexo_exercise_angles"))
    parser.add_argument("--keypoints-dir", type=Path, default=Path("results/egoexo_exercise_angles"))
    parser.add_argument("--clip-features-root", type=Path, default=Path("notebooks/data/egoexo_fitness_full/features_open/visual"))
    parser.add_argument("--split", default="train")
    parser.add_argument("--user-id", required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("results/personalization"))
    parser.add_argument("--adapter-dim", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    if not args.base_checkpoint.is_file():
        print(f"Missing {args.base_checkpoint}", file=sys.stderr)
        return 1
    if not args.index_csv.is_file():
        print(f"Missing {args.index_csv}", file=sys.stderr)
        return 1

    checkpoint = torch.load(args.base_checkpoint, map_location="cpu", weights_only=False)
    rows = load_index_rows(args.index_csv)
    split_rows = [row for row in rows if (row.get("split") or "train") == args.split]
    if not split_rows:
        print(f"No rows found for split={args.split}", file=sys.stderr)
        return 1

    classes = list(checkpoint["classes"])
    class_to_idx = dict(checkpoint["class_to_idx"])
    feature_mode = str(checkpoint.get("feature_mode", "mixed"))
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")

    needs_pose = feature_mode in ("angles", "coords", "mixed")
    needs_collate = feature_mode in ("annotation", "clip")
    dataset = EgoExoXLSTMDataset(
        args.index_csv,
        class_to_idx,
        args.split,
        feature_mode=feature_mode,
        quality_encoding=str(checkpoint.get("quality_encoding", "unit")),
        angles_dir=args.angles_dir if needs_pose else None,
        keypoints_dir=args.keypoints_dir if feature_mode in ("coords", "mixed") else None,
        clip_features_root=args.clip_features_root if feature_mode == "clip" else None,
        window=int(checkpoint.get("window", 60)) if needs_pose else 0,
        stride=int(checkpoint.get("stride", 30)) if needs_pose else 0,
    )
    if len(dataset) == 0:
        print("No session windows found for adapter training.", file=sys.stderr)
        return 1

    if checkpoint.get("mean") is not None and checkpoint.get("std") is not None:
        apply_feature_standardizer(
            dataset.samples,
            checkpoint["mean"],
            checkpoint["std"],
        )

    nqc_ck = int(checkpoint.get("num_quality_classes", 1))
    if nqc_ck > 1:
        edges_raw = checkpoint.get("comment_quality_bucket_edges")
        if not edges_raw:
            print("Classification checkpoint missing comment_quality_bucket_edges; cannot rebuild bucket labels.", file=sys.stderr)
            return 1
        dataset.apply_quality_bucket_labels(tuple(float(e) for e in edges_raw))

    model = xLSTMExerciseClassifier(
        input_size=int(checkpoint["input_size"]),
        hidden_size=int(checkpoint.get("hidden", 128)),
        num_layers=int(checkpoint.get("layers", 4)),
        num_classes=len(classes),
        dropout=float(checkpoint.get("dropout", 0.15)),
        num_heads=int(checkpoint.get("num_heads", 4)),
        conv_kernel_size=int(checkpoint.get("conv_kernel_size", 4)),
        projection_factor=float(checkpoint.get("projection_factor", 4.0 / 3.0)),
        num_error_tags=len(checkpoint.get("error_tags", [])),
        num_quality_classes=int(checkpoint.get("num_quality_classes", 1)),
        quality_scale=float(checkpoint.get("quality_scale", 1.0)),
        quality_output_low=float(checkpoint.get("quality_output_low", 0.0)),
        linear_classifier=bool(checkpoint.get("linear_classifier", False)),
        block_pattern=checkpoint.get("block_pattern") if isinstance(checkpoint.get("block_pattern"), str) else None,
        quality_class_conditioning=bool(checkpoint.get("teacher_force_quality", False)),
    ).to(device)
    model.load_state_dict(checkpoint["model"])

    for parameter in model.parameters():
        parameter.requires_grad = False

    adapter = BottleneckAdapter(
        dim=int(checkpoint.get("hidden", 128)),
        bottleneck_dim=args.adapter_dim,
        dropout=float(checkpoint.get("dropout", 0.15)),
    ).to(device)
    model.attach_adapter(adapter)

    optimizer = torch.optim.AdamW(adapter.parameters(), lr=args.lr, weight_decay=1e-4)
    collate = egoexo_collate_fn if needs_collate else None
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=False, collate_fn=collate)

    for epoch in range(1, args.epochs + 1):
        adapter.train()
        total = 0.0
        seen = 0
        for xb, y_cls, y_qf, y_q_bucket, y_err in loader:
            xb = xb.to(device)
            y_cls = y_cls.to(device)
            y_qf = y_qf.to(device)
            y_q_bucket = y_q_bucket.to(device)
            y_err = y_err.to(device)
            optimizer.zero_grad()
            outputs = model(xb)
            if len(outputs) == 2:
                logits, pred_q = outputs
                if model.quality_is_classification:
                    loss = F.cross_entropy(logits, y_cls) + args.reg_weight * F.cross_entropy(pred_q, y_q_bucket)
                else:
                    loss = F.cross_entropy(logits, y_cls) + args.reg_weight * F.smooth_l1_loss(pred_q.squeeze(-1), y_qf)
            else:
                logits, pred_q, pred_err = outputs
                if model.quality_is_classification:
                    loss = (
                        F.cross_entropy(logits, y_cls)
                        + args.reg_weight * F.cross_entropy(pred_q, y_q_bucket)
                        + 0.6 * F.binary_cross_entropy_with_logits(pred_err, y_err)
                    )
                else:
                    loss = (
                        F.cross_entropy(logits, y_cls)
                        + 0.7 * F.smooth_l1_loss(pred_q.squeeze(-1), y_qf)
                        + 0.6 * F.binary_cross_entropy_with_logits(pred_err, y_err)
                    )
            optimizer.step()
            total += float(loss.item()) * xb.size(0)
            seen += xb.size(0)
        print(f"epoch {epoch:03d} adapter_loss={total / max(seen, 1):.4f}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.output_dir / f"{args.user_id}_adapter.pt"
    torch.save(
        {
            "user_id": args.user_id,
            "base_checkpoint": str(args.base_checkpoint),
            "hidden_size": int(checkpoint.get("hidden", 128)),
            "adapter_dim": int(args.adapter_dim),
            "dropout": float(checkpoint.get("dropout", 0.15)),
            "adapter": adapter.state_dict(),
        },
        out_path,
    )
    meta_path = args.output_dir / f"{args.user_id}_adapter.json"
    meta_path.write_text(
        json.dumps(
            {
                "user_id": args.user_id,
                "base_checkpoint": str(args.base_checkpoint),
                "adapter_checkpoint": str(out_path),
                "epochs": args.epochs,
                "split": args.split,
            },
            indent=2,
        )
    )
    print(f"Saved adapter to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
