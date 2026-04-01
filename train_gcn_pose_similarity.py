#!/usr/bin/env python3
"""
Train topology-aware GCN pose encoder with contrastive regression (Zeng et al.,
arXiv:2511.01194): Siamese shared weights, cosine distance Dc, loss Eq. (4).

Uses COCO-17 keypoints from *_keypoints.npz and the same index CSV as the BiLSTM pipeline.

Example:
  ./venv/bin/python train_gcn_pose_similarity.py \\
    --index-csv results/exercise_training_index_long_range.csv \\
    --keypoints-dir results/processed_keypoints_mediapipe \\
    --epochs 50
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from gcn_pose_dataset import GCNPosePairDataset, build_stem_metadata_from_index
from gcn_pose_model import TopologyAwareGCNEncoder, contrastive_regression_loss, cosine_distance


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    opt: torch.optim.Optimizer,
    device: torch.device,
    margin: float,
):
    model.train()
    total = 0.0
    n = 0
    for p1, p2, y in loader:
        p1 = p1.to(device)
        p2 = p2.to(device)
        y = y.to(device)
        opt.zero_grad()
        e1 = model(p1)
        e2 = model(p2)
        dc = cosine_distance(e1, e2)
        loss = contrastive_regression_loss(dc, y, margin=margin)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
        opt.step()
        total += float(loss.item()) * p1.size(0)
        n += p1.size(0)
    return total / max(n, 1)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    margin: float,
):
    model.eval()
    total = 0.0
    n = 0
    for p1, p2, y in loader:
        p1 = p1.to(device)
        p2 = p2.to(device)
        y = y.to(device)
        e1 = model(p1)
        e2 = model(p2)
        dc = cosine_distance(e1, e2)
        loss = contrastive_regression_loss(dc, y, margin=margin)
        total += float(loss.item()) * p1.size(0)
        n += p1.size(0)
    return total / max(n, 1)


def main() -> int:
    ap = argparse.ArgumentParser(description="Train GCN pose similarity (contrastive regression)")
    ap.add_argument("--index-csv", default="./results/exercise_training_index.csv")
    ap.add_argument("--keypoints-dir", default="./results/processed_keypoints_mediapipe")
    ap.add_argument("--output-dir", default="./results/gcn_pose_similarity")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--margin", type=float, default=1.35)
    ap.add_argument("--pairs-per-epoch", type=int, default=4096)
    ap.add_argument("--val-pairs", type=int, default=512)
    ap.add_argument("--gcn-hidden", type=int, default=64)
    ap.add_argument("--embed-dim", type=int, default=50)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()

    index_path = Path(args.index_csv)
    kp_dir = Path(args.keypoints_dir)
    if not index_path.is_file():
        print(f"Missing {index_path}", file=sys.stderr)
        return 1

    class_to_stems, stem_quality, stem_split = build_stem_metadata_from_index(index_path)
    train_stems = [s for s, sp in stem_split.items() if sp == "train" and (kp_dir / f"{s}_keypoints.npz").is_file()]
    if not train_stems:
        print(
            "No train stems with keypoints — check index split and --keypoints-dir.",
            file=sys.stderr,
        )
        return 1

    has_val = any(stem_split.get(s) == "val" for s in stem_split) or any(
        stem_split.get(s) == "test" for s in stem_split
    )
    val_split = "val" if any(stem_split.get(s) == "val" for s in stem_split) else (
        "test" if any(stem_split.get(s) == "test" for s in stem_split) else None
    )

    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")

    train_ds = GCNPosePairDataset(
        index_path,
        kp_dir,
        class_to_stems,
        stem_quality,
        stem_split,
        split="train",
        pairs_per_epoch=args.pairs_per_epoch,
        seed=args.seed,
    )
    if len(train_ds.stems) == 0:
        print("No training stems with keypoints after filtering.", file=sys.stderr)
        return 1

    val_loader = None
    if has_val and val_split:
        val_ds = GCNPosePairDataset(
            index_path,
            kp_dir,
            class_to_stems,
            stem_quality,
            stem_split,
            split=val_split,
            pairs_per_epoch=args.val_pairs,
            seed=args.seed + 1,
        )
        if len(val_ds.stems) > 0:
            val_loader = DataLoader(
                val_ds,
                batch_size=min(args.batch_size, 64),
                shuffle=True,
                num_workers=0,
                drop_last=False,
            )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        drop_last=False,
    )

    model = TopologyAwareGCNEncoder(
        gcn_hidden=args.gcn_hidden,
        embed_dim=args.embed_dim,
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    best_val = float("inf")
    for epoch in range(1, args.epochs + 1):
        tr = train_one_epoch(model, train_loader, opt, device, args.margin)
        if val_loader is not None:
            va = evaluate(model, val_loader, device, args.margin)
            if va < best_val:
                best_val = va
                torch.save(
                    {
                        "model": model.state_dict(),
                        "embed_dim": args.embed_dim,
                        "gcn_hidden": args.gcn_hidden,
                        "margin": args.margin,
                        "num_nodes": 17,
                        "in_dim": 2,
                    },
                    out_dir / "gcn_pose_similarity_best.pt",
                )
            print(f"epoch {epoch:03d}  train_loss={tr:.6f}  val_loss={va:.6f}")
        else:
            torch.save(
                {
                    "model": model.state_dict(),
                    "embed_dim": args.embed_dim,
                    "gcn_hidden": args.gcn_hidden,
                    "margin": args.margin,
                    "num_nodes": 17,
                    "in_dim": 2,
                },
                out_dir / "gcn_pose_similarity_best.pt",
            )
            print(f"epoch {epoch:03d}  train_loss={tr:.6f}  (no val split)")

    meta = {
        "paper": "Zeng et al., Topology-Aware GCN for Pose Similarity / AQA, arXiv:2511.01194",
        "train_stems": len(train_ds.stems),
        "index_csv": str(index_path.resolve()),
        "keypoints_dir": str(kp_dir.resolve()),
    }
    with open(out_dir / "training_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"Checkpoint: {out_dir / 'gcn_pose_similarity_best.pt'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
