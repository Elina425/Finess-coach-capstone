#!/usr/bin/env python3
"""
Load trained GCN-PSN checkpoint and compare two single-frame poses (COCO-17, normalized).

Similarity score mapping (paper Eq. 6): f(x) = σ·exp(-½(x/u)²), σ=100, u=0.3, x = Dc.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

from fitness_coach.models.gcn_pose_model import TopologyAwareGCNEncoder, cosine_distance
from fitness_coach.models.gcn_pose_topology import minmax_normalize_xy_nan_safe


def gaussian_similarity_score(dc: float, sigma: float = 100.0, u: float = 0.3) -> float:
    return float(sigma * np.exp(-0.5 * (dc / u) ** 2))


@torch.no_grad()
def main() -> int:
    p = argparse.ArgumentParser(description="GCN pose similarity between two frames")
    p.add_argument("--checkpoint", default="./results/gcn_pose_similarity/gcn_pose_similarity_best.pt")
    p.add_argument("--npz-a", required=True, help="*_keypoints.npz path")
    p.add_argument("--frame-a", type=int, default=0)
    p.add_argument("--npz-b", required=True)
    p.add_argument("--frame-b", type=int, default=0)
    p.add_argument("--cpu", action="store_true")
    args = p.parse_args()

    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.is_file():
        print(f"Missing {ckpt_path}", file=sys.stderr)
        return 1

    try:
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    except TypeError:
        ckpt = torch.load(ckpt_path, map_location=device)

    embed_dim = int(ckpt.get("embed_dim", 50))
    gcn_hidden = int(ckpt.get("gcn_hidden", 64))
    model = TopologyAwareGCNEncoder(
        gcn_hidden=gcn_hidden,
        embed_dim=embed_dim,
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    def frame_tensor(npz_path: Path, fi: int) -> torch.Tensor:
        kp = np.asarray(np.load(npz_path, allow_pickle=True)["keypoints"], dtype=np.float32)
        if kp.ndim != 3 or kp.shape[1:] != (17, 2):
            raise ValueError(f"Bad keypoints in {npz_path}")
        x = minmax_normalize_xy_nan_safe(kp[fi])
        return torch.from_numpy(x).float().unsqueeze(0).to(device)

    pa = frame_tensor(Path(args.npz_a), args.frame_a)
    pb = frame_tensor(Path(args.npz_b), args.frame_b)
    ea = model(pa)
    eb = model(pb)
    dc = float(cosine_distance(ea, eb).item())
    score = gaussian_similarity_score(dc)

    print(f"Cosine distance Dc: {dc:.4f}")
    print(f"Mapped similarity score (0–100 scale, paper Eq. 6): {score:.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
