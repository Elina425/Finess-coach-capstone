"""
Skeleton graph and coordinate normalization for topology-aware GCN pose models.

Implements bbox min–max normalization (Eq. 1–2) and adjacency with self-loops
(Ĉ = C + I) as in Zeng et al., "A Topology-Aware Graph Convolutional Network for
Human Pose Similarity and Action Quality Assessment" (arXiv:2511.01194).
COCO-17 keypoint order matches biomechanical_features / MediaPipe mapping.
"""

from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np
import torch


def coco17_skeleton_edges() -> List[Tuple[int, int]]:
    """Undirected bone / kinematic edges for MS COCO 17 (person)."""
    return [
        (0, 1),
        (0, 2),
        (1, 3),
        (2, 4),
        (0, 5),
        (0, 6),
        (5, 6),
        (5, 7),
        (7, 9),
        (6, 8),
        (8, 10),
        (5, 11),
        (6, 12),
        (11, 12),
        (11, 13),
        (13, 15),
        (12, 14),
        (14, 16),
    ]


def adjacency_matrix(
    num_nodes: int,
    edges: Sequence[Tuple[int, int]],
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Undirected edge set C (no self-loops); use Ĉ = C + I before normalization."""
    c = torch.zeros(num_nodes, num_nodes, dtype=dtype)
    for i, j in edges:
        c[i, j] = 1.0
        c[j, i] = 1.0
    return c


def symmetric_normalized_adjacency(c_hat: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """D̂^{-1/2} Ĉ D̂^{-1/2} for undirected graph (Eq. 3)."""
    deg = c_hat.sum(dim=1).clamp_min(eps)
    d_inv_sqrt = deg.pow(-0.5)
    d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.0
    d_mat = torch.diag(d_inv_sqrt)
    return d_mat @ c_hat @ d_mat


def minmax_normalize_xy_coco17(
    kp: np.ndarray,
    eps: float = 1e-6,
) -> np.ndarray:
    """
    Per-frame min–max normalization over all joints (paper Eq. 1–2).
    kp: (17, 2) or (T, 17, 2). Invalid (0,0) should be handled before or masked.
    """
    x = np.asarray(kp, dtype=np.float32)
    single = x.ndim == 2
    if single:
        x = x[np.newaxis, ...]
    out = np.zeros_like(x)
    for t in range(x.shape[0]):
        frame = x[t]
        xmin = float(np.nanmin(frame[:, 0]))
        xmax = float(np.nanmax(frame[:, 0]))
        ymin = float(np.nanmin(frame[:, 1]))
        ymax = float(np.nanmax(frame[:, 1]))
        dx = max(xmax - xmin, eps)
        dy = max(ymax - ymin, eps)
        out[t, :, 0] = (frame[:, 0] - xmin) / dx
        out[t, :, 1] = (frame[:, 1] - ymin) / dy
    if single:
        return out[0]
    return out


def mask_invalid_coco17(kp: np.ndarray) -> np.ndarray:
    """Replace missing (0,0) joints with NaN so min/max ignores them if needed."""
    x = np.asarray(kp, dtype=np.float32).copy()
    bad = (np.abs(x[..., 0]) < 1e-8) & (np.abs(x[..., 1]) < 1e-8)
    x[bad] = np.nan
    return x


def minmax_normalize_xy_nan_safe(kp: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Min–max using finite values only; missing joints filled with 0 after norm."""
    x = mask_invalid_coco17(kp)
    single = x.ndim == 2
    if single:
        x = x[np.newaxis, ...]
    out = np.zeros((x.shape[0], 17, 2), dtype=np.float32)
    for t in range(x.shape[0]):
        frame = x[t]
        fx = frame[:, 0][np.isfinite(frame[:, 0])]
        fy = frame[:, 1][np.isfinite(frame[:, 1])]
        if fx.size == 0 or fy.size == 0:
            continue
        xmin, xmax = float(fx.min()), float(fx.max())
        ymin, ymax = float(fy.min()), float(fy.max())
        dx = max(xmax - xmin, eps)
        dy = max(ymax - ymin, eps)
        for j in range(17):
            if not np.isfinite(frame[j, 0]):
                continue
            out[t, j, 0] = (frame[j, 0] - xmin) / dx
            out[t, j, 1] = (frame[j, 1] - ymin) / dy
    if single:
        return out[0]
    return out
