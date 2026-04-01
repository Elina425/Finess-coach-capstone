"""Spatial–temporal GCN on COCO-17 skeleton (ST-GCN style) for exercise classification + quality."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# COCO-17 undirected edges (same as pose_estimation_core.COCO_KEYPOINTS.SKELETON)
_COCO_EDGES = (
    (0, 1),
    (0, 2),
    (1, 3),
    (2, 4),
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
)


def coco_adjacency_matrix(num_nodes: int = 17) -> np.ndarray:
    a = np.zeros((num_nodes, num_nodes), dtype=np.float32)
    for i, j in _COCO_EDGES:
        a[i, j] = 1.0
        a[j, i] = 1.0
    return a


def normalized_adjacency(A: np.ndarray, self_loop: bool = True) -> np.ndarray:
    """Symmetric normalization: Ã = D^{-1/2} (A + I) D^{-1/2}."""
    a = np.asarray(A, dtype=np.float64)
    if self_loop:
        a = a + np.eye(a.shape[0], dtype=np.float64)
    deg = a.sum(axis=1)
    d_inv_sqrt = np.zeros_like(deg)
    nz = deg > 0
    d_inv_sqrt[nz] = np.power(deg[nz], -0.5)
    d = np.diag(d_inv_sqrt)
    return (d @ a @ d).astype(np.float32)


class STGCNBlock(nn.Module):
    """One spatial GCN (neighbor aggregate + 1×1 conv) + temporal conv over T."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        A: torch.Tensor,
        kernel_size: int = 9,
        stride: int = 1,
        dropout: float = 0.25,
    ):
        super().__init__()
        self.register_buffer("A", A)
        self.gcn = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        pad = (kernel_size - 1) // 2
        self.tcn = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=(kernel_size, 1),
                padding=(pad, 0),
                stride=(stride, 1),
            ),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(dropout),
        )
        self.use_res = in_channels != out_channels or stride != 1
        if self.use_res:
            self.res = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=1,
                stride=(stride, 1),
            )
        else:
            self.res = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: N, C, T, V
        xa = torch.einsum("nctu,vu->nctv", x, self.A)
        xa = self.gcn(xa)
        y = self.tcn(xa)
        res = self.res(x) if self.use_res else x
        return F.relu(y + res)


class ExerciseSTGCN(nn.Module):
    """
    Input: (N, 2, T, V) with V=17 joints, 2 channels = x,y (normalized coords).
    Output: logits (N, num_classes), quality (N,).
    """

    def __init__(
        self,
        num_classes: int,
        num_joints: int = 17,
        in_channels: int = 2,
        base_channels: int = 64,
        dropout: float = 0.25,
    ):
        super().__init__()
        A = torch.from_numpy(normalized_adjacency(coco_adjacency_matrix(num_joints)))
        self.num_joints = num_joints
        c1, c2, c3 = base_channels, base_channels * 2, base_channels * 4
        self.st1 = STGCNBlock(in_channels, c1, A, kernel_size=9, stride=1, dropout=dropout)
        self.st2 = STGCNBlock(c1, c2, A, kernel_size=9, stride=1, dropout=dropout)
        self.st3 = STGCNBlock(c2, c3, A, kernel_size=9, stride=1, dropout=dropout)
        self.pool = nn.AdaptiveAvgPool2d(1)
        d = c3
        self.drop = nn.Dropout(0.35)
        self.fc_cls = nn.Linear(d, num_classes)
        self.fc_reg = nn.Linear(d, 1)

    def forward(self, x: torch.Tensor):
        # N, C, T, V
        x = self.st1(x)
        x = self.st2(x)
        x = self.st3(x)
        x = self.pool(x).squeeze(-1).squeeze(-1)
        h = self.drop(x)
        return self.fc_cls(h), self.fc_reg(h).squeeze(-1)
