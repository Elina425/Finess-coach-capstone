"""
Topology-aware GCN pose encoder + MLP embedding (GCN-PSN style, arXiv:2511.01194).

Two graph convolution layers with symmetric normalized adjacency, ReLU,
then flatten node features and MLP to fixed embedding size (default 50).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from fitness_coach.models.gcn_pose_topology import adjacency_matrix, coco17_skeleton_edges, symmetric_normalized_adjacency


class TopologyAwareGCNEncoder(nn.Module):
    """
    Siamese-shared encoder: (B, 17, 2) normalized coords → (B, embed_dim).
    """

    def __init__(
        self,
        num_nodes: int = 17,
        in_dim: int = 2,
        gcn_hidden: int = 64,
        embed_dim: int = 50,
        mlp_hidden: int = 256,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.num_nodes = num_nodes
        self.in_dim = in_dim
        self.gcn_hidden = gcn_hidden
        self.embed_dim = embed_dim

        edges = coco17_skeleton_edges()
        c = adjacency_matrix(num_nodes, edges)
        c_hat = c + torch.eye(num_nodes, dtype=c.dtype, device=c.device)
        adj = symmetric_normalized_adjacency(c_hat)
        self.register_buffer("adj_norm", adj)

        self.lin1 = nn.Linear(in_dim, gcn_hidden)
        self.lin2 = nn.Linear(gcn_hidden, gcn_hidden)
        flat = num_nodes * gcn_hidden
        self.mlp = nn.Sequential(
            nn.Linear(flat, mlp_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, embed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, 17, 2) min–max normalized coordinates.
        """
        a = self.adj_norm
        h = torch.einsum("ij,bjf->bif", a, x)
        h = F.relu(self.lin1(h))
        h = torch.einsum("ij,bjf->bif", a, h)
        h = F.relu(self.lin2(h))
        h = h.reshape(h.size(0), -1)
        return self.mlp(h)


def cosine_distance(e1: torch.Tensor, e2: torch.Tensor) -> torch.Tensor:
    """Dc = 1 - cos(e1, e2) with L2-normalized embeddings (paper Eq. 5)."""
    e1 = F.normalize(e1, dim=-1, eps=1e-8)
    e2 = F.normalize(e2, dim=-1, eps=1e-8)
    cos = (e1 * e2).sum(dim=-1)
    return 1.0 - cos


class GCNSequenceExerciseNet(nn.Module):
    """
    Per-frame GCN on min–max normalized poses, mean-pool over time, then classification + quality.
    Same task heads as BiLSTM for comparable evaluation (exercise class + scalar quality).
    """

    def __init__(
        self,
        num_classes: int,
        gcn_hidden: int = 64,
        embed_dim: int = 50,
        head_dropout: float = 0.2,
    ):
        super().__init__()
        self.encoder = TopologyAwareGCNEncoder(
            gcn_hidden=gcn_hidden,
            embed_dim=embed_dim,
        )
        d = embed_dim
        self.drop = nn.Dropout(head_dropout)
        self.fc_cls = nn.Linear(d, num_classes)
        self.fc_reg = nn.Linear(d, 1)

    def forward(self, x: torch.Tensor):
        """
        x: (B, T, 17, 2) per-frame bbox-normalized COCO keypoints.
        """
        b, t, v, d = x.shape
        x = x.reshape(b * t, v, d)
        e = self.encoder(x)
        e = e.reshape(b, t, -1).mean(dim=1)
        h = self.drop(e)
        return self.fc_cls(h), self.fc_reg(h).squeeze(-1)


def contrastive_regression_loss(
    dc: torch.Tensor,
    y: torch.Tensor,
    margin: float = 1.35,
) -> torch.Tensor:
    """
    Eq. (4): 0.5 * Y * Dc^2 + 0.5 * (1-Y) * max(0, m - Dc)^2
    y: 1.0 similar poses, 0.0 dissimilar.
    """
    y = y.float()
    pos = 0.5 * y * (dc ** 2)
    neg = 0.5 * (1.0 - y) * (F.relu(margin - dc) ** 2)
    return (pos + neg).mean()
