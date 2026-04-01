"""Shared BiLSTM architecture for exercise classification + quality regression."""

from __future__ import annotations

import torch
import torch.nn as nn


class ExerciseBiLSTM(nn.Module):
    """Shared BiLSTM trunk + classification head + quality regression head."""

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        hidden: int = 128,
        num_layers: int = 2,
        dropout: float = 0.35,  # Table 4 BiLSTM ~0.217 (Riccio, arXiv:2411.11548)
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim,
            hidden,
            num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        d = hidden * 2
        self.drop = nn.Dropout(dropout)
        self.fc_cls = nn.Linear(d, num_classes)
        self.fc_reg = nn.Linear(d, 1)

    def forward(self, x: torch.Tensor):
        o, _ = self.lstm(x)
        pooled = o.mean(dim=1)
        h = self.drop(pooled)
        logits = self.fc_cls(h)
        quality = self.fc_reg(h).squeeze(-1)
        return logits, quality
