"""BiLSTM over annotation sequences (EgoExo interpretable judgements; no pose)."""

from __future__ import annotations

import torch
import torch.nn as nn


class ExerciseAnnotationBiLSTM(nn.Module):
    """
    Packed BiLSTM over (B, T, F) with variable lengths → classification + quality regression.
    """

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        hidden: int = 128,
        num_layers: int = 2,
        dropout: float = 0.35,
    ):
        super().__init__()
        self.hidden = hidden
        self.num_layers = num_layers
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

    def forward(self, x: torch.Tensor, lengths: torch.Tensor, text=None):
        # text unused (API symmetry with pose BiLSTM)
        packed = nn.utils.rnn.pack_padded_sequence(
            x,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        _, (hn, _) = self.lstm(packed)
        forward_h = hn[-2, :, :]
        backward_h = hn[-1, :, :]
        h = torch.cat([forward_h, backward_h], dim=-1)
        h = self.drop(h)
        return self.fc_cls(h), self.fc_reg(h).squeeze(-1)
