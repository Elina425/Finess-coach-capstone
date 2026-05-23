"""
Matrix-LSTM (mLSTM) model for exercise classification on keypoint sequences.

This implementation follows the recommendation for residual stacked mLSTM
blocks that use multiplicative memory / matrix-style gating, with
bidirectional processing and global pooling.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple


class MLSTMCell(nn.Module):
    """Multiplicative LSTM cell for parallelizable matrix memory."""

    def __init__(self, input_size: int, hidden_size: int, bias: bool = True):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        # Standard LSTM-like gate projections
        self.input_proj = nn.Linear(input_size, 4 * hidden_size, bias=bias)
        self.hidden_proj = nn.Linear(hidden_size, 4 * hidden_size, bias=False)

        # Multiplicative modulation path: m_t = tanh(W_m x_t * U_m h_{t-1})
        self.mod_x = nn.Linear(input_size, hidden_size, bias=False)
        self.mod_h = nn.Linear(hidden_size, hidden_size, bias=False)

        self.layer_norm = nn.LayerNorm(hidden_size)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.input_proj.weight)
        nn.init.orthogonal_(self.hidden_proj.weight)
        nn.init.xavier_uniform_(self.mod_x.weight)
        nn.init.orthogonal_(self.mod_h.weight)
        if self.input_proj.bias is not None:
            nn.init.zeros_(self.input_proj.bias)

    def forward(
        self,
        x: torch.Tensor,
        h_prev: torch.Tensor,
        c_prev: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        gates = self.input_proj(x) + self.hidden_proj(h_prev)
        i_gate, f_gate, o_gate, g_gate = gates.chunk(4, dim=-1)

        mod = torch.tanh(self.mod_x(x) * self.mod_h(h_prev))

        i = torch.sigmoid(i_gate + mod)
        f = torch.sigmoid(f_gate + mod)
        o = torch.sigmoid(o_gate + mod)
        g = torch.tanh(g_gate)

        c_new = f * c_prev + i * g
        h_new = o * torch.tanh(self.layer_norm(c_new))
        return h_new, c_new


class ResidualMLSTM(nn.Module):
    """Residual stack of bidirectional mLSTM layers."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 4,
        dropout: float = 0.25,
        bidirectional: bool = True,
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.dropout_rate = dropout if num_layers > 1 else 0.0

        self.cells = nn.ModuleList()
        self.residual_projections = nn.ModuleList()

        for layer_idx in range(num_layers):
            layer_input_size = input_size if layer_idx == 0 else hidden_size * (2 if bidirectional else 1)

            forward_cell = MLSTMCell(input_size=layer_input_size, hidden_size=hidden_size)
            self.cells.append(forward_cell)
            if bidirectional:
                backward_cell = MLSTMCell(input_size=layer_input_size, hidden_size=hidden_size)
                self.cells.append(backward_cell)

            if layer_input_size != hidden_size * (2 if bidirectional else 1):
                self.residual_projections.append(
                    nn.Linear(layer_input_size, hidden_size * (2 if bidirectional else 1))
                )
            else:
                self.residual_projections.append(nn.Identity())

        self.dropouts = nn.ModuleList([
            nn.Dropout(self.dropout_rate) for _ in range(max(0, num_layers - 1))
        ]) if self.dropout_rate > 0 else None

    def forward(
        self,
        x: torch.Tensor,
        h_states: Optional[List[torch.Tensor]] = None,
        c_states: Optional[List[torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[List[torch.Tensor], List[torch.Tensor]]]:
        batch_size, seq_len, _ = x.shape
        device = x.device
        dtype = x.dtype

        num_dirs = 2 if self.bidirectional else 1
        total_cells = self.num_layers * num_dirs

        if h_states is None:
            h_states = [torch.zeros(batch_size, self.hidden_size, device=device, dtype=dtype) for _ in range(total_cells)]
        if c_states is None:
            c_states = [torch.zeros(batch_size, self.hidden_size, device=device, dtype=dtype) for _ in range(total_cells)]

        layer_input = x
        for layer_idx in range(self.num_layers):
            forward_idx = layer_idx * num_dirs
            backward_idx = forward_idx + 1 if self.bidirectional else None

            f_h, f_c = h_states[forward_idx], c_states[forward_idx]
            forward_outputs = []
            for t in range(seq_len):
                f_h, f_c = self.cells[forward_idx](layer_input[:, t, :], f_h, f_c)
                forward_outputs.append(f_h)
            forward_outputs = torch.stack(forward_outputs, dim=1)
            h_states[forward_idx] = f_h
            c_states[forward_idx] = f_c

            if self.bidirectional:
                b_h, b_c = h_states[backward_idx], c_states[backward_idx]
                backward_outputs = []
                for t in range(seq_len - 1, -1, -1):
                    b_h, b_c = self.cells[backward_idx](layer_input[:, t, :], b_h, b_c)
                    backward_outputs.append(b_h)
                backward_outputs.reverse()
                backward_outputs = torch.stack(backward_outputs, dim=1)
                h_states[backward_idx] = b_h
                c_states[backward_idx] = b_c

                layer_output = torch.cat([forward_outputs, backward_outputs], dim=-1)
            else:
                layer_output = forward_outputs

            residual = self.residual_projections[layer_idx](layer_input)
            layer_output = layer_output + residual

            if self.dropouts is not None and layer_idx < self.num_layers - 1:
                layer_output = self.dropouts[layer_idx](layer_output)

            layer_input = layer_output

        return layer_input, (h_states, c_states)


class MLSTMExerciseClassifier(nn.Module):
    """Exercise classifier using residual stacked mLSTM encoding."""

    def __init__(
        self,
        input_size: int,
        embed_dim: int = 256,
        hidden_size: int = 256,
        num_layers: int = 4,
        num_classes: int = 5,
        dropout: float = 0.25,
        bidirectional: bool = True,
    ):
        super().__init__()
        self.input_size = input_size
        self.embed_dim = embed_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.bidirectional = bidirectional

        self.input_norm = nn.LayerNorm(input_size)
        self.embedding = nn.Linear(input_size, embed_dim)
        self.embed_activation = nn.GELU()

        self.encoder = ResidualMLSTM(
            input_size=embed_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            bidirectional=bidirectional,
        )

        encoded_dim = hidden_size * (2 if bidirectional else 1)
        self.pool = nn.AdaptiveAvgPool1d(1)

        self.class_head = nn.Sequential(
            nn.Linear(encoded_dim, encoded_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(encoded_dim, num_classes),
        )

        self.quality_head = nn.Sequential(
            nn.Linear(encoded_dim, encoded_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(encoded_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.input_norm(x)
        x = self.embedding(x)
        x = self.embed_activation(x)

        encoded, _ = self.encoder(x)
        pooled = self.pool(encoded.transpose(1, 2)).squeeze(-1)

        class_logits = self.class_head(pooled)
        quality_scores = torch.sigmoid(self.quality_head(pooled)) * 5.0
        return class_logits, quality_scores

    def get_loss(
        self,
        logits: torch.Tensor,
        quality: torch.Tensor,
        labels: torch.Tensor,
        quality_targets: torch.Tensor,
        class_weight: float = 1.0,
        quality_weight: float = 0.5,
    ) -> torch.Tensor:
        ce = F.cross_entropy(logits, labels)
        mse = F.mse_loss(quality.squeeze(-1), quality_targets)
        return class_weight * ce + quality_weight * mse
