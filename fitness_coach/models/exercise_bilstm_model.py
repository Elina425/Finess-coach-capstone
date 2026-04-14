"""BiLSTM architectures for exercise classification + optional quality regression."""

from __future__ import annotations

from typing import Optional, Tuple, Union

import torch
import torch.nn as nn


def _num_heads_for_embed(embed_dim: int, prefer: int = 4) -> int:
    """Pick a valid ``num_heads`` for MultiheadAttention (embed_dim divisible by num_heads)."""
    for h in (prefer, 3, 2, 1):
        if embed_dim % h == 0:
            return h
    return 1


class DilatedTemporalCNN(nn.Module):
    """
1D convolutions over time with increasing dilation (receptive field1, 2, 4, …) for multi-scale
    temporal motion before the BiLSTM.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        dropout: float,
        dilations: Tuple[int, ...] = (1, 2, 4),
    ):
        super().__init__()
        layers: list[nn.Module] = []
        c_in = in_channels
        for i, d in enumerate(dilations):
            c_out = hidden_channels
            k = 3
            pad = d * (k - 1) // 2
            layers.append(nn.Conv1d(c_in, c_out, kernel_size=k, padding=pad, dilation=d))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout(dropout))
            c_in = c_out
        self.net = nn.Sequential(*layers)
        self.out_channels = c_in

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, F) -> (B, F, T) for Conv1d
        x = x.transpose(1, 2)
        x = self.net(x)
        return x.transpose(1, 2)


class ExerciseBiLSTM(nn.Module):
    """
    Shared BiLSTM trunk + classification head + quality regression head.

    If ``text_dim > 0``, ``forward`` expects coaching embeddings concatenated with mean-pooled LSTM
    features. Pass ``text=None`` for zeros in the text subspace.
    """

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        hidden: int = 128,
        num_layers: int = 2,
        dropout: float = 0.35,
        text_dim: int = 0,
    ):
        super().__init__()
        self.text_dim = int(text_dim)
        self.architecture = "plain"
        self.lstm = nn.LSTM(
            input_dim,
            hidden,
            num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        d_lstm = hidden * 2
        d_head = d_lstm + self.text_dim
        self.drop = nn.Dropout(dropout)
        self.fc_cls = nn.Linear(d_head, num_classes)
        self.fc_reg = nn.Linear(d_head, 1)

    def forward(self, x: torch.Tensor, text: Optional[torch.Tensor] = None):
        o, _ = self.lstm(x)
        pooled = o.mean(dim=1)
        if self.text_dim > 0:
            if text is None:
                text = pooled.new_zeros(pooled.size(0), self.text_dim)
            else:
                if text.dim() != 2 or text.size(0) != pooled.size(0) or text.size(1) != self.text_dim:
                    raise ValueError(
                        f"text must be (batch, {self.text_dim}), got {tuple(text.shape)}"
                    )
            h = torch.cat([pooled, text], dim=-1)
        else:
            h = pooled
        h = self.drop(h)
        logits = self.fc_cls(h)
        quality = self.fc_reg(h).squeeze(-1)
        return logits, quality


class ExerciseBiLSTMCnnAttention(nn.Module):
    """
    Dilated temporal CNN front-end → BiLSTM → multi-head self-attention (residual + LayerNorm)
    → mean pool → classification + regression heads.

    Self-attention lets the model re-weight time steps before pooling (complements mean pooling in
    the plain BiLSTM). CNN front-end expands receptive field over motion features.
    """

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        hidden: int = 128,
        num_layers: int = 2,
        dropout: float = 0.35,
        text_dim: int = 0,
        *,
        cnn_hidden: int = 64,
        attn_heads: int = 4,
        dilations: Tuple[int, ...] = (1, 2, 4),
    ):
        super().__init__()
        self.text_dim = int(text_dim)
        self.architecture = "cnn_attn"
        self.cnn_hidden = int(cnn_hidden)
        self.attn_heads_requested = int(attn_heads)

        self.front = DilatedTemporalCNN(
            input_dim, cnn_hidden, dropout, dilations=dilations
        )
        lstm_in = self.front.out_channels
        self.lstm = nn.LSTM(
            lstm_in,
            hidden,
            num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        d_lstm = hidden * 2
        n_head = _num_heads_for_embed(d_lstm, attn_heads)
        self.attn_heads = n_head
        self.mha = nn.MultiheadAttention(
            d_lstm,
            num_heads=n_head,
            dropout=min(0.2, dropout),
            batch_first=True,
        )
        self.norm = nn.LayerNorm(d_lstm)
        d_head = d_lstm + self.text_dim
        self.drop = nn.Dropout(dropout)
        self.fc_cls = nn.Linear(d_head, num_classes)
        self.fc_reg = nn.Linear(d_head, 1)

    def forward(self, x: torch.Tensor, text: Optional[torch.Tensor] = None):
        z = self.front(x)
        o, _ = self.lstm(z)
        attn_out, _ = self.mha(o, o, o, need_weights=False)
        h_seq = self.norm(o + attn_out)
        pooled = h_seq.mean(dim=1)
        if self.text_dim > 0:
            if text is None:
                text = pooled.new_zeros(pooled.size(0), self.text_dim)
            else:
                if text.dim() != 2 or text.size(0) != pooled.size(0) or text.size(1) != self.text_dim:
                    raise ValueError(
                        f"text must be (batch, {self.text_dim}), got {tuple(text.shape)}"
                    )
            h = torch.cat([pooled, text], dim=-1)
        else:
            h = pooled
        h = self.drop(h)
        logits = self.fc_cls(h)
        quality = self.fc_reg(h).squeeze(-1)
        return logits, quality


def build_exercise_bilstm(
    architecture: str,
    input_dim: int,
    num_classes: int,
    hidden: int,
    num_layers: int,
    dropout: float,
    text_dim: int = 0,
    *,
    cnn_hidden: int = 64,
    attn_heads: int = 4,
    dilations: Tuple[int, ...] = (1, 2, 4),
) -> Union[ExerciseBiLSTM, ExerciseBiLSTMCnnAttention]:
    arch = (architecture or "plain").strip().lower()
    if arch in ("plain", "bilstm"):
        return ExerciseBiLSTM(
            input_dim=input_dim,
            num_classes=num_classes,
            hidden=hidden,
            num_layers=num_layers,
            dropout=dropout,
            text_dim=text_dim,
        )
    if arch in ("cnn_attn", "cnn-attention", "enhanced"):
        return ExerciseBiLSTMCnnAttention(
            input_dim=input_dim,
            num_classes=num_classes,
            hidden=hidden,
            num_layers=num_layers,
            dropout=dropout,
            text_dim=text_dim,
            cnn_hidden=cnn_hidden,
            attn_heads=attn_heads,
            dilations=dilations,
        )
    raise ValueError(
        f"Unknown architecture {architecture!r}; use 'plain' or 'cnn_attn'."
    )


def build_exercise_bilstm_from_checkpoint(ckpt: dict) -> Union[ExerciseBiLSTM, ExerciseBiLSTMCnnAttention]:
    """Restore the same architecture as saved in ``ckpt`` (defaults to ``plain`` for older runs)."""
    arch = str(ckpt.get("architecture", "plain")).strip().lower()
    text_dim = int(ckpt.get("text_dim", 0))
    ch = int(ckpt.get("cnn_hidden", 0) or 0)
    if arch == "cnn_attn" and ch <= 0:
        ch = 64
    kw = dict(
        architecture=arch,
        input_dim=int(ckpt.get("feat_dim", 8)),
        num_classes=int(ckpt["num_classes"]),
        hidden=int(ckpt.get("hidden", 128)),
        num_layers=int(ckpt.get("layers", 2)),
        dropout=float(ckpt.get("dropout", 0.35)),
        text_dim=text_dim,
        cnn_hidden=ch if ch > 0 else 64,
        attn_heads=int(ckpt.get("attn_heads", 4) or 4),
    )
    return build_exercise_bilstm(**kw)
