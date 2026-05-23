"""BiLSTM architectures for exercise classification + optional quality regression."""

from __future__ import annotations

from typing import Any, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from fitness_coach.models.personalization import DoRALinear


def unpack_bilstm_outputs(out: Any) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """``(logits, quality)`` or ``(logits,)`` for classification-only checkpoints."""
    if isinstance(out, tuple):
        if len(out) == 1:
            return out[0], None
        return out[0], out[1]
    return out, None


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


class ExerciseBiLSTMCNN(nn.Module):
    """Paper-faithful BiLSTM-CNN classifier for PosePulse (Riccio §3.3).

    Pipeline on a ``(N, 30, 256)`` window of ViTPose-S embeddings:

    1. BiLSTM with ``bilstm_hidden`` units **per direction** (paper: 4 → 8 concatenated).
    2. Reshape ``(N, 30, 2·hidden)`` → ``(N, 1, 30, 2·hidden)`` — single-channel 2-D map.
    3. Three Conv2D + ReLU stages with filter counts ``(128, 256, 64)`` using ``3×3``
       kernels, ``padding=1``, ``stride=1`` (spatial dims preserved).
    4. Final Conv2D collapses the channel dim to 1.
    5. Flatten → Dropout → Linear → ``num_classes`` (CE handles the softmax).

    Classification-only — no quality head.
    """

    has_regression_head: bool = False

    def __init__(
        self,
        input_dim: int = 256,
        num_classes: int = 4,
        bilstm_hidden: int = 4,
        seq_len: int = 30,
        conv_filters: Tuple[int, int, int] = (128, 256, 64),
        dropout: float = 0.3,
    ):
        super().__init__()
        self.architecture = "bilstm_cnn"
        self.seq_len = int(seq_len)
        self.bilstm_hidden = int(bilstm_hidden)
        self._bidir_dim = 2 * self.bilstm_hidden

        self.bilstm = nn.LSTM(
            input_dim, bilstm_hidden, num_layers=1, batch_first=True, bidirectional=True,
        )

        c1, c2, c3 = conv_filters
        self.conv1   = nn.Conv2d(1,  c1, kernel_size=3, padding=1)
        self.conv2   = nn.Conv2d(c1, c2, kernel_size=3, padding=1)
        self.conv3   = nn.Conv2d(c2, c3, kernel_size=3, padding=1)
        self.conv_out = nn.Conv2d(c3, 1, kernel_size=3, padding=1)

        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.seq_len * self._bidir_dim, num_classes)

    def forward(self, x: torch.Tensor, text: Optional[torch.Tensor] = None):
        del text
        out, _ = self.bilstm(x)                # (N, T, 2*hidden)
        z = out.unsqueeze(1)                    # (N, 1, T, 2*hidden)
        z = F.relu(self.conv1(z))
        z = F.relu(self.conv2(z))
        z = F.relu(self.conv3(z))
        z = self.conv_out(z)                    # (N, 1, T, 2*hidden)
        z = z.flatten(start_dim=1)              # (N, T * 2*hidden)
        z = self.dropout(z)
        logits = self.classifier(z)
        return logits, None


class ExerciseBiLSTM(nn.Module):
    """
    Shared BiLSTM trunk + classification head + quality regression head.

    If ``text_dim > 0``, ``forward`` expects coaching embeddings concatenated with mean-pooled LSTM
    features. Pass ``text=None`` for zeros in the text subspace.
    """

    has_regression_head: bool = True

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

    has_regression_head: bool = True

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


class ExerciseBiLSTMPosePulseCnn(nn.Module):
    """
    PosePulse-style **BiLSTM → Conv2D** stack (paper description): bidirectional LSTM with a small
    per-direction hidden size, then three ReLU Conv2d refinements (128 → 256 → 64) and a 1-channel
    reduction before a linear classifier.

    Expects ``input_dim=42`` (8 angles + 34 normalized coords per frame) and fixed ``sequence_len``
    (default 30) for the temporal axis after the final conv.
    """

    has_regression_head: bool = True

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        *,
        sequence_len: int = 30,
        lstm_hidden_per_dir: int = 4,
        dropout: float = 0.25,
        dora_head_rank: int = 0,
        dora_head_alpha: float = 8.0,
    ):
        super().__init__()
        self.architecture = "posepulse_bilstm_cnn"
        self.sequence_len = int(sequence_len)
        self.lstm_hidden_per_dir = int(lstm_hidden_per_dir)
        self.dora_head_rank = int(dora_head_rank)
        self.dora_head_alpha = float(dora_head_alpha)
        self.lstm = nn.LSTM(
            input_dim,
            lstm_hidden_per_dir,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
            dropout=0.0,
        )
        lstm_out = lstm_hidden_per_dir * 2  # 8 when per_dir=4
        # (B, T, C) → (B, C, T, 1) for 2-D conv over (time × pseudo-width)
        self.conv1 = nn.Conv2d(lstm_out, 128, kernel_size=(3, 1), padding=(1, 0))
        self.conv2 = nn.Conv2d(128, 256, kernel_size=(3, 1), padding=(1, 0))
        self.conv3 = nn.Conv2d(256, 64, kernel_size=(3, 1), padding=(1, 0))
        self.conv4 = nn.Conv2d(64, 1, kernel_size=(3, 1), padding=(1, 0))
        self.drop = nn.Dropout(dropout)
        flat_dim = self.sequence_len  # (B, 1, T, 1) → (B, T)
        fc_cls_lin = nn.Linear(flat_dim, num_classes)
        fc_reg_lin = nn.Linear(flat_dim, 1)
        if self.dora_head_rank > 0:
            self.fc_cls = DoRALinear.from_linear(
                fc_cls_lin, self.dora_head_rank, self.dora_head_alpha
            )
            self.fc_reg = DoRALinear.from_linear(
                fc_reg_lin, self.dora_head_rank, self.dora_head_alpha
            )
        else:
            self.fc_cls = fc_cls_lin
            self.fc_reg = fc_reg_lin

    def forward(self, x: torch.Tensor, text: Optional[torch.Tensor] = None):
        if text is not None and text.numel() > 0:
            raise ValueError("ExerciseBiLSTMPosePulseCnn does not support text embeddings")
        o, _ = self.lstm(x)  # (B, T, 8)
        if o.size(1) != self.sequence_len:
            raise ValueError(
                f"Expected time dimension T={self.sequence_len}, got {o.size(1)} "
                "(train with --window matching this model)"
            )
        z = o.transpose(1, 2).unsqueeze(-1)  # (B, 8, T, 1)
        z = torch.relu(self.conv1(z))
        z = torch.relu(self.conv2(z))
        z = torch.relu(self.conv3(z))
        z = self.conv4(z)  # (B, 1, T, 1)
        z = z.squeeze(1).squeeze(-1)  # (B, T)
        z = self.drop(z)
        logits = self.fc_cls(z)
        quality = self.fc_reg(z).squeeze(-1)
        return logits, quality


class ExerciseBiLSTMPaperRiccioMapCnn(nn.Module):
    """
    Riccio / capstone **paper** BiLSTM–CNN: one BiLSTM layer with **4 units per direction** (8-D
    per frame), reshape to a **(30 × 8)** “map” with **one input channel** ``(N, 1, 30, 8)``,
    three **3×3** Conv2d stages (**128 → 256 → 64** channels) with **ReLU**, a final **1-channel**
    conv, then **flatten + linear** classifier. **No quality head** (classification-only).

    Optional **DoRA** (rank ``dora_head_rank``) wraps the final ``Linear`` only; LSTM internals
    stay standard ``nn.LSTM`` (full AdamW updates).
    """

    has_regression_head: bool = False

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        *,
        sequence_len: int = 30,
        lstm_hidden_per_dir: int = 4,
        dropout: float = 0.0,
        dora_head_rank: int = 0,
        dora_head_alpha: float = 8.0,
    ):
        super().__init__()
        self.architecture = "paper_riccio_bilstm_cnn"
        self.sequence_len = int(sequence_len)
        self.lstm_hidden_per_dir = int(lstm_hidden_per_dir)
        self.dora_head_rank = int(dora_head_rank)
        self.dora_head_alpha = float(dora_head_alpha)
        self.lstm = nn.LSTM(
            input_dim,
            lstm_hidden_per_dir,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
            dropout=0.0,
        )
        lstm_out = lstm_hidden_per_dir * 2  # 8
        # (B, 1, T, W) with T=time, W=8 hidden features (paper tensor layout).
        self.conv1 = nn.Conv2d(1, 128, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(256, 64, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(64, 1, kernel_size=3, padding=1)
        self.drop = nn.Dropout(dropout)
        flat_dim = self.sequence_len * lstm_out
        fc_lin = nn.Linear(flat_dim, num_classes)
        if self.dora_head_rank > 0:
            self.fc_cls = DoRALinear.from_linear(fc_lin, self.dora_head_rank, self.dora_head_alpha)
        else:
            self.fc_cls = fc_lin

    def forward(self, x: torch.Tensor, text: Optional[torch.Tensor] = None):
        if text is not None and text.numel() > 0:
            raise ValueError("ExerciseBiLSTMPaperRiccioMapCnn does not support text embeddings")
        o, _ = self.lstm(x)
        if o.size(1) != self.sequence_len:
            raise ValueError(
                f"Expected time dimension T={self.sequence_len}, got {o.size(1)} "
                "(train with --window matching this model)"
            )
        z = o.unsqueeze(1)
        z = F.relu(self.conv1(z))
        z = F.relu(self.conv2(z))
        z = F.relu(self.conv3(z))
        z = F.relu(self.conv4(z))
        z = z.flatten(1)
        z = self.drop(z)
        logits = self.fc_cls(z)
        return (logits,)


class ExerciseBiLSTMPosePulseDiagramNet(nn.Module):
    """
    BiLSTM–CNN stack aligned with the PosePulse diagram: **2-layer BiLSTM** (hidden 128 per direction),
    **256-d** per-frame embeddings, **Conv2d + BatchNorm + ReLU** (effective 3×1 over time×width-1),
    **stride-2** temporal downsample, **global average pooling**, **single classifier** (no quality head).

    Returns ``(logits,)`` only so training/export stay classification-focused and ONNX stays a single
    ``class_logits`` output for concise Netron graphs (cf. Riccio-style clarity).
    """

    has_regression_head: bool = False

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        *,
        sequence_len: int = 30,
        lstm_hidden: int = 128,
        lstm_layers: int = 2,
        lstm_dropout: float = 0.2,
    ):
        super().__init__()
        self.architecture = "posepulse_diagram"
        self.sequence_len = int(sequence_len)
        self.lstm_hidden = int(lstm_hidden)
        self.lstm_layers = int(lstm_layers)
        lstm_dropout = float(lstm_dropout)
        if self.lstm_layers < 2:
            lstm_dropout = 0.0
        self.bilstm = nn.LSTM(
            input_dim,
            self.lstm_hidden,
            num_layers=self.lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=lstm_dropout,
        )
        c_lstm = 2 * self.lstm_hidden  # 256
        # (B, T, C) → (B, C, T, 1): one spatial column so 3×1 acts along time (diagram “3×3” on time×features layout).
        self.temporal_cnn = nn.Sequential(
            nn.Conv2d(c_lstm, 128, kernel_size=(3, 1), stride=(1, 1), padding=(1, 0)),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=(3, 1), stride=(2, 1), padding=(1, 0)),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 64, kernel_size=(3, 1), stride=(1, 1), padding=(1, 0)),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(64, num_classes)

    def forward(self, x: torch.Tensor, text: Optional[torch.Tensor] = None):
        if text is not None and text.numel() > 0:
            raise ValueError("ExerciseBiLSTMPosePulseDiagramNet does not support text embeddings")
        o, _ = self.bilstm(x)
        if o.size(1) != self.sequence_len:
            raise ValueError(
                f"Expected time dimension T={self.sequence_len}, got {o.size(1)} "
                "(train with --window matching this model)"
            )
        z = o.transpose(1, 2).unsqueeze(-1)
        z = self.temporal_cnn(z)
        z = self.pool(z).flatten(1)
        logits = self.classifier(z)
        return (logits,)


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
    sequence_len: int = 30,
    dora_head_rank: int = 0,
    dora_head_alpha: float = 8.0,
) -> Union[
    ExerciseBiLSTM,
    ExerciseBiLSTMCnnAttention,
    ExerciseBiLSTMPosePulseCnn,
    ExerciseBiLSTMPaperRiccioMapCnn,
    ExerciseBiLSTMPosePulseDiagramNet,
]:
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
    if arch in ("posepulse_bilstm_cnn", "posepulse", "bilstm_cnn_posepulse"):
        if int(text_dim) != 0:
            raise ValueError("posepulse_bilstm_cnn does not support text_dim > 0")
        return ExerciseBiLSTMPosePulseCnn(
            input_dim=input_dim,
            num_classes=num_classes,
            sequence_len=int(sequence_len),
            lstm_hidden_per_dir=4,
            dropout=float(dropout),
            dora_head_rank=int(dora_head_rank),
            dora_head_alpha=float(dora_head_alpha),
        )
    if arch in ("posepulse_diagram", "posepulse_diagram_cnn"):
        if int(text_dim) != 0:
            raise ValueError("posepulse_diagram does not support text_dim > 0")
        return ExerciseBiLSTMPosePulseDiagramNet(
            input_dim=input_dim,
            num_classes=num_classes,
            sequence_len=int(sequence_len),
            lstm_dropout=float(dropout),
        )
    if arch in ("paper_riccio_bilstm_cnn", "paper_riccio", "riccio_paper_bilstm"):
        if int(text_dim) != 0:
            raise ValueError("paper_riccio_bilstm_cnn does not support text_dim > 0")
        return ExerciseBiLSTMPaperRiccioMapCnn(
            input_dim=input_dim,
            num_classes=num_classes,
            sequence_len=int(sequence_len),
            lstm_hidden_per_dir=4,
            dropout=float(dropout),
            dora_head_rank=int(dora_head_rank),
            dora_head_alpha=float(dora_head_alpha),
        )
    raise ValueError(
        f"Unknown architecture {architecture!r}; use 'plain', 'cnn_attn', 'posepulse_bilstm_cnn', "
        "'posepulse_diagram', or 'paper_riccio_bilstm_cnn'."
    )


def build_exercise_bilstm_from_checkpoint(
    ckpt: dict,
) -> Union[
    ExerciseBiLSTM,
    ExerciseBiLSTMCnnAttention,
    ExerciseBiLSTMPosePulseCnn,
    ExerciseBiLSTMPaperRiccioMapCnn,
    ExerciseBiLSTMPosePulseDiagramNet,
]:
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
        sequence_len=int(ckpt.get("window", ckpt.get("posepulse_sequence_len", 30))),
        dora_head_rank=int(ckpt.get("dora_head_rank", 0) or 0),
        dora_head_alpha=float(ckpt.get("dora_head_alpha", 8.0) or 8.0),
    )
    return build_exercise_bilstm(**kw)
