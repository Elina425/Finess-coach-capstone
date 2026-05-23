"""
Paper-aligned xLSTM sequence model for exercise analysis.

This implementation follows the sLSTM/xLSTM block structure described by Beck et al. (2024):
https://arxiv.org/abs/2405.04517

Key paper-aligned choices implemented here:
- residual pre-LayerNorm backbone
- **mLSTM** (matrix memory, head-wise ``C_t``, ``n_t``, exponential gates) and **sLSTM** (scalar memory) mixers
- configurable block stack, e.g. ``"mmmmmmms"`` for Beck-style **xLSTM[7:1]** (7 mLSTM + 1 sLSTM over 8 blocks)
- sLSTM: head-wise block-diagonal projections and hidden-hidden recurrence
- mLSTM: queries/keys/values from the current frame; stabilised normaliser in the denominator
- optional causal depthwise convolution before sequence mixing
- head-wise GroupNorm after recurrent mixing
- gated MLP after each block (sLSTM post-style; mLSTM block uses the same MLP wrapper for parity with this codebase)

The public `xLSTMExerciseClassifier` API stays compatible with the repo's existing training scripts.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _bucket_interval_centres(
    bucket_edges: Tuple[float, ...],
    *,
    domain_lo: float = 0.0,
    domain_hi: float = 1.0,
) -> Tuple[float, ...]:
    """Midpoints of buckets cut into ``[domain_lo, domain_hi]`` by ``bucket_edges``."""
    edges = tuple(float(e) for e in bucket_edges)
    dl, dh = float(domain_lo), float(domain_hi)
    cuts = [dl] + list(edges) + [dh]
    return tuple(0.5 * (cuts[i] + cuts[i + 1]) for i in range(len(cuts) - 1))


def _resolve_num_heads(hidden_size: int, requested_heads: int) -> int:
    requested_heads = max(1, int(requested_heads))
    if hidden_size % requested_heads == 0:
        return requested_heads
    for candidate in range(requested_heads, 0, -1):
        if hidden_size % candidate == 0:
            return candidate
    return 1


class BlockDiagonalLinear(nn.Module):
    """Head-wise linear layer used for xLSTM's block-diagonal projections."""

    def __init__(self, in_features: int, out_features: int, num_heads: int, bias: bool = True):
        super().__init__()
        if in_features % num_heads != 0 or out_features % num_heads != 0:
            raise ValueError("in_features and out_features must be divisible by num_heads")
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.num_heads = int(num_heads)
        self.in_head = self.in_features // self.num_heads
        self.out_head = self.out_features // self.num_heads
        self.weight = nn.Parameter(torch.empty(self.num_heads, self.out_head, self.in_head))
        self.bias = nn.Parameter(torch.zeros(self.out_features)) if bias else None
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for h in range(self.num_heads):
            nn.init.xavier_uniform_(self.weight[h])
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_shape = x.shape[:-1]
        xh = x.reshape(-1, self.num_heads, self.in_head)
        yh = torch.einsum("bhi,hoi->bho", xh, self.weight)
        y = yh.reshape(*orig_shape, self.out_features)
        if self.bias is not None:
            y = y + self.bias
        return y


class CausalDepthwiseConv1d(nn.Module):
    """Dimension-wise causal convolution used in paper xLSTM blocks."""

    def __init__(self, dim: int, kernel_size: int = 4):
        super().__init__()
        self.kernel_size = int(max(1, kernel_size))
        self.conv = nn.Conv1d(
            dim,
            dim,
            kernel_size=self.kernel_size,
            groups=dim,
            bias=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D)
        xt = x.transpose(1, 2)
        pad = self.kernel_size - 1
        xt = F.pad(xt, (pad, 0))
        yt = self.conv(xt)
        return yt.transpose(1, 2)


class GatedMLP(nn.Module):
    """Post up-projection gated MLP with GELU, as in the sLSTM block."""

    def __init__(self, dim: int, projection_factor: float = 4.0 / 3.0, dropout: float = 0.0):
        super().__init__()
        inner = max(dim, int(round(dim * float(projection_factor))))
        self.up = nn.Linear(dim, inner)
        self.gate = nn.Linear(dim, inner)
        self.down = nn.Linear(inner, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        up = F.gelu(self.up(x))
        gate = torch.sigmoid(self.gate(x))
        return self.down(self.dropout(up * gate))


class sLSTMCell(nn.Module):
    """Scalar-memory xLSTM cell with exponential gating and stabilizer state."""

    def __init__(self, dim: int, num_heads: int):
        super().__init__()
        self.dim = int(dim)
        self.num_heads = int(num_heads)
        self.in_i = BlockDiagonalLinear(dim, dim, num_heads)
        self.in_f = BlockDiagonalLinear(dim, dim, num_heads)
        self.in_o = BlockDiagonalLinear(dim, dim, num_heads)
        self.in_z = BlockDiagonalLinear(dim, dim, num_heads)
        self.rec_i = BlockDiagonalLinear(dim, dim, num_heads, bias=False)
        self.rec_f = BlockDiagonalLinear(dim, dim, num_heads, bias=False)
        self.rec_o = BlockDiagonalLinear(dim, dim, num_heads, bias=False)
        self.rec_z = BlockDiagonalLinear(dim, dim, num_heads, bias=False)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for module in (self.in_i, self.in_f, self.in_o, self.in_z, self.rec_i, self.rec_f, self.rec_o, self.rec_z):
            module.reset_parameters()
        if self.in_f.bias is not None:
            nn.init.constant_(self.in_f.bias, 1.0)

    def forward(
        self,
        x_t: torch.Tensor,
        h_prev: torch.Tensor,
        c_prev: torch.Tensor,
        n_prev: torch.Tensor,
        m_prev: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        z_t = torch.tanh(self.in_z(x_t) + self.rec_z(h_prev))
        i_tilde = self.in_i(x_t) + self.rec_i(h_prev)
        f_tilde = self.in_f(x_t) + self.rec_f(h_prev)
        o_t = torch.sigmoid(self.in_o(x_t) + self.rec_o(h_prev))

        m_t = torch.maximum(f_tilde + m_prev, i_tilde)
        i_t = torch.exp(i_tilde - m_t)
        f_t = torch.exp(f_tilde + m_prev - m_t)

        c_t = f_t * c_prev + i_t * z_t
        n_t = f_t * n_prev + i_t
        h_t = o_t * (c_t / n_t.clamp_min(1e-6))
        return h_t, c_t, n_t, m_t


class sLSTMSequenceLayer(nn.Module):
    """Sequential sLSTM mixing over a full sequence."""

    def __init__(self, dim: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        self.dim = int(dim)
        self.num_heads = int(num_heads)
        self.cell = sLSTMCell(dim, num_heads)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq_len, dim = x.shape
        h_t = x.new_zeros(batch, dim)
        c_t = x.new_zeros(batch, dim)
        n_t = x.new_zeros(batch, dim)
        m_t = x.new_zeros(batch, dim)
        outputs = []
        for step in range(seq_len):
            h_t, c_t, n_t, m_t = self.cell(x[:, step, :], h_t, c_t, n_t, m_t)
            outputs.append(self.dropout(h_t))
        return torch.stack(outputs, dim=1)


class sLSTMBlock(nn.Module):
    """
    Residual sLSTM block in the paper's post up-projection style.

    Structure:
      pre-LN -> optional causal conv + swish -> sLSTM -> head-wise GroupNorm -> gated MLP -> residual
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        dropout: float = 0.1,
        conv_kernel_size: int = 4,
        projection_factor: float = 4.0 / 3.0,
    ):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.causal_conv = CausalDepthwiseConv1d(dim, kernel_size=conv_kernel_size)
        self.sequence_mixer = sLSTMSequenceLayer(dim, num_heads=num_heads, dropout=dropout)
        self.group_norm = nn.GroupNorm(num_groups=num_heads, num_channels=dim)
        self.mlp = GatedMLP(dim, projection_factor=projection_factor, dropout=dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        y = self.norm(x)
        y = F.silu(self.causal_conv(y))
        y = self.sequence_mixer(y)
        y = self.group_norm(y.transpose(1, 2)).transpose(1, 2)
        y = self.mlp(y)
        y = self.dropout(y)
        return residual + y


class MLSTMSequenceLayer(nn.Module):
    """
    Sequential matrix-memory mLSTM (Beck et al., 2024) over ``(B, T, D)``.

    Memory is **head-wise**: each of ``num_heads`` heads keeps ``C ∈ R^{d_h × d_h}``,
    ``n ∈ R^{d_h}``, with ``d_h = D // num_heads``, matching the block-diagonal spirit of the paper
    while keeping ``O(D^2 / H)`` state per step tractable.

    Per time step ``t`` (with stabiliser ``m`` for exponential gates, same construction as sLSTM):

    - ``C_t = f_t ⊙ C_{t-1} + i_t (v_t k_t^T)``
    - ``n_t = f_t ⊙ n_{t-1} + i_t ⊙ k_t``
    - ``\\tilde{h}_t = (C_t q_t) / \\max(|n_t^T q_t|, 1)``
    - ``h_t = o_t ⊙ \\tilde{h}_t``
    """

    def __init__(self, dim: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        self.dim = int(dim)
        self.num_heads = int(num_heads)
        if self.dim % self.num_heads != 0:
            raise ValueError(f"dim {self.dim} must be divisible by num_heads {self.num_heads}")
        self.dh = self.dim // self.num_heads
        self.proj_q = nn.Linear(self.dim, self.dim)
        self.proj_k = nn.Linear(self.dim, self.dim)
        self.proj_v = nn.Linear(self.dim, self.dim)
        self.proj_if = nn.Linear(self.dim, 2 * self.dim)
        self.proj_o = nn.Linear(self.dim, self.dim)
        self.dropout = nn.Dropout(dropout)
        self._scale = 1.0 / math.sqrt(float(self.dh))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq_len, dim = x.shape
        h = self.num_heads
        dh = self.dh
        c_state = x.new_zeros(batch, h, dh, dh)
        n_state = x.new_zeros(batch, h, dh)
        m_state = x.new_zeros(batch, h, dh)
        outputs: list[torch.Tensor] = []
        for step in range(seq_len):
            xt = x[:, step, :]
            q = self.proj_q(xt).view(batch, h, dh)
            k = self.proj_k(xt).view(batch, h, dh) * self._scale
            v = self.proj_v(xt).view(batch, h, dh)
            if_pair = self.proj_if(xt).view(batch, h, dh, 2)
            i_tilde, f_tilde = if_pair[..., 0], if_pair[..., 1]
            o = torch.sigmoid(self.proj_o(xt)).view(batch, h, dh)

            m_new = torch.maximum(f_tilde + m_state, i_tilde)
            i_gate = torch.exp(i_tilde - m_new)
            f_gate = torch.exp(f_tilde + m_state - m_new)
            m_state = m_new

            outer = v.unsqueeze(-1) * k.unsqueeze(-2)
            c_state = c_state * f_gate.unsqueeze(-1) + i_gate.unsqueeze(-1) * outer
            n_state = n_state * f_gate + i_gate * k

            cq = torch.einsum("bhde,bhe->bhd", c_state, q)
            dot = (n_state * q).sum(dim=-1, keepdim=True)
            den = dot.abs().clamp(min=1.0)
            h_t = o * (cq / den)
            outputs.append(self.dropout(h_t.reshape(batch, dim)))
        return torch.stack(outputs, dim=1)


class mLSTMBlock(nn.Module):
    """
    Residual mLSTM block (matrix-memory mixer), same outer wrapper as ``sLSTMBlock`` in this repo.

    pre-LN → causal conv (SiLU) → mLSTM → GroupNorm → gated MLP → dropout → residual
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        dropout: float = 0.1,
        conv_kernel_size: int = 4,
        projection_factor: float = 4.0 / 3.0,
    ):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.causal_conv = CausalDepthwiseConv1d(dim, kernel_size=conv_kernel_size)
        self.sequence_mixer = MLSTMSequenceLayer(dim, num_heads=num_heads, dropout=dropout)
        self.group_norm = nn.GroupNorm(num_groups=num_heads, num_channels=dim)
        self.mlp = GatedMLP(dim, projection_factor=projection_factor, dropout=dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        y = self.norm(x)
        y = F.silu(self.causal_conv(y))
        y = self.sequence_mixer(y)
        y = self.group_norm(y.transpose(1, 2)).transpose(1, 2)
        y = self.mlp(y)
        y = self.dropout(y)
        return residual + y


def _parse_block_pattern(pattern: str) -> list[str]:
    p = pattern.strip().lower()
    if not p:
        raise ValueError("block_pattern must be non-empty")
    for ch in p:
        if ch not in ("m", "s"):
            raise ValueError(f"block_pattern must contain only 'm' or 's', got {pattern!r}")
    return list(p)


class xLSTM(nn.Module):
    """Residual xLSTM stack: interleaved **mLSTM** and **sLSTM** blocks (Beck et al., 2024).

    Pass ``block_pattern`` as a string of ``'m'`` / ``'s'`` characters, e.g. ``"mmmmmmms"`` for
    **xLSTM[7:1]** (seven matrix-memory blocks and one scalar-memory block). If ``block_pattern``
    is ``None``, the stack is ``num_layers`` homogeneous **sLSTM** blocks (legacy behaviour).
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 2,
        dropout: float = 0.1,
        bidirectional: bool = True,  # kept for repo compatibility; xLSTM itself is causal/recurrent
        num_heads: int = 4,
        conv_kernel_size: int = 4,
        projection_factor: float = 4.0 / 3.0,
        block_pattern: Optional[str] = None,
    ):
        super().__init__()
        del bidirectional
        self.input_size = int(input_size)
        self.hidden_size = int(hidden_size)
        self.num_heads = _resolve_num_heads(hidden_size, num_heads)
        self.block_pattern = block_pattern
        if block_pattern is not None:
            types = _parse_block_pattern(block_pattern)
            self._depth = len(types)
        else:
            types = ["s"] * int(num_layers)
            self._depth = len(types)
        self.num_layers = self._depth
        self.input_proj = nn.Linear(self.input_size, self.hidden_size)
        self.blocks = nn.ModuleList()
        for bt in types:
            if bt == "m":
                self.blocks.append(
                    mLSTMBlock(
                        dim=self.hidden_size,
                        num_heads=self.num_heads,
                        dropout=dropout,
                        conv_kernel_size=conv_kernel_size,
                        projection_factor=projection_factor,
                    )
                )
            else:
                self.blocks.append(
                    sLSTMBlock(
                        dim=self.hidden_size,
                        num_heads=self.num_heads,
                        dropout=dropout,
                        conv_kernel_size=conv_kernel_size,
                        projection_factor=projection_factor,
                    )
                )
        self.final_norm = nn.LayerNorm(self.hidden_size)

    def forward(self, x: torch.Tensor, *_, **__) -> Tuple[torch.Tensor, Tuple[None, None]]:
        y = self.input_proj(x)
        for block in self.blocks:
            y = block(y)
        y = self.final_norm(y)
        return y, (None, None)


class MultiTaskFusion(nn.Module):
    """Two PLN → Linear branches with task-specific routing.

    Realises the whiteboard architecture: Branch A is the *only* path to the
    classification head and Branch B is the *only* path to the quality head, so
    each branch is forced to specialise on its task by virtue of gradient flow.
    The sum ``z = a + b`` carries both specialised signals and is fed to the
    comment head — giving the language model access to the class-discriminative
    and quality-discriminative information that the two branches just extracted.

    ``forward`` returns the triple ``(a, b, z)`` so the caller can route each
    output to the appropriate head:

        a → classification head
        b → quality head
        z → comment head
    """

    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.1):
        super().__init__()
        self.branch_a = nn.Sequential(
            nn.LayerNorm(in_dim), nn.Linear(in_dim, out_dim), nn.GELU(),
        )
        self.branch_b = nn.Sequential(
            nn.LayerNorm(in_dim), nn.Linear(in_dim, out_dim), nn.GELU(),
        )
        self.dropout = nn.Dropout(dropout)
        self.out_dim = int(out_dim)

    def forward(self, pooled: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        a = self.branch_a(pooled)
        b = self.branch_b(pooled)
        z = self.dropout(a + b)
        return a, b, z


class AttentionPooling(nn.Module):
    """Single-head additive attention pooling over the time axis.

    Replaces global average pooling with a soft-attention readout: the model learns
    a per-step scalar score, softmaxes it across time, and returns the weighted sum.
    Empirically this gives a small but reliable lift on action-clip recognition
    versus plain GAP because not all frames carry the same amount of evidence.
    """

    def __init__(self, dim: int, hidden: int = 64, dropout: float = 0.0):
        super().__init__()
        self.score = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, seq: torch.Tensor) -> torch.Tensor:
        # seq: (B, T, D)
        scores = self.score(seq).squeeze(-1)  # (B, T)
        weights = torch.softmax(scores, dim=1).unsqueeze(-1)  # (B, T, 1)
        return (seq * weights).sum(dim=1)  # (B, D)


class CommentGenerationHead(nn.Module):
    """Frozen Flan-T5-small comment generator (Approach B, hybrid).

    The model conditions on **two** complementary signals:

    1. **Soft prefix** — a small learnable projection ``ℝ²⁵⁶ → ℝ^{n_prefix·d_model}`` that
       turns the xLSTM's pooled clip embedding into ``n_prefix`` "soft tokens" prepended
       to the T5 encoder input. This is the only trainable component inside the head.
    2. **Hard text prompt** — built at train/inference time from the **predicted error
       tags** + exercise name, e.g.
       ``"Exercise: kneeling pushing-ups. Detected errors: back_not_straight,
       insufficient_depth. Write a short coaching tip to fix the form."``
       During training we use the *gold* error tags so the comment loss does not
       cascade through a noisy classifier; at inference we switch to predicted tags.

    The T5 weights are kept frozen — on EgoExo's ~1k clips, fine-tuning the LM would
    overfit hard. Trainable param count for ``flan-t5-small`` (n_prefix=16) is
    ``256 * 16 * 512 ≈ 2.1 M``, which fits the data scale.
    """

    def __init__(
        self,
        encoder_dim: int,
        error_tags: Sequence[str],
        model_name: str = "google/flan-t5-small",
        n_prefix_tokens: int = 16,
        max_target_len: int = 48,
        max_prompt_len: int = 96,
    ):
        super().__init__()
        try:
            from transformers import T5ForConditionalGeneration, T5Tokenizer  # type: ignore
        except ImportError as exc:  # pragma: no cover - import-time guard
            raise ImportError(
                "CommentGenerationHead requires `transformers` and `sentencepiece`. "
                "Install with: pip install 'transformers>=4.30' sentencepiece"
            ) from exc

        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        self.lm = T5ForConditionalGeneration.from_pretrained(model_name)
        for p in self.lm.parameters():
            p.requires_grad_(False)
        self.lm.eval()

        self.d_model = int(self.lm.config.d_model)
        self.n_prefix = int(n_prefix_tokens)
        self.max_target_len = int(max_target_len)
        self.max_prompt_len = int(max_prompt_len)
        self.error_tags = tuple(error_tags)

        # Trainable: project the pooled embedding to a stack of soft prefix tokens.
        self.prefix_proj = nn.Sequential(
            nn.LayerNorm(encoder_dim),
            nn.Linear(encoder_dim, self.n_prefix * self.d_model),
        )

    def train(self, mode: bool = True):  # type: ignore[override]
        super().train(mode)
        # Keep the LM frozen + in eval mode (disables dropout) regardless of outer state.
        self.lm.eval()
        return self

    def parameters(self, recurse: bool = True):
        # Yield only the trainable prefix projector — handy when building optimizers.
        return self.prefix_proj.parameters(recurse=recurse)

    def all_parameters(self, recurse: bool = True):
        """Including the frozen LM (rarely needed)."""
        return super().parameters(recurse=recurse)

    # ------------------------------------------------------------------ helpers

    def _build_prompts(
        self,
        error_targets: torch.Tensor,
        class_names: Sequence[str],
        threshold: float = 0.5,
    ) -> List[str]:
        """Convert (B, K) error vectors + class names into one text prompt per sample.

        ``error_targets`` may be either binary {0,1} (gold) or sigmoid probabilities
        (predicted). A 0.5 threshold is applied to the latter.
        """
        prompts: List[str] = []
        if error_targets.dtype != torch.bool:
            mask = (error_targets >= threshold).cpu().numpy()
        else:
            mask = error_targets.cpu().numpy()
        for i in range(mask.shape[0]):
            tags = [self.error_tags[j].replace("_", " ") for j in range(mask.shape[1]) if bool(mask[i, j])]
            tag_str = ", ".join(tags) if tags else "form looks correct"
            cls = (class_names[i] or "exercise").strip().lower() or "exercise"
            prompts.append(
                f"Exercise: {cls}. Detected errors: {tag_str}. "
                "Write a short coaching tip to correct the form."
            )
        return prompts

    def _tokenize(self, texts: Sequence[str], max_length: int, device: torch.device):
        enc = self.tokenizer(
            list(texts),
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        return enc["input_ids"].to(device), enc["attention_mask"].to(device)

    def _encoder_inputs(
        self,
        pooled_emb: torch.Tensor,
        prompts: Sequence[str],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        device = pooled_emb.device
        b = pooled_emb.size(0)
        prompt_ids, prompt_mask = self._tokenize(prompts, self.max_prompt_len, device)
        # Hard prompt token embeddings (frozen embedding table).
        with torch.no_grad():
            prompt_emb = self.lm.get_input_embeddings()(prompt_ids)
        # Soft prefix from the xLSTM clip embedding (trainable).
        soft = self.prefix_proj(pooled_emb).view(b, self.n_prefix, self.d_model)
        inputs_embeds = torch.cat([soft, prompt_emb], dim=1)
        soft_mask = torch.ones(b, self.n_prefix, dtype=prompt_mask.dtype, device=device)
        attn_mask = torch.cat([soft_mask, prompt_mask], dim=1)
        return inputs_embeds, attn_mask

    # --------------------------------------------------------------------- API

    def compute_loss(
        self,
        pooled_emb: torch.Tensor,
        gold_error_targets: torch.Tensor,
        class_names: Sequence[str],
        target_comments: Sequence[str],
    ) -> torch.Tensor:
        """Teacher-forced cross-entropy on gold comment tokens.

        Skips samples with empty comments (returns 0-loss in that case).
        """
        device = pooled_emb.device
        keep = [i for i, c in enumerate(target_comments) if c and c.strip()]
        if not keep:
            return pooled_emb.new_zeros(())
        if len(keep) != pooled_emb.size(0):
            pooled_emb = pooled_emb[keep]
            gold_error_targets = gold_error_targets[keep]
            class_names = [class_names[i] for i in keep]
            target_comments = [target_comments[i] for i in keep]

        prompts = self._build_prompts(gold_error_targets, class_names)
        inputs_embeds, attn_mask = self._encoder_inputs(pooled_emb, prompts)
        target_ids, _ = self._tokenize(list(target_comments), self.max_target_len, device)
        # T5 expects -100 to mask pad tokens out of the loss.
        labels = target_ids.clone()
        labels[labels == self.tokenizer.pad_token_id] = -100
        out = self.lm(inputs_embeds=inputs_embeds, attention_mask=attn_mask, labels=labels)
        return out.loss

    @torch.no_grad()
    def generate(
        self,
        pooled_emb: torch.Tensor,
        predicted_error_logits: torch.Tensor,
        class_names: Sequence[str],
        threshold: float = 0.5,
        num_beams: int = 4,
    ) -> List[str]:
        probs = torch.sigmoid(predicted_error_logits)
        prompts = self._build_prompts(probs, class_names, threshold=threshold)
        inputs_embeds, attn_mask = self._encoder_inputs(pooled_emb, prompts)
        gen = self.lm.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attn_mask,
            max_new_tokens=self.max_target_len,
            num_beams=num_beams,
            no_repeat_ngram_size=3,
            early_stopping=True,
        )
        return self.tokenizer.batch_decode(gen, skip_special_tokens=True)


class BranchedFusionT5CommentHead(nn.Module):
    """Branched-fusion comment path with a **frozen Flan-T5** encoder-decoder.

    Mirrors ``ClassConditionedCommentHead`` at the API level (consumes fused ``z``, short
    class-only coach prompts + soft prefix) but uses ``T5ForConditionalGeneration`` so
    ``--lm-name google/flan-t5-*`` works with ``--class-conditioned-comment``. Causal
    LMs remain on ``ClassConditionedCommentHead`` (e.g. Gemma).
    """

    def __init__(
        self,
        encoder_dim: int,
        model_name: str = "google/flan-t5-small",
        n_prefix_tokens: int = 16,
        max_target_len: int = 48,
        max_prompt_len: int = 96,
    ):
        super().__init__()
        try:
            from transformers import T5ForConditionalGeneration, T5Tokenizer  # type: ignore
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "BranchedFusionT5CommentHead requires `transformers` and `sentencepiece`. "
                "Install with: pip install 'transformers>=4.30' sentencepiece"
            ) from exc

        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        self.lm = T5ForConditionalGeneration.from_pretrained(model_name)
        for p in self.lm.parameters():
            p.requires_grad_(False)
        self.lm.eval()

        self.d_model = int(self.lm.config.d_model)
        self.n_prefix = int(n_prefix_tokens)
        self.max_target_len = int(max_target_len)
        self.max_prompt_len = int(max_prompt_len)
        self.model_name = str(model_name)

        self.prefix_proj = nn.Sequential(
            nn.LayerNorm(int(encoder_dim)),
            nn.Linear(int(encoder_dim), self.n_prefix * self.d_model),
        )

    def train(self, mode: bool = True):  # type: ignore[override]
        super().train(mode)
        self.lm.eval()
        return self

    def parameters(self, recurse: bool = True):
        return self.prefix_proj.parameters(recurse=recurse)

    @staticmethod
    def _build_prompts(class_names: Sequence[str]) -> List[str]:
        prompts: List[str] = []
        for n in class_names:
            cls = (n or "exercise").strip().lower() or "exercise"
            prompts.append(
                f"You are a fitness coach. The athlete just performed "
                f'"{cls}". Write one short corrective coaching tip for their form:'
            )
        return prompts

    def _tokenize(self, texts: Sequence[str], max_length: int, device: torch.device):
        enc = self.tokenizer(
            list(texts),
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        return enc["input_ids"].to(device), enc["attention_mask"].to(device)

    def _encoder_inputs(
        self, pooled_emb: torch.Tensor, prompts: Sequence[str]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        device = pooled_emb.device
        b = pooled_emb.size(0)
        prompt_ids, prompt_mask = self._tokenize(prompts, self.max_prompt_len, device)
        with torch.no_grad():
            prompt_emb = self.lm.get_input_embeddings()(prompt_ids)
        soft = self.prefix_proj(pooled_emb).view(b, self.n_prefix, self.d_model)
        inputs_embeds = torch.cat([soft, prompt_emb], dim=1)
        soft_mask = torch.ones(b, self.n_prefix, dtype=prompt_mask.dtype, device=device)
        attn_mask = torch.cat([soft_mask, prompt_mask], dim=1)
        return inputs_embeds, attn_mask

    def compute_loss(
        self,
        pooled_emb: torch.Tensor,
        class_indices: torch.Tensor,
        class_names: Sequence[str],
        target_comments: Sequence[str],
    ) -> torch.Tensor:
        """Teacher-forced cross-entropy; ``class_indices`` unused (matches causal-head API).

        Prompts use **gold** class names per batch sample (CSV), same rationale as ``ClassConditionedCommentHead``.
        """
        del class_indices
        device = pooled_emb.device
        keep = [i for i, c in enumerate(target_comments) if c and c.strip()]
        if not keep:
            return pooled_emb.new_zeros(())
        if len(keep) != pooled_emb.size(0):
            pooled_emb = pooled_emb[keep]
            class_names = [class_names[i] for i in keep]
            target_comments = [target_comments[i] for i in keep]

        prompts = self._build_prompts(class_names)
        inputs_embeds, attn_mask = self._encoder_inputs(pooled_emb, prompts)
        target_ids, _ = self._tokenize(list(target_comments), self.max_target_len, device)
        labels = target_ids.clone()
        labels[labels == self.tokenizer.pad_token_id] = -100
        out = self.lm(inputs_embeds=inputs_embeds, attention_mask=attn_mask, labels=labels)
        return out.loss

    @torch.no_grad()
    def generate(
        self,
        pooled_emb: torch.Tensor,
        predicted_class_indices: torch.Tensor,
        class_names: Sequence[str],
        num_beams: int = 4,
    ) -> List[str]:
        del predicted_class_indices
        prompts = self._build_prompts(class_names)
        inputs_embeds, attn_mask = self._encoder_inputs(pooled_emb, prompts)
        gen = self.lm.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attn_mask,
            max_new_tokens=self.max_target_len,
            num_beams=num_beams,
            no_repeat_ngram_size=3,
            early_stopping=True,
        )
        return self.tokenizer.batch_decode(gen, skip_special_tokens=True)


class ClassConditionedCommentHead(nn.Module):
    """Coaching-comment head with frozen decoder-only LM (Gemma-2-2B default).

    Consumes the fused vector ``z = a + b`` produced by the branched
    ``MultiTaskFusion`` tower (Branch A specialises on classification,
    Branch B on quality). Because each branch is the only direct path to its
    respective task head, gradient flow forces the branches to specialise on
    different features — and ``z`` then carries both class-discriminative and
    quality-discriminative signals into the comment head implicitly. No explicit
    class-embedding lookup is required at the comment-head input.

    The optional ``class_emb_dim > 0`` argument is kept for the legacy
    "explicit class conditioning" variant (where ``c = E_cls[ŷ]`` is
    concatenated to the fused vector before the soft-prefix projection).
    Set ``class_emb_dim=0`` to use the pure branched-fusion design.

    Trainable: only the optional class-embedding table ``E_cls`` and the
    soft-prefix projection ``W_p``. All LM weights stay frozen.
    """

    def __init__(
        self,
        encoder_dim: int,
        num_classes: int,
        class_emb_dim: int = 0,
        model_name: str = "google/gemma-2-2b",
        n_prefix_tokens: int = 16,
        max_target_len: int = 64,
        max_prompt_len: int = 96,
        hf_token: Optional[str] = None,
        device_map: Optional[str] = None,
    ):
        super().__init__()
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM  # type: ignore
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "ClassConditionedCommentHead requires `transformers>=4.42`. "
                "Install with: pip install 'transformers>=4.42' accelerate sentencepiece"
            ) from exc

        kwargs: Dict[str, Any] = {}
        if hf_token:
            kwargs["token"] = hf_token
        if device_map:
            kwargs["device_map"] = device_map
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, **kwargs)
        # decoder-only causal LM (Gemma, Llama, Qwen, Phi, etc.)
        self.lm = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
        for p in self.lm.parameters():
            p.requires_grad_(False)
        self.lm.eval()

        # Ensure a pad token exists (some causal LMs ship without one).
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.d_model = int(getattr(self.lm.config, "hidden_size", None)
                           or getattr(self.lm.config, "d_model", None))
        if self.d_model is None:
            raise ValueError(f"Could not infer hidden_size from {model_name} config.")

        self.n_prefix = int(n_prefix_tokens)
        self.max_target_len = int(max_target_len)
        self.max_prompt_len = int(max_prompt_len)
        self.num_classes = int(num_classes)
        self.class_emb_dim = int(class_emb_dim)
        self.model_name = model_name

        # Trainable components.
        # In the branched-fusion design class_emb_dim=0 — Branch A already
        # delivers the class signal via z = a + b. The explicit class
        # embedding is only used when class_emb_dim > 0 (legacy variant).
        if self.class_emb_dim > 0:
            self.class_embeddings = nn.Embedding(self.num_classes, self.class_emb_dim)
        else:
            self.class_embeddings = None
        proj_in_dim = encoder_dim + self.class_emb_dim
        self.prefix_proj = nn.Sequential(
            nn.LayerNorm(proj_in_dim),
            nn.Linear(proj_in_dim, self.n_prefix * self.d_model),
        )

    def train(self, mode: bool = True):  # type: ignore[override]
        super().train(mode)
        self.lm.eval()  # keep frozen LM in eval (no dropout)
        return self

    def parameters(self, recurse: bool = True):
        # Optimiser-facing iterator — only trainable params.
        if self.class_embeddings is not None:
            yield from self.class_embeddings.parameters(recurse=recurse)
        yield from self.prefix_proj.parameters(recurse=recurse)

    # ------------------------------------------------------------------ helpers

    def _class_names_to_indices(
        self, class_names: Sequence[str], class_to_idx: Dict[str, int]
    ) -> torch.Tensor:
        return torch.tensor(
            [class_to_idx.get((n or "").strip().lower(), 0) for n in class_names],
            dtype=torch.long,
        )

    def _build_prompts(self, class_names: Sequence[str]) -> List[str]:
        """Build per-sample text prompt that pairs with the soft prefix."""
        prompts: List[str] = []
        for n in class_names:
            cls = (n or "exercise").strip().lower() or "exercise"
            prompts.append(
                f"You are a fitness coach. The athlete just performed "
                f"\"{cls}\". Write one short corrective coaching tip for their form:"
            )
        return prompts

    def _build_inputs_embeds(
        self,
        pooled_emb: torch.Tensor,
        class_indices: torch.Tensor,
        prompts: Sequence[str],
        target_ids: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """Return (inputs_embeds, attention_mask, labels-for-causal-LM-or-None)."""
        device = pooled_emb.device
        b = pooled_emb.size(0)

        # Conditioning vector → soft prefix (b, n_prefix, d_model).
        # Branched-fusion design (class_emb_dim=0): v = z directly (the fused
        # vector already carries class info via Branch A).
        # Legacy design (class_emb_dim>0): v = [z ; c] with c = E_cls[ŷ].
        if self.class_embeddings is not None:
            c = self.class_embeddings(class_indices.to(device))
            v = torch.cat([pooled_emb, c], dim=-1)
        else:
            v = pooled_emb
        soft = self.prefix_proj(v).view(b, self.n_prefix, self.d_model)

        prompt_enc = self.tokenizer(
            list(prompts), padding=True, truncation=True,
            max_length=self.max_prompt_len, return_tensors="pt",
        )
        prompt_ids = prompt_enc["input_ids"].to(device)
        prompt_mask = prompt_enc["attention_mask"].to(device)
        with torch.no_grad():
            prompt_embeds = self.lm.get_input_embeddings()(prompt_ids)

        if target_ids is None:
            # Inference path
            inputs_embeds = torch.cat([soft, prompt_embeds], dim=1)
            soft_mask = torch.ones(b, self.n_prefix, dtype=prompt_mask.dtype, device=device)
            attn = torch.cat([soft_mask, prompt_mask], dim=1)
            return inputs_embeds, attn, None

        # Training path — concatenate target tokens after the prompt.
        target_ids = target_ids.to(device)
        with torch.no_grad():
            target_embeds = self.lm.get_input_embeddings()(target_ids)
        target_mask = (target_ids != self.tokenizer.pad_token_id).long()

        inputs_embeds = torch.cat([soft, prompt_embeds, target_embeds], dim=1)
        soft_mask = torch.ones(b, self.n_prefix, dtype=prompt_mask.dtype, device=device)
        attn = torch.cat([soft_mask, prompt_mask, target_mask], dim=1)

        # Labels: -100 everywhere except the target-token positions (so only target
        # tokens contribute to the cross-entropy loss).
        n_soft = self.n_prefix
        n_prompt = prompt_ids.size(1)
        n_target = target_ids.size(1)
        labels = inputs_embeds.new_full(
            (b, n_soft + n_prompt + n_target), fill_value=-100, dtype=torch.long
        )
        target_labels = target_ids.clone()
        target_labels[target_labels == self.tokenizer.pad_token_id] = -100
        labels[:, n_soft + n_prompt:] = target_labels
        return inputs_embeds, attn, labels

    # --------------------------------------------------------------------- API

    def compute_loss(
        self,
        pooled_emb: torch.Tensor,
        class_indices: torch.Tensor,
        class_names: Sequence[str],
        target_comments: Sequence[str],
    ) -> torch.Tensor:
        """Teacher-forced LM-CE loss using the *gold* class index.

        Skips samples with empty comments; returns a zero-grad tensor if none remain.
        """
        device = pooled_emb.device
        keep = [i for i, c in enumerate(target_comments) if c and c.strip()]
        if not keep:
            return pooled_emb.new_zeros(())
        if len(keep) != pooled_emb.size(0):
            pooled_emb = pooled_emb[keep]
            class_indices = class_indices[keep]
            class_names = [class_names[i] for i in keep]
            target_comments = [target_comments[i] for i in keep]

        prompts = self._build_prompts(class_names)
        # Tokenize targets (with EOS appended so the model learns to stop).
        targets = [c.strip() + self.tokenizer.eos_token for c in target_comments]
        target_enc = self.tokenizer(
            targets, padding=True, truncation=True,
            max_length=self.max_target_len, return_tensors="pt",
        )
        inputs_embeds, attn, labels = self._build_inputs_embeds(
            pooled_emb, class_indices, prompts, target_ids=target_enc["input_ids"],
        )
        out = self.lm(inputs_embeds=inputs_embeds, attention_mask=attn, labels=labels)
        return out.loss

    @torch.no_grad()
    def generate(
        self,
        pooled_emb: torch.Tensor,
        predicted_class_indices: torch.Tensor,
        class_names: Sequence[str],
        max_new_tokens: Optional[int] = None,
        num_beams: int = 4,
        do_sample: bool = False,
    ) -> List[str]:
        """Generate coaching comments using the classifier's *predicted* class."""
        prompts = self._build_prompts(class_names)
        inputs_embeds, attn, _ = self._build_inputs_embeds(
            pooled_emb, predicted_class_indices, prompts, target_ids=None,
        )
        gen = self.lm.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attn,
            max_new_tokens=max_new_tokens or self.max_target_len,
            num_beams=num_beams,
            do_sample=do_sample,
            no_repeat_ngram_size=3,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        # Strip soft-prefix + prompt portion if present, decode only generated tokens.
        return self.tokenizer.batch_decode(gen, skip_special_tokens=True)


class BottleneckAdapter(nn.Module):
    """Small per-user adapter trained while the base model stays frozen."""

    def __init__(self, dim: int, bottleneck_dim: int = 32, dropout: float = 0.1):
        super().__init__()
        bottleneck_dim = max(4, int(bottleneck_dim))
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, bottleneck_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(bottleneck_dim, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class xLSTMExerciseClassifier(nn.Module):
    """Shared xLSTM encoder with exercise, quality, and optional error-tag heads.

    Temporal readout (``temporal_pool`` / legacy ``use_attention_pool``): **mean** (GAP),
    **last** (final causal state — uses full prefix), or **attention** (learned softmax).
    Optional **input_dropout** regularizes raw frame features before the stack (still causal).
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        num_classes: int = 5,
        dropout: float = 0.1,
        bidirectional: bool = True,
        num_heads: int = 4,
        conv_kernel_size: int = 4,
        projection_factor: float = 4.0 / 3.0,
        num_error_tags: int = 0,
        quality_scale: float = 1.0,
        quality_output_low: float = 0.0,
        num_quality_classes: int = 1,
        adapter_dim: int = 0,
        linear_classifier: bool = False,
        block_pattern: Optional[str] = None,
        use_attention_pool: bool = False,
        temporal_pool: Optional[str] = None,
        input_dropout: float = 0.0,
        use_fusion: bool = False,
        fusion_dim: int = 128,
        quality_class_conditioning: bool = False,
        task_specific_pools: bool = False,
        soft_class_conditioning: bool = True,
    ):
        super().__init__()
        self.xlstm = xLSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            bidirectional=bidirectional,
            num_heads=num_heads,
            conv_kernel_size=conv_kernel_size,
            projection_factor=projection_factor,
            block_pattern=block_pattern,
        )
        idrop = float(input_dropout)
        self.input_dropout_rate = idrop
        self.input_drop = nn.Dropout(idrop) if idrop > 1e-8 else nn.Identity()

        if use_attention_pool:
            tp = "attention"
        elif temporal_pool is not None and str(temporal_pool).strip():
            tp = str(temporal_pool).strip().lower()
        else:
            tp = "mean"
        if tp not in ("mean", "last", "attention"):
            raise ValueError(f"temporal_pool must be mean|last|attention; got {tp!r}")

        self._temporal_pool = tp
        self.use_attention_pool = tp == "attention"
        self.task_specific_pools = bool(task_specific_pools) and tp == "attention"
        self.attn_pool: Optional[AttentionPooling]
        self.attn_pool_quality: Optional[AttentionPooling]
        self.pool: Optional[nn.AdaptiveAvgPool1d]
        if tp == "attention":
            # When task_specific_pools is True, attn_pool plays the role of the
            # classification-stream pool, and attn_pool_quality is a parallel
            # pool with its own learnable temporal scoring MLP for the quality
            # stream.  This lets each task pick a *different* frame weighting
            # (e.g. peak-motion frames for cls vs depth/extension frames for
            # quality) without changing the shared 8-block backbone above.
            self.attn_pool = AttentionPooling(hidden_size, hidden=max(32, hidden_size // 4), dropout=dropout)
            self.attn_pool_quality = (
                AttentionPooling(hidden_size, hidden=max(32, hidden_size // 4), dropout=dropout)
                if self.task_specific_pools else None
            )
            self.pool = None
        elif tp == "mean":
            self.attn_pool = None
            self.attn_pool_quality = None
            self.pool = nn.AdaptiveAvgPool1d(1)
        else:
            self.attn_pool = None
            self.attn_pool_quality = None
            self.pool = None
        self.dropout = nn.Dropout(dropout)
        self.soft_class_conditioning = bool(soft_class_conditioning)

        # Optional shared task-tower with additive fusion (whiteboard "+").
        # When enabled, the cls and quality heads consume the fused vector;
        # the comment head still consumes the raw pooled embedding because the
        # LM benefits from a higher-dim, less-compressed signal.
        self.use_fusion = bool(use_fusion)
        if self.use_fusion:
            self.fusion = MultiTaskFusion(hidden_size, int(fusion_dim), dropout=dropout)
            head_in = int(fusion_dim)
        else:
            self.fusion = None
            head_in = hidden_size

        # Paper-style capstone: global pool → dropout → single linear. Default remains small MLP.
        if linear_classifier:
            self.class_head = nn.Linear(head_in, num_classes)
        else:
            self.class_head = nn.Sequential(
                nn.Linear(head_in, head_in),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(head_in, num_classes),
            )
        q_out = max(1, int(num_quality_classes))
        self.num_quality_classes = q_out
        self.quality_is_classification = q_out > 1
        if self.quality_is_classification:
            q_dim = q_out
        else:
            q_dim = 1
        self.quality_class_conditioning = bool(quality_class_conditioning)
        self.quality_class_emb: Optional[nn.Embedding]
        if self.quality_class_conditioning:
            self.quality_class_emb = nn.Embedding(int(num_classes), head_in)
        else:
            self.quality_class_emb = None
        q_feat_in = head_in
        self.quality_head = nn.Sequential(
            nn.Linear(q_feat_in, head_in),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(head_in, q_dim),
        )
        self.error_head = (
            nn.Sequential(
                nn.Linear(head_in, head_in),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(head_in, num_error_tags),
            )
            if num_error_tags > 0
            else None
        )

        # Guidance registry: predicted class index → canonical action_guidance text.
        # Populated by the training script after the dataset is built (see
        # ``set_guidance_table``). Stored as a plain dict so it survives ``state_dict``
        # round-trips via the checkpoint sidecar.
        self._guidance_table: Dict[int, str] = {}
        # Comment registry: (predicted class, quality-bucket) → corrective text.
        # Populated by the training script via ``set_comment_table`` after the
        # train dataset has been scanned. Empty until then; ``lookup_comment``
        # returns empty strings when the table is missing.
        self._comment_table: Dict[Tuple[int, int], str] = {}
        self._quality_bucket_edges: Tuple[float, ...] = (0.4, 0.7)
        self._quality_domain_lo: float = 0.0
        self._quality_domain_hi: float = 1.0
        self._quality_bucket_centres: Tuple[float, ...] = _bucket_interval_centres(
            self._quality_bucket_edges,
            domain_lo=self._quality_domain_lo,
            domain_hi=self._quality_domain_hi,
        )
        self._idx_to_class: Dict[int, str] = {}
        self.adapter = BottleneckAdapter(hidden_size, bottleneck_dim=adapter_dim, dropout=dropout) if adapter_dim > 0 else None
        self.num_classes = int(num_classes)
        self.num_error_tags = int(num_error_tags)
        self.quality_scale = float(quality_scale)
        self.quality_output_low = float(quality_output_low)
        self.hidden_size = int(hidden_size)
        self.num_layers = int(self.xlstm.num_layers)
        self.dropout_rate = float(dropout)
        self.num_heads = self.xlstm.num_heads
        self.conv_kernel_size = int(conv_kernel_size)
        self.projection_factor = float(projection_factor)
        self.linear_classifier = bool(linear_classifier)
        self.block_pattern = self.xlstm.block_pattern

    def attach_adapter(self, adapter: Optional[nn.Module]) -> None:
        self.adapter = adapter

    def _pool_seq(self, seq_out: torch.Tensor, *, pool_module: Optional[AttentionPooling] = None) -> torch.Tensor:
        """Pool a sequence ``(B, T, D)`` → ``(B, D)`` using the configured strategy.

        ``pool_module`` overrides ``self.attn_pool`` for attention mode (used by
        the task-specific quality pool path).
        """
        tp = self._temporal_pool
        if tp == "attention":
            pool = pool_module if pool_module is not None else self.attn_pool
            assert pool is not None
            return pool(seq_out)
        if tp == "last":
            return seq_out[:, -1, :]
        assert self.pool is not None
        return self.pool(seq_out.transpose(1, 2)).squeeze(-1)

    def encode(self, x: torch.Tensor, adapter: Optional[nn.Module] = None) -> torch.Tensor:
        """Encode a clip to its raw pooled embedding (256-d). Comment head consumes this directly.

        With ``task_specific_pools=True`` this returns the **classification** stream
        pool; the quality stream pool is computed separately by ``encode_pair``.
        """
        seq_out, _ = self.xlstm(self.input_drop(x))
        pooled = self._pool_seq(seq_out)
        pooled = self.dropout(pooled)
        active_adapter = adapter if adapter is not None else self.adapter
        if active_adapter is not None:
            pooled = active_adapter(pooled)
        return pooled

    def encode_pair(
        self,
        x: torch.Tensor,
        adapter: Optional[nn.Module] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode with **two** task-specific pools.

        Returns ``(pooled_cls, pooled_quality)``. Both come from the same 8-block
        backbone — only the temporal attention is task-specific.  When
        ``task_specific_pools=False`` (or when not using attention pooling) the
        two returned tensors are the same object (back-compat fast path).
        """
        seq_out, _ = self.xlstm(self.input_drop(x))
        pooled_cls = self._pool_seq(seq_out)
        if self.task_specific_pools and self.attn_pool_quality is not None:
            pooled_quality = self._pool_seq(seq_out, pool_module=self.attn_pool_quality)
        else:
            pooled_quality = pooled_cls
        active_adapter = adapter if adapter is not None else self.adapter

        def _finish(t: torch.Tensor) -> torch.Tensor:
            t = self.dropout(t)
            return active_adapter(t) if active_adapter is not None else t

        pooled_cls = _finish(pooled_cls)
        # If the quality pool is the same object, dropout/adapter were applied
        # once via _finish(pooled_cls); only re-finish when we actually have a
        # separate tensor.
        pooled_quality = _finish(pooled_quality) if pooled_quality is not pooled_cls else pooled_cls
        return pooled_cls, pooled_quality

    def fuse(self, pooled: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Apply the branched fusion tower.

        Returns ``(a, b, z)`` where:
            a → consumed by the classification head only
            b → consumed by the quality head only
            z = a + b → carries both signals; consumed by the comment head

        If fusion is disabled (``use_fusion=False``), returns ``(pooled, pooled, pooled)``
        so all three heads receive the raw clip embedding.
        """
        if self.fusion is None:
            return pooled, pooled, pooled
        return self.fusion(pooled)

    def quality_branch_feat(
        self,
        branch_b: torch.Tensor,
        class_logits: torch.Tensor,
        explicit_class_one_hot: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Features fed into ``quality_head`` = branch-B + class-embedding offset.

        Two conditioning modes:

        * ``soft_class_conditioning=True`` (default): mix the class-embedding rows
          by the **soft** class distribution.

              p = explicit_one_hot if given else softmax(class_logits)
              offset = p @ class_emb.weight            # (B, C) @ (C, D) → (B, D)

          This is **fully differentiable end-to-end** — quality loss can push
          gradients back through ``class_logits``, so the classifier learns to
          produce class distributions that help the quality head, not just hit
          the cls-loss target.

        * ``soft_class_conditioning=False`` (legacy): hard lookup of
          ``class_emb[argmax]`` — non-differentiable, the two heads are
          decoupled at the conditioning step.

        Returns ``branch_b`` unchanged if ``quality_class_conditioning=False``.
        """
        if not self.quality_class_conditioning:
            return branch_b
        assert self.quality_class_emb is not None

        if self.soft_class_conditioning:
            if explicit_class_one_hot is not None:
                p = explicit_class_one_hot.to(device=branch_b.device, dtype=branch_b.dtype)
            else:
                p = F.softmax(class_logits, dim=-1).to(dtype=branch_b.dtype)
            offset = p @ self.quality_class_emb.weight.to(dtype=branch_b.dtype)
            return branch_b + offset

        # Legacy hard-argmax path
        if explicit_class_one_hot is not None:
            oh = explicit_class_one_hot.to(device=branch_b.device)
            idx = oh.argmax(dim=-1).long().clamp(min=0, max=self.num_classes - 1)
        else:
            idx = class_logits.argmax(dim=-1).long().clamp(min=0, max=self.num_classes - 1)
        e = self.quality_class_emb(idx).to(dtype=branch_b.dtype)
        return branch_b + e

    # -------------------------------- guidance lookup helpers --------------------------------

    def set_guidance_table(self, guidance: Dict[int, str], idx_to_class: Dict[int, str]) -> None:
        """Register the deterministic class → ``action_guidance`` map.

        Built once from the training set in ``train_xlstm_egoexo_multitask.py`` and
        also persisted into the checkpoint so inference is self-contained.
        """
        self._guidance_table = {int(k): str(v) for k, v in guidance.items()}
        self._idx_to_class = {int(k): str(v) for k, v in idx_to_class.items()}

    def lookup_guidance(self, class_indices: torch.Tensor) -> List[str]:
        out: List[str] = []
        for idx in class_indices.detach().cpu().tolist():
            out.append(self._guidance_table.get(int(idx), ""))
        return out

    def set_comment_table(
        self,
        table: Dict[Tuple[int, int], str],
        bucket_edges: Tuple[float, ...],
        *,
        domain_lo: float = 0.0,
        domain_hi: float = 1.0,
    ) -> None:
        """Register the (class, quality-bucket) → corrective comment lookup.

        Built once from the training set in
        ``train_xlstm_egoexo_multitask.py`` via
        ``EgoExoXLSTMDataset.build_comment_table`` and persisted in the
        checkpoint so inference is self-contained.

        ``domain_lo`` / ``domain_hi`` bound the same axis as CSV quality after
        ``canonical_quality_score`` (``[0,1]`` vs ``[1,5]`` Likert).
        """
        self._comment_table = {
            (int(k[0]), int(k[1])): str(v) for k, v in table.items()
        }
        self._quality_bucket_edges = tuple(float(e) for e in bucket_edges)
        self._quality_domain_lo = float(domain_lo)
        self._quality_domain_hi = float(domain_hi)
        self._quality_bucket_centres = _bucket_interval_centres(
            self._quality_bucket_edges,
            domain_lo=self._quality_domain_lo,
            domain_hi=self._quality_domain_hi,
        )

    def lookup_comment(
        self,
        class_indices: torch.Tensor,
        quality_scores: torch.Tensor,
        *,
        quality_bucket_indices: Optional[torch.Tensor] = None,
    ) -> List[str]:
        """Retrieve the corrective comment for each (predicted class, quality) pair.

        ``quality_scores`` are either regression scalars on the training quality axis
        (shape ``[B]`` / ``[B,1]``), or logits/probabilities over ``K`` quality buckets
        (shape ``[B, K]``). When ``quality_bucket_indices`` is provided (integer buckets),
        it overrides bucketing from continuous scalars / logits.
        """
        edges = self._quality_bucket_edges
        n_buckets = len(edges) + 1
        out: List[str] = []

        if quality_bucket_indices is not None:
            for cls_idx, bkt in zip(
                class_indices.detach().cpu().tolist(),
                quality_bucket_indices.detach().cpu().tolist(),
            ):
                out.append(self._comment_table.get((int(cls_idx), int(bkt)), ""))
            return out

        # Infer from tensor rank: logits/probs per bucket vs scalar regression
        qs = quality_scores
        if qs.dim() == 2 and qs.size(-1) == self.num_quality_classes and self.num_quality_classes > 1:
            buckets = qs.argmax(dim=1).detach().cpu().tolist()
            for cls_idx, bucket in zip(class_indices.detach().cpu().tolist(), buckets):
                out.append(self._comment_table.get((int(cls_idx), int(bucket)), ""))
            return out

        scalars = qs.squeeze(-1) if qs.dim() == 2 else qs
        for cls_idx, q in zip(
            class_indices.detach().cpu().tolist(),
            scalars.detach().cpu().tolist(),
        ):
            q = float(q)
            bucket = n_buckets - 1
            for i, e in enumerate(edges):
                if q < e:
                    bucket = i
                    break
            out.append(self._comment_table.get((int(cls_idx), int(bucket)), ""))
        return out

    def freeze_backbone(self, freeze: bool = True, last_n_unfrozen: int = 0) -> None:
        """Optionally freeze the xLSTM stack (and input projection) for fast head fine-tuning.

        Useful for the personalisation / capstone ablation: warm-start from a
        Riccio-pretrained checkpoint, freeze everything except the heads + last
        ``last_n_unfrozen`` blocks.
        """
        for p in self.xlstm.input_proj.parameters():
            p.requires_grad_(not freeze)
        n = len(self.xlstm.blocks)
        for i, block in enumerate(self.xlstm.blocks):
            unfreeze = (not freeze) or (i >= n - max(0, int(last_n_unfrozen)))
            for p in block.parameters():
                p.requires_grad_(unfreeze)

    def forward(
        self,
        x: torch.Tensor,
        adapter: Optional[nn.Module] = None,
        *,
        quality_explicit_class_one_hot: Optional[torch.Tensor] = None,
    ):
        # Task-specific pools (when enabled) produce one pooled embedding per
        # task — the classification stream and quality stream see different
        # frame weightings over the same 8-block backbone.  When disabled the
        # two returned tensors are the same.
        pooled_cls, pooled_q = self.encode_pair(x, adapter=adapter)
        # Branched fusion: a feeds cls only, b feeds quality only, z = a+b is
        # available for the comment head via the helper below.
        a_cls, _b_cls, _z_cls = self.fuse(pooled_cls)
        if pooled_q is pooled_cls:
            a_for_z, b, _ = self.fuse(pooled_q)
            _z = a_for_z + b  # comment head sees the shared fused vector
        else:
            _a_q, b, _ = self.fuse(pooled_q)
            _z = a_cls + b    # cls path's a + quality path's b for the comment head
        class_logits = self.class_head(a_cls)
        b_q = self.quality_branch_feat(b, class_logits, quality_explicit_class_one_hot)
        quality_raw = self.quality_head(b_q)
        if self.quality_is_classification:
            quality_scores = quality_raw
        else:
            quality_scores = torch.sigmoid(quality_raw) * self.quality_scale + self.quality_output_low
        if self.error_head is None:
            return class_logits, quality_scores
        # The error head retains the fused vector since it's a shared auxiliary.
        error_logits = self.error_head(_z)
        return class_logits, quality_scores, error_logits

    @torch.no_grad()
    def infer(
        self,
        x: torch.Tensor,
        comment_head: Optional["CommentGenerationHead"] = None,
    ) -> List[Dict[str, object]]:
        """User-facing inference. Returns one dict per clip with the three deliverables:

        ``{ "quality": float, "guidance": str, "comment": str, "exercise": str }``

        Classification logits are still computed (we need the predicted class for the
        guidance lookup and as conditioning for the comment head) but are not
        surfaced as a separate field — the predicted exercise name is included for
        UI convenience.
        """
        pooled_cls, pooled_q = self.encode_pair(x)
        # Branched fusion per stream: a from cls pool, b from quality pool.
        a, _, _ = self.fuse(pooled_cls)
        _, b, _ = self.fuse(pooled_q)
        logits = self.class_head(a)
        cls_idx = logits.argmax(dim=1)
        exercises = [self._idx_to_class.get(int(i), "") for i in cls_idx.cpu().tolist()]
        guidance = self.lookup_guidance(cls_idx)
        b_q = self.quality_branch_feat(b, logits, None)
        q_raw = self.quality_head(b_q)
        if self.quality_is_classification:
            probs = torch.softmax(q_raw, dim=-1)
            centres = q_raw.new_tensor(list(self._quality_bucket_centres))
            q_expect = (probs * centres).sum(dim=-1)
            qb = q_raw.argmax(dim=1)
            quality = q_expect.detach().cpu().tolist()
            comments = self.lookup_comment(cls_idx, q_raw, quality_bucket_indices=qb)
            qb_list = qb.detach().cpu().tolist()
            return [
                {"exercise": ex, "quality": float(q), "quality_bucket": int(bi), "guidance": g, "comment": c}
                for ex, q, bi, g, c in zip(exercises, quality, qb_list, guidance, comments)
            ]

        q_tensor = torch.sigmoid(q_raw).squeeze(-1) * self.quality_scale + self.quality_output_low
        quality = q_tensor.detach().cpu().tolist()
        comments = self.lookup_comment(cls_idx, q_tensor)

        return [
            {"exercise": ex, "quality": float(q), "guidance": g, "comment": c}
            for ex, q, g, c in zip(exercises, quality, guidance, comments)
        ]

    def get_loss(
        self,
        class_logits: torch.Tensor,
        quality_scores: torch.Tensor,
        labels: torch.Tensor,
        quality_targets: torch.Tensor,
        class_weight: float = 1.0,
        quality_weight: float = 0.5,
        error_logits: Optional[torch.Tensor] = None,
        error_targets: Optional[torch.Tensor] = None,
        error_weight: float = 0.5,
        error_pos_weight: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        ce_loss = F.cross_entropy(class_logits, labels)
        if self.quality_is_classification:
            reg_loss = F.cross_entropy(quality_scores, quality_targets.long())
        else:
            reg_loss = F.mse_loss(quality_scores.squeeze(-1), quality_targets)
        total_loss = class_weight * ce_loss
        if quality_weight > 0:
            total_loss = total_loss + quality_weight * reg_loss
        if error_logits is not None and error_targets is not None:
            bce = F.binary_cross_entropy_with_logits(
                error_logits,
                error_targets,
                pos_weight=error_pos_weight,
            )
            total_loss = total_loss + error_weight * bce
        return total_loss


if __name__ == "__main__":
    batch_size = 4
    seq_len = 60
    input_size = 34
    model = xLSTMExerciseClassifier(
        input_size=input_size,
        hidden_size=128,
        num_layers=3,
        num_classes=5,
        num_error_tags=8,
        adapter_dim=16,
    )
    x = torch.randn(batch_size, seq_len, input_size)
    cls, quality, errors = model(x)
    print("xLSTM paper-style block test passed")
    print("class logits:", tuple(cls.shape))
    print("quality:", tuple(quality.shape))
    print("errors:", tuple(errors.shape))
