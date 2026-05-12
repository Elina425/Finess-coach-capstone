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
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


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
    """Two PLN → Linear branches summed before the per-task heads.

    This realises the additive fusion the instructor sketched on the whiteboard
    (the ``+`` box between the two PLN→Linear branches). Each branch is a
    LayerNorm → Linear → GELU stack — same shape, different weights — so they
    can specialise during training; the sum produces the shared task-tower
    representation that feeds both the classification and quality heads.
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

    def forward(self, pooled: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.branch_a(pooled) + self.branch_b(pooled))


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
    """Shared xLSTM encoder with exercise, quality, and optional error-tag heads."""

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
        adapter_dim: int = 0,
        linear_classifier: bool = False,
        block_pattern: Optional[str] = None,
        use_attention_pool: bool = False,
        use_fusion: bool = False,
        fusion_dim: int = 128,
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
        self.use_attention_pool = bool(use_attention_pool)
        if self.use_attention_pool:
            self.attn_pool = AttentionPooling(hidden_size, hidden=max(32, hidden_size // 4), dropout=dropout)
        else:
            self.pool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(dropout)

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
        self.quality_head = nn.Sequential(
            nn.Linear(head_in, head_in),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(head_in, 1),
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
        self._idx_to_class: Dict[int, str] = {}
        self.adapter = BottleneckAdapter(hidden_size, bottleneck_dim=adapter_dim, dropout=dropout) if adapter_dim > 0 else None
        self.num_classes = int(num_classes)
        self.num_error_tags = int(num_error_tags)
        self.quality_scale = float(quality_scale)
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

    def encode(self, x: torch.Tensor, adapter: Optional[nn.Module] = None) -> torch.Tensor:
        """Encode a clip to its raw pooled embedding (256-d). Comment head consumes this directly."""
        seq_out, _ = self.xlstm(x)
        if self.use_attention_pool:
            pooled = self.attn_pool(seq_out)
        else:
            pooled = self.pool(seq_out.transpose(1, 2)).squeeze(-1)
        pooled = self.dropout(pooled)
        active_adapter = adapter if adapter is not None else self.adapter
        if active_adapter is not None:
            pooled = active_adapter(pooled)
        return pooled

    def fuse(self, pooled: torch.Tensor) -> torch.Tensor:
        """Apply the additive fusion tower (no-op if ``use_fusion=False``)."""
        return self.fusion(pooled) if self.fusion is not None else pooled

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

    def forward(self, x: torch.Tensor, adapter: Optional[nn.Module] = None):
        pooled = self.encode(x, adapter=adapter)
        fused = self.fuse(pooled)
        class_logits = self.class_head(fused)
        quality_raw = self.quality_head(fused)
        quality_scores = torch.sigmoid(quality_raw) * self.quality_scale
        if self.error_head is None:
            return class_logits, quality_scores
        error_logits = self.error_head(fused)
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
        pooled = self.encode(x)
        fused = self.fuse(pooled)
        logits = self.class_head(fused)
        cls_idx = logits.argmax(dim=1)
        quality = (torch.sigmoid(self.quality_head(fused)).squeeze(-1) * self.quality_scale).cpu().tolist()
        exercises = [self._idx_to_class.get(int(i), "") for i in cls_idx.cpu().tolist()]
        guidance = self.lookup_guidance(cls_idx)

        if comment_head is not None:
            # Use predicted error logits if the head exists, else zeros — the comment
            # head will still produce a sensible coaching tip from the class name alone.
            err_logits = self.error_head(fused) if self.error_head is not None else \
                torch.zeros(pooled.size(0), len(comment_head.error_tags), device=pooled.device)
            comments = comment_head.generate(pooled, err_logits, exercises)
        else:
            comments = ["" for _ in exercises]

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
