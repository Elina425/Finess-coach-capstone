"""
ViTPose-S (COCO, top-down heatmap) **backbone** weights for paper-aligned frame features.

Loads OpenMMLab ``td-hm_ViTPose-small_8xb64-210e_coco-256x192`` checkpoints without ``mmpose``/``mmcv``:
the tensor names match ``mmpretrain.VisionTransformer`` (MAE-pretrained ViT-S stem + COCO pose head
weights in the same ``.pth``; we **only run the backbone** and discard the heatmap head).

- Input: BGR crop resized to **H×W = 256×192** (dataset ``input_size`` order in mmpose codec).
- Preprocess: RGB, ``mean/std`` from the official config (PoseDataPreprocessor).
- Tokens: patch tokens only (checkpoint ``pos_embed`` includes a unused cls slot; we drop it).
- Pooling: **mean over spatial tokens** (equivalent to GAP on the **C×H×W** feature map).
- Output: **384 → 256** via a trainable ``nn.Linear`` (official backbone is 384-D; the paper fixes **256-D**).

**DoRA (optional):** ``inject_dora_last_two_blocks`` wraps attention + FFN ``nn.Linear`` layers in the last
two encoder blocks with ``DoRALinear`` (rank ``r``) for stage-2 adaptation; base weights stay frozen inside
each ``DoRALinear``.
"""

from __future__ import annotations

import math
import os
import shutil
import tempfile
import urllib.error
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from fitness_coach.models.personalization import DoRALinear

# Official ViTPose-S COCO (256×192) — see mmpose ``configs/.../vitpose_coco.yml``
VITPOSE_SMALL_COCO_URL = (
    "https://download.openmmlab.com/mmpose/v1/body_2d_keypoint/topdown_heatmap/coco/"
    "td-hm_ViTPose-small_8xb64-210e_coco-256x192-62d7a712_20230314.pth"
)

IMG_H, IMG_W = 256, 192  # height × width for top-down ViTPose-S COCO
MEAN = (123.675, 116.28, 103.53)
STD = (58.395, 57.12, 57.375)
BACKBONE_DIM = 384
OUT_DIM = 256


def default_vitpose_cache_path() -> Path:
    return Path.home() / ".cache" / "fitness_coach" / "vitpose"


def ensure_vitpose_small_checkpoint(
    checkpoint_path: Optional[Path] = None,
    *,
    url: str = VITPOSE_SMALL_COCO_URL,
) -> Path:
    """Download ViTPose-S COCO weights once into ``~/.cache/fitness_coach/vitpose/``."""
    if checkpoint_path is not None:
        p = Path(checkpoint_path).expanduser().resolve()
        if not p.is_file():
            raise FileNotFoundError(str(p))
        return p
    cache_override = os.environ.get("FITNESS_COACH_VITPOSE_CACHE", "").strip()
    cache = Path(cache_override).expanduser() if cache_override else default_vitpose_cache_path()
    try:
        cache.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        raise OSError(f"Cannot create ViTPose cache directory {cache}: {e}") from e

    name = url.rsplit("/", 1)[-1]
    dest = cache / name
    min_bytes = 10_000_000

    if dest.is_file() and dest.stat().st_size >= min_bytes:
        return dest

    # Unique temp path so parallel pipeline workers don't clobber each other's downloads.
    fd, tmp_str = tempfile.mkstemp(prefix=f"{dest.stem}_", suffix=".partial", dir=str(cache))
    os.close(fd)
    tmp = Path(tmp_str)
    try:
        try:
            with urllib.request.urlopen(url, timeout=600) as resp:
                with open(tmp, "wb") as out:
                    shutil.copyfileobj(resp, out)
        except (urllib.error.URLError, OSError, TimeoutError) as e:
            raise RuntimeError(
                f"ViTPose-S download failed from {url!r}. Check network, disk space, and cache path {cache}."
            ) from e

        sz = tmp.stat().st_size
        if sz < min_bytes:
            raise RuntimeError(
                f"Incomplete ViTPose checkpoint ({sz} bytes); expected at least {min_bytes}. "
                f"Delete {tmp} / {dest} if corrupt and retry."
            )

        # Another worker may have finished first.
        if dest.is_file() and dest.stat().st_size >= min_bytes:
            tmp.unlink(missing_ok=True)
            return dest

        os.replace(tmp, dest)
        tmp = None  # Successfully renamed; skip unlink in finally
        return dest
    finally:
        if tmp is not None and Path(tmp_str).exists():
            try:
                Path(tmp_str).unlink()
            except OSError:
                pass


class _ViTAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int = 12, qkv_bias: bool = True) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, n, c = x.shape
        qkv = self.qkv(x).reshape(b, n, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(b, n, c)
        return self.proj(x)


class _FFN(nn.Module):
    """Matches checkpoint keys ``ffn.layers.0.0`` (Linear+GELU) and ``ffn.layers.1``."""

    def __init__(self, dim: int, hidden: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [
                nn.ModuleList([nn.Linear(dim, hidden)]),
                nn.Linear(hidden, dim),
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.gelu(self.layers[0][0](x))
        return self.layers[1](x)


class _EncoderLayer(nn.Module):
    """Pre-LayerNorm ViT block (same signal flow as mmpretrain ``TransformerEncoderLayer``)."""

    def __init__(self, dim: int, num_heads: int, ffn_hidden: int) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = _ViTAttention(dim, num_heads=num_heads, qkv_bias=True)
        self.ln2 = nn.LayerNorm(dim, eps=1e-6)
        self.ffn = _FFN(dim, ffn_hidden)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


class _PatchEmbedConv(nn.Module):
    """Matches mmpretrain ``PatchEmbed`` state keys ``patch_embed.projection.*``."""

    def __init__(self, embed_dim: int) -> None:
        super().__init__()
        self.projection = nn.Conv2d(3, embed_dim, kernel_size=16, stride=16, padding=0, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.projection(x)


class ViTPoseSmallBackbone256(nn.Module):
    """
    ViTPose-S backbone + linear head to 256-D. ``forward_bgr`` expects **BGR** uint8/float crops.
    """

    def __init__(self, *, proj_dim: int = OUT_DIM) -> None:
        super().__init__()
        embed = BACKBONE_DIM
        self.patch_embed = _PatchEmbedConv(embed)
        self.pos_embed = nn.Parameter(torch.zeros(1, 192, embed))
        self.layers = nn.ModuleList([_EncoderLayer(embed, 12, 1536) for _ in range(12)])
        self.ln1 = nn.LayerNorm(embed, eps=1e-6)
        self.proj = nn.Linear(embed, int(proj_dim))
        nn.init.xavier_uniform_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def load_vitpose_checkpoint(self, ckpt_path: Path, *, strict_backbone: bool = True) -> None:
        ck = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
        state = ck["state_dict"] if isinstance(ck, dict) and "state_dict" in ck else ck
        sd: Dict[str, torch.Tensor] = {}
        for k, v in state.items():
            if not k.startswith("backbone."):
                continue
            sd[k[len("backbone.") :]] = v
        pos = sd["pos_embed"]
        if pos.shape[1] == 193:
            sd["pos_embed"] = pos[:, 1:, :].contiguous()
        incompatible = self.load_state_dict(sd, strict=False)
        if strict_backbone and len(incompatible.missing_keys) > 0:
            bad = [m for m in incompatible.missing_keys if not m.startswith("proj.")]
            if bad:
                raise RuntimeError(f"ViTPose backbone missing keys: {bad[:12]}")

    def preprocess_bgr(self, crop_bgr: np.ndarray) -> torch.Tensor:
        """(1,3,H,W) float RGB, ImageNet-style norm (mmpose PoseDataPreprocessor)."""
        if crop_bgr is None or crop_bgr.size == 0:
            raise ValueError("empty crop")
        rgb = cv2_cvt_bgr_rgb(crop_bgr)
        t = torch.from_numpy(rgb).float().permute(2, 0, 1).unsqueeze(0)
        t = F.interpolate(t, size=(IMG_H, IMG_W), mode="bilinear", align_corners=False)
        mean = torch.tensor(MEAN, dtype=t.dtype, device=t.device).view(1, 3, 1, 1)
        std = torch.tensor(STD, dtype=t.dtype, device=t.device).view(1, 3, 1, 1)
        t = (t - mean) / std
        return t

    def forward_tokens(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(x)
        _, _, h, w = x.shape
        x = x.flatten(2).transpose(1, 2)
        if x.size(1) != self.pos_embed.size(1):
            raise ValueError(f"patch count {x.size(1)} != pos_embed {self.pos_embed.size(1)} for input spatial")
        x = x + self.pos_embed
        for blk in self.layers:
            x = blk(x)
        x = self.ln1(x)
        return x.mean(dim=1)

    def forward_bgr(self, crop_bgr: np.ndarray, device: torch.device) -> torch.Tensor:
        t = self.preprocess_bgr(crop_bgr).to(device)
        z = self.forward_tokens(t)
        return self.proj(z)

    def freeze_all_for_export(self) -> None:
        """Freeze every parameter (ViTPose + 256-D head) for frozen feature extraction."""
        for p in self.parameters():
            p.requires_grad = False

    def unfreeze_all(self) -> None:
        for p in self.parameters():
            p.requires_grad = True

    def inject_dora_last_two_blocks(self, rank: int = 8, alpha: float = 8.0) -> None:
        """Wrap attention + FFN Linears in layers 10–11 with ``DoRALinear`` (base frozen)."""
        for idx in (10, 11):
            layer = self.layers[idx]
            layer.attn.qkv = DoRALinear.from_linear(layer.attn.qkv, rank=rank, alpha=alpha)
            layer.attn.proj = DoRALinear.from_linear(layer.attn.proj, rank=rank, alpha=alpha)
            layer.ffn.layers[0][0] = DoRALinear.from_linear(layer.ffn.layers[0][0], rank=rank, alpha=alpha)
            layer.ffn.layers[1] = DoRALinear.from_linear(layer.ffn.layers[1], rank=rank, alpha=alpha)


def cv2_cvt_bgr_rgb(arr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(np.asarray(arr), cv2.COLOR_BGR2RGB)


_ENCODER: Optional[ViTPoseSmallBackbone256] = None
_ENCODER_SPEC: Optional[Tuple[str, str, str]] = None  # (ckpt, device, kind)


def get_vitpose_paper_encoder(
    *,
    checkpoint_path: Optional[Path],
    device: str,
    vit_encoder: str,
) -> Tuple[ViTPoseSmallBackbone256, int]:
    """
    Return a singleton encoder for ``(checkpoint_path, device, vit_encoder)``.

    ``vit_encoder`` must be ``paper`` (ViTPose-S). Do **not** call ``inject_dora_last_two_blocks`` on
    this shared instance; construct a fresh ``ViTPoseSmallBackbone256()`` for adapter fine-tuning.
    """
    global _ENCODER, _ENCODER_SPEC
    ck = str(Path(checkpoint_path).resolve()) if checkpoint_path else str(ensure_vitpose_small_checkpoint())
    spec = (ck, str(device), str(vit_encoder))
    if _ENCODER is None or _ENCODER_SPEC != spec:
        dev = resolve_torch_device(device)
        m = ViTPoseSmallBackbone256()
        m.load_vitpose_checkpoint(Path(ck))
        m.freeze_all_for_export()
        m.eval()
        m.to(dev)
        _ENCODER = m
        _ENCODER_SPEC = spec
    assert _ENCODER is not None
    return _ENCODER, OUT_DIM


def resolve_torch_device(device: str) -> torch.device:
    s = str(device).lower()
    if s == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if s == "mps" and getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@torch.inference_mode()
def embed_crop_bgr_paper(encoder: ViTPoseSmallBackbone256, crop_bgr: np.ndarray, device: torch.device) -> np.ndarray:
    if crop_bgr is None or crop_bgr.size == 0 or crop_bgr.shape[0] < 2 or crop_bgr.shape[1] < 2:
        return np.zeros(OUT_DIM, dtype=np.float32)
    was_training = encoder.training
    encoder.eval()
    z = encoder.forward_bgr(crop_bgr, device)
    if was_training:
        encoder.train()
    return z.squeeze(0).float().cpu().numpy().astype(np.float32)
