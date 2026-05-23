"""
Extract OpenAI CLIP ViT-B/32 **512-D** frame embeddings from a video, compatible with
EgoExo ``clip_vit_b32_vid_frame_feat.pth`` features used by ``EgoExoXLSTMDataset``.

Uses HuggingFace ``openai/clip-vit-base-patch32`` (same architecture as the dataset).
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


def _frame_indices(total_frames: int, stride: int, max_frames: int) -> List[int]:
    if total_frames <= 0:
        return []
    raw = np.arange(0, total_frames, max(1, stride), dtype=np.int64)
    if raw.size == 0:
        raw = np.array([0], dtype=np.int64)
    if raw.size > max_frames:
        pick = np.linspace(0, raw.size - 1, max_frames, dtype=int)
        raw = raw[pick]
    return [int(x) for x in raw.tolist()]


@dataclass
class CLIPViTB32Extractor:
    """Loads HF CLIP once; reuse across many videos."""

    model_name: str = "openai/clip-vit-base-patch32"
    device: str = "cpu"

    def __post_init__(self) -> None:
        import torch
        from transformers import CLIPModel, CLIPProcessor

        self._torch = torch
        self._processor = CLIPProcessor.from_pretrained(self.model_name)
        d = torch.device(self.device)
        self._model = CLIPModel.from_pretrained(self.model_name).to(d)
        self._model.eval()
        self._device = d

    def embed_frames_rgb(self, frames_rgb: List[np.ndarray], *, batch_size: int = 16) -> np.ndarray:
        from PIL import Image

        if not frames_rgb:
            return np.zeros((0, 512), dtype=np.float32)
        torch_device = self._device
        feats_list: List[np.ndarray] = []
        bs = max(1, int(batch_size))
        with self._torch.no_grad():
            for i in range(0, len(frames_rgb), bs):
                batch = frames_rgb[i : i + bs]
                pil = [Image.fromarray(arr) for arr in batch]
                inputs = self._processor(images=pil, return_tensors="pt").to(torch_device)
                out = self._model.get_image_features(pixel_values=inputs["pixel_values"])
                pooled = getattr(out, "pooler_output", None)
                if pooled is None:
                    pooled = out[0] if isinstance(out, tuple) else out
                feats_list.append(pooled.float().cpu().numpy())
        return np.concatenate(feats_list, axis=0).astype(np.float32, copy=False)


def clip_vit_b32_frames_from_video(
    video_path: Path,
    *,
    max_frames: int = 300,
    subsample_stride: int = 3,
    clip_model_name: str = "openai/clip-vit-base-patch32",
    device: str = "cpu",
    batch_size: int = 16,
    extractor: Optional[CLIPViTB32Extractor] = None,
) -> Optional[Tuple[np.ndarray, Dict[str, Any]]]:
    """Decode ``video_path``, sample frames (stride + cap), run CLIP vision tower.

    Returns ``(features, meta)`` where ``features`` is ``(T, 512)`` float32 — same layout as
    EgoExo precomputed tensors before train-set standardization.

    Pass a shared ``CLIPViTB32Extractor`` to avoid reloading weights for every file.
    """
    try:
        import cv2
    except ImportError as e:
        print(f"clip_frame_features: missing dependency: {e}", file=sys.stderr)
        return None

    path = Path(video_path)
    if not path.is_file():
        print(f"clip_frame_features: missing file {path}", file=sys.stderr)
        return None

    ext = extractor or CLIPViTB32Extractor(model_name=clip_model_name, device=device)

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        print(f"clip_frame_features: could not open {path}", file=sys.stderr)
        return None

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    if total <= 0:
        # Some codecs report 0 — count by reading
        total = 0
        while True:
            ok, _ = cap.read()
            if not ok:
                break
            total += 1
        cap.release()
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            return None

    indices = _frame_indices(total, subsample_stride, max_frames)
    if not indices:
        cap.release()
        return None

    frames_rgb: List[np.ndarray] = []
    for fi in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, float(fi))
        ok, bgr = cap.read()
        if not ok or bgr is None:
            continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        frames_rgb.append(rgb)
    cap.release()

    if not frames_rgb:
        return None

    feats = ext.embed_frames_rgb(frames_rgb, batch_size=batch_size)
    meta = {
        "video_path": str(path),
        "n_frames_read": int(len(frames_rgb)),
        "n_frames_total_reported": int(total),
        "fps_reported": fps,
        "frame_indices": indices[: len(frames_rgb)],
        "subsample_stride": int(subsample_stride),
        "max_frames": int(max_frames),
        "feature_dim": int(feats.shape[-1]),
        "clip_model": clip_model_name,
    }
    return feats, meta
