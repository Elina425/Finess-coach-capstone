"""
YOLO26-Pose COCO-17 → PosePulse diagram **Row 1** (steps 2–4): **spatial imputation** → **torso
normalisation** → **temporal imputation** via ``POSEPULSE_DIAGRAM_TECHNIQUES_FRAME_ALIGNED`` /
``apply_keypoint_preprocessing_pipeline``, **without** ``fps_sync`` so each timestep stays aligned with
a real video frame for RGB person crops.

**Paper path (default, ``vit_feature_encoder='paper'``):** person crops come **only** from the
conceptual cleaned skeleton — ``apply_keypoint_preprocessing_pipeline`` outputs
``final_keypoints[t]``, mapped back to pixels with ``keypoints_pixels_for_crop`` (torso denorm meta)
for the bbox ViTPose reads from RGB. Raw YOLO joints are **not** used for crops when
``strict_pipeline_crops=True`` (default); frames with no visible joints after denorm yield a **zero**
embedding (no detector fallback). Set ``strict_pipeline_crops=False`` to allow YOLO box / raw
keypoints for crops on those frames.

Optional ``return_conceptual_cleaned_keypoints=True`` attaches ``conceptual_cleaned_keypoints``
``(T, 17, 2)`` = stacked ``final_keypoints`` (**normalized** coordinates when normalization ran in the
pipeline — same space as angle / mixed features after Row 1). Use for late fusion or for building
42-D mixed vectors next to ``frame_features``.

**ViTPose-S** COCO weights (MAE ViT-S stem + COCO fine-tuning) run as a **frozen** backbone; patch
tokens are **mean-pooled** and mapped with a frozen **384→256** linear head to ``v_t ∈ R^{256}``.

**Legacy path (``vit_feature_encoder='timm'``):** ImageNet ViT (timm) on the same crops — optional
for ablations; not the ViTPose-S checkpoint.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from fitness_coach.core.pose_estimation_core import (
    POSEPULSE_DIAGRAM_TECHNIQUES_FRAME_ALIGNED,
    UltralyticsCOCOPoseDetector,
    VideoProcessor,
    apply_keypoint_preprocessing_pipeline,
    frame_passes_bilateral_coco17,
    keypoints_pixels_for_crop,
    BILATERAL_LIMB_PAIRS,
    BILATERAL_LIMB_PAIRS_WITH_ANKLES,
)


def bbox_from_coco17_pixels(
    kp: np.ndarray,
    width: int,
    height: int,
    margin: float = 0.12,
) -> Tuple[int, int, int, int]:
    """Axis-aligned box from visible joints, expanded by ``margin`` fraction of size, clipped."""
    kp = np.asarray(kp, dtype=np.float64).reshape(17, 2)
    valid = np.any(kp != 0.0, axis=1)
    if not valid.any():
        return 0, 0, max(1, width - 1), max(1, height - 1)
    xy = kp[valid]
    x0, y0 = float(xy[:, 0].min()), float(xy[:, 1].min())
    x1, y1 = float(xy[:, 0].max()), float(xy[:, 1].max())
    bw, bh = max(1.0, x1 - x0), max(1.0, y1 - y0)
    mx, my = margin * bw, margin * bh
    x0, x1 = x0 - mx, x1 + mx
    y0, y1 = y0 - my, y1 + my
    x0i = int(max(0, np.floor(x0)))
    y0i = int(max(0, np.floor(y0)))
    x1i = int(min(width - 1, np.ceil(x1)))
    y1i = int(min(height - 1, np.ceil(y1)))
    if x1i <= x0i:
        x1i = min(width - 1, x0i + 1)
    if y1i <= y0i:
        y1i = min(height - 1, y0i + 1)
    return x0i, y0i, x1i, y1i


def _bbox_from_conceptual_cleaned_pixels(
    kp_px: np.ndarray,
    width: int,
    height: int,
    bbox_margin: float,
    *,
    strict_pipeline_crops: bool,
    pose_result: Any,
) -> Tuple[int, int, int, int]:
    """Crop box from denormalised pipeline keypoints, optional YOLO/raw fallback."""
    kp_px = np.asarray(kp_px, dtype=np.float32)
    vis = np.any(kp_px != 0.0, axis=1)
    if vis.any():
        return bbox_from_coco17_pixels(kp_px, width, height, margin=float(bbox_margin))
    if strict_pipeline_crops:
        return 0, 0, 1, 1
    box = getattr(pose_result, "person_box_xyxy", None)
    if box is not None:
        return expand_xyxy_margin_pixels(box, width, height, float(bbox_margin))
    return bbox_from_coco17_pixels(pose_result.keypoints, width, height, margin=float(bbox_margin))


def expand_xyxy_margin_pixels(
    xyxy: np.ndarray,
    width: int,
    height: int,
    margin: float,
) -> Tuple[int, int, int, int]:
    """Expand a pixel ``xyxy`` box by ``margin`` times local width/height; clip to image."""
    x0, y0, x1, y1 = [float(v) for v in np.asarray(xyxy, dtype=np.float64).reshape(4)]
    bw, bh = max(1.0, x1 - x0), max(1.0, y1 - y0)
    mx, my = float(margin) * bw, float(margin) * bh
    x0, x1 = x0 - mx, x1 + mx
    y0, y1 = y0 - my, y1 + my
    x0i = int(max(0, np.floor(x0)))
    y0i = int(max(0, np.floor(y0)))
    x1i = int(min(width - 1, np.ceil(x1)))
    y1i = int(min(height - 1, np.ceil(y1)))
    if x1i <= x0i:
        x1i = min(width - 1, x0i + 1)
    if y1i <= y0i:
        y1i = min(height - 1, y0i + 1)
    return x0i, y0i, x1i, y1i


_VIT_ENCODER: Optional["PersonCropViTBackbone"] = None
_VIT_SPEC: Optional[Tuple[str, str]] = None


class PersonCropViTBackbone(torch.nn.Module):
    """ImageNet ViT (timm) on fixed-size person crops → one embedding per crop."""

    def __init__(self, model_name: str = "vit_small_patch16_224", device: str = "cpu"):
        super().__init__()
        try:
            import timm
            from timm.data import resolve_model_data_config
        except ImportError as e:
            raise ImportError(
                "timm is required for ViT backbone features. Install: pip install timm"
            ) from e

        self.device = torch.device("cpu")
        if str(device) == "cuda" and torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif str(device) == "mps" and getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            self.device = torch.device("mps")

        self.backbone = timm.create_model(str(model_name), pretrained=True, num_classes=0)
        self.backbone.eval()
        self.backbone.to(self.device)
        data_config = resolve_model_data_config(self.backbone)
        mean = torch.tensor(data_config["mean"], dtype=torch.float32, device=self.device).view(
            1, 3, 1, 1
        )
        std = torch.tensor(data_config["std"], dtype=torch.float32, device=self.device).view(
            1, 3, 1, 1
        )
        self.register_buffer("_mean", mean, persistent=False)
        self.register_buffer("_std", std, persistent=False)
        self.img_size = int(data_config["input_size"][-1])
        self.out_dim = int(self.backbone.num_features)

    @torch.inference_mode()
    def embed_crop_bgr(self, crop_bgr: np.ndarray) -> np.ndarray:
        if crop_bgr is None or crop_bgr.size == 0 or crop_bgr.shape[0] < 2 or crop_bgr.shape[1] < 2:
            return np.zeros(self.out_dim, dtype=np.float32)
        rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        t = torch.from_numpy(rgb).float().permute(2, 0, 1).unsqueeze(0) / 255.0
        t = F.interpolate(
            t,
            size=(self.img_size, self.img_size),
            mode="bilinear",
            align_corners=False,
        )
        t = t.to(self.device)
        t = (t - self._mean) / self._std
        z = self.backbone(t)
        return z.squeeze(0).float().cpu().numpy().astype(np.float32)


def _get_vit_encoder(model_name: str, device: str) -> PersonCropViTBackbone:
    global _VIT_ENCODER, _VIT_SPEC
    spec = (str(model_name), str(device))
    if _VIT_ENCODER is None or _VIT_SPEC != spec:
        _VIT_ENCODER = PersonCropViTBackbone(model_name=spec[0], device=spec[1])
        _VIT_SPEC = spec
    return _VIT_ENCODER


def vit_frame_features_from_yolo_video(
    video_path: Path,
    max_frames: Optional[int],
    *,
    yolo_pose_model: str = "yolo26n-pose.pt",
    bilateral_filter: bool = False,
    bilateral_conf_tau: float = 0.3,
    bilateral_include_ankles: bool = False,
    detection_stride: int = 1,
    detection_max_long_edge: int = 0,
    source_fps: Optional[float] = None,
    target_fps: float = 30.0,
    preprocessing_techniques: Optional[List[str]] = None,
    vit_feature_encoder: str = "paper",
    vitpose_checkpoint: Optional[str] = None,
    vit_model_name: str = "vit_small_patch16_224",
    vit_device: str = "cpu",
    bbox_margin: float = 0.12,
    strict_pipeline_crops: bool = True,
    return_conceptual_cleaned_keypoints: bool = True,
) -> Optional[Tuple[np.ndarray, Dict[str, Any]]]:
    """
    YOLO26 → bilateral (optional) → keypoint preprocessing **without fps_sync** → per-frame crop embeddings.

    Returns ``(frame_features, meta)`` with ``frame_features`` of shape ``(T, D)`` (``D=256`` for
    ``vit_feature_encoder='paper'``).
    """
    mode = str(vit_feature_encoder or "paper").strip().lower()
    if mode not in ("paper", "timm"):
        print(f"vit_frame_features: unknown vit_feature_encoder {mode!r}, use paper|timm", file=sys.stderr)
        return None

    yolo = UltralyticsCOCOPoseDetector(model_name=str(yolo_pose_model), quiet=True)
    if not yolo.available:
        print(f"Ultralytics pose unavailable ({yolo_pose_model})", file=sys.stderr)
        return None

    vp = VideoProcessor(str(video_path))
    stride = max(1, int(detection_stride))
    pose_results = vp.process_with_detector(
        yolo,
        max_frames=max_frames,
        detection_stride=stride,
        detection_max_long_edge=int(detection_max_long_edge),
    )
    measured_fps = float(vp.fps) if vp.fps and vp.fps > 1e-3 else 30.0
    if stride > 1:
        measured_fps = measured_fps / float(stride)
    w_img, h_img = int(vp.width), int(vp.height)
    vp.close()

    if not pose_results:
        return None

    pairs = BILATERAL_LIMB_PAIRS_WITH_ANKLES if bilateral_include_ankles else BILATERAL_LIMB_PAIRS
    if bilateral_filter:
        pose_results = [
            r
            for r in pose_results
            if frame_passes_bilateral_coco17(
                r.confidence, float(bilateral_conf_tau), pairs=pairs
            )
        ]
    if not pose_results:
        return None

    kp_seq = [np.asarray(r.keypoints, dtype=np.float32).copy() for r in pose_results]
    conf_seq = [np.asarray(r.confidence, dtype=np.float32).copy() for r in pose_results]

    tech = (
        list(preprocessing_techniques)
        if preprocessing_techniques
        else list(POSEPULSE_DIAGRAM_TECHNIQUES_FRAME_ALIGNED)
    )
    tech = [t for t in tech if t != "fps_sync"]
    _has_imp = (
        "imputation" in tech
        or "spatial_imputation" in tech
        or "temporal_impute" in tech
        or "temporal_imputation" in tech
    )
    if not _has_imp and "normalization" not in tech:
        tech = list(POSEPULSE_DIAGRAM_TECHNIQUES_FRAME_ALIGNED)

    src = float(source_fps) if source_fps is not None and source_fps > 1e-6 else measured_fps
    processed = apply_keypoint_preprocessing_pipeline(
        kp_seq,
        conf_seq,
        preprocessing_techniques=tech,
        target_fps=float(target_fps),
        source_fps=src,
        original_frames=len(kp_seq),
    )
    if len(processed["final_keypoints"]) != len(pose_results):
        print(
            "vit_frame_features: preprocessing changed length (unexpected without fps_sync); aborting.",
            file=sys.stderr,
        )
        return None

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None
    feats: List[np.ndarray] = []

    if mode == "paper":
        from fitness_coach.preprocessing.vitpose_paper_encoder import (
            OUT_DIM,
            embed_crop_bgr_paper,
            get_vitpose_paper_encoder,
            resolve_torch_device,
        )

        ck = Path(vitpose_checkpoint).expanduser().resolve() if vitpose_checkpoint else None
        try:
            enc, _out_dim_chk = get_vitpose_paper_encoder(
                checkpoint_path=ck,
                device=str(vit_device),
                vit_encoder="paper",
            )
        except Exception as e:
            print(f"ViTPose-S encoder failed ({e})", file=sys.stderr)
            cap.release()
            return None
        dev = resolve_torch_device(str(vit_device))
        vit_ck_meta = str(ck) if ck is not None else "default_cache"
        out_dim_fill = int(OUT_DIM)
        for i, r in enumerate(pose_results):
            fi = int(r.frame_idx)
            cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
            ok, frame = cap.read()
            if not ok or frame is None:
                feats.append(np.zeros(out_dim_fill, dtype=np.float32))
                continue
            h, w = frame.shape[:2]
            kp_px = keypoints_pixels_for_crop(processed, i)
            x0, y0, x1, y1 = _bbox_from_conceptual_cleaned_pixels(
                kp_px,
                w,
                h,
                float(bbox_margin),
                strict_pipeline_crops=bool(strict_pipeline_crops),
                pose_result=r,
            )
            crop = frame[y0 : y1 + 1, x0 : x1 + 1]
            feats.append(embed_crop_bgr_paper(enc, crop, dev))
        meta_encoder = "vitpose_s_coco_mmpose_weights"
        meta_dim = int(OUT_DIM)
    else:
        try:
            enc = _get_vit_encoder(vit_model_name, vit_device)
        except ImportError as e:
            print(str(e), file=sys.stderr)
            cap.release()
            return None
        meta_encoder = f"timm:{vit_model_name}"
        meta_dim = int(enc.out_dim)
        for i, r in enumerate(pose_results):
            fi = int(r.frame_idx)
            cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
            ok, frame = cap.read()
            if not ok or frame is None:
                feats.append(np.zeros(enc.out_dim, dtype=np.float32))
                continue
            h, w = frame.shape[:2]
            kp_px = keypoints_pixels_for_crop(processed, i)
            x0, y0, x1, y1 = _bbox_from_conceptual_cleaned_pixels(
                kp_px,
                w,
                h,
                float(bbox_margin),
                strict_pipeline_crops=bool(strict_pipeline_crops),
                pose_result=r,
            )
            crop = frame[y0 : y1 + 1, x0 : x1 + 1]
            feats.append(enc.embed_crop_bgr(crop))

    cap.release()

    fe = np.stack(feats, axis=0)
    meta: Dict[str, Any] = {
        "pose_backend": "yolo26_vit_backbone",
        "yolo_pose_model": str(yolo_pose_model),
        "vit_feature_encoder": mode,
        "vit_model_name": str(vit_model_name) if mode == "timm" else "vitpose_small_coco_256",
        "vit_device": str(vit_device),
        "feat_dim": int(fe.shape[1]),
        "techniques_json": json.dumps(processed.get("techniques_applied", {})),
        "preprocessing_techniques": tech,
        "vit_encoder_description": meta_encoder,
        "crop_geometry": (
            "conceptual_cleaned_denorm_pixels_strict"
            if strict_pipeline_crops
            else "conceptual_cleaned_denorm_pixels_with_detector_fallback"
        ),
        "strict_pipeline_crops": strict_pipeline_crops,
    }
    if return_conceptual_cleaned_keypoints:
        ck = np.stack(
            [np.asarray(k, dtype=np.float32) for k in processed["final_keypoints"]],
            axis=0,
        )
        if ck.shape[0] != fe.shape[0]:
            print(
                "vit_frame_features: conceptual_cleaned_keypoints length mismatch; omitting.",
                file=sys.stderr,
            )
        else:
            meta["conceptual_cleaned_keypoints"] = ck
    if mode == "paper":
        meta["vitpose_checkpoint"] = vit_ck_meta
        meta["frame_feature_dim_paper"] = meta_dim
    return fe, meta
