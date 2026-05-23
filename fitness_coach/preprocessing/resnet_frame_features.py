"""
YOLO26 pose → person crops → frozen **torchvision ResNet** ImageNet embeddings per frame.

Mirrors ``vit_frame_features_from_yolo_video``: crops use **pixels reconstructed from**
``apply_keypoint_preprocessing_pipeline`` with ``POSEPULSE_DIAGRAM_TECHNIQUES_FRAME_ALIGNED``
(spatial → torso normalization → temporal)
conceptual cleaned skeleton (``final_keypoints`` → denorm for bbox), with the same
``strict_pipeline_crops`` / ``return_conceptual_cleaned_keypoints`` behaviour as ViT.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
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
from fitness_coach.preprocessing.vit_frame_features import _bbox_from_conceptual_cleaned_pixels


_RESNET_ENCODER: Optional["PersonCropResNetBackbone"] = None
_RESNET_SPEC: Optional[Tuple[str, str]] = None


class PersonCropResNetBackbone(nn.Module):
    """ImageNet ResNet on 224×224 person crops → one embedding per crop (fc removed)."""

    def __init__(self, variant: str = "resnet50", device: str = "cpu"):
        super().__init__()
        try:
            import torchvision.models as tvm
        except ImportError as e:  # pragma: no cover
            raise ImportError("torchvision is required for ResNet frame features.") from e

        v = str(variant).strip().lower().replace("-", "_")
        if v == "resnet18":
            m = tvm.resnet18(weights=tvm.ResNet18_Weights.IMAGENET1K_V1)
        elif v == "resnet34":
            m = tvm.resnet34(weights=tvm.ResNet34_Weights.IMAGENET1K_V1)
        elif v == "resnet50":
            m = tvm.resnet50(weights=tvm.ResNet50_Weights.IMAGENET1K_V1)
        elif v == "resnet101":
            m = tvm.resnet101(weights=tvm.ResNet101_Weights.IMAGENET1K_V1)
        else:
            raise ValueError(f"Unsupported ResNet variant {variant!r}; use resnet18|34|50|101")

        backbone_dim = int(m.fc.in_features)
        m.fc = nn.Identity()
        self.device = torch.device("cpu")
        if str(device) == "cuda" and torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif str(device) == "mps" and getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            self.device = torch.device("mps")

        self.backbone = m
        self.backbone.eval()
        self.backbone.to(self.device)

        self.out_dim = backbone_dim
        mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32, device=self.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32, device=self.device).view(1, 3, 1, 1)
        self.register_buffer("_mean", mean, persistent=False)
        self.register_buffer("_std", std, persistent=False)
        self.img_size = 224
        self.variant = v

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


def _get_resnet_encoder(variant: str, device: str) -> PersonCropResNetBackbone:
    global _RESNET_ENCODER, _RESNET_SPEC
    spec = (str(variant).strip().lower(), str(device))
    if _RESNET_ENCODER is None or _RESNET_SPEC != spec:
        _RESNET_ENCODER = PersonCropResNetBackbone(variant=spec[0], device=spec[1])
        _RESNET_SPEC = spec
    return _RESNET_ENCODER


def resnet_frame_features_from_yolo_video(
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
    resnet_variant: str = "resnet50",
    resnet_device: str = "cpu",
    bbox_margin: float = 0.12,
    strict_pipeline_crops: bool = True,
    return_conceptual_cleaned_keypoints: bool = True,
) -> Optional[Tuple[np.ndarray, Dict[str, Any]]]:
    """
    Same detector / preprocessing stack as ViT crop embeddings; encoder is torchvision ResNet.

    Returns ``(frame_features, meta)`` with ``frame_features`` of shape ``(T, D)`` where ``D`` is 512 for
    ResNet-{18,34} and 2048 for ResNet-{50,101}.
    """
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
            "resnet_frame_features: preprocessing changed length (unexpected without fps_sync); aborting.",
            file=sys.stderr,
        )
        return None

    enc = _get_resnet_encoder(resnet_variant, resnet_device)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None
    feats: List[np.ndarray] = []

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
        "pose_backend": "yolo26_resnet_backbone",
        "yolo_pose_model": str(yolo_pose_model),
        "resnet_variant": str(enc.variant),
        "resnet_device": str(resnet_device),
        "feat_dim": int(fe.shape[1]),
        "techniques_json": json.dumps(processed.get("techniques_applied", {})),
        "preprocessing_techniques": tech,
        "vit_encoder_description": f"torchvision.{enc.variant}",
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
                "resnet_frame_features: conceptual_cleaned_keypoints length mismatch; omitting.",
                file=sys.stderr,
            )
        else:
            meta["conceptual_cleaned_keypoints"] = ck
    return fe, meta
