"""
Dataset utilities for EgoExo xLSTM multi-task training.

Each sample consists of:
- temporal features (CLIP visual, pose NPZ, or annotation-derived)
- exercise class label
- quality score normalized to [0, 1]
- multi-hot error-tag targets derived from interpretable judgement annotations

Feature modes:
- "clip" (recommended): loads pre-extracted CLIP ViT-B/32 frame features (512-dim)
  from .pth files, sliced by action frame boundaries.
- "annotation": encodes verification checks + text into features from CSV alone.
- "angles", "coords", "mixed": require extracted pose NPZ files on disk.
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

try:
    from fitness_coach.core.biomechanical_features import (
        compute_coords_only_sequence_features,
        compute_mixed_sequence_features,
        keypoints_npz_coords_already_normalized,
    )
    from fitness_coach.datasets.exercise_bilstm_dataset import load_angles_npz, make_windows

    _HAS_POSE = True
except ImportError:
    _HAS_POSE = False


ANNOTATION_FEATURE_DIM = 64
CLIP_FEATURE_DIM = 512
DEFAULT_CLIP_VIEW = "ego_l"
CLIP_SUBDIR = "EgoExo_Fitness_CLIP_Vid_Feat_w_Rotate"

_clip_cache: Dict[str, np.ndarray] = {}


def load_clip_segment(
    clip_features_root: Path,
    record_id: str,
    frame_start: int,
    frame_end: int,
    view: str = DEFAULT_CLIP_VIEW,
    max_frames: int = 300,
    subsample_stride: int = 0,
) -> Optional[np.ndarray]:
    """Load a CLIP feature segment for one action clip, with optional subsampling."""
    cache_key = f"{record_id}/{view}"
    if cache_key not in _clip_cache:
        pth_path = clip_features_root / CLIP_SUBDIR / record_id / view / "clip_vit_b32_vid_frame_feat.pth"
        if not pth_path.is_file():
            for fallback in ("exo_m", "ego_m", "exo_l"):
                alt = clip_features_root / CLIP_SUBDIR / record_id / fallback / "clip_vit_b32_vid_frame_feat.pth"
                if alt.is_file():
                    pth_path = alt
                    break
            else:
                return None
        data = torch.load(pth_path, map_location="cpu", weights_only=False)
        _clip_cache[cache_key] = data["clip_feat"].numpy().astype(np.float32)
    all_frames = _clip_cache[cache_key]
    end = min(frame_end, all_frames.shape[0])
    start = max(0, frame_start)
    if end <= start:
        return None
    segment = all_frames[start:end]
    if subsample_stride > 1:
        segment = segment[::subsample_stride]
    if segment.shape[0] > max_frames:
        indices = np.linspace(0, segment.shape[0] - 1, max_frames, dtype=int)
        segment = segment[indices]
    return segment


ERROR_TAGS: Tuple[str, ...] = (
    "alignment",
    "balance_stability",
    "back_not_straight",
    "elbows_flared",
    "hip_position",
    "incomplete_extension",
    "insufficient_depth",
    "knees_too_far_forward",
    "range_of_motion",
    "shoulder_instability",
    "tempo_control",
)

ERROR_KEYWORDS: Dict[str, Tuple[str, ...]] = {
    "alignment": (
        "straight line",
        "alignment",
        "line from the side",
        "body in a straight line",
    ),
    "balance_stability": (
        "balance",
        "stable",
        "stability",
        "wobble",
        "control your body",
    ),
    "back_not_straight": (
        "back straight",
        "back was not kept straight",
        "rounded back",
        "lean forward",
        "leaning forward",
        "torso",
    ),
    "elbows_flared": (
        "elbows",
        "arms too wide",
        "wider than shoulder-width",
        "flare",
    ),
    "hip_position": (
        "hip",
        "hips",
        "sag",
        "pike",
    ),
    "incomplete_extension": (
        "stretch your arms",
        "straighten",
        "extend",
        "lockout",
        "fully extended",
    ),
    "insufficient_depth": (
        "depth",
        "descent",
        "insufficient",
        "not low enough",
        "lower down",
    ),
    "knees_too_far_forward": (
        "knees over toes",
        "knees too far forward",
        "knees forward",
    ),
    "range_of_motion": (
        "range of motion",
        "full range",
        "restore",
        "not complete",
        "insufficient",
    ),
    "shoulder_instability": (
        "shoulder",
        "shrug",
        "unstable shoulder",
    ),
    "tempo_control": (
        "too fast",
        "too slow",
        "tempo",
        "control",
    ),
}


def normalize_quality_score(value: float) -> float:
    """Keep [0,1] as-is; map [1,5] to [0,1]; clamp otherwise."""
    value = float(value)
    if 0.0 <= value <= 1.0:
        return value
    if 1.0 <= value <= 5.0:
        return max(0.0, min(1.0, (value - 1.0) / 4.0))
    return max(0.0, min(1.0, value))


def _collect_error_text(row: Dict[str, str]) -> str:
    texts: List[str] = []
    verification_raw = (row.get("verification_json") or "").strip()
    if verification_raw:
        try:
            verification_items = json.loads(verification_raw)
        except json.JSONDecodeError:
            verification_items = []
        if isinstance(verification_items, list):
            for item in verification_items:
                if not isinstance(item, dict):
                    continue
                if bool(item.get("ok")):
                    continue
                text = str(item.get("text") or "").strip()
                if text:
                    texts.append(text)
    for field in ("comment", "action_guidance"):
        text = (row.get(field) or "").strip()
        if text:
            texts.append(text)
    return " ".join(texts).lower()


def derive_error_targets(row: Dict[str, str]) -> np.ndarray:
    text = _collect_error_text(row)
    target = np.zeros(len(ERROR_TAGS), dtype=np.float32)
    if text:
        for idx, tag in enumerate(ERROR_TAGS):
            patterns = ERROR_KEYWORDS.get(tag, ())
            if any(pattern in text for pattern in patterns):
                target[idx] = 1.0
    if target.sum() == 0:
        quality = normalize_quality_score(float(row.get("quality", 0.5)))
        if quality < 0.45:
            target[ERROR_TAGS.index("alignment")] = 1.0
    return target


def _hash_text_to_floats(text: str, dim: int) -> np.ndarray:
    blob = hashlib.sha256(text.encode("utf-8", errors="replace")).digest()
    need = dim * 4
    buf = (blob * ((need // len(blob)) + 2))[:need]
    ints = np.frombuffer(buf, dtype=np.uint32).astype(np.float32)
    return (ints / np.float32(2**31)) - 1.0


def _encode_verification_step(instruction: str, ok: bool, dim: int) -> np.ndarray:
    vec = np.zeros(dim, dtype=np.float32)
    vec[0] = 1.0 if ok else -1.0
    hashed = _hash_text_to_floats(instruction.lower().strip(), dim - 1)
    vec[1:] = hashed[: dim - 1]
    return vec


def encode_annotation_row(row: Dict[str, str], feature_dim: int = ANNOTATION_FEATURE_DIM) -> np.ndarray:
    """Encode one CSV row's annotations into a (T, feature_dim) sequence."""
    steps: List[np.ndarray] = []
    vj_raw = (row.get("verification_json") or "").strip()
    if vj_raw:
        try:
            items = json.loads(vj_raw)
        except json.JSONDecodeError:
            items = []
        if isinstance(items, list):
            for item in items:
                if not isinstance(item, dict):
                    continue
                text = str(item.get("text") or "").strip()
                ok = bool(item.get("ok"))
                if text:
                    steps.append(_encode_verification_step(text, ok, feature_dim))

    comment = (row.get("comment") or "").strip()
    if comment:
        steps.append(_encode_verification_step(comment, True, feature_dim))
    guidance = (row.get("action_guidance") or "").strip()
    if guidance:
        steps.append(_encode_verification_step(guidance, True, feature_dim))
    if not steps:
        steps.append(np.zeros(feature_dim, dtype=np.float32))
    return np.stack(steps, axis=0)


def fit_feature_standardizer(samples: Sequence[Tuple[np.ndarray, int, float, np.ndarray]]) -> Tuple[np.ndarray, np.ndarray]:
    xs = np.concatenate([sample[0].reshape(-1, sample[0].shape[-1]) for sample in samples], axis=0)
    mean = xs.mean(axis=0).astype(np.float32)
    std = (xs.std(axis=0) + 1e-8).astype(np.float32)
    return mean, std


def apply_feature_standardizer(
    samples: List[Tuple[np.ndarray, int, float, np.ndarray]],
    mean: np.ndarray,
    std: np.ndarray,
) -> None:
    for idx, (x, y, q, e) in enumerate(samples):
        xn = (x.astype(np.float32) - mean) / std
        samples[idx] = (xn.astype(np.float32), y, q, e)


class EgoExoXLSTMDataset(Dataset):
    """EgoExo dataset for multitask xLSTM training.

    Feature modes:
    - "clip": pre-extracted CLIP ViT-B/32 visual features (512-dim per frame).
    - "annotation": encodes verification checks + text from CSV only.
    - "angles"/"coords"/"mixed": requires extracted pose NPZ files.
    """

    def __init__(
        self,
        index_csv: Path,
        class_to_idx: Dict[str, int],
        split: str,
        *,
        feature_mode: str = "clip",
        angles_dir: Optional[Path] = None,
        keypoints_dir: Optional[Path] = None,
        clip_features_root: Optional[Path] = None,
        clip_view: str = DEFAULT_CLIP_VIEW,
        clip_max_frames: int = 300,
        clip_subsample_stride: int = 3,
        window: int = 0,
        stride: int = 0,
        max_seq_len: int = 96,
    ):
        self.samples: List[Tuple[np.ndarray, int, float, np.ndarray]] = []
        # Parallel metadata used by the comment-generation head; aligned 1:1 with samples.
        # Each entry: {"comment": str, "class_name": str, "guidance": str}.
        self.metadata: List[Dict[str, str]] = []
        self.idx_to_class: Dict[int, str] = {idx: name for name, idx in class_to_idx.items()}
        self.feature_mode = str(feature_mode)
        self.window = int(window)
        self.stride = int(stride)
        self.max_seq_len = int(max_seq_len)

        with open(index_csv, newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                if (row.get("split") or "train") != split:
                    continue
                cls_name = (row.get("exercise_class") or "").strip()
                if cls_name not in class_to_idx:
                    continue
                stem = (row.get("video_stem") or "").strip()
                if not stem:
                    continue

                if self.feature_mode == "clip":
                    sequence = self._load_clip_sequence(
                        row,
                        clip_features_root=clip_features_root,
                        clip_view=clip_view,
                        max_frames=clip_max_frames,
                        subsample_stride=clip_subsample_stride,
                    )
                elif self.feature_mode == "annotation":
                    sequence = encode_annotation_row(row, ANNOTATION_FEATURE_DIM)
                    if sequence.shape[0] > self.max_seq_len:
                        sequence = sequence[-self.max_seq_len:]
                else:
                    sequence = self._load_pose_sequence(
                        stem,
                        feature_mode=self.feature_mode,
                        angles_dir=angles_dir,
                        keypoints_dir=keypoints_dir,
                    )
                if sequence is None or sequence.shape[0] == 0:
                    continue

                y_cls = class_to_idx[cls_name]
                quality = normalize_quality_score(float(row["quality"]))
                error_targets = derive_error_targets(row)

                comment_text = (row.get("comment") or "").strip()
                guidance_text = (row.get("action_guidance") or "").strip()
                meta = {
                    "comment": comment_text,
                    "class_name": cls_name,
                    "guidance": guidance_text,
                }

                use_windowing = self.feature_mode not in ("annotation", "clip") and self.window > 0 and _HAS_POSE
                if use_windowing:
                    for windowed in make_windows(sequence, self.window, self.stride or max(1, self.window // 2)):
                        self.samples.append((windowed, y_cls, quality, error_targets.copy()))
                        self.metadata.append(dict(meta))
                else:
                    self.samples.append((sequence, y_cls, quality, error_targets.copy()))
                    self.metadata.append(dict(meta))

    @staticmethod
    def _load_clip_sequence(
        row: Dict[str, str],
        *,
        clip_features_root: Optional[Path],
        clip_view: str,
        max_frames: int,
        subsample_stride: int,
    ) -> Optional[np.ndarray]:
        if clip_features_root is None:
            return None
        jk = (row.get("judgement_key") or "").strip()
        if not jk or "_action_" not in jk:
            return None
        record_id = jk.split("_action_")[0]
        try:
            fs = int(row.get("frame_start", 0))
            fe = int(row.get("frame_end", 0))
        except (ValueError, TypeError):
            return None
        if fe <= fs:
            return None
        return load_clip_segment(
            clip_features_root,
            record_id,
            fs,
            fe,
            view=clip_view,
            max_frames=max_frames,
            subsample_stride=subsample_stride,
        )

    @staticmethod
    def _load_pose_sequence(
        stem: str,
        *,
        feature_mode: str,
        angles_dir: Optional[Path],
        keypoints_dir: Optional[Path],
    ) -> Optional[np.ndarray]:
        if not _HAS_POSE:
            return None
        if feature_mode == "angles":
            if angles_dir is None:
                return None
            angle_path = angles_dir / f"{stem}_biomechanics.npz"
            if not angle_path.is_file():
                return None
            try:
                return load_angles_npz(angle_path).astype(np.float32)
            except Exception:
                return None
        if keypoints_dir is None:
            return None
        keypoint_path = keypoints_dir / f"{stem}_keypoints.npz"
        if not keypoint_path.is_file():
            return None
        try:
            data = np.load(keypoint_path, allow_pickle=True)
            keypoints = np.asarray(data["keypoints"], dtype=np.float64)
            coords_ok = keypoints_npz_coords_already_normalized(data)
            if feature_mode == "coords":
                return compute_coords_only_sequence_features(keypoints, coords_already_normalized=coords_ok).astype(np.float32)
            if feature_mode == "mixed":
                mixed, _ = compute_mixed_sequence_features(keypoints, coords_already_normalized=coords_ok)
                return mixed.astype(np.float32)
        except Exception:
            return None
        return None

    def feature_dim(self) -> int:
        if self.samples:
            return int(self.samples[0][0].shape[-1])
        if self.feature_mode == "clip":
            return CLIP_FEATURE_DIM
        if self.feature_mode == "annotation":
            return ANNOTATION_FEATURE_DIM
        return 0

    def __len__(self) -> int:
        return len(self.samples)

    def build_guidance_table(self, class_to_idx: Dict[str, int]) -> Dict[int, str]:
        """Build a {class_idx → action_guidance} map from the training metadata.

        Picks the most frequent guidance string per class (handles minor annotator
        wording drift). Empty strings are skipped.
        """
        from collections import Counter

        per_class: Dict[int, Counter] = {}
        for meta in self.metadata:
            cls_name = meta.get("class_name", "")
            guide = (meta.get("guidance") or "").strip()
            if not cls_name or not guide:
                continue
            idx = class_to_idx.get(cls_name)
            if idx is None:
                continue
            per_class.setdefault(idx, Counter())[guide] += 1
        return {idx: counter.most_common(1)[0][0] for idx, counter in per_class.items()}

    def __getitem__(self, idx: int):
        x, y, q, e = self.samples[idx]
        meta = self.metadata[idx] if idx < len(self.metadata) else {"comment": "", "class_name": "", "guidance": ""}
        return (
            torch.from_numpy(np.asarray(x, dtype=np.float32)),
            torch.tensor(y, dtype=torch.long),
            torch.tensor(q, dtype=torch.float32),
            torch.from_numpy(np.asarray(e, dtype=np.float32)),
            meta.get("comment", ""),
            meta.get("class_name", ""),
        )


def egoexo_collate_fn(batch):
    """Pad variable-length annotation/CLIP sequences to the longest in the batch.

    Returns a 6-tuple: (features, y_cls, y_q, y_err, comments, class_names) where
    `comments` and `class_names` are plain Python lists (consumed by the comment head).
    """
    items = list(zip(*batch))
    if len(items) == 4:
        xs, yc, yq, ye = items
        comments: Tuple[str, ...] = tuple("" for _ in xs)
        class_names: Tuple[str, ...] = tuple("" for _ in xs)
    else:
        xs, yc, yq, ye, comments, class_names = items
    lengths = [x.shape[0] for x in xs]
    max_t = max(lengths)
    feat_dim = xs[0].shape[-1]
    padded = torch.zeros(len(xs), max_t, feat_dim, dtype=torch.float32)
    for i, x in enumerate(xs):
        padded[i, : x.shape[0]] = x
    return (
        padded,
        torch.stack(yc),
        torch.stack(yq),
        torch.stack(ye),
        list(comments),
        list(class_names),
    )
