"""
Dataset utilities for EgoExo xLSTM multi-task training.

Each sample consists of:
- temporal features (CLIP visual, pose NPZ, or annotation-derived)
- exercise class label
- quality targets: ``unit`` (mapped to ``[0, 1]``) or ``likert`` native ``[1, 5]`` (EgoExo paper form)
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
    allow_fallback: bool = False,
) -> Optional[np.ndarray]:
    """Load a CLIP feature segment for one action clip, with optional subsampling.

    By default, ``allow_fallback=False`` — when the requested view's feature
    file is missing, ``None`` is returned. This is the correct behaviour when
    multiple views are being enumerated explicitly (e.g. all six standard
    views): missing-view clips should simply be skipped, not silently
    substituted with a different view's features. Set ``allow_fallback=True``
    only for legacy single-view loads where a stand-in is acceptable.
    """
    cache_key = f"{record_id}/{view}"
    if cache_key not in _clip_cache:
        pth_path = clip_features_root / CLIP_SUBDIR / record_id / view / "clip_vit_b32_vid_frame_feat.pth"
        if not pth_path.is_file():
            if not allow_fallback:
                return None
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


LIKERT_Q_MIN = 1.0
LIKERT_Q_MAX = 5.0


def normalize_quality_score(value: float) -> float:
    """Legacy: squeeze everything to ``[0, 1]`` (``--quality-encoding unit``)."""
    value = float(value)
    if 0.0 <= value <= 1.0:
        return value
    if LIKERT_Q_MIN <= value <= LIKERT_Q_MAX:
        return max(0.0, min(1.0, (value - LIKERT_Q_MIN) / (LIKERT_Q_MAX - LIKERT_Q_MIN)))
    return max(0.0, min(1.0, value))


def canonical_quality_score(raw: float, encoding: str) -> float:
    """Return quality on the supervised axis chosen by ``encoding``.

    ``unit`` — canonical ``[0, 1]`` (``[1,5]`` Likert folded with ``normalize_quality_score``).
    ``likert`` — canonical ``[1, 5]``; values already on ``[0, 1]`` are expanded with ``1 + 4*v``
    (inverse of folding) so older indices stay usable.
    """
    enc = (encoding or "unit").strip().lower()
    v = float(raw)
    if enc == "likert":
        if LIKERT_Q_MIN <= v <= LIKERT_Q_MAX:
            return max(LIKERT_Q_MIN, min(LIKERT_Q_MAX, v))
        if 0.0 <= v <= 1.0:
            return max(LIKERT_Q_MIN, min(LIKERT_Q_MAX, LIKERT_Q_MIN + (LIKERT_Q_MAX - LIKERT_Q_MIN) * v))
        return max(LIKERT_Q_MIN, min(LIKERT_Q_MAX, v))
    # unit path
    return normalize_quality_score(v)


def quality_score_to_bucket(q: float, bucket_edges: Tuple[float, ...]) -> int:
    """Assign ``q`` to ``0 … len(bucket_edges)`` by upper-exclusive bin edges."""
    q = float(q)
    for i, edge in enumerate(bucket_edges):
        if q < edge:
            return i
    return len(bucket_edges)


def quality_bucket_centres(
    bucket_edges: Tuple[float, ...],
    *,
    domain_lo: float = 0.0,
    domain_hi: float = 1.0,
) -> Tuple[float, ...]:
    """Interval midpoints between ``domain_lo``, ``edges``, and ``domain_hi``."""
    dl, dh = float(domain_lo), float(domain_hi)
    cuts = [dl] + list(bucket_edges) + [dh]
    return tuple(0.5 * (cuts[i] + cuts[i + 1]) for i in range(len(cuts) - 1))


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


def derive_error_targets(row: Dict[str, str], *, encoding: str = "unit") -> np.ndarray:
    text = _collect_error_text(row)
    target = np.zeros(len(ERROR_TAGS), dtype=np.float32)
    enc = (encoding or "unit").strip().lower()
    q_lo, q_hi = ((0.0, 1.0) if enc != "likert" else (LIKERT_Q_MIN, LIKERT_Q_MAX))
    default_raw = "3.0" if enc == "likert" else "0.5"
    if text:
        for idx, tag in enumerate(ERROR_TAGS):
            patterns = ERROR_KEYWORDS.get(tag, ())
            if any(pattern in text for pattern in patterns):
                target[idx] = 1.0
    if target.sum() == 0:
        quality = canonical_quality_score(float(row.get("quality", default_raw)), encoding)
        if quality < q_lo + 0.45 * (q_hi - q_lo):
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
    for idx, sample in enumerate(samples):
        x = sample[0]
        xn = (x.astype(np.float32) - mean) / std
        x_new = xn.astype(np.float32)
        if len(sample) == 4:
            _, y, q, e = sample
            samples[idx] = (x_new, int(y), float(q), e)
        elif len(sample) == 5:
            _, y, q, qb, e = sample
            samples[idx] = (x_new, int(y), float(q), int(qb), e)
        else:
            raise ValueError(f"unsupported sample length {len(sample)}")


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
        quality_encoding: str = "likert",
        angles_dir: Optional[Path] = None,
        keypoints_dir: Optional[Path] = None,
        clip_features_root: Optional[Path] = None,
        clip_view: str = DEFAULT_CLIP_VIEW,
        clip_max_frames: int = 300,
        clip_subsample_stride: int = 3,
        window: int = 0,
        stride: int = 0,
        max_seq_len: int = 96,
        filter_null_comments: bool = False,
    ):
        # When True, clips whose `comment` field is null/empty after stripping
        # whitespace are dropped at construction time. Used in the retrieval-based
        # feedback design (no LM head) where null comments contribute nothing to
        # the comment-lookup table and only inflate per-batch memory.
        self.filter_null_comments = bool(filter_null_comments)
        qe = str(quality_encoding or "likert").strip().lower()
        self.quality_encoding: str = qe if qe in ("unit", "likert") else "unit"
        self.q_domain_lo, self.q_domain_hi = (
            (0.0, 1.0) if self.quality_encoding == "unit" else (LIKERT_Q_MIN, LIKERT_Q_MAX)
        )
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

                # Resolve which camera views to load. `clip_view` may be:
                #   - a single string (e.g. "ego_l")  → one sample per row
                #   - a comma-separated string ("ego_l,ego_m,exo_m")
                #   - the literal "all" → all six standard views
                # When multiple views resolve, EACH view produces a separate
                # training sample with identical labels/metadata. This is
                # multi-view data augmentation, not view ensembling.
                views_to_load = self._resolve_views(clip_view)

                y_cls = class_to_idx[cls_name]
                quality = canonical_quality_score(float(row["quality"]), self.quality_encoding)
                error_targets = derive_error_targets(row, encoding=self.quality_encoding)

                comment_text = (row.get("comment") or "").strip()
                guidance_text = (row.get("action_guidance") or "").strip()
                # Optionally drop clips whose annotator comment is null/empty.
                # Useful for the retrieval-based feedback design where empty
                # comments contribute nothing to the comment-lookup table.
                if self.filter_null_comments and (not comment_text or comment_text.lower() in ("null", "none", "nan")):
                    continue
                base_meta = {
                    "comment": comment_text,
                    "class_name": cls_name,
                    "guidance": guidance_text,
                }

                for view_name in views_to_load:
                    if self.feature_mode == "clip":
                        sequence = self._load_clip_sequence(
                            row,
                            clip_features_root=clip_features_root,
                            clip_view=view_name,
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
                    meta = dict(base_meta)
                    meta["view"] = view_name
                    use_windowing = self.feature_mode not in ("annotation", "clip") and self.window > 0 and _HAS_POSE
                    if use_windowing:
                        for windowed in make_windows(sequence, self.window, self.stride or max(1, self.window // 2)):
                            self.samples.append((windowed, y_cls, quality, error_targets.copy()))
                            self.metadata.append(dict(meta))
                    else:
                        self.samples.append((sequence, y_cls, quality, error_targets.copy()))
                        self.metadata.append(dict(meta))
                    # For non-clip feature modes the sequence is view-agnostic;
                    # don't duplicate identical samples per view.
                    if self.feature_mode != "clip":
                        break

    # Canonical EgoExo-Fitness view list.
    ALL_VIEWS: Tuple[str, ...] = ("ego_l", "ego_m", "ego_r", "exo_l", "exo_m", "exo_r")

    @classmethod
    def _resolve_views(cls, spec: str) -> Tuple[str, ...]:
        """Parse a ``--clip-view`` spec into the ordered tuple of views to load.

        Accepts:
            "ego_l"                → ("ego_l",)
            "ego_l,exo_m"          → ("ego_l", "exo_m")
            "all"                  → all six canonical views
            "ego" / "exo"          → all three of that family
        """
        s = (spec or "").strip().lower()
        if s in ("all", "*"):
            return cls.ALL_VIEWS
        if s == "ego":
            return tuple(v for v in cls.ALL_VIEWS if v.startswith("ego_"))
        if s == "exo":
            return tuple(v for v in cls.ALL_VIEWS if v.startswith("exo_"))
        if "," in s:
            parts = tuple(p.strip() for p in s.split(",") if p.strip())
            return parts or (DEFAULT_CLIP_VIEW,)
        return (s or DEFAULT_CLIP_VIEW,)

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

    def build_comment_table(
        self,
        class_to_idx: Dict[str, int],
        num_quality_buckets: int = 5,
        bucket_edges: Optional[Tuple[float, ...]] = None,
    ) -> Tuple[Dict[Tuple[int, int], str], Tuple[float, ...]]:
        """Build a {(class_idx, quality_bucket) → comment} lookup table.

        For each (class, quality-bucket) cell, picks the comment from the training
        sample whose quality score is closest to the bucket centre. Quality scalars
        share the supervised axis (`unit` ``[0,1]`` or `likert` ``[1,5]``) from the
        dataset constructor.

        Returns a tuple ``(table, bucket_edges)`` where ``bucket_edges`` is the
        sequence of cumulative-distribution boundaries used for bucketing, so the
        inference path can quantise predicted quality the same way.

        For empty cells (a class with no clips in a given quality bucket), the
        implementation falls back to the nearest populated **same-class** bucket
        (never a different exercise class).
        """
        dl, dh = self.q_domain_lo, self.q_domain_hi
        kb = int(num_quality_buckets)
        if kb < 2:
            raise ValueError(f"num_quality_buckets must be >= 2 (got {kb})")

        # Default buckets: ordinal Likert 1…5 splits at 1.5…4.5 when K==5 matches the paper scale;
        # otherwise equal-frequency quantiles within the supervised domain (or uniform cuts if empty).
        if bucket_edges is None:
            if self.quality_encoding == "likert" and kb == 5:
                bucket_edges = (1.5, 2.5, 3.5, 4.5)
            else:
                qs = sorted(float(s[2]) for s in self.samples)
                if not qs:
                    bucket_edges = tuple(float(x) for x in np.linspace(dl, dh, kb + 1)[1:-1])
                else:
                    qs_arr = np.asarray(qs)
                    edges = [float(np.quantile(qs_arr, (i + 1) / kb)) for i in range(kb - 1)]
                    bucket_edges = tuple(edges)

        centres = quality_bucket_centres(bucket_edges, domain_lo=dl, domain_hi=dh)
        num_buckets = len(bucket_edges) + 1
        if num_buckets != kb:
            raise ValueError(f"requested K={kb} but edges imply {num_buckets} buckets")

        def _bucket_of(q: float) -> int:
            return quality_score_to_bucket(float(q), bucket_edges)

        # Group comments by (class_idx, quality_bucket). The retrieval-based
        # feedback design REQUIRES strict (class, quality) routing — no
        # cross-class or cross-quality fallback is permitted, because the whole
        # contract of this lookup is that the returned comment matches both the
        # predicted exercise AND the predicted quality. We instead fall back to
        # the NEAREST POPULATED quality bucket of the SAME class when a cell is
        # empty, never to a different class and never to a different quality
        # regime (e.g. a poor-quality squat clip will not receive a good-quality
        # squat comment if any nearer-quality cell of the same class exists).
        cell_comments: Dict[Tuple[int, int], List[Tuple[float, str]]] = {}
        for sample, meta in zip(self.samples, self.metadata):
            cls_name = meta.get("class_name", "")
            comment = (meta.get("comment") or "").strip()
            if not cls_name or not comment:
                continue
            idx = class_to_idx.get(cls_name)
            if idx is None:
                continue
            q = float(sample[2])
            bucket = _bucket_of(q)
            cell_comments.setdefault((idx, bucket), []).append((q, comment))

        def _bucket_centre(bucket_idx: int) -> float:
            return float(centres[bucket_idx])

        # Pick the medoid comment (training comment with quality closest to
        # the bucket centre) for each populated (class, bucket) cell. Cells
        # with no in-bucket clips fall back to the SAME class's nearest
        # populated quality bucket — never to a different class.
        table: Dict[Tuple[int, int], str] = {}
        for cls_idx in class_to_idx.values():
            # Pre-compute populated buckets for this class, sorted by distance to centre.
            populated = sorted(b for b in range(num_buckets)
                               if (cls_idx, b) in cell_comments)
            for bucket in range(num_buckets):
                key = (cls_idx, bucket)
                if key in cell_comments and cell_comments[key]:
                    centre = _bucket_centre(bucket)
                    pick = min(cell_comments[key], key=lambda qc: abs(qc[0] - centre))
                    table[key] = pick[1]
                elif populated:
                    # Same-class fallback: take the nearest populated quality bucket.
                    nearest = min(populated, key=lambda b: abs(b - bucket))
                    centre = _bucket_centre(nearest)
                    pick = min(cell_comments[(cls_idx, nearest)],
                               key=lambda qc: abs(qc[0] - centre))
                    table[key] = pick[1]
                # else: leave the cell unset → lookup_comment returns "" for it
                # (an honest "no information available" instead of a wrong comment).
        return table, bucket_edges

    def apply_quality_bucket_labels(self, bucket_edges: Tuple[float, ...]) -> None:
        """Attach discrete quality-bucket IDs (0 … K-1) for classification training.

        Each sample tuple becomes ``(x, y_cls, q_float, q_bucket_idx, errors)``.
        Call **after** ``build_comment_table`` so bucket edges match the retrieval tables.
        """
        edges = tuple(float(e) for e in bucket_edges)
        new_samples: List[Tuple[np.ndarray, int, float, int, np.ndarray]] = []
        for tup in self.samples:
            x, y, q, e = tup
            qb = quality_score_to_bucket(float(q), edges)
            new_samples.append((x, int(y), float(q), int(qb), e))
        self.samples = new_samples  # type: ignore[assignment]

    def __getitem__(self, idx: int):
        parts = self.samples[idx]
        if len(parts) == 4:
            x, y, q, e = parts
            q_bucket = -1
        elif len(parts) == 5:
            x, y, q, q_bucket, e = parts
        else:
            raise ValueError(f"unexpected sample tuple length {len(parts)}")
        meta = self.metadata[idx] if idx < len(self.metadata) else {"comment": "", "class_name": "", "guidance": ""}
        return (
            torch.from_numpy(np.asarray(x, dtype=np.float32)),
            torch.tensor(y, dtype=torch.long),
            torch.tensor(q, dtype=torch.float32),
            torch.tensor(q_bucket, dtype=torch.long),
            torch.from_numpy(np.asarray(e, dtype=np.float32)),
            meta.get("comment", ""),
            meta.get("class_name", ""),
        )


def egoexo_collate_fn(batch):
    """Pad variable-length annotation/CLIP sequences to the longest in the batch.

    Returns a 7-tuple:
    ``(features, y_cls, y_q_scalar, y_q_bucket, y_err, comments, class_names)``.
    ``y_q_bucket`` is ``-1`` when samples have not been labelled with
    ``apply_quality_bucket_labels`` (regression heads ignore it).
    """
    xs, yc, yqf, yqb, ye, comments, class_names = list(zip(*batch))
    lengths = [x.shape[0] for x in xs]
    max_t = max(lengths)
    feat_dim = xs[0].shape[-1]
    padded = torch.zeros(len(xs), max_t, feat_dim, dtype=torch.float32)
    for i, x in enumerate(xs):
        padded[i, : x.shape[0]] = x
    return (
        padded,
        torch.stack(yc),
        torch.stack(yqf),
        torch.stack(yqb),
        torch.stack(ye),
        list(comments),
        list(class_names),
    )
