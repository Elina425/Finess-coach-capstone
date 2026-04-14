"""
PyTorch Dataset: 30-frame windows over angle sequences (T, F) with exercise class + quality.

F=8 angles; F=34 coords-only; F=42 mixed (Riccio-style, arXiv:2411.11548).
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

try:
    from sklearn.model_selection import train_test_split
except ImportError:  # pragma: no cover
    train_test_split = None  # type: ignore

from fitness_coach.core.biomechanical_features import (
    compute_coords_only_sequence_features,
    compute_mixed_sequence_features,
    keypoints_npz_coords_already_normalized,
)


def load_angles_npz(path: Path) -> np.ndarray:
    data = np.load(path, allow_pickle=True)
    a = np.asarray(data["angles"], dtype=np.float32)
    if a.ndim != 2:
        raise ValueError(f"Bad angles shape in {path}: {a.shape}")
    return a


def nan_to_num_angles(x: np.ndarray) -> np.ndarray:
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    return x.astype(np.float32)


def make_windows(
    angles: np.ndarray,
    window: int,
    stride: int,
) -> List[np.ndarray]:
    """Non-overlapping / strided windows; pad if shorter than window."""
    angles = nan_to_num_angles(angles)
    T = angles.shape[0]
    if T == 0:
        return []
    if T < window:
        pad = np.zeros((window - T, angles.shape[1]), dtype=np.float32)
        return [np.vstack([angles, pad])]
    out: List[np.ndarray] = []
    for start in range(0, T - window + 1, stride):
        out.append(angles[start : start + window].copy())
    if not out:
        out.append(angles[-window:].copy())
    return out


def fit_standardizer_from_dataset(ds: "ExerciseAngleWindowDataset") -> Tuple[np.ndarray, np.ndarray]:
    """Per-feature mean/std over all training windows (StandardScaler-style; Riccio §3.3.1)."""
    if not ds.samples:
        raise ValueError("empty dataset")
    xs = np.stack([s[0] for s in ds.samples], axis=0)
    x2 = xs.reshape(-1, xs.shape[-1])
    mean = x2.mean(axis=0).astype(np.float32)
    std = x2.std(axis=0).astype(np.float32) + np.float32(1e-8)
    return mean, std


class ExerciseAngleWindowDataset(Dataset):
    """
    Each row in index_csv must have: video_stem, exercise_class, quality, split
    Angles file: angles_dir / f"{video_stem}_biomechanics.npz"
    Mixed mode: keypoints_dir / f"{video_stem}_keypoints.npz" (key 'keypoints', (T,17,2))

    Optional columns ``comment`` and ``action_guidance`` populate ``_text_raw`` per window for
    multimodal training (``--text-supervision`` in ``train_exercise_bilstm.py``).
    """

    def __init__(
        self,
        index_csv: Path,
        angles_dir: Path,
        class_to_idx: Dict[str, int],
        split: str,
        window: int = 30,
        stride: int = 15,
        feature_mode: str = "angles",
        keypoints_dir: Optional[Path] = None,
        scale_mean: Optional[np.ndarray] = None,
        scale_std: Optional[np.ndarray] = None,
    ):
        self.window = window
        self.stride = stride
        self.class_to_idx = class_to_idx
        self.feature_mode = feature_mode
        self.keypoints_dir = keypoints_dir
        self.scale_mean = scale_mean
        self.scale_std = scale_std
        self.samples: List[Tuple[np.ndarray, int, float]] = []
        self._text_raw: List[str] = []
        self._text_feat: Optional[np.ndarray] = None

        with open(index_csv, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get("split", "train") != split:
                    continue
                stem = row["video_stem"]
                cls_name = row["exercise_class"]
                if cls_name not in class_to_idx:
                    continue
                seq = self._load_sequence(stem, angles_dir)
                if seq is None:
                    continue
                y_cls = class_to_idx[cls_name]
                q = float(row["quality"])
                c = (row.get("comment") or "").strip()
                g = (row.get("action_guidance") or "").strip()
                row_text = (c + " " + g).strip()
                for w in make_windows(seq, window, stride):
                    if w.shape != (window, seq.shape[1]):
                        continue
                    self.samples.append((w, y_cls, q))
                    self._text_raw.append(row_text)

    def set_text_features(self, emb: np.ndarray) -> None:
        """Per-window coaching embeddings (same order as ``samples``), from TF–IDF/SVD or similar."""
        if emb.shape[0] != len(self.samples):
            raise ValueError(
                f"text emb rows {emb.shape[0]} != num samples {len(self.samples)}"
            )
        self._text_feat = np.asarray(emb, dtype=np.float32)

    def _load_sequence(self, stem: str, angles_dir: Path) -> Optional[np.ndarray]:
        if self.feature_mode in ("mixed", "coords"):
            if self.keypoints_dir is None:
                return None
            kp_path = self.keypoints_dir / f"{stem}_keypoints.npz"
            if not kp_path.is_file():
                return None
            try:
                data = np.load(kp_path, allow_pickle=True)
                kp = np.asarray(data["keypoints"], dtype=np.float64)
                if kp.ndim != 3 or kp.shape[1:] != (17, 2):
                    return None
                coords_ok = keypoints_npz_coords_already_normalized(data)
                if self.feature_mode == "mixed":
                    seq, _ = compute_mixed_sequence_features(
                        kp, coords_already_normalized=coords_ok
                    )
                    return seq
                return compute_coords_only_sequence_features(
                    kp, coords_already_normalized=coords_ok
                )
            except Exception:
                return None
        ap = angles_dir / f"{stem}_biomechanics.npz"
        if not ap.is_file():
            return None
        try:
            return load_angles_npz(ap)
        except Exception:
            return None

    def apply_standardizer(self, mean: np.ndarray, std: np.ndarray) -> None:
        self.scale_mean = mean.astype(np.float32)
        self.scale_std = std.astype(np.float32)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, i: int):
        x, y, q = self.samples[i]
        if self.scale_mean is not None and self.scale_std is not None:
            x = (x - self.scale_mean) / self.scale_std
        x_t = torch.from_numpy(np.asarray(x, dtype=np.float32))
        y_t = torch.tensor(y, dtype=torch.long)
        q_t = torch.tensor(q, dtype=torch.float32)
        if self._text_feat is not None:
            t_t = torch.from_numpy(self._text_feat[i].copy())
            return x_t, y_t, q_t, t_t
        return x_t, y_t, q_t


def build_class_map(train_rows: List[Dict[str, str]]) -> Dict[str, int]:
    names = sorted({r["exercise_class"] for r in train_rows})
    return {n: i for i, n in enumerate(names)}


def load_index_rows(path: Path) -> List[Dict[str, str]]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def coarse_exercise_from_pose(pose: str) -> str:
    """
    Map phase labels (e.g. squats_up, jumping_jacks_down) to base exercise name
    for classification (squats, jumping_jacks).
    """
    s = str(pose).strip()
    if not s:
        return "unknown"
    if "_" in s:
        return s.rsplit("_", 1)[0]
    return s


def load_kaggle_angles_and_labels(
    angles_dir: Path,
    stem: str = "kaggle_exercise_recognition",
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """Load angles (T, 8), per-frame pose strings, and optional per-frame video_id (int).

    ``video_id`` is stored by ``riccio_kaggle_video_pipeline`` so train/val/test can be split by
    source video (stratified by class) instead of shuffling overlapping windows across videos.
    """
    ang_path = angles_dir / f"{stem}_biomechanics.npz"
    lab_path = angles_dir / f"{stem}_labels.npz"
    if not ang_path.is_file():
        raise FileNotFoundError(f"Missing {ang_path}")
    if not lab_path.is_file():
        raise FileNotFoundError(f"Missing {lab_path} (need per-frame pose labels)")
    ang = load_angles_npz(ang_path)
    d = np.load(lab_path, allow_pickle=True)
    if "pose" not in d:
        raise ValueError(f"{lab_path} must contain 'pose' array")
    poses = np.asarray(d["pose"], dtype=object)
    if len(poses) != len(ang):
        raise ValueError(
            f"pose length {len(poses)} != angles length {len(ang)} for {stem}"
        )
    video_id: Optional[np.ndarray] = None
    if "video_id" in d:
        video_id = np.asarray(d["video_id"])
        if video_id.shape[0] != len(ang):
            raise ValueError(
                f"video_id length {video_id.shape[0]} != angles length {len(ang)} for {stem}"
            )
        video_id = video_id.astype(np.int32, copy=False)
    return ang, poses, video_id


def build_kaggle_angle_window_samples(
    angles: np.ndarray,
    poses: np.ndarray,
    *,
    window: int,
    stride: int,
    quality_default: float = 0.75,
) -> List[Tuple[np.ndarray, str, float]]:
    """Sliding windows; label = coarse exercise from last frame in window."""
    ang = np.asarray(angles, dtype=np.float32)
    ang = np.nan_to_num(ang, nan=0.0, posinf=0.0, neginf=0.0)
    T = ang.shape[0]
    out: List[Tuple[np.ndarray, str, float]] = []
    if T < window:
        pad = np.zeros((window - T, ang.shape[1]), dtype=np.float32)
        ang = np.vstack([ang, pad])
        T = window
        cls = coarse_exercise_from_pose(str(poses[-1]))
        out.append((ang.copy(), cls, float(quality_default)))
        return out
    for start in range(0, T - window + 1, stride):
        w = ang[start : start + window].copy()
        last_pose = poses[start + window - 1]
        cls = coarse_exercise_from_pose(str(last_pose))
        out.append((w, cls, float(quality_default)))
    return out


def build_kaggle_angle_window_samples_by_video(
    angles: np.ndarray,
    poses: np.ndarray,
    video_id: np.ndarray,
    *,
    window: int,
    stride: int,
    quality_default: float = 0.75,
) -> List[Tuple[np.ndarray, str, float, int]]:
    """Windows only within each video segment; no window crosses a ``video_id`` boundary.

    Returns (window, coarse_class, quality, video_id) per window.
    """
    ang = np.asarray(angles, dtype=np.float32)
    ang = np.nan_to_num(ang, nan=0.0, posinf=0.0, neginf=0.0)
    vid = np.asarray(video_id, dtype=np.int32).ravel()
    if vid.shape[0] != ang.shape[0]:
        raise ValueError("video_id length must match angles time dimension")
    out: List[Tuple[np.ndarray, str, float, int]] = []
    for v in np.unique(vid):
        m = vid == int(v)
        ang_v = ang[m]
        pose_v = poses[m]
        for w, c, q in build_kaggle_angle_window_samples(
            ang_v,
            pose_v,
            window=window,
            stride=stride,
            quality_default=quality_default,
        ):
            out.append((w, c, q, int(v)))
    return out


def stratified_train_val_test_split(
    indices: np.ndarray,
    y: np.ndarray,
    *,
    test_ratio: float = 0.15,
    val_ratio: float = 0.15,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split indices into train / val / test; stratify on y when possible."""
    if train_test_split is None:
        raise RuntimeError("install scikit-learn for stratified Kaggle splits: pip install scikit-learn")
    n = len(indices)
    test_n = max(1, int(round(n * test_ratio)))
    val_n = max(1, int(round(n * val_ratio)))
    train_n = n - val_n - test_n
    if train_n < 1:
        # Too few windows: all train
        return indices.copy(), indices[:0], indices[:0]

    if test_n >= n:
        test_n = max(1, n - 1)

    # First reserve test
    try:
        train_val_idx, test_idx = train_test_split(
            indices,
            test_size=test_n,
            stratify=y[indices],
            random_state=seed,
        )
    except ValueError:
        train_val_idx, test_idx = train_test_split(
            indices, test_size=test_n, random_state=seed
        )

    # Split train_val into train and val
    y_tv = y[train_val_idx]
    if len(train_val_idx) <= val_n + 1:
        train_idx, val_idx = train_val_idx, np.array([], dtype=int)
    else:
        rel_val_ratio = val_n / len(train_val_idx)
        try:
            train_idx, val_idx = train_test_split(
                train_val_idx,
                test_size=rel_val_ratio,
                stratify=y_tv,
                random_state=seed + 1,
            )
        except ValueError:
            train_idx, val_idx = train_test_split(
                train_val_idx, test_size=rel_val_ratio, random_state=seed + 1
            )
    return np.asarray(train_idx), np.asarray(val_idx), np.asarray(test_idx)


def build_kaggle_angle_datasets(
    angles_dir: Path,
    *,
    stem: str = "kaggle_exercise_recognition",
    window: int = 30,
    stride: int = 15,
    test_ratio: float = 0.15,
    val_ratio: float = 0.15,
    seed: int = 42,
    standardize: bool = True,
    quality_default: float = 0.75,
) -> Tuple[Dataset, Dataset, Dataset, Dict[str, int], Dict[int, str], Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Build train/val/test datasets from Kaggle pipeline biomechanics + labels NPZs.
    Each window is labeled by coarse exercise (phase suffix stripped).

    If ``*_labels.npz`` contains ``video_id`` (Riccio pipeline) and there is more than one distinct
    video, splits are **by video** (stratified by each video's exercise class) so windows from the
    same recording never appear in more than one split. Otherwise splits are stratified by window
    label (legacy / single-timeline exports).
    """
    angles, poses, video_id = load_kaggle_angles_and_labels(angles_dir, stem=stem)
    use_video_split = (
        video_id is not None and int(np.unique(video_id).size) > 1
    )

    if use_video_split:
        assert video_id is not None
        raw_v = build_kaggle_angle_window_samples_by_video(
            angles,
            poses,
            video_id,
            window=window,
            stride=stride,
            quality_default=quality_default,
        )
        if not raw_v:
            raise ValueError("No windows built from Kaggle angles (per-video)")
        class_names = sorted({c for _, c, _, _ in raw_v})
        class_to_idx = {n: i for i, n in enumerate(class_names)}
        idx_to_class = {i: n for n, i in class_to_idx.items()}

        uniq_vids = np.unique(np.asarray([t[3] for t in raw_v], dtype=np.int32))
        vid_to_label = {}
        for v in uniq_vids:
            frame0 = int(np.nonzero(video_id == v)[0][0])
            vid_to_label[int(v)] = coarse_exercise_from_pose(str(poses[frame0]))
        y_vid = np.array([class_to_idx[vid_to_label[int(v)]] for v in uniq_vids], dtype=int)
        vidx = np.arange(len(uniq_vids), dtype=int)
        train_vi, val_vi, test_vi = stratified_train_val_test_split(
            vidx,
            y_vid,
            test_ratio=test_ratio,
            val_ratio=val_ratio,
            seed=seed,
        )
        train_videos = set(uniq_vids[train_vi].tolist())
        val_videos = set(uniq_vids[val_vi].tolist())
        test_videos = set(uniq_vids[test_vi].tolist())

        train_s: List[Tuple[np.ndarray, int, float]] = []
        val_s: List[Tuple[np.ndarray, int, float]] = []
        test_s: List[Tuple[np.ndarray, int, float]] = []
        for w, c, q, vid in raw_v:
            triplet = (w, class_to_idx[c], q)
            if vid in train_videos:
                train_s.append(triplet)
            elif vid in val_videos:
                val_s.append(triplet)
            elif vid in test_videos:
                test_s.append(triplet)
    else:
        raw = build_kaggle_angle_window_samples(
            angles,
            poses,
            window=window,
            stride=stride,
            quality_default=quality_default,
        )
        if not raw:
            raise ValueError("No windows built from Kaggle angles")

        class_names = sorted({c for _, c, _ in raw})
        class_to_idx = {n: i for i, n in enumerate(class_names)}
        idx_to_class = {i: n for n, i in class_to_idx.items()}

        samples_idxed: List[Tuple[np.ndarray, int, float]] = [
            (w, class_to_idx[c], q) for w, c, q in raw
        ]
        indices = np.arange(len(samples_idxed), dtype=int)
        y_all = np.array([samples_idxed[i][1] for i in range(len(samples_idxed))], dtype=int)

        train_idx, val_idx, test_idx = stratified_train_val_test_split(
            indices,
            y_all,
            test_ratio=test_ratio,
            val_ratio=val_ratio,
            seed=seed,
        )

        def take(idxs: np.ndarray) -> List[Tuple[np.ndarray, int, float]]:
            return [samples_idxed[int(i)] for i in idxs]

        train_s = take(train_idx)
        val_s = take(val_idx)
        test_s = take(test_idx)

    scale_mean: Optional[np.ndarray] = None
    scale_std: Optional[np.ndarray] = None
    if standardize and train_s:
        xs = np.stack([s[0] for s in train_s], axis=0)
        flat = xs.reshape(-1, xs.shape[-1])
        scale_mean = flat.mean(axis=0).astype(np.float32)
        scale_std = (flat.std(axis=0) + 1e-8).astype(np.float32)

    train_ds = TensorWindowDataset(train_s, scale_mean, scale_std)
    val_ds = TensorWindowDataset(val_s, scale_mean, scale_std)
    test_ds = TensorWindowDataset(test_s, scale_mean, scale_std)
    return train_ds, val_ds, test_ds, class_to_idx, idx_to_class, scale_mean, scale_std


class TensorWindowDataset(Dataset):
    """In-memory angle windows (same tensor layout as ExerciseAngleWindowDataset)."""

    def __init__(
        self,
        samples: List[Tuple[np.ndarray, int, float]],
        scale_mean: Optional[np.ndarray],
        scale_std: Optional[np.ndarray],
    ) -> None:
        self.samples = samples
        self.scale_mean = scale_mean
        self.scale_std = scale_std

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        x, y, q = self.samples[idx]
        x = np.asarray(x, dtype=np.float32)
        if self.scale_mean is not None and self.scale_std is not None:
            x = (x - self.scale_mean) / self.scale_std
        return (
            torch.from_numpy(x),
            torch.tensor(y, dtype=torch.long),
            torch.tensor(q, dtype=torch.float32),
        )
