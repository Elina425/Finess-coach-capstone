"""
Datasets for ST-GCN: sliding windows of shape (T, V, C) keypoints → tensor (C, T, V).
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from fitness_coach.datasets.exercise_bilstm_dataset import (
    coarse_exercise_from_pose,
    load_index_rows,
    stratified_train_val_test_split,
)


def load_keypoints_npz(path: Path) -> np.ndarray:
    data = np.load(path, allow_pickle=True)
    kp = np.asarray(data["keypoints"], dtype=np.float32)
    if kp.ndim != 3 or kp.shape[1:] != (17, 2):
        raise ValueError(f"Bad keypoints shape in {path}: {kp.shape}")
    return np.nan_to_num(kp, nan=0.0, posinf=0.0, neginf=0.0)


def load_kaggle_keypoints_and_labels(
    data_dir: Path,
    stem: str = "kaggle_exercise_recognition",
) -> Tuple[np.ndarray, np.ndarray]:
    kp_path = data_dir / f"{stem}_keypoints.npz"
    lab_path = data_dir / f"{stem}_labels.npz"
    if not kp_path.is_file():
        raise FileNotFoundError(f"Missing {kp_path}")
    if not lab_path.is_file():
        raise FileNotFoundError(f"Missing {lab_path}")
    kp = load_keypoints_npz(kp_path)
    d = np.load(lab_path, allow_pickle=True)
    if "pose" not in d:
        raise ValueError(f"{lab_path} must contain 'pose'")
    poses = np.asarray(d["pose"], dtype=object)
    if len(poses) != len(kp):
        raise ValueError(f"pose length {len(poses)} != keypoints length {len(kp)}")
    return kp, poses


def build_keypoint_window_samples(
    kp: np.ndarray,
    poses: np.ndarray,
    *,
    window: int,
    stride: int,
    quality_default: float = 0.75,
) -> List[Tuple[np.ndarray, str, float]]:
    """Windows (window, 17, 2); label = coarse exercise from last frame pose."""
    kp = np.asarray(kp, dtype=np.float32)
    kp = np.nan_to_num(kp, nan=0.0, posinf=0.0, neginf=0.0)
    T = kp.shape[0]
    out: List[Tuple[np.ndarray, str, float]] = []
    if T < window:
        pad = np.zeros((window - T, 17, 2), dtype=np.float32)
        w = np.vstack([kp, pad])
        cls = coarse_exercise_from_pose(str(poses[-1]))
        out.append((w, cls, float(quality_default)))
        return out
    for start in range(0, T - window + 1, stride):
        w = kp[start : start + window].copy()
        last_pose = poses[start + window - 1]
        cls = coarse_exercise_from_pose(str(last_pose))
        out.append((w, cls, float(quality_default)))
    return out


class TensorSTGCNDataset(Dataset):
    """Samples: (window, 17, 2); optional per-joint mean/std (17, 2). Returns (2, T, 17)."""

    def __init__(
        self,
        samples: List[Tuple[np.ndarray, int, float]],
        mean: Optional[np.ndarray],
        std: Optional[np.ndarray],
    ) -> None:
        self.samples = samples
        self.mean = mean
        self.std = std

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        x, y, q = self.samples[idx]
        x = np.asarray(x, dtype=np.float32)
        if self.mean is not None and self.std is not None:
            x = (x - self.mean) / self.std
        # (T, V, C) -> (C, T, V)
        x = np.transpose(x, (2, 0, 1)).copy()
        return (
            torch.from_numpy(x),
            torch.tensor(y, dtype=torch.long),
            torch.tensor(q, dtype=torch.float32),
        )


def fit_stgcn_standardizer(
    samples: List[Tuple[np.ndarray, int, float]],
) -> Tuple[np.ndarray, np.ndarray]:
    xs = np.stack([s[0] for s in samples], axis=0)
    flat = xs.reshape(-1, 17, 2)
    mean = flat.mean(axis=0).astype(np.float32)
    std = (flat.std(axis=0) + 1e-8).astype(np.float32)
    return mean, std


def build_kaggle_stgcn_datasets(
    data_dir: Path,
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
    kp, poses = load_kaggle_keypoints_and_labels(data_dir, stem=stem)
    raw = build_keypoint_window_samples(
        kp,
        poses,
        window=window,
        stride=stride,
        quality_default=quality_default,
    )
    if not raw:
        raise ValueError("No windows from Kaggle keypoints")

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

    mean = std = None
    if standardize and train_s:
        mean, std = fit_stgcn_standardizer(train_s)

    train_ds = TensorSTGCNDataset(train_s, mean, std)
    val_ds = TensorSTGCNDataset(val_s, mean, std)
    test_ds = TensorSTGCNDataset(test_s, mean, std)
    return train_ds, val_ds, test_ds, class_to_idx, idx_to_class, mean, std


class ExerciseKeypointIndexDataset(Dataset):
    """Index CSV + keypoints NPZ → (2, T, 17) windows."""

    def __init__(
        self,
        index_csv: Path,
        keypoints_dir: Path,
        class_to_idx: Dict[str, int],
        split: str,
        window: int = 30,
        stride: int = 15,
        mean: Optional[np.ndarray] = None,
        std: Optional[np.ndarray] = None,
    ) -> None:
        self.window = window
        self.stride = stride
        self.class_to_idx = class_to_idx
        self.mean = mean
        self.std = std
        self.samples: List[Tuple[np.ndarray, int, float]] = []

        with open(index_csv, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get("split", "train") != split:
                    continue
                stem = row["video_stem"]
                cls_name = row["exercise_class"]
                if cls_name not in class_to_idx:
                    continue
                kp_path = keypoints_dir / f"{stem}_keypoints.npz"
                if not kp_path.is_file():
                    continue
                try:
                    data = np.load(kp_path, allow_pickle=True)
                    kp = np.asarray(data["keypoints"], dtype=np.float32)
                    if kp.ndim != 3 or kp.shape[1:] != (17, 2):
                        continue
                    kp = np.nan_to_num(kp, nan=0.0, posinf=0.0, neginf=0.0)
                except Exception:
                    continue
                y_cls = class_to_idx[cls_name]
                q = float(row["quality"])
                T = kp.shape[0]
                if T < window:
                    pad = np.zeros((window - T, 17, 2), dtype=np.float32)
                    w = np.vstack([kp, pad])
                    self.samples.append((w, y_cls, q))
                else:
                    for start in range(0, T - window + 1, stride):
                        self.samples.append((kp[start : start + window].copy(), y_cls, q))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, i: int):
        x, y, q = self.samples[i]
        x = np.asarray(x, dtype=np.float32)
        if self.mean is not None and self.std is not None:
            x = (x - self.mean) / self.std
        x = np.transpose(x, (2, 0, 1)).copy()
        return (
            torch.from_numpy(x),
            torch.tensor(y, dtype=torch.long),
            torch.tensor(q, dtype=torch.float32),
        )


def apply_stgcn_standardizer(
    ds: ExerciseKeypointIndexDataset,
    mean: np.ndarray,
    std: np.ndarray,
) -> None:
    ds.mean = mean.astype(np.float32)
    ds.std = std.astype(np.float32)
