"""
30-frame windows of COCO-17 keypoints (T, 17, 2) with per-frame bbox min–max normalization
for GCN sequence models (same CSV index as BiLSTM).
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from fitness_coach.models.gcn_pose_topology import minmax_normalize_xy_nan_safe


def load_keypoints_npz(path: Path) -> Optional[np.ndarray]:
    try:
        kp = np.asarray(np.load(path, allow_pickle=True)["keypoints"], dtype=np.float32)
        if kp.ndim != 3 or kp.shape[1:] != (17, 2):
            return None
        return kp
    except Exception:
        return None


def make_windows_keypoints(
    kp: np.ndarray,
    window: int,
    stride: int,
) -> List[np.ndarray]:
    """kp: (T, 17, 2) raw; returns list of (window, 17, 2) with per-frame min–max norm."""
    kp = np.nan_to_num(kp, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    T = kp.shape[0]
    if T == 0:
        return []
    out: List[np.ndarray] = []

    def norm_window(w: np.ndarray) -> np.ndarray:
        return np.stack(
            [minmax_normalize_xy_nan_safe(w[t]) for t in range(w.shape[0])],
            axis=0,
        ).astype(np.float32)

    if T < window:
        pad = np.zeros((window - T, 17, 2), dtype=np.float32)
        w = np.vstack([kp, pad])
        return [norm_window(w)]
    for start in range(0, T - window + 1, stride):
        w = kp[start : start + window].copy()
        out.append(norm_window(w))
    if not out:
        out.append(norm_window(kp[-window:].copy()))
    return out


class ExerciseKeypointWindowDataset(Dataset):
    """One row per window; labels from CSV (exercise_class, quality)."""

    def __init__(
        self,
        index_csv: Path,
        keypoints_dir: Path,
        class_to_idx: Dict[str, int],
        split: str,
        window: int = 30,
        stride: int = 15,
    ):
        self.window = window
        self.stride = stride
        self.class_to_idx = class_to_idx
        self.keypoints_dir = Path(keypoints_dir)
        self.samples: List[Tuple[np.ndarray, int, float]] = []

        with open(index_csv, newline="") as f:
            for row in csv.DictReader(f):
                if row.get("split", "train") != split:
                    continue
                stem = row["video_stem"]
                cls_name = row["exercise_class"]
                if cls_name not in class_to_idx:
                    continue
                p = self.keypoints_dir / f"{stem}_keypoints.npz"
                if not p.is_file():
                    continue
                kp = load_keypoints_npz(p)
                if kp is None:
                    continue
                y_cls = class_to_idx[cls_name]
                q = float(row["quality"])
                for w in make_windows_keypoints(kp, window, stride):
                    if w.shape != (window, 17, 2):
                        continue
                    self.samples.append((w, y_cls, q))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, i: int):
        x, y, q = self.samples[i]
        return (
            torch.from_numpy(x),
            torch.tensor(y, dtype=torch.long),
            torch.tensor(q, dtype=torch.float32),
        )
