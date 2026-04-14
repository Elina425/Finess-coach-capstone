"""
Pair sampling for contrastive regression (GCN-PSN, arXiv:2511.01194).

Positive pairs: two frames from the same clip (same movement context).
Negative pairs: random frames from two clips of different exercise_class.
"""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set, Tuple

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


class GCNPosePairDataset(Dataset):
    """
    On-the-fly random pairs. __len__ = pairs_per_epoch (virtual epochs).
    """

    def __init__(
        self,
        index_csv: Path,
        keypoints_dir: Path,
        class_to_stems: Dict[str, List[str]],
        stem_quality: Dict[str, float],
        stem_split: Dict[str, str],
        split: str,
        pairs_per_epoch: int,
        seed: int,
        pos_same_video: float = 0.5,
    ):
        self.keypoints_dir = Path(keypoints_dir)
        self.pairs_per_epoch = pairs_per_epoch
        self.rng = np.random.default_rng(seed)
        self.pos_same_video = pos_same_video

        self.stems: List[str] = []
        for c, stems in class_to_stems.items():
            for s in stems:
                if stem_split.get(s) != split:
                    continue
                kp_path = self.keypoints_dir / f"{s}_keypoints.npz"
                if kp_path.is_file():
                    self.stems.append(s)
        self.stems = sorted(set(self.stems))
        self.stem_quality = stem_quality
        self.class_to_stems = {k: [x for x in v if x in stem_split and stem_split[x] == split] for k, v in class_to_stems.items()}
        self.class_to_stems = {k: [x for x in v if (self.keypoints_dir / f"{x}_keypoints.npz").is_file()] for k, v in self.class_to_stems.items()}
        self.class_to_stems = {k: v for k, v in self.class_to_stems.items() if len(v) > 0}
        self.classes = sorted(self.class_to_stems.keys())

        self._cache: Dict[str, np.ndarray] = {}
        self._cache_order: List[str] = []
        self._cache_max = 64

    def _get_kp(self, stem: str) -> Optional[np.ndarray]:
        if stem in self._cache:
            return self._cache[stem]
        p = self.keypoints_dir / f"{stem}_keypoints.npz"
        kp = load_keypoints_npz(p)
        if kp is None:
            return None
        if len(self._cache_order) >= self._cache_max:
            old = self._cache_order.pop(0)
            self._cache.pop(old, None)
        self._cache[stem] = kp
        self._cache_order.append(stem)
        return kp

    def _sample_pair(self) -> Tuple[np.ndarray, np.ndarray, float]:
        if len(self.stems) < 1:
            raise RuntimeError("No stems with keypoints for this split")

        want_pos = self.rng.random() < self.pos_same_video

        if want_pos or len(self.classes) < 2:
            for _ in range(50):
                stem = self.stems[int(self.rng.integers(0, len(self.stems)))]
                kp = self._get_kp(stem)
                if kp is None or kp.shape[0] < 1:
                    continue
                T = kp.shape[0]
                i = int(self.rng.integers(0, T))
                j = int(self.rng.integers(0, T))
                if T > 1:
                    while j == i:
                        j = int(self.rng.integers(0, T))
                p1 = minmax_normalize_xy_nan_safe(kp[i])
                p2 = minmax_normalize_xy_nan_safe(kp[j])
                return p1, p2, 1.0

        if len(self.classes) >= 2:
            idx = self.rng.choice(len(self.classes), size=2, replace=False)
            ex1, ex2 = self.classes[int(idx[0])], self.classes[int(idx[1])]
            pool1 = self.class_to_stems[ex1]
            pool2 = self.class_to_stems[ex2]
            if not pool1 or not pool2:
                return self._sample_pair()
            s1 = pool1[int(self.rng.integers(0, len(pool1)))]
            s2 = pool2[int(self.rng.integers(0, len(pool2)))]
            kp1 = self._get_kp(s1)
            kp2 = self._get_kp(s2)
            if kp1 is None or kp2 is None:
                return self._sample_pair()
            i1 = int(self.rng.integers(0, kp1.shape[0]))
            i2 = int(self.rng.integers(0, kp2.shape[0]))
            p1 = minmax_normalize_xy_nan_safe(kp1[i1])
            p2 = minmax_normalize_xy_nan_safe(kp2[i2])
            return p1, p2, 0.0

        for _ in range(50):
            if len(self.stems) < 2:
                break
            s1, s2 = self.rng.choice(self.stems, size=2, replace=False)
            kp1 = self._get_kp(s1)
            kp2 = self._get_kp(s2)
            if kp1 is None or kp2 is None:
                continue
            i1 = int(self.rng.integers(0, kp1.shape[0]))
            i2 = int(self.rng.integers(0, kp2.shape[0]))
            p1 = minmax_normalize_xy_nan_safe(kp1[i1])
            p2 = minmax_normalize_xy_nan_safe(kp2[i2])
            return p1, p2, 0.0

        stem = self.stems[0]
        kp = self._get_kp(stem)
        T = max(kp.shape[0], 2)
        p1 = minmax_normalize_xy_nan_safe(kp[0])
        p2 = minmax_normalize_xy_nan_safe(kp[min(1, T - 1)])
        return p1, p2, 1.0

    def __len__(self) -> int:
        return self.pairs_per_epoch

    def __getitem__(self, i: int):
        p1, p2, y = self._sample_pair()
        return (
            torch.from_numpy(p1).float(),
            torch.from_numpy(p2).float(),
            torch.tensor(y, dtype=torch.float32),
        )


def build_stem_metadata_from_index(
    index_csv: Path,
    default_split: str = "train",
) -> Tuple[Dict[str, List[str]], Dict[str, float], Dict[str, str]]:
    """exercise_class -> stems, stem -> quality, stem -> split."""
    class_to_stems: Dict[str, List[str]] = defaultdict(list)
    stem_quality: Dict[str, float] = {}
    stem_split: Dict[str, str] = {}
    seen: Set[str] = set()

    with open(index_csv, newline="") as f:
        for row in csv.DictReader(f):
            stem = (row.get("video_stem") or "").strip()
            if not stem:
                continue
            ex = (row.get("exercise_class") or "unknown").strip()
            q = float(row.get("quality", 0.5))
            sp = (row.get("split") or default_split).strip()
            if stem not in seen:
                seen.add(stem)
                class_to_stems[ex].append(stem)
                stem_quality[stem] = q
                stem_split[stem] = sp
            else:
                stem_quality[stem] = q
    return dict(class_to_stems), stem_quality, stem_split
