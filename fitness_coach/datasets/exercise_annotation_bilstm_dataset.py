"""
One sample per index row: variable-length **annotation sequence** (no video / pose).

Each timestep encodes an interpretable key-point check or a summary step built from
``comment`` + ``action_guidance``. Built for EgoExo ``verification_json`` from
``build_egoexo_fitness_index.py``.
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

STEP_DIM = 32


def _step_vector(instruction: str, ok: bool) -> np.ndarray:
    """Fixed-size encoding: dim 0 = verdict float, rest from hashed instruction."""
    v = np.zeros(STEP_DIM, dtype=np.float32)
    v[0] = 1.0 if ok else 0.0
    if STEP_DIM <= 1:
        return v
    blob = hashlib.sha256(
        instruction.encode("utf-8", errors="replace")
    ).digest()
    need = (STEP_DIM - 1) * 4
    buf = (blob * ((need // len(blob)) + 1))[:need]
    ints = np.frombuffer(buf, dtype=np.uint32).astype(np.float32)
    ints = (ints / np.float32(2**31)) - 1.0
    v[1:] = ints[: STEP_DIM - 1]
    return v


def row_to_sequence(
    row: Dict[str, str],
    *,
    max_steps: int,
) -> np.ndarray:
    """Return float32 (T, STEP_DIM), T >= 1, T <= max_steps."""
    steps: List[np.ndarray] = []
    raw = (row.get("verification_json") or "").strip()
    if raw:
        try:
            items = json.loads(raw)
            if isinstance(items, list):
                for it in items:
                    if isinstance(it, dict):
                        steps.append(
                            _step_vector(
                                str(it.get("text") or ""),
                                bool(it.get("ok")),
                            )
                        )
        except json.JSONDecodeError:
            pass
    coach = (
        (row.get("comment") or "").strip()
        + " \n "
        + (row.get("action_guidance") or "").strip()
    )
    if coach.strip():
        steps.append(_step_vector(coach, True))
    if not steps:
        steps.append(_step_vector("", False))
    if len(steps) > max_steps:
        steps = steps[-max_steps:]
    return np.stack(steps, axis=0)


def fit_annotation_standardizer(
    samples: List[Tuple[np.ndarray, int, float]],
) -> Tuple[np.ndarray, np.ndarray]:
    flat = np.concatenate([s[0].reshape(-1, STEP_DIM) for s in samples], axis=0)
    mean = flat.mean(axis=0).astype(np.float32)
    std = flat.std(axis=0).astype(np.float32) + np.float32(1e-8)
    return mean, std


def apply_annotation_standardizer(
    samples: List[Tuple[np.ndarray, int, float]],
    mean: np.ndarray,
    std: np.ndarray,
) -> None:
    for i, (x, y, q) in enumerate(samples):
        xn = (x.astype(np.float32) - mean) / std
        samples[i] = (xn.astype(np.float32), y, q)


class ExerciseAnnotationSequenceDataset(Dataset):
    """Index CSV with exercise_class, quality, split, optional verification_json."""

    def __init__(
        self,
        index_csv: Path,
        class_to_idx: Dict[str, int],
        split: str,
        max_steps: int = 96,
    ):
        self.max_steps = int(max_steps)
        self.samples: List[Tuple[np.ndarray, int, float]] = []
        with open(index_csv, newline="") as f:
            for row in csv.DictReader(f):
                if (row.get("split") or "train") != split:
                    continue
                cls_name = (row.get("exercise_class") or "").strip()
                if cls_name not in class_to_idx:
                    continue
                try:
                    q = float(row["quality"])
                except (KeyError, ValueError):
                    continue
                x = row_to_sequence(row, max_steps=self.max_steps)
                y = class_to_idx[cls_name]
                self.samples.append((x, y, q))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, i: int):
        x, y, q = self.samples[i]
        return (
            torch.from_numpy(np.asarray(x, dtype=np.float32)),
            torch.tensor(y, dtype=torch.long),
            torch.tensor(q, dtype=torch.float32),
        )


def annotation_collate_fn(batch):
    xs, yc, yq = zip(*batch)
    lengths = torch.tensor([x.shape[0] for x in xs], dtype=torch.long)
    max_t = int(lengths.max().item())
    b = len(xs)
    f = xs[0].shape[-1]
    padded = torch.zeros(b, max_t, f, dtype=torch.float32)
    for i, x in enumerate(xs):
        padded[i, : x.shape[0]] = x
    return padded, lengths, torch.stack(yc), torch.stack(yq)
