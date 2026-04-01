#!/usr/bin/env python3
"""
KMeans + within-cluster KNN imputation for skeleton sequences, following the data-driven
imputation idea in:

  Y. Chen et al., "Exploring Self-Supervised Skeleton-Based Human Action Recognition
  under Occlusions", arXiv:2309.12029 (OPSTL) — Sec. II-D KNN Imputation.
  https://arxiv.org/html/2309.12029v2
  Code reference: https://github.com/cyfml/OPSTL

**Difference from the paper:** OPSTL uses embeddings from a **self-supervised pre-trained**
encoder for KMeans, then KNN inside each cluster. Here we use **flattened, time-resampled**
2D COCO-17 coordinates as clustering features when no encoder is available (practical
offline batch step). You can swap ``cluster_features`` for SSL embeddings later.

**When to use:** Requires **multiple** sequences (same layout ``(T, 17, 2)``). Not for a
single clip in isolation — the paper imputes using *neighboring training samples*.

Typical call: ``batch_knn_impute_keypoints.py`` after standard per-sequence preprocessing.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np

try:
    from sklearn.cluster import KMeans
except ImportError as e:  # pragma: no cover
    raise ImportError("knn_skeleton_imputation requires scikit-learn") from e


def resample_time_uniform(kp: np.ndarray, target_t: int) -> np.ndarray:
    """Linear resample (T, 17, 2) along time to ``target_t`` frames."""
    kp = np.asarray(kp, dtype=np.float64)
    if kp.ndim != 3 or kp.shape[1:] != (17, 2):
        raise ValueError(f"Expected (T, 17, 2), got {kp.shape}")
    t0 = kp.shape[0]
    if t0 == target_t:
        return kp.copy()
    if t0 <= 1:
        return np.repeat(kp[:1], target_t, axis=0)
    x_old = np.linspace(0.0, 1.0, t0)
    x_new = np.linspace(0.0, 1.0, target_t)
    out = np.zeros((target_t, 17, 2), dtype=np.float64)
    for j in range(17):
        for c in range(2):
            out[:, j, c] = np.interp(x_new, x_old, kp[:, j, c])
    return out


def validity_mask(
    kp: np.ndarray,
    confidence: Optional[np.ndarray] = None,
    conf_threshold: float = 0.3,
) -> np.ndarray:
    """
    Boolean mask (T, 17, 2) — True where coordinate is considered observed.
    A joint is invalid if both coords ~0 or confidence below threshold (when given).
    """
    kp = np.asarray(kp, dtype=np.float64)
    m = np.ones(kp.shape, dtype=bool)
    # zero / missing
    joint_zero = np.all(np.abs(kp) < 1e-8, axis=-1)  # (T, 17)
    m[joint_zero] = False
    if confidence is not None:
        c = np.asarray(confidence, dtype=np.float64)
        if c.ndim == 2 and c.shape[0] == kp.shape[0]:
            low = c < conf_threshold
            m[low] = False
        elif c.ndim == 1 and len(c) == 17:
            low_j = c < conf_threshold
            m[:, low_j, :] = False
    return m


def _dist_opstl_style(
    xa: np.ndarray,
    xb: np.ndarray,
    ma: np.ndarray,
    mb: np.ndarray,
    total_dims: int,
) -> float:
    """
    Distance similar to arXiv:2309.12029 Eq. (3): sqrt(w * d_ignore), with w = total/present
    and d_ignore Euclidean ignoring coordinates missing in either sample.
    """
    both = ma & mb
    if not np.any(both):
        return float("inf")
    diff = xa[both] - xb[both]
    d_ignore = float(np.mean(diff * diff))  # mean squared error on overlap
    cnt = int(np.sum(both))
    w = float(total_dims) / float(max(cnt, 1))
    return float(np.sqrt(w * d_ignore))


def cluster_features_from_sequence(
    kp: np.ndarray,
    target_t: int = 32,
    confidence: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns:
      feat: (D,) flattened resampled sequence for KMeans
      flat: (D,) same as feat (for imputation we use same layout)
      mask_flat: (D,) validity on flattened vector
    """
    rs = resample_time_uniform(kp, target_t)
    vm = validity_mask(rs, confidence=confidence)
    flat = rs.reshape(-1)
    mflat = vm.reshape(-1)
    feat = np.where(mflat, flat, 0.0)
    return feat.astype(np.float64), flat.astype(np.float64), mflat


def knn_impute_flat_sample(
    x_self: np.ndarray,
    m_self: np.ndarray,
    neighbors: List[Tuple[np.ndarray, np.ndarray]],
    k: int,
    total_dims: int,
) -> np.ndarray:
    """
    Impute missing entries using K nearest *neighbors* (other sequences), weighted by 1/dist
    (arXiv:2309.12029 Eq. (4)–(5)). Only neighbors with a valid coordinate at a missing index
    contribute to that coordinate.
    """
    out = x_self.copy()
    miss_idx = np.where(~m_self)[0]
    if miss_idx.size == 0:
        return out

    scored: List[Tuple[float, np.ndarray, np.ndarray]] = []
    for xc, mc in neighbors:
        d = _dist_opstl_style(x_self, xc, m_self, mc, total_dims)
        if np.isfinite(d) and d > 1e-12:
            scored.append((d, xc, mc))
    scored.sort(key=lambda t: t[0])
    if not scored:
        # e.g. all pairwise distances infinite — use all neighbors unweighted mean per dim
        for d_idx in miss_idx:
            vals = [float(xc[d_idx]) for xc, mc in neighbors if mc[d_idx]]
            if vals:
                out[d_idx] = float(np.mean(vals))
        return out

    top = scored[: max(1, min(k, len(scored)))]

    for d_idx in miss_idx:
        num = 0.0
        den = 0.0
        for dist, xc, mc in top:
            if mc[d_idx]:
                rj = 1.0 / dist
                num += rj * float(xc[d_idx])
                den += rj
        if den > 1e-12:
            out[d_idx] = num / den
        else:
            vals = [float(xc[d_idx]) for dist, xc, mc in top if mc[d_idx]]
            if vals:
                out[d_idx] = float(np.mean(vals))

    return out


def knn_impute_batch(
    sequences: List[np.ndarray],
    confidences: Optional[List[Optional[np.ndarray]]] = None,
    *,
    target_t: int = 32,
    n_clusters: int = 8,
    k_neighbors: int = 5,
    random_state: int = 42,
) -> List[np.ndarray]:
    """
    OPSTL-style two-stage imputation (KMeans on features, then KNN within cluster).

    Parameters
    ----------
    sequences
        List of (T, 17, 2) arrays (same preprocessing as your pipeline).
    confidences
        Optional per-frame (T, 17) or None per sequence.
    n_clusters
        KMeans clusters (capped by n_samples).
    k_neighbors
        K for KNN inside each cluster.
    """
    n = len(sequences)
    if n < 2:
        raise ValueError("Need at least 2 sequences for KMeans+KNN imputation")

    feats: List[np.ndarray] = []
    flats: List[np.ndarray] = []
    masks: List[np.ndarray] = []
    total_dims = target_t * 17 * 2

    for i, kp in enumerate(sequences):
        conf = confidences[i] if confidences else None
        f, fl, mf = cluster_features_from_sequence(
            kp, target_t=target_t, confidence=conf
        )
        feats.append(f)
        flats.append(fl)
        masks.append(mf)

    X = np.stack(feats, axis=0)
    n_clust = max(1, min(int(n_clusters), n))
    km = KMeans(n_clusters=n_clust, random_state=random_state, n_init=10)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        labels = km.fit_predict(X)

    out_flat: List[np.ndarray] = []

    for i in range(n):
        lab = labels[i]
        pool_j = [j for j in range(n) if labels[j] == lab and j != i]
        neighbors: List[Tuple[np.ndarray, np.ndarray]] = [
            (flats[j], masks[j]) for j in pool_j
        ]
        if not neighbors:
            # single sample in cluster: use all other sequences
            neighbors = [(flats[j], masks[j]) for j in range(n) if j != i]

        imputed = knn_impute_flat_sample(
            flats[i],
            masks[i],
            neighbors,
            k=k_neighbors,
            total_dims=total_dims,
        )
        out_flat.append(imputed)

    # reshape back to (target_t, 17, 2) then resample to original T per sequence
    results: List[np.ndarray] = []
    for i, kp_orig in enumerate(sequences):
        arr = out_flat[i].reshape(target_t, 17, 2)
        t_orig = kp_orig.shape[0]
        if t_orig == target_t:
            results.append(arr.astype(np.float32))
        else:
            results.append(resample_time_uniform(arr, t_orig).astype(np.float32))

    return results


def load_npz_keypoints(path: Path) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    data = np.load(path, allow_pickle=True)
    kp = np.asarray(data["keypoints"], dtype=np.float64)
    conf = None
    if "confidence" in data.files:
        conf = np.asarray(data["confidence"], dtype=np.float64)
    return kp, conf


def impute_directory(
    input_dir: Path,
    output_dir: Path,
    *,
    pattern: str = "*_keypoints.npz",
    target_t: int = 32,
    n_clusters: int = 8,
    k_neighbors: int = 5,
    seed: int = 42,
    max_files: int = 0,
) -> int:
    """
    Load all ``*_keypoints.npz``, run batch KNN imputation, save ``*_keypoints_knn.npz``.
    Returns number of files written.
    """
    paths = sorted(input_dir.glob(pattern))
    if max_files and max_files > 0:
        paths = paths[: max_files]
    if len(paths) < 2:
        raise ValueError(f"Need at least 2 files matching {pattern} in {input_dir}")

    seqs: List[np.ndarray] = []
    confs: List[Optional[np.ndarray]] = []
    stems: List[str] = []
    valid_paths: List[Path] = []

    for p in paths:
        kp, c = load_npz_keypoints(p)
        if kp.ndim != 3 or kp.shape[1:] != (17, 2):
            continue
        seqs.append(kp)
        confs.append(c)
        stems.append(p.stem.replace("_keypoints", ""))
        valid_paths.append(p)

    if len(seqs) < 2:
        raise ValueError("Not enough valid (T,17,2) sequences")

    imputed = knn_impute_batch(
        seqs,
        confs,
        target_t=target_t,
        n_clusters=n_clusters,
        k_neighbors=k_neighbors,
        random_state=seed,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    n_out = 0
    for stem, orig_p, kp_new, conf in zip(stems, valid_paths, imputed, confs):
        out_path = output_dir / f"{stem}_keypoints_knn.npz"
        payload = {
            "keypoints": kp_new.astype(np.float32),
            "source_npz": str(orig_p.name),
            "knn_imputation": "opstl-style-kmeans-knn",
            "arxiv_ref": "2309.12029",
        }
        if conf is not None:
            payload["confidence"] = np.asarray(conf, dtype=np.float32)
        np.savez_compressed(out_path, **payload)
        n_out += 1

    meta = {
        "method": "KMeans+KNN (OPSTL Sec II-D, feature=resampled skeleton)",
        "reference": "https://arxiv.org/html/2309.12029v2",
        "n_sequences": len(seqs),
        "n_clusters": min(n_clusters, len(seqs)),
        "k_neighbors": k_neighbors,
        "target_t": target_t,
        "seed": seed,
    }
    import json

    with open(output_dir / "knn_imputation_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    return n_out
