"""TF–IDF + SVD on coaching text (comment + action_guidance) for multimodal BiLSTM."""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Tuple

import numpy as np


@dataclass
class TextCoachingEncoder:
    """Fitted on training split only; transform val/test with the same vectorizer + SVD."""

    vectorizer: Any
    svd: Any
    dim: int

    def transform(self, texts: List[str]) -> np.ndarray:
        X = self.vectorizer.transform(texts)
        Z = self.svd.transform(X).astype(np.float32)
        return Z

    def dumps(self) -> bytes:
        return pickle.dumps({"vectorizer": self.vectorizer, "svd": self.svd, "dim": self.dim})

    @classmethod
    def loads(cls, blob: bytes) -> "TextCoachingEncoder":
        d = pickle.loads(blob)
        return cls(vectorizer=d["vectorizer"], svd=d["svd"], dim=int(d["dim"]))

    def save(self, path: Path) -> None:
        path.write_bytes(self.dumps())

    @classmethod
    def load(cls, path: Path) -> "TextCoachingEncoder":
        return cls.loads(path.read_bytes())


def fit_text_coaching_encoder(
    train_texts: List[str],
    *,
    svd_dim: int = 64,
    max_features: int = 8192,
    min_df: int = 1,
    random_state: int = 42,
) -> Tuple[TextCoachingEncoder, np.ndarray]:
    """Fit on training strings; returns encoder and train embeddings (N_train, dim)."""
    from sklearn.decomposition import TruncatedSVD
    from sklearn.feature_extraction.text import TfidfVectorizer

    texts = [t if t else " " for t in train_texts]
    vec = TfidfVectorizer(
        max_features=int(max_features),
        min_df=int(min_df),
        strip_accents="unicode",
        ngram_range=(1, 2),
    )
    X = vec.fit_transform(texts)
    n_comp = min(int(svd_dim), max(1, X.shape[1] - 1), max(1, X.shape[0] - 1))
    svd = TruncatedSVD(n_components=n_comp, random_state=random_state)
    svd.fit(X)
    Z = svd.transform(X).astype(np.float32)
    enc = TextCoachingEncoder(vectorizer=vec, svd=svd, dim=n_comp)
    return enc, Z
