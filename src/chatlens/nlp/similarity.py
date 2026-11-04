"""Semantic / lexical similarity search and near-duplicate detection.

Backends: ``tfidf`` (word + char n-grams; handles Roman-Urdu spelling variation reasonably) and
``dense`` (any ``encoder(list[str]) -> ndarray``, e.g. multilingual sentence-transformers)."""
from __future__ import annotations

import re
from collections.abc import Callable

import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import normalize

from ..preprocessing.normalize import normalize_for_classification, tokenize


def _prep(t: str) -> str:
    return re.sub(r"[^\w\s]|_", " ", normalize_for_classification(t))


class SimilarityIndex:
    def __init__(self, backend: str = "tfidf", encoder: Callable | None = None):
        if backend not in ("tfidf", "dense"):
            raise ValueError(backend)
        if backend == "dense" and encoder is None:
            raise ValueError("dense backend needs an encoder")
        self.backend, self.encoder = backend, encoder
        self.texts: list[str] = []

    def fit(self, texts: list[str]) -> SimilarityIndex:
        self.texts = list(texts)
        if self.backend == "tfidf":
            self.vec = FeatureUnion([
                ("w", TfidfVectorizer(preprocessor=_prep, sublinear_tf=True,
                                      tokenizer=tokenize, token_pattern=None, lowercase=False,
                                      dtype=np.float32)),
                ("c", TfidfVectorizer(preprocessor=_prep, analyzer="char_wb",
                                      ngram_range=(3, 4), sublinear_tf=True, dtype=np.float32))])
            self.M = normalize(self.vec.fit_transform(self.texts))
        else:
            self.M = normalize(np.asarray(self.encoder(self.texts), dtype=np.float32))
        return self

    def _encode(self, qs):
        return normalize(self.vec.transform(qs)) if self.backend == "tfidf" else normalize(
            np.asarray(self.encoder(qs), dtype=np.float32))

    def query(self, text: str, k: int = 5, exclude_self: bool = False) -> list[tuple[int, float]]:
        q = self._encode([text])
        sims = (self.M @ q.T)
        sims = np.asarray(sims.todense()).ravel() if sparse.issparse(sims) else np.asarray(sims).ravel()
        order = np.argsort(-sims)
        if exclude_self:
            order = [i for i in order if self.texts[i] != text]
        return [(int(i), float(sims[i])) for i in order[:k]]

    def near_duplicates(self, threshold: float = 0.9, max_pairs: int = 1000,
                        chunk: int = 500) -> list[tuple[int, int, float]]:
        """Pairs (i<j) with cosine >= threshold, computed in blocks (bounded memory)."""
        n = self.M.shape[0]
        pairs: list[tuple[int, int, float]] = []
        for s in range(0, n, chunk):
            blk = self.M[s:s + chunk] @ self.M.T
            blk = blk.toarray() if sparse.issparse(blk) else np.asarray(blk)
            for r in range(blk.shape[0]):
                i = s + r
                cols = np.where(blk[r, i + 1:] >= threshold)[0] + i + 1
                pairs += [(i, int(j), float(blk[r, j])) for j in cols]
                if len(pairs) >= max_pairs:
                    return sorted(pairs, key=lambda p: -p[2])[:max_pairs]
        return sorted(pairs, key=lambda p: -p[2])
