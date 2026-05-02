"""Stage 4a: deduplication.

Improvement B over the paper: SOTA dedup via sentence embeddings with a
cosine-similarity threshold of 0.92 (configurable). Falls back to character
n-gram Jaccard (threshold 0.85) when no encoder is available so the pipeline
remains functional offline.

Deduplication is scoped to nuggets that share a key (subject, predicate,
scope); we compare object values to catch alias variation ("Sundar Pichai"
vs. "S. Pichai"). An encoder is any callable ``(list[str]) -> np.ndarray`` of
shape ``(n, d)``.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from nuggetindex.core.models import Nugget


def jaccard_ngram(a: str, b: str, n: int = 3) -> float:
    """Character-level n-gram Jaccard similarity, case-insensitive."""

    def grams(s: str) -> set[str]:
        lowered = s.lower()
        if len(lowered) < n:
            return {lowered} if lowered else set()
        return {lowered[i : i + n] for i in range(len(lowered) - n + 1)}

    ga, gb = grams(a), grams(b)
    if not ga and not gb:
        return 1.0
    if not ga or not gb:
        return 0.0
    inter = len(ga & gb)
    union = len(ga | gb)
    return inter / union if union else 0.0


class Deduplicator:
    """Find duplicates among candidate nuggets sharing the same key.

    ``encoder`` is an optional callable that maps a list of strings to a
    numpy array of embeddings; when provided, cosine similarity >=
    ``cosine_threshold`` flags a duplicate. Otherwise the Jaccard path fires
    with ``jaccard_threshold``.
    """

    def __init__(
        self,
        encoder: Callable[[list[str]], Any] | None = None,
        *,
        cosine_threshold: float = 0.92,
        jaccard_threshold: float = 0.85,
    ) -> None:
        self.encoder = encoder
        self.cosine_threshold = cosine_threshold
        self.jaccard_threshold = jaccard_threshold

    async def afind_duplicate(
        self,
        candidate: Nugget,
        existing: list[Nugget],
    ) -> Nugget | None:
        """Return an existing nugget considered equivalent to ``candidate``, or None."""
        same_key = [e for e in existing if e.key == candidate.key]
        if not same_key:
            return None

        if self.encoder is None:
            # Jaccard fallback: compare object values.
            best: Nugget | None = None
            best_score = 0.0
            for e in same_key:
                score = jaccard_ngram(e.fact.object, candidate.fact.object)
                if score >= self.jaccard_threshold and score > best_score:
                    best = e
                    best_score = score
            return best

        # Semantic path: encode objects and compare cosine similarity.
        import numpy as np

        texts = [candidate.fact.object, *(e.fact.object for e in same_key)]
        vecs = np.asarray(self.encoder(texts), dtype=float)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        # Guard against zero vectors (would NaN the division).
        norms = np.where(norms == 0, 1.0, norms)
        vecs = vecs / norms
        cand = vecs[0]
        sims = vecs[1:] @ cand
        idx = int(np.argmax(sims))
        if float(sims[idx]) >= self.cosine_threshold:
            return same_key[idx]
        return None
