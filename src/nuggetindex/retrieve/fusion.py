"""Fusion strategies for sparse/dense retrieval results.

- Reciprocal Rank Fusion (RRF): tuning-free, SOTA in TREC/MSMARCO. Default.
- Weighted min-max: the paper's original scheme; kept for reproducibility.

Both functions accept lists of ``(id, score)`` tuples where the input order is
assumed to be the ranking (rank 1 first). Returned lists are sorted by fused
score descending; RRF ties break by first-appearance order across the input
rankings (deterministic stable sort).
"""

from __future__ import annotations

from collections import defaultdict


def reciprocal_rank_fusion(
    rankings: list[list[tuple[str, float]]],
    *,
    k: int = 60,
) -> list[tuple[str, float]]:
    """Fuse multiple ``(id, score)`` rankings via RRF.

    Each ranking contributes ``1 / (k + rank_i)`` per document. The raw scores
    are ignored; only input order matters. Returns ``(id, rrf_score)`` sorted
    descending by score. Ties break on first-appearance index across all input
    rankings, giving a stable, deterministic ordering.

    Parameters
    ----------
    rankings:
        List of rankings, each a list of ``(id, score)`` tuples in rank order.
        Empty inner rankings are silently skipped, which makes the sparse-only
        path a no-op when ``dense_backend`` is ``None``.
    k:
        RRF constant. ``60`` is the standard setting from Cormack et al.
    """
    scores: dict[str, float] = defaultdict(float)
    order: dict[str, int] = {}
    idx = 0
    for ranking in rankings:
        for rank, (doc_id, _score) in enumerate(ranking, start=1):
            scores[doc_id] += 1.0 / (k + rank)
            if doc_id not in order:
                order[doc_id] = idx
                idx += 1
    return sorted(scores.items(), key=lambda kv: (-kv[1], order[kv[0]]))


def weighted_minmax_fusion(
    sparse: list[tuple[str, float]],
    dense: list[tuple[str, float]],
    *,
    alpha: float = 0.4,
    beta: float = 0.5,
) -> list[tuple[str, float]]:
    """Min-max normalize each ranking, then weighted sum.

    Missing IDs in either ranking default to ``0.0``. If a ranking has zero
    range (all scores equal), it falls back to ``1.0`` to avoid division by
    zero; callers relying on tight differences should prefer RRF.

    Defaults ``alpha=0.4`` (sparse weight) and ``beta=0.5`` (dense weight)
    match the original paper's scheme for reproducibility.
    """

    def minmax(ranking: list[tuple[str, float]]) -> dict[str, float]:
        if not ranking:
            return {}
        vals = [s for _, s in ranking]
        lo, hi = min(vals), max(vals)
        rng = (hi - lo) or 1.0
        return {d: (s - lo) / rng for d, s in ranking}

    sp = minmax(sparse)
    dn = minmax(dense)
    all_ids = set(sp) | set(dn)
    combined = {d: alpha * sp.get(d, 0.0) + beta * dn.get(d, 0.0) for d in all_ids}
    return sorted(combined.items(), key=lambda kv: -kv[1])
