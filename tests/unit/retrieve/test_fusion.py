"""Unit tests for fusion primitives (RRF + weighted min-max)."""
from nuggetindex.retrieve.fusion import (
    reciprocal_rank_fusion,
    weighted_minmax_fusion,
)


def test_rrf_on_two_overlapping_rankings():
    sparse = [("a", 0.9), ("b", 0.7), ("c", 0.5)]
    dense = [("b", 0.95), ("a", 0.8), ("d", 0.6)]
    fused = reciprocal_rank_fusion([sparse, dense], k=60)
    ids = [f[0] for f in fused]
    assert set(ids) == {"a", "b", "c", "d"}
    # "a" in rank 1 (sparse) + rank 2 (dense), "b" in rank 2 + rank 1 -> top 2
    assert ids[0] in ("a", "b")
    assert ids[1] in ("a", "b")


def test_rrf_handles_ties_stably():
    # If two docs have identical RRF scores, order is deterministic (stable
    # sort follows first-appearance in the first non-empty ranking).
    sparse = [("a", 1.0), ("b", 1.0)]
    dense = [("a", 1.0), ("b", 1.0)]
    fused = reciprocal_rank_fusion([sparse, dense], k=60)
    assert fused[0][0] == "a"
    assert fused[1][0] == "b"


def test_rrf_single_ranking():
    """Dense-disabled path: single-input rankings must sort cleanly."""
    sparse = [("a", 0.9), ("b", 0.7), ("c", 0.5)]
    fused = reciprocal_rank_fusion([sparse], k=60)
    ids = [f[0] for f in fused]
    assert ids == ["a", "b", "c"]


def test_rrf_empty_rankings():
    fused = reciprocal_rank_fusion([], k=60)
    assert fused == []


def test_rrf_skips_empty_rankings():
    sparse = [("a", 0.9), ("b", 0.7)]
    fused = reciprocal_rank_fusion([sparse, []], k=60)
    ids = [f[0] for f in fused]
    assert ids == ["a", "b"]


def test_rrf_score_values():
    """Concrete check on k=60 arithmetic."""
    sparse = [("a", 1.0)]
    dense = [("a", 1.0)]
    fused = reciprocal_rank_fusion([sparse, dense], k=60)
    # a at rank 1 in both -> 1/61 + 1/61 = 2/61
    assert fused[0][0] == "a"
    assert abs(fused[0][1] - 2.0 / 61.0) < 1e-9


def test_weighted_minmax_normalizes_and_combines():
    sparse = [("a", 10.0), ("b", 5.0)]
    dense = [("a", 0.8), ("c", 0.6)]
    fused = weighted_minmax_fusion(sparse, dense, alpha=0.4, beta=0.5)
    ids = [f[0] for f in fused]
    assert "a" in ids and "b" in ids and "c" in ids
    # "a" has both sparse-top (normalized 1.0) and dense-top (1.0) ->
    # combined 0.4 + 0.5 = 0.9. Should be highest.
    assert ids[0] == "a"


def test_weighted_minmax_missing_defaults_zero():
    sparse = [("a", 10.0), ("b", 5.0)]
    dense: list[tuple[str, float]] = []
    fused = weighted_minmax_fusion(sparse, dense, alpha=0.4, beta=0.5)
    ids = [f[0] for f in fused]
    assert ids[0] == "a"  # sparse-top dominates
    assert set(ids) == {"a", "b"}


def test_weighted_minmax_identical_scores():
    """All scores equal -> range is zero, no NaN."""
    sparse = [("a", 1.0), ("b", 1.0)]
    dense = [("a", 0.5), ("b", 0.5)]
    fused = weighted_minmax_fusion(sparse, dense, alpha=0.4, beta=0.5)
    assert len(fused) == 2
    # with rng=1 fallback, both score 0 (since each subtracts its own min);
    # this should still sort without error.
    for _, s in fused:
        assert s == 0.0
