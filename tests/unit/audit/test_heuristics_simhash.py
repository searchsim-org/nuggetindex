"""Tests for the SimHash helper backing dedup (Fix 4)."""

from __future__ import annotations

from nuggetindex.audit.heuristics.sample import _hamming, _simhash


def test_simhash_identical_texts_equal() -> None:
    """Two docs with identical text hash to the same simhash."""
    text = "The quick brown fox jumps over the lazy dog near the river bank."
    assert _simhash(text) == _simhash(text)


def test_simhash_near_duplicate_small_hamming() -> None:
    """A single-word change in a long text produces a small Hamming distance (<10 of 64 bits).

    SimHash signal strength scales with token count; the assertion uses a
    paragraph-length passage so a single-token swap measures as a small
    Hamming perturbation rather than a coincidental hash collision artefact.
    """
    a = (
        "The quick brown fox jumps over the lazy dog near the river bank on a bright "
        "summer morning. Birds sing overhead while the old mill wheel turns slowly. "
        "Children play along the path, gathering wildflowers and smooth stones from "
        "the stream. A farmer waves from the far field as the bells in the village "
        "church ring the hour. Everything feels peaceful and unhurried beneath the "
        "warm sun, and the breeze carries the scent of fresh hay across the meadow."
    )
    b = a.replace("fox", "cat")
    dist = _hamming(_simhash(a), _simhash(b))
    assert dist < 10, f"expected Hamming < 10 for near-duplicate, got {dist}"


def test_simhash_different_texts_large_hamming() -> None:
    """Two unrelated long texts should produce a Hamming distance > 15."""
    a = (
        "Economic outlook remains uncertain as central banks raise interest rates "
        "across several major economies, leaving investors worried about a slowdown "
        "in consumer spending. Bond markets have priced in further hikes while "
        "equity traders rotate into defensive sectors such as healthcare and "
        "utilities. Policymakers continue to stress that bringing inflation back to "
        "target is the priority, even at the cost of slower growth."
    )
    b = (
        "The cat sat on the mat while the dog barked loudly at strangers passing "
        "by the garden fence. A warm afternoon breeze stirred the leaves, and the "
        "hummingbird darted between the pink azaleas near the porch. Children "
        "laughed as they chased each other around the old oak tree, their voices "
        "rising above the distant hum of lawnmowers in the neighbourhood."
    )
    dist = _hamming(_simhash(a), _simhash(b))
    assert dist > 15, f"expected Hamming > 15 for unrelated texts, got {dist}"


def test_simhash_empty_text_returns_zero() -> None:
    """Empty input yields a sentinel 0 — verified so callers don't get surprises."""
    assert _simhash("") == 0
    assert _simhash("   ") == 0
