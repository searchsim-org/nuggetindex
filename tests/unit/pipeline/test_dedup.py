"""Tests for Stage 4a deduplication."""

from datetime import UTC, datetime

import pytest

from nuggetindex.core.enums import NuggetKind
from nuggetindex.core.models import (
    EpistemicState,
    FactTriple,
    Nugget,
    ProvenanceRecord,
    ValidityInterval,
)
from nuggetindex.pipeline.dedup import Deduplicator, jaccard_ngram


def _n(*, subject: str = "Google", predicate: str = "ceo", obj: str = "Sundar Pichai") -> Nugget:
    return Nugget.new(
        kind=NuggetKind.SEMANTIC_FACT,
        fact=FactTriple(subject=subject, predicate=predicate, object=obj, text="x"),
        validity=ValidityInterval(start=datetime(2019, 1, 1, tzinfo=UTC)),
        epistemic=EpistemicState(),
        provenance=(ProvenanceRecord(source_id="d", evidence_span="x"),),
    )


# --- jaccard_ngram ---------------------------------------------------------


def test_jaccard_ngram_identical() -> None:
    assert jaccard_ngram("Sundar Pichai", "Sundar Pichai") == 1.0


def test_jaccard_ngram_similar() -> None:
    assert jaccard_ngram("Sundar Pichai", "Pichai Sundar") > 0.5


def test_jaccard_ngram_different() -> None:
    assert jaccard_ngram("Sundar Pichai", "Larry Page") < 0.3


def test_jaccard_ngram_empty_strings_equal() -> None:
    assert jaccard_ngram("", "") == 1.0


def test_jaccard_ngram_one_empty_zero() -> None:
    assert jaccard_ngram("", "abc") == 0.0


def test_jaccard_ngram_short_string_handles_minimum_length() -> None:
    # Both shorter than n=3 but identical -> 1.0
    assert jaccard_ngram("ab", "ab") == 1.0
    # Shorter, different
    assert jaccard_ngram("ab", "cd") == 0.0


# --- Deduplicator (Jaccard fallback) --------------------------------------


@pytest.mark.asyncio
async def test_dedup_without_encoder_uses_jaccard_match_exact() -> None:
    dedup = Deduplicator(encoder=None)
    a = _n(obj="Sundar Pichai")
    b = _n(obj="Sundar Pichai")
    match = await dedup.afind_duplicate(a, [b])
    assert match is b


@pytest.mark.asyncio
async def test_dedup_without_encoder_no_match_when_different_object() -> None:
    dedup = Deduplicator(encoder=None)
    a = _n(obj="Sundar Pichai")
    b = _n(obj="Larry Page")
    match = await dedup.afind_duplicate(a, [b])
    assert match is None


@pytest.mark.asyncio
async def test_dedup_ignores_nuggets_with_different_key() -> None:
    dedup = Deduplicator(encoder=None)
    a = _n(subject="Google", obj="Sundar Pichai")
    b = _n(subject="Alphabet", obj="Sundar Pichai")  # different subject
    match = await dedup.afind_duplicate(a, [b])
    assert match is None


@pytest.mark.asyncio
async def test_dedup_empty_existing_returns_none() -> None:
    dedup = Deduplicator(encoder=None)
    match = await dedup.afind_duplicate(_n(), [])
    assert match is None


# --- Deduplicator (encoder path) ------------------------------------------


@pytest.mark.asyncio
async def test_dedup_with_encoder_uses_cosine() -> None:
    np = pytest.importorskip("numpy")

    def stub_encoder(texts: list[str]) -> "np.ndarray":
        mapping = {
            "Sundar Pichai": [1.0, 0.0],
            "S. Pichai": [0.98, 0.02],
            "Larry Page": [0.0, 1.0],
        }
        return np.array([mapping[t] for t in texts], dtype=float)

    dedup = Deduplicator(encoder=stub_encoder, cosine_threshold=0.92)

    a = _n(obj="Sundar Pichai")
    near = _n(obj="S. Pichai")
    far = _n(obj="Larry Page")

    match = await dedup.afind_duplicate(a, [near, far])
    assert match is near


@pytest.mark.asyncio
async def test_dedup_with_encoder_below_threshold_none() -> None:
    np = pytest.importorskip("numpy")

    def stub_encoder(texts: list[str]) -> "np.ndarray":
        mapping = {
            "Sundar Pichai": [1.0, 0.0],
            "Pichai variant": [0.5, 0.5],
        }
        return np.array([mapping[t] for t in texts], dtype=float)

    dedup = Deduplicator(encoder=stub_encoder, cosine_threshold=0.92)
    match = await dedup.afind_duplicate(_n(obj="Sundar Pichai"), [_n(obj="Pichai variant")])
    assert match is None
