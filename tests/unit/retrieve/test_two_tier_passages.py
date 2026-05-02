"""Integration test: retrieve nuggets, then resolve source passages.

Exercises the two-tier retrieval contract: semantic BM25/fusion over nugget
text picks the relevant records, then ``aget_source_passages`` hydrates the
original document text via each nugget's provenance source_id.
"""
from __future__ import annotations

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
from nuggetindex.store import NuggetStore


@pytest.mark.asyncio
async def test_two_tier_passage_resolution(tmp_db_path):
    store = NuggetStore(db_path=tmp_db_path)

    # Two source documents with distinct passages.
    passage_a = (
        "Sundar Pichai was promoted to CEO of Google in October 2015, "
        "replacing Larry Page who moved to lead Alphabet."
    )
    passage_b = (
        "Apple Inc. is headquartered at Apple Park in Cupertino, California."
    )
    await store.backend.aupsert_passage("doc-google", "https://ex.com/google", passage_a)
    await store.backend.aupsert_passage("doc-apple", "https://ex.com/apple", passage_b)

    n_google = Nugget.new(
        kind=NuggetKind.SEMANTIC_FACT,
        fact=FactTriple(
            subject="Google",
            predicate="ceo",
            object="Pichai",
            text="Sundar Pichai is CEO of Google",
        ),
        validity=ValidityInterval(start=datetime(2015, 10, 1, tzinfo=UTC)),
        epistemic=EpistemicState(),
        provenance=(
            ProvenanceRecord(
                source_id="doc-google",
                evidence_span="Sundar Pichai was promoted to CEO of Google",
                char_start=0,
                char_end=44,
            ),
        ),
    )
    n_apple = Nugget.new(
        kind=NuggetKind.SEMANTIC_FACT,
        fact=FactTriple(
            subject="Apple",
            predicate="hq",
            object="Cupertino",
            text="Apple is headquartered in Cupertino",
        ),
        validity=ValidityInterval(start=datetime(2015, 10, 1, tzinfo=UTC)),
        epistemic=EpistemicState(),
        provenance=(
            ProvenanceRecord(
                source_id="doc-apple",
                evidence_span="Apple Inc. is headquartered at Apple Park in Cupertino",
                char_start=0,
                char_end=54,
            ),
        ),
    )
    await store.aadd(n_google)
    await store.aadd(n_apple)

    # Retrieve and resolve passages.
    results = await store.aretrieve("CEO of Google", top_k=5)
    assert len(results) >= 1
    top_nugget = results[0].nugget
    assert top_nugget.id == n_google.id

    retrieved_nuggets = [r.nugget for r in results]
    passages = await store.aget_source_passages(retrieved_nuggets)
    # The google doc passage must be resolved for the Pichai nugget.
    assert "doc-google" in passages
    assert passages["doc-google"] == passage_a

    await store.aclose()


@pytest.mark.asyncio
async def test_two_tier_missing_passages_silently_dropped(tmp_db_path):
    """If a nugget's source_id has no stored passage, aget_source_passages
    silently omits it rather than raising. Validates the contract that
    callers can iterate results['doc-id'] with a fallback."""
    store = NuggetStore(db_path=tmp_db_path)
    n = Nugget.new(
        kind=NuggetKind.SEMANTIC_FACT,
        fact=FactTriple(
            subject="X", predicate="p", object="Y", text="X p Y"
        ),
        validity=ValidityInterval(start=datetime(2020, 1, 1, tzinfo=UTC)),
        epistemic=EpistemicState(),
        provenance=(ProvenanceRecord(source_id="absent-doc", evidence_span="x"),),
    )
    await store.aadd(n)
    passages = await store.aget_source_passages([n])
    assert passages == {}
    await store.aclose()
