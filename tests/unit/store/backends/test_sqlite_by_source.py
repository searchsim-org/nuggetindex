"""``SQLiteBackend.aget_nuggets_by_source`` — SQL JOIN on provenance."""

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
from nuggetindex.store.backends.sqlite import SQLiteBackend


def _make_nugget(subject: str, obj: str, source_id: str) -> Nugget:
    return Nugget.new(
        kind=NuggetKind.SEMANTIC_FACT,
        fact=FactTriple(subject=subject, predicate="is", object=obj, text=f"{subject} is {obj}"),
        validity=ValidityInterval(start=datetime(2024, 1, 1, tzinfo=UTC)),
        epistemic=EpistemicState(confidence=0.9),
        provenance=(ProvenanceRecord(source_id=source_id, evidence_span=f"{subject} is {obj}"),),
        extraction_confidence=0.9,
    )


@pytest.mark.asyncio
async def test_returns_nuggets_for_source(tmp_path):
    backend = SQLiteBackend(tmp_path / "b.db")
    n1 = _make_nugget("Alpha", "one", "doc-a")
    n2 = _make_nugget("Beta", "two", "doc-a")
    n3 = _make_nugget("Gamma", "three", "doc-b")
    await backend.aupsert(n1)
    await backend.aupsert(n2)
    await backend.aupsert(n3)

    results = await backend.aget_nuggets_by_source("doc-a")
    ids = {n.id for n in results}
    assert ids == {n1.id, n2.id}
    await backend.aclose()


@pytest.mark.asyncio
async def test_empty_for_unknown_source(tmp_path):
    backend = SQLiteBackend(tmp_path / "b.db")
    n1 = _make_nugget("Alpha", "one", "doc-a")
    await backend.aupsert(n1)
    results = await backend.aget_nuggets_by_source("unknown")
    assert results == []
    await backend.aclose()


@pytest.mark.asyncio
async def test_distinct_when_multiple_provenance(tmp_path):
    """If a nugget has multiple provenance rows for same source, don't double-return."""
    backend = SQLiteBackend(tmp_path / "b.db")
    n = Nugget.new(
        kind=NuggetKind.SEMANTIC_FACT,
        fact=FactTriple(subject="A", predicate="is", object="B", text="A is B"),
        validity=ValidityInterval(start=datetime(2024, 1, 1, tzinfo=UTC)),
        epistemic=EpistemicState(confidence=0.9),
        provenance=(
            ProvenanceRecord(source_id="doc-a", evidence_span="A is B", char_start=0, char_end=6),
            ProvenanceRecord(
                source_id="doc-a",
                evidence_span="A is B again",
                char_start=20,
                char_end=30,
            ),
        ),
    )
    await backend.aupsert(n)
    results = await backend.aget_nuggets_by_source("doc-a")
    assert len(results) == 1
    assert results[0].id == n.id
    await backend.aclose()
