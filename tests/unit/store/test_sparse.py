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


@pytest.mark.asyncio
async def test_bm25_ranks_relevant_higher(tmp_db_path):
    b = SQLiteBackend(tmp_db_path)
    docs = [
        ("Sundar Pichai is CEO of Google", "Pichai"),
        ("Apple is headquartered in Cupertino", "Cupertino"),
        ("Google was founded in 1998", "1998"),
    ]
    for text, obj in docs:
        await b.aupsert(
            Nugget.new(
                kind=NuggetKind.SEMANTIC_FACT,
                fact=FactTriple(subject="X", predicate="p", object=obj, text=text),
                validity=ValidityInterval(start=datetime(2020, 1, 1, tzinfo=UTC)),
                epistemic=EpistemicState(),
                provenance=(ProvenanceRecord(source_id="d", evidence_span="x"),),
            )
        )
    results = await b.abm25_search("CEO of Google", candidate_ids=None, top_k=3)
    assert len(results) >= 1
    # First result should be the Pichai one (has "CEO" and "Google")
    first = await b.aget(results[0][0])
    assert first is not None
    assert "CEO" in first.fact.text
    assert "Google" in first.fact.text
    await b.aclose()


@pytest.mark.asyncio
async def test_bm25_candidate_filter(tmp_db_path):
    b = SQLiteBackend(tmp_db_path)
    n1 = Nugget.new(
        kind=NuggetKind.SEMANTIC_FACT,
        fact=FactTriple(subject="X", predicate="p", object="A", text="A talks about Google"),
        validity=ValidityInterval(start=datetime(2020, 1, 1, tzinfo=UTC)),
        epistemic=EpistemicState(),
        provenance=(ProvenanceRecord(source_id="d", evidence_span="x"),),
    )
    n2 = Nugget.new(
        kind=NuggetKind.SEMANTIC_FACT,
        fact=FactTriple(subject="Y", predicate="p", object="B", text="B also mentions Google"),
        validity=ValidityInterval(start=datetime(2020, 1, 1, tzinfo=UTC)),
        epistemic=EpistemicState(),
        provenance=(ProvenanceRecord(source_id="d", evidence_span="x"),),
    )
    await b.aupsert(n1)
    await b.aupsert(n2)
    # Filter to only n1
    results = await b.abm25_search("Google", candidate_ids=[n1.id], top_k=10)
    returned_ids = {r[0] for r in results}
    assert returned_ids == {n1.id}
    await b.aclose()
