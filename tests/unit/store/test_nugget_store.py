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


def _n(obj: str = "Pichai") -> Nugget:
    return Nugget.new(
        kind=NuggetKind.SEMANTIC_FACT,
        fact=FactTriple(
            subject="Google", predicate="ceo", object=obj, text=f"{obj} is CEO"
        ),
        validity=ValidityInterval(start=datetime(2015, 10, 1, tzinfo=UTC)),
        epistemic=EpistemicState(),
        provenance=(ProvenanceRecord(source_id="doc-1", evidence_span="x"),),
    )


@pytest.mark.asyncio
async def test_store_add_and_count(tmp_db_path):
    store = NuggetStore(db_path=tmp_db_path)
    result = await store.aadd(_n())
    assert result.created is True
    assert await store.acount() == 1
    await store.aclose()


@pytest.mark.asyncio
async def test_store_add_idempotent(tmp_db_path):
    store = NuggetStore(db_path=tmp_db_path)
    n = _n()
    await store.aadd(n)
    result = await store.aadd(n)
    assert await store.acount() == 1
    assert result.created is False  # second add merged
    await store.aclose()


def test_sync_add_works(tmp_db_path):
    store = NuggetStore(db_path=tmp_db_path)
    store.add(_n())
    assert store.count() == 1
    store.close()


def test_top_level_export():
    """NuggetStore must be importable from top-level nuggetindex package."""
    from nuggetindex import AddResult, IngestResult, NuggetStore

    assert NuggetStore is not None
    assert AddResult is not None
    assert IngestResult is not None
