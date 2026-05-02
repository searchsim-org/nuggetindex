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


def _nugget(
    subject: str = "Google",
    obj: str = "Pichai",
    start: datetime = datetime(2015, 10, 1, tzinfo=UTC),
) -> Nugget:
    return Nugget.new(
        kind=NuggetKind.SEMANTIC_FACT,
        fact=FactTriple(
            subject=subject, predicate="ceo", object=obj, text=f"{obj} is CEO of {subject}"
        ),
        validity=ValidityInterval(start=start),
        epistemic=EpistemicState(),
        provenance=(ProvenanceRecord(source_id="doc-1", evidence_span=f"{obj} is CEO"),),
    )


@pytest.mark.asyncio
async def test_upsert_then_get(tmp_db_path):
    backend = SQLiteBackend(tmp_db_path)
    n = _nugget()
    await backend.aupsert(n)
    restored = await backend.aget(n.id)
    assert restored == n
    await backend.aclose()


@pytest.mark.asyncio
async def test_upsert_idempotent(tmp_db_path):
    backend = SQLiteBackend(tmp_db_path)
    n = _nugget()
    await backend.aupsert(n)
    await backend.aupsert(n)
    assert await backend.acount() == 1
    await backend.aclose()


@pytest.mark.asyncio
async def test_get_missing_returns_none(tmp_db_path):
    backend = SQLiteBackend(tmp_db_path)
    assert await backend.aget("deadbeefdeadbeef") is None
    await backend.aclose()


@pytest.mark.asyncio
async def test_find_by_key_groups_same_key(tmp_db_path):
    backend = SQLiteBackend(tmp_db_path)
    a = _nugget(obj="Pichai", start=datetime(2015, 10, 1, tzinfo=UTC))
    b = _nugget(obj="Page", start=datetime(2011, 4, 1, tzinfo=UTC))
    await backend.aupsert(a)
    await backend.aupsert(b)
    rows = await backend.afind_by_key(("Google", "ceo", "global"))
    assert len(rows) == 2
    assert {r.fact.object for r in rows} == {"Pichai", "Page"}
    await backend.aclose()
