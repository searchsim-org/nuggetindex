"""Tests for the writer-queue + read-your-own-write behaviour of SQLiteBackend.

The v0.2 backend routes every mutation through a single writer task that owns
its own ``sqlite3.Connection``. The contract these tests lock in:

* ``await store.aupsert(n)`` does not return until the enqueued write has
  COMMITted — an immediate ``aget`` must see the new row.
* Concurrent ``asyncio.gather`` of many writes serialises cleanly (no lost
  updates, no SQLite "database is locked" flakiness).
* ``aclose()`` drains any pending writes before tearing the connection down.
"""
from __future__ import annotations

import asyncio
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


def _n(obj: str) -> Nugget:
    return Nugget.new(
        kind=NuggetKind.SEMANTIC_FACT,
        fact=FactTriple(subject="S", predicate="p", object=obj, text=f"S p {obj}"),
        validity=ValidityInterval(start=datetime(2020, 1, 1, tzinfo=UTC)),
        epistemic=EpistemicState(),
        provenance=(ProvenanceRecord(source_id="d", evidence_span=f"S p {obj}"),),
    )


@pytest.mark.asyncio
async def test_read_your_own_write(tmp_db_path):
    b = SQLiteBackend(tmp_db_path)
    n = _n("A")
    await b.aupsert(n)  # must block until COMMIT
    got = await b.aget(n.id)  # immediate read MUST see it
    assert got == n
    await b.aclose()


@pytest.mark.asyncio
async def test_concurrent_writes_serialize_correctly(tmp_db_path):
    b = SQLiteBackend(tmp_db_path)
    await asyncio.gather(*(b.aupsert(_n(f"A{i}")) for i in range(50)))
    assert await b.acount() == 50
    await b.aclose()


@pytest.mark.asyncio
async def test_close_drains_pending_writes(tmp_db_path):
    b = SQLiteBackend(tmp_db_path)
    tasks = [asyncio.create_task(b.aupsert(_n(f"B{i}"))) for i in range(20)]
    await asyncio.gather(*tasks)
    await b.aclose()
    # Reopen the DB and confirm all 20 writes landed.
    b2 = SQLiteBackend(tmp_db_path)
    assert await b2.acount() == 20
    await b2.aclose()


@pytest.mark.asyncio
async def test_writer_exception_propagates_to_caller(tmp_db_path):
    """A failing write surfaces as an exception on the awaiting caller without
    killing the writer task (subsequent writes still succeed).
    """
    b = SQLiteBackend(tmp_db_path)

    # Boom-submit a write that raises inside the writer task.
    def _boom(conn):  # type: ignore[no-untyped-def]
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError, match="boom"):
        await b._submit_write(_boom)

    # Writer is still alive — a real write still commits.
    await b.aupsert(_n("after-boom"))
    assert await b.acount() == 1
    await b.aclose()


@pytest.mark.asyncio
async def test_aclose_is_idempotent(tmp_db_path):
    b = SQLiteBackend(tmp_db_path)
    await b.aupsert(_n("X"))
    await b.aclose()
    # Second close is a no-op, not an error.
    await b.aclose()


@pytest.mark.asyncio
async def test_writer_not_spawned_for_read_only_backend(tmp_db_path):
    """Constructing + only reading never spawns the writer task."""
    b = SQLiteBackend(tmp_db_path)
    _ = await b.acount()
    assert b._writer_task is None
    await b.aclose()


@pytest.mark.asyncio
async def test_mixed_reads_and_writes_see_each_other(tmp_db_path):
    """After a gathered mix of reads + writes, every committed write is
    visible to a subsequent read.
    """
    b = SQLiteBackend(tmp_db_path)
    nuggets = [_n(f"M{i}") for i in range(10)]
    # Interleave the reads and writes on the event loop.
    ops: list = []
    for n in nuggets:
        ops.append(b.aupsert(n))
        ops.append(b.acount())
    await asyncio.gather(*ops)
    # All 10 nuggets must be present.
    assert await b.acount() == 10
    for n in nuggets:
        got = await b.aget(n.id)
        assert got is not None and got.id == n.id
    await b.aclose()
