"""Verify that a failed write rolls back the writer-connection transaction.

In v0.2 the writer owns a private ``sqlite3.Connection`` (``_writer_conn``)
and every mutation runs inside ``_tx(conn)``. To exercise the rollback path
we wrap that connection with a proxy that raises on the Nth ``execute`` call,
then confirm the error propagates to the caller and leaves the DB empty.
"""
from __future__ import annotations

import sqlite3
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


class _FlakyConn:
    """Wraps a sqlite3 connection and makes the Nth execute() call raise."""

    def __init__(self, real: sqlite3.Connection, fail_on_call: int) -> None:
        self._real = real
        self._fail_on_call = fail_on_call
        self._calls = 0

    def execute(self, *args, **kwargs):
        self._calls += 1
        if self._calls == self._fail_on_call:
            raise sqlite3.IntegrityError("simulated")
        return self._real.execute(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(self._real, name)


@pytest.mark.asyncio
async def test_rollback_on_error(tmp_db_path):
    b = SQLiteBackend(tmp_db_path)
    real_conn = b._writer_conn

    # Call sequence inside _upsert_sync (all on the writer connection):
    #   1) BEGIN
    #   2) INSERT OR REPLACE INTO nuggets ...
    #   3) DELETE FROM provenance ...   <- we will make this one fail
    # After the raise, _tx runs ROLLBACK via the *real* connection in the
    # except clause; our flaky proxy also increments, but ROLLBACK is
    # call 4 so we don't interfere.
    flaky = _FlakyConn(real_conn, fail_on_call=3)
    b._writer_conn = flaky  # type: ignore[assignment]

    n = Nugget.new(
        kind=NuggetKind.SEMANTIC_FACT,
        fact=FactTriple(subject="X", predicate="p", object="Y", text="X p Y"),
        validity=ValidityInterval(start=datetime(2020, 1, 1, tzinfo=UTC)),
        epistemic=EpistemicState(),
        provenance=(ProvenanceRecord(source_id="d", evidence_span="x"),),
    )

    with pytest.raises(sqlite3.IntegrityError):
        await b.aupsert(n)

    # Restore real connection so acount works normally.
    b._writer_conn = real_conn
    # State should have rolled back — count still 0.
    assert await b.acount() == 0
    await b.aclose()
