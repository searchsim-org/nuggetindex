"""SQLite backend: single-file, FTS5-enabled, WAL-mode.

Default metadata + sparse backend. All operations are transactional per call.

Concurrency model (v0.2.0)
--------------------------
- **Reads** use a per-thread connection pool (:class:`_ConnectionPool`). SQLite
  in WAL mode supports an arbitrary number of concurrent readers as long as
  each reader uses a distinct connection, so giving each executor thread its
  own connection unlocks real read parallelism.

- **Writes** are serialised through a single writer task that consumes a queue
  of ``(fn, Future)`` pairs and executes each ``fn`` against a dedicated
  writer connection. ``aupsert`` / ``aupsert_passage`` await the per-operation
  future, giving callers a **read-your-own-write** guarantee: the awaitable
  does not return until the write has been COMMITted.

This replaces v0.1's single shared ``sqlite3.Connection`` + ``asyncio.Lock``
arrangement, which caused flaky behaviour under heavy ``asyncio.gather`` load
because the single connection was shared across executor threads (requiring
``check_same_thread=False``) while the lock was held async-ly.
"""
from __future__ import annotations

import asyncio
import sqlite3
import threading
from collections.abc import Callable, Iterable, Iterator
from contextlib import contextmanager, suppress
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, TypeVar

from nuggetindex.core.enums import LifecycleStatus
from nuggetindex.core.models import Nugget
from nuggetindex.store.base import ViewMode

_R = TypeVar("_R")

# Allowlist for extra_filters in afilter() — prevents SQL-identifier injection.
# Every column here must exist on the nuggets table (see _SCHEMA below).
_ALLOWED_FILTER_COLUMNS: frozenset[str] = frozenset({
    "subject", "predicate", "object", "scope", "kind", "status", "rank", "parent_id",
})

_SCHEMA = """
CREATE TABLE IF NOT EXISTS nuggets (
    id TEXT PRIMARY KEY,
    key TEXT NOT NULL,
    kind TEXT NOT NULL,
    status TEXT NOT NULL,
    rank TEXT NOT NULL,
    subject TEXT NOT NULL,
    predicate TEXT NOT NULL,
    object TEXT NOT NULL,
    text TEXT NOT NULL,
    validity_start TEXT NOT NULL,
    validity_end TEXT,
    scope TEXT NOT NULL,
    confidence REAL NOT NULL,
    extraction_confidence REAL NOT NULL,
    parent_id TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    data TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_nuggets_key       ON nuggets(key);
CREATE INDEX IF NOT EXISTS idx_nuggets_validity  ON nuggets(validity_start, validity_end);
CREATE INDEX IF NOT EXISTS idx_nuggets_status    ON nuggets(status);

CREATE TABLE IF NOT EXISTS provenance (
    nugget_id TEXT NOT NULL,
    source_id TEXT NOT NULL,
    evidence_span TEXT NOT NULL,
    char_start INTEGER NOT NULL,
    char_end INTEGER NOT NULL,
    created_at TEXT NOT NULL,
    PRIMARY KEY (nugget_id, source_id, char_start),
    FOREIGN KEY (nugget_id) REFERENCES nuggets(id) ON DELETE CASCADE
);

CREATE VIRTUAL TABLE IF NOT EXISTS nuggets_fts USING fts5(
    text, subject, object,
    content='nuggets', content_rowid='rowid',
    tokenize='porter unicode61'
);

CREATE TRIGGER IF NOT EXISTS nuggets_ai AFTER INSERT ON nuggets BEGIN
    INSERT INTO nuggets_fts(rowid, text, subject, object)
    VALUES (new.rowid, new.text, new.subject, new.object);
END;
CREATE TRIGGER IF NOT EXISTS nuggets_ad AFTER DELETE ON nuggets BEGIN
    INSERT INTO nuggets_fts(nuggets_fts, rowid, text, subject, object)
    VALUES ('delete', old.rowid, old.text, old.subject, old.object);
END;
CREATE TRIGGER IF NOT EXISTS nuggets_au AFTER UPDATE ON nuggets BEGIN
    INSERT INTO nuggets_fts(nuggets_fts, rowid, text, subject, object)
    VALUES ('delete', old.rowid, old.text, old.subject, old.object);
    INSERT INTO nuggets_fts(rowid, text, subject, object)
    VALUES (new.rowid, new.text, new.subject, new.object);
END;

CREATE TABLE IF NOT EXISTS passages (
    source_id TEXT PRIMARY KEY,
    uri TEXT,
    text TEXT NOT NULL,
    retrieved_at TEXT NOT NULL,
    meta_json TEXT
);
"""


class _ConnectionPool:
    """Per-thread SQLite connection pool.

    Each executor thread that calls :meth:`get` lazily opens its own
    ``sqlite3.Connection`` with WAL + foreign-keys pragmas applied and a
    ``Row`` row factory. Connections are tracked in a module-local list so
    :meth:`close_all` can tear them all down at shutdown.

    The pool is read-only by design. Schema initialisation happens once on
    the writer connection (see :class:`SQLiteBackend`), not per pooled read
    connection.
    """

    def __init__(self, db_path: Path) -> None:
        self._db_path = db_path
        self._tls = threading.local()
        self._all_conns: list[sqlite3.Connection] = []
        self._lock = threading.Lock()

    def get(self) -> sqlite3.Connection:
        conn = getattr(self._tls, "conn", None)
        if conn is None:
            conn = sqlite3.connect(self._db_path, isolation_level=None)
            conn.execute("PRAGMA journal_mode = WAL")
            conn.execute("PRAGMA foreign_keys = ON")
            conn.row_factory = sqlite3.Row
            self._tls.conn = conn
            with self._lock:
                self._all_conns.append(conn)
        return conn

    def close_all(self) -> None:
        with self._lock:
            for c in self._all_conns:
                # Already closed on its owning thread -> ignore.
                with suppress(sqlite3.ProgrammingError):
                    c.close()
            self._all_conns.clear()


# Writer queue item: a callable taking the writer connection plus a future the
# writer loop resolves with the callable's result (or exception).
_WriteFn = Callable[[sqlite3.Connection], Any]
_WriteItem = tuple[_WriteFn, "asyncio.Future[Any]"] | None


class SQLiteBackend:
    """SQLite-backed metadata + sparse store.

    See the module docstring for the concurrency model. Public API is
    unchanged from v0.1; the per-thread pool + writer-queue refactor is an
    internal implementation detail.
    """

    def __init__(self, db_path: Path | str) -> None:
        self.db_path = Path(db_path)
        self._pool = _ConnectionPool(self.db_path)
        # The writer connection is opened once and owned exclusively by the
        # writer task (once it's spawned). Schema initialisation runs here
        # so pooled read connections never need to re-run migrations.
        self._writer_conn = self._open_writer_connection()
        self._writer_conn.executescript(_SCHEMA)
        # Additive migration for pre-0.2 DBs that pre-date the
        # ``passages.meta_json`` column. CREATE TABLE IF NOT EXISTS won't add
        # the column to an existing table, so probe via PRAGMA table_info and
        # ALTER TABLE when missing. Safe on fresh DBs too: the column is
        # already there from the CREATE, and the ``if ... not in cols`` guard
        # short-circuits.
        cols = {
            r[1]
            for r in self._writer_conn.execute(
                "PRAGMA table_info(passages)"
            ).fetchall()
        }
        if "meta_json" not in cols:
            self._writer_conn.execute(
                "ALTER TABLE passages ADD COLUMN meta_json TEXT"
            )
        self._writer_queue: asyncio.Queue[_WriteItem] | None = None
        self._writer_task: asyncio.Task[None] | None = None
        # The event loop that owns the current writer task/queue. Sync
        # wrappers go through ``asyncio.run`` which spins up a *new* loop for
        # each call, so we must detect loop-changes and rebind the writer
        # onto the current loop. ``None`` means "no writer yet".
        self._writer_loop: asyncio.AbstractEventLoop | None = None
        self._closed = False

    # ------------------------------------------------------------------
    # Connection management
    # ------------------------------------------------------------------

    def _open_writer_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, isolation_level=None)
        conn.execute("PRAGMA journal_mode = WAL")
        conn.execute("PRAGMA foreign_keys = ON")
        conn.row_factory = sqlite3.Row
        return conn

    @contextmanager
    def _tx(self, conn: sqlite3.Connection) -> Iterator[sqlite3.Connection]:
        """Run a write in an explicit transaction against ``conn``.

        Used by the writer task against its private ``self._writer_conn``; no
        other coroutine / thread touches that connection, so we don't need
        any additional locking.
        """
        conn.execute("BEGIN")
        try:
            yield conn
            conn.execute("COMMIT")
        except Exception:
            conn.execute("ROLLBACK")
            raise

    async def _ensure_writer(self) -> None:
        """Lazily spawn (or re-spawn) the writer task on the running loop.

        Spawning lazily means ``__init__`` stays synchronous (no running event
        loop needed), and a read-only ``SQLiteBackend`` never pays the cost of
        a dangling writer task.

        When the current running loop differs from the one that owns the
        existing writer (the usual case under sync wrappers, which call
        ``asyncio.run`` per top-level op), we abandon the old writer/queue
        and start fresh. Any old task is already unreachable because its loop
        has been closed — there's nothing to cancel or drain.
        """
        current_loop = asyncio.get_running_loop()
        if (
            self._writer_task is not None
            and not self._writer_task.done()
            and self._writer_loop is current_loop
        ):
            return
        self._writer_queue = asyncio.Queue()
        self._writer_loop = current_loop
        self._writer_task = current_loop.create_task(self._writer_loop_coro())

    async def _writer_loop_coro(self) -> None:
        """Consume one write at a time from the queue.

        Exits cleanly when it sees the ``None`` poison pill. Exceptions from
        the submitted callable are forwarded to the caller via the per-op
        future rather than killing the loop.
        """
        assert self._writer_queue is not None
        while True:
            item = await self._writer_queue.get()
            try:
                if item is None:
                    return
                fn, fut = item
                if fut.cancelled():
                    continue
                try:
                    result = fn(self._writer_conn)
                except BaseException as e:  # noqa: BLE001 — forward to caller
                    if not fut.done():
                        fut.set_exception(e)
                else:
                    if not fut.done():
                        fut.set_result(result)
            finally:
                self._writer_queue.task_done()

    async def _submit_write(self, fn: _WriteFn) -> Any:
        """Enqueue ``fn(writer_conn)`` onto the writer task.

        Returns once the writer has COMMITted the transaction (read-your-own-
        write). If ``fn`` raises, the exception propagates to the caller.
        """
        if self._closed:
            raise RuntimeError("SQLiteBackend is closed")
        await self._ensure_writer()
        assert self._writer_queue is not None
        fut: asyncio.Future[Any] = asyncio.get_event_loop().create_future()
        await self._writer_queue.put((fn, fut))
        return await fut

    async def _run_read(self, fn: Callable[[sqlite3.Connection], _R]) -> _R:
        """Execute ``fn(pool_conn)`` in the default executor.

        The pool's ``get()`` resolves the connection on the executor thread so
        each thread sees its own ``sqlite3.Connection``.
        """
        loop = asyncio.get_event_loop()

        def _runner() -> _R:
            return fn(self._pool.get())

        return await loop.run_in_executor(None, _runner)

    # ------------------------------------------------------------------
    # CRUD — writes go through _submit_write, reads through _run_read
    # ------------------------------------------------------------------

    async def aupsert(self, nugget: Nugget) -> None:
        await self._submit_write(lambda conn: self._upsert_sync(conn, nugget))

    def _upsert_sync(self, conn: sqlite3.Connection, nugget: Nugget) -> None:
        data_json = nugget.model_dump_json()
        key = (
            f"{nugget.fact.subject}|{nugget.fact.predicate}|{nugget.validity.scope}"
        )
        with self._tx(conn):
            conn.execute(
                """INSERT OR REPLACE INTO nuggets
                   (id, key, kind, status, rank, subject, predicate, object, text,
                    validity_start, validity_end, scope, confidence, extraction_confidence,
                    parent_id, created_at, updated_at, data)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    nugget.id,
                    key,
                    str(nugget.kind),
                    str(nugget.epistemic.status),
                    str(nugget.epistemic.rank),
                    nugget.fact.subject,
                    nugget.fact.predicate,
                    nugget.fact.object,
                    nugget.fact.text,
                    nugget.validity.start.isoformat(),
                    nugget.validity.end.isoformat() if nugget.validity.end else None,
                    nugget.validity.scope,
                    nugget.epistemic.confidence,
                    nugget.extraction_confidence,
                    nugget.parent_id,
                    nugget.created_at.isoformat(),
                    nugget.updated_at.isoformat(),
                    data_json,
                ),
            )
            # Replace provenance: delete + insert fresh
            conn.execute("DELETE FROM provenance WHERE nugget_id = ?", (nugget.id,))
            for p in nugget.provenance:
                conn.execute(
                    """INSERT OR IGNORE INTO provenance
                       (nugget_id, source_id, evidence_span, char_start, char_end, created_at)
                       VALUES (?,?,?,?,?,?)""",
                    (
                        nugget.id,
                        p.source_id,
                        p.evidence_span,
                        p.char_start,
                        p.char_end,
                        p.created_at.isoformat(),
                    ),
                )

    async def aget(self, nugget_id: str) -> Nugget | None:
        return await self._run_read(lambda conn: self._get_sync(conn, nugget_id))

    def _get_sync(
        self, conn: sqlite3.Connection, nugget_id: str
    ) -> Nugget | None:
        row = conn.execute(
            "SELECT data FROM nuggets WHERE id = ?", (nugget_id,)
        ).fetchone()
        if row is None:
            return None
        return Nugget.model_validate_json(row[0])

    async def afind_by_key(self, key: tuple[str, str, str]) -> list[Nugget]:
        return await self._run_read(
            lambda conn: self._find_by_key_sync(conn, key)
        )

    def _find_by_key_sync(
        self, conn: sqlite3.Connection, key: tuple[str, str, str]
    ) -> list[Nugget]:
        key_str = "|".join(key)
        rows = conn.execute(
            "SELECT data FROM nuggets WHERE key = ?", (key_str,)
        ).fetchall()
        return [Nugget.model_validate_json(r[0]) for r in rows]

    async def aget_nuggets_by_source(self, source_id: str) -> list[Nugget]:
        """Return all nuggets whose provenance includes ``source_id``.

        Used by the Tier-1 governance postprocessor to look up which nuggets
        were sourced from a given retrieved passage so it can decide whether
        to filter (all DEPRECATED) or flag (any CONTESTED) that passage.
        """
        return await self._run_read(
            lambda conn: self._get_nuggets_by_source_sync(conn, source_id)
        )

    def _get_nuggets_by_source_sync(
        self, conn: sqlite3.Connection, source_id: str
    ) -> list[Nugget]:
        rows = conn.execute(
            """SELECT DISTINCT n.data FROM nuggets n
               JOIN provenance p ON n.id = p.nugget_id
               WHERE p.source_id = ?""",
            (source_id,),
        ).fetchall()
        return [Nugget.model_validate_json(r[0]) for r in rows]

    async def acount(self, status: LifecycleStatus | None = None) -> int:
        return await self._run_read(lambda conn: self._count_sync(conn, status))

    def _count_sync(
        self, conn: sqlite3.Connection, status: LifecycleStatus | None
    ) -> int:
        if status is None:
            row = conn.execute("SELECT COUNT(*) FROM nuggets").fetchone()
            return int(row[0])
        row = conn.execute(
            "SELECT COUNT(*) FROM nuggets WHERE status = ?", (str(status),)
        ).fetchone()
        return int(row[0])

    # --- Filter ---

    async def afilter(
        self,
        *,
        query_time: datetime,
        view: ViewMode,
        extra_filters: dict[str, Any] | None = None,
    ) -> list[str]:
        return await self._run_read(
            lambda conn: self._filter_sync(conn, query_time, view, extra_filters)
        )

    def _filter_sync(
        self,
        conn: sqlite3.Connection,
        query_time: datetime,
        view: ViewMode,
        extra_filters: dict[str, Any] | None,
    ) -> list[str]:
        qt = query_time.isoformat()
        status_clause = {
            "active": "status = 'active'",
            "active_contested": "status IN ('active', 'contested')",
            "all": "1=1",
        }[view]
        sql = f"""
            SELECT id FROM nuggets
            WHERE validity_start <= ?
              AND (validity_end IS NULL OR validity_end > ?)
              AND {status_clause}
        """
        params: list[Any] = [qt, qt]
        if extra_filters:
            for col, val in extra_filters.items():
                if col not in _ALLOWED_FILTER_COLUMNS:
                    raise ValueError(
                        f"unknown filter column: {col!r} "
                        f"(allowed: {sorted(_ALLOWED_FILTER_COLUMNS)})"
                    )
                sql += f" AND {col} = ?"
                params.append(val)
        rows = conn.execute(sql, params).fetchall()
        return [r[0] for r in rows]

    # --- Chain helpers ---

    async def asuccession_for_key(
        self,
        key: str,
        as_of: datetime | None,
        statuses: list[str],
        limit: int,
    ) -> list[Nugget]:
        """Return nuggets sharing ``key`` ordered by ``validity_start``.

        ``statuses`` is the allow-list for the ``nuggets.status`` column
        (e.g. ``["active", "deprecated"]``). ``as_of``, when provided,
        restricts the result to nuggets whose ``validity_start <= as_of``.

        Used by :meth:`NuggetStore.achain_succession`.
        """
        return await self._run_read(
            lambda conn: self._succession_for_key_sync(
                conn, key, as_of, statuses, limit
            )
        )

    def _succession_for_key_sync(
        self,
        conn: sqlite3.Connection,
        key: str,
        as_of: datetime | None,
        statuses: list[str],
        limit: int,
    ) -> list[Nugget]:
        if not statuses:
            return []
        placeholders = ",".join("?" * len(statuses))
        as_of_iso = as_of.isoformat() if as_of is not None else None
        sql = (
            f"SELECT data FROM nuggets "
            f"WHERE key = ? "
            f"  AND (? IS NULL OR validity_start <= ?) "
            f"  AND status IN ({placeholders}) "
            f"ORDER BY validity_start ASC "
            f"LIMIT ?"
        )
        params: list[Any] = [key, as_of_iso, as_of_iso, *statuses, limit]
        rows = conn.execute(sql, params).fetchall()
        return [Nugget.model_validate_json(r[0]) for r in rows]

    async def arename_candidates(
        self,
        *,
        subject: str,
        as_of: datetime | None,
        renaming_predicates: frozenset[str],
        direction: str = "forward",
        include_contested: bool = False,
        limit: int = 3,
    ) -> list[Nugget]:
        """Return up to ``limit`` candidates for a single rename step.

        ``direction="forward"`` queries by ``subject = ?`` (the renaming
        originates *at* ``subject``). ``direction="backward"`` queries by
        ``object = ?`` (something renames *into* ``subject``).

        Only nuggets whose canonical predicate is in ``renaming_predicates``
        are returned, ordered by ``validity_start ASC``. Any second hit
        short-circuits (``limit=3``) so the caller can detect ambiguity
        without scanning the whole table.
        """
        return await self._run_read(
            lambda conn: self._rename_candidates_sync(
                conn,
                subject=subject,
                as_of=as_of,
                renaming_predicates=renaming_predicates,
                direction=direction,
                include_contested=include_contested,
                limit=limit,
            )
        )

    def _rename_candidates_sync(
        self,
        conn: sqlite3.Connection,
        *,
        subject: str,
        as_of: datetime | None,
        renaming_predicates: frozenset[str],
        direction: str,
        include_contested: bool,
        limit: int,
    ) -> list[Nugget]:
        if not renaming_predicates:
            return []
        preds = sorted(renaming_predicates)
        pred_placeholders = ",".join("?" * len(preds))
        column = "subject" if direction == "forward" else "object"
        statuses = ["active", "deprecated"]
        if include_contested:
            statuses.append("contested")
        status_placeholders = ",".join("?" * len(statuses))
        as_of_iso = as_of.isoformat() if as_of is not None else None
        sql = (
            f"SELECT data FROM nuggets "
            f"WHERE {column} = ? "
            f"  AND predicate IN ({pred_placeholders}) "
            f"  AND (? IS NULL OR validity_start <= ?) "
            f"  AND status IN ({status_placeholders}) "
            f"ORDER BY validity_start ASC "
            f"LIMIT ?"
        )
        params: list[Any] = [subject, *preds, as_of_iso, as_of_iso, *statuses, limit]
        rows = conn.execute(sql, params).fetchall()
        return [Nugget.model_validate_json(r[0]) for r in rows]

    # --- Candidate-key discovery (v0.2.1) ---

    async def acandidate_keys(
        self,
        *,
        subject_contains: str | None = None,
        predicate_contains: str | None = None,
        scope: str = "global",
        limit: int = 20,
    ) -> list[tuple[str, str, str]]:
        """Return distinct ``(subject, predicate, scope)`` triples matching filters.

        Case-insensitive substring match on ``subject`` and/or ``predicate``.
        Both filters are optional: when ``None``, that dimension is not
        restricted. Designed for the CLI ``--discover`` flag and interactive
        exploration of what keys a store actually contains.
        """
        return await self._run_read(
            lambda conn: self._candidate_keys_sync(
                conn,
                subject_contains=subject_contains,
                predicate_contains=predicate_contains,
                scope=scope,
                limit=limit,
            )
        )

    def _candidate_keys_sync(
        self,
        conn: sqlite3.Connection,
        *,
        subject_contains: str | None,
        predicate_contains: str | None,
        scope: str,
        limit: int,
    ) -> list[tuple[str, str, str]]:
        sql = (
            "SELECT DISTINCT subject, predicate, scope FROM nuggets "
            "WHERE scope = ? "
            "  AND (? IS NULL OR subject LIKE '%' || ? || '%' COLLATE NOCASE) "
            "  AND (? IS NULL OR predicate LIKE '%' || ? || '%' COLLATE NOCASE) "
            "LIMIT ?"
        )
        params = (
            scope,
            subject_contains,
            subject_contains,
            predicate_contains,
            predicate_contains,
            limit,
        )
        rows = conn.execute(sql, params).fetchall()
        return [(r[0], r[1], r[2]) for r in rows]

    async def acontested_keys(self) -> list[tuple[str, str, str, int]]:
        """Return ``(subject, predicate, scope, n_contested)`` rows for every
        key that has at least one ``CONTESTED`` member, ordered by
        descending number of contested nuggets.

        Used by ``nuggetindex resolve`` (Phase 0.5) to walk the human-
        adjudication queue. Cheap to compute: a single grouped SELECT.
        """
        return await self._run_read(self._contested_keys_sync)

    def _contested_keys_sync(
        self, conn: sqlite3.Connection
    ) -> list[tuple[str, str, str, int]]:
        rows = conn.execute(
            "SELECT subject, predicate, scope, COUNT(*) AS n "
            "FROM nuggets "
            "WHERE status = 'contested' "
            "GROUP BY subject, predicate, scope "
            "ORDER BY n DESC, subject ASC, predicate ASC"
        ).fetchall()
        return [(r[0], r[1], r[2], r[3]) for r in rows]

    async def adistinct_entities(self) -> list[str]:
        """Return the de-duplicated union of every distinct subject + object
        string currently stored in the ``nuggets`` table.

        Used by :meth:`nuggetindex.store.base.NuggetStore._ensure_alias_resolver`
        to seed the store-scoped alias pool on first ingest so cross-document
        alias merging works (e.g. "Microsoft" from doc A collapses
        "Microsoft Corporation" from doc B).
        """
        return await self._run_read(self._distinct_entities_sync)

    def _distinct_entities_sync(self, conn: sqlite3.Connection) -> list[str]:
        rows = conn.execute(
            "SELECT subject FROM nuggets "
            "UNION "
            "SELECT object FROM nuggets"
        ).fetchall()
        # ``fetchall`` gives us one-element tuples; flatten and drop any
        # empty strings (defensive — the FactTriple min_length=1 guard
        # should prevent those from ever landing in the DB).
        return [r[0] for r in rows if r[0]]

    # --- BM25 search ---

    async def abm25_search(
        self,
        query: str,
        *,
        candidate_ids: list[str] | None = None,
        top_k: int = 20,
    ) -> list[tuple[str, float]]:
        return await self._run_read(
            lambda conn: self._bm25_sync(conn, query, candidate_ids, top_k)
        )

    def _bm25_sync(
        self,
        conn: sqlite3.Connection,
        query: str,
        candidate_ids: list[str] | None,
        top_k: int,
    ) -> list[tuple[str, float]]:
        # Sanitize query for FTS5: escape double quotes then wrap as phrase
        safe = query.replace('"', '""')
        if candidate_ids:
            placeholders = ",".join("?" * len(candidate_ids))
            sql = f"""
                SELECT n.id AS id, bm25(nuggets_fts) AS score
                FROM nuggets_fts f JOIN nuggets n ON f.rowid = n.rowid
                WHERE nuggets_fts MATCH ?
                  AND n.id IN ({placeholders})
                ORDER BY score LIMIT ?
            """
            params: list[Any] = [safe, *candidate_ids, top_k]
        else:
            sql = """
                SELECT n.id AS id, bm25(nuggets_fts) AS score
                FROM nuggets_fts f JOIN nuggets n ON f.rowid = n.rowid
                WHERE nuggets_fts MATCH ?
                ORDER BY score LIMIT ?
            """
            params = [safe, top_k]
        rows = conn.execute(sql, params).fetchall()
        # bm25() returns negative-good scores; normalize to positive: -score
        return [(r["id"], -r["score"]) for r in rows]

    # --- Passages ---

    async def aupsert_passage(
        self, source_id: str, uri: str | None, text: str
    ) -> None:
        await self._submit_write(
            lambda conn: self._upsert_passage_sync(conn, source_id, uri, text)
        )

    def _upsert_passage_sync(
        self,
        conn: sqlite3.Connection,
        source_id: str,
        uri: str | None,
        text: str,
    ) -> None:
        with self._tx(conn):
            conn.execute(
                """INSERT OR REPLACE INTO passages (source_id, uri, text, retrieved_at)
                   VALUES (?,?,?,?)""",
                (source_id, uri, text, datetime.now(UTC).isoformat()),
            )

    async def aget_passages(self, source_ids: Iterable[str]) -> dict[str, str]:
        ids = list(source_ids)
        return await self._run_read(lambda conn: self._get_passages_sync(conn, ids))

    def _get_passages_sync(
        self, conn: sqlite3.Connection, source_ids: list[str]
    ) -> dict[str, str]:
        if not source_ids:
            return {}
        placeholders = ",".join("?" * len(source_ids))
        rows = conn.execute(
            f"SELECT source_id, text FROM passages WHERE source_id IN ({placeholders})",
            source_ids,
        ).fetchall()
        return {r["source_id"]: r["text"] for r in rows}

    # --- Passage bulk ops (used by Haystack DocumentStore glue) ---

    async def acount_passages(self) -> int:
        """Return the total number of rows in the passages table.

        Distinct from :meth:`acount` (which counts nuggets). Used by
        ``NuggetDocumentStore.count_documents``.
        """
        return await self._run_read(self._count_passages_sync)

    def _count_passages_sync(self, conn: sqlite3.Connection) -> int:
        row = conn.execute("SELECT COUNT(*) FROM passages").fetchone()
        return int(row[0])

    async def alist_source_ids(self) -> list[str]:
        """Return every ``source_id`` currently stored in ``passages``.

        Order is unspecified. Used by ``NuggetDocumentStore.filter_documents``
        when asked for the full corpus (``filters=None``).
        """
        return await self._run_read(self._list_source_ids_sync)

    def _list_source_ids_sync(self, conn: sqlite3.Connection) -> list[str]:
        rows = conn.execute("SELECT source_id FROM passages").fetchall()
        return [r["source_id"] for r in rows]

    async def aget_passage_records(
        self, source_ids: Iterable[str]
    ) -> dict[str, tuple[str, str | None]]:
        """Return ``{source_id: (text, meta_json)}`` for each given id.

        ``meta_json`` is the raw JSON string previously stored via
        :meth:`aupsert_passage` (or ``None`` if none was stored). Useful for
        reconstructing full Haystack ``Document`` objects in the
        ``NuggetDocumentStore`` glue.
        """
        ids = list(source_ids)
        return await self._run_read(
            lambda conn: self._get_passage_records_sync(conn, ids)
        )

    def _get_passage_records_sync(
        self, conn: sqlite3.Connection, source_ids: list[str]
    ) -> dict[str, tuple[str, str | None]]:
        if not source_ids:
            return {}
        placeholders = ",".join("?" * len(source_ids))
        rows = conn.execute(
            f"SELECT source_id, text, meta_json FROM passages "
            f"WHERE source_id IN ({placeholders})",
            source_ids,
        ).fetchall()
        return {r["source_id"]: (r["text"], r["meta_json"]) for r in rows}

    async def adelete_by_source_ids(self, ids: list[str]) -> None:
        """Delete passages + their derived nuggets for the given source ids.

        Cascade semantics: any nugget whose provenance references one of
        ``ids`` is dropped along with its provenance rows (via the
        ``ON DELETE CASCADE`` foreign key from ``provenance`` to ``nuggets``).
        A passage with no extracted nuggets (the Haystack ``write_documents``
        path) simply drops one ``passages`` row.

        Runs through the writer queue so subsequent reads see the result
        (read-your-own-write). Non-existing ids are silently ignored.
        """
        if not ids:
            return
        await self._submit_write(
            lambda conn: self._delete_by_source_ids_sync(conn, ids)
        )

    def _delete_by_source_ids_sync(
        self, conn: sqlite3.Connection, source_ids: list[str]
    ) -> None:
        if not source_ids:
            return
        placeholders = ",".join("?" * len(source_ids))
        with self._tx(conn):
            # 1. Drop any nuggets whose provenance points at these sources.
            #    Pulling the ids explicitly (vs a single DELETE with
            #    subquery) keeps the provenance CASCADE paths predictable and
            #    lets us audit the deletion in tests if needed.
            nugget_rows = conn.execute(
                f"SELECT DISTINCT nugget_id FROM provenance "
                f"WHERE source_id IN ({placeholders})",
                source_ids,
            ).fetchall()
            if nugget_rows:
                nids = [r[0] for r in nugget_rows]
                np = ",".join("?" * len(nids))
                conn.execute(
                    f"DELETE FROM nuggets WHERE id IN ({np})", nids
                )
            # 2. Drop passages themselves (independent of nugget presence).
            conn.execute(
                f"DELETE FROM passages WHERE source_id IN ({placeholders})",
                source_ids,
            )

    async def aupsert_passage_with_meta(
        self,
        source_id: str,
        uri: str | None,
        text: str,
        meta_json: str | None,
    ) -> None:
        """Upsert a passage row with an attached JSON blob.

        Companion to :meth:`aupsert_passage` used by the Haystack glue to
        round-trip ``Document.to_dict()``. Kept as a separate method rather
        than overloading ``aupsert_passage`` so existing callers (the main
        ingest path) are unaffected.
        """
        await self._submit_write(
            lambda conn: self._upsert_passage_with_meta_sync(
                conn, source_id, uri, text, meta_json
            )
        )

    def _upsert_passage_with_meta_sync(
        self,
        conn: sqlite3.Connection,
        source_id: str,
        uri: str | None,
        text: str,
        meta_json: str | None,
    ) -> None:
        with self._tx(conn):
            conn.execute(
                """INSERT OR REPLACE INTO passages
                   (source_id, uri, text, retrieved_at, meta_json)
                   VALUES (?,?,?,?,?)""",
                (
                    source_id,
                    uri,
                    text,
                    datetime.now(UTC).isoformat(),
                    meta_json,
                ),
            )

    async def apassage_exists(self, source_id: str) -> bool:
        """Return ``True`` iff a passage row with ``source_id`` exists."""
        return await self._run_read(
            lambda conn: self._passage_exists_sync(conn, source_id)
        )

    def _passage_exists_sync(
        self, conn: sqlite3.Connection, source_id: str
    ) -> bool:
        row = conn.execute(
            "SELECT 1 FROM passages WHERE source_id = ? LIMIT 1", (source_id,)
        ).fetchone()
        return row is not None

    # --- Lifecycle ---

    async def aclose(self) -> None:
        """Drain the writer queue, stop the writer task, close all connections.

        Idempotent — subsequent calls are no-ops.
        """
        if self._closed:
            return
        self._closed = True
        if self._writer_task is not None:
            # Only drain the writer when we're on its owning loop. Under sync
            # wrappers (``store.close()`` → ``asyncio.run``) the writer may
            # have been spawned on an earlier, now-closed loop; there's
            # nothing to drain in that case — stranded tasks are cleaned up
            # with their loop.
            current_loop: asyncio.AbstractEventLoop | None
            try:
                current_loop = asyncio.get_running_loop()
            except RuntimeError:
                current_loop = None
            if (
                self._writer_loop is current_loop
                and self._writer_queue is not None
                and not self._writer_task.done()
            ):
                # Poison pill: any items already enqueued ahead of this one
                # will be processed first, which guarantees pending writes
                # land on disk before we tear the connection down.
                await self._writer_queue.put(None)
                await self._writer_task
            self._writer_task = None
            self._writer_queue = None
            self._writer_loop = None
        with suppress(sqlite3.ProgrammingError):
            self._writer_conn.close()
        self._pool.close_all()
