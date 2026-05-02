"""Per-passage extraction cache for the sidecar's just-in-time mode.

For a live retriever that returns overlapping passage sets across similar
queries (``"who was Google's CEO in 2013?"`` and ``"who ran Google in 2013?"``
hit the same top passages), re-running the extractor on every passage for
every query is pure waste. :class:`JITPassageCache` memoizes extractions at
the passage granularity, keyed by ``SHA-256(text)``, so repeated passages
return cached results in ~1 ms instead of paying the extractor's latency.

Relationship to :class:`CachedExtractor`
----------------------------------------

* :class:`CachedExtractor` lives at the *extractor* level. Its key is
  ``SHA-256(text + '|' + extractor_id)``. It's meant for ingest pipelines
  where identical text flows through the same extractor repeatedly.
* :class:`JITPassageCache` lives at the *sidecar* level. Its key is
  ``SHA-256(text)`` alone; the sidecar already fixes the extractor. It's
  meant for query-time workloads where overlapping top-K passages repeat
  across semantically similar queries.

Both caches can coexist — if the sidecar wraps a :class:`CachedExtractor`
and also has a :class:`JITPassageCache`, the JIT cache short-circuits first
(no extractor call at all), and only genuine misses fall through to the
inner cache which may then serve its own hit.

In-memory + optional on-disk persistence
----------------------------------------

The default is a pure-in-memory LRU with ``max_entries=10_000`` slots. Pass
``cache_path=`` to persist entries to a SQLite file for cross-process reuse
— the schema is identical in shape to :class:`CachedExtractor`'s cache
file so operators can spot-check it with ``sqlite3`` directly.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import sqlite3
from collections import OrderedDict
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover -- type-check-only imports
    from nuggetindex.extractors.base import BaseExtractor, ExtractionResult


_EXPECTED_COLUMNS: tuple[str, ...] = (
    "content_hash",
    "results_json",
    "created_at",
)


def _hash_text(text: str) -> str:
    """SHA-256 hex digest of ``text`` (UTF-8 encoded)."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class JITPassageCache:
    """LRU cache of :class:`ExtractionResult` lists keyed by passage content.

    The in-memory bucket is an :class:`collections.OrderedDict` used as an
    LRU: ``move_to_end`` on access, ``popitem(last=False)`` when full. The
    optional on-disk bucket is a small SQLite table (``jit_passage_cache``).

    The cache's contract with callers: :meth:`get_or_extract` returns a
    list of :class:`ExtractionResult` for the given ``text``, either from
    cache or by awaiting the supplied extractor. Both hit and miss paths
    update the LRU ordering so the hottest entries stay resident.
    """

    def __init__(
        self,
        *,
        max_entries: int = 10_000,
        cache_path: Path | str | None = None,
    ) -> None:
        if max_entries <= 0:
            raise ValueError(
                f"max_entries must be positive, got {max_entries!r}"
            )
        self._max_entries = int(max_entries)
        self._cache_path = Path(cache_path) if cache_path is not None else None
        # Ordered by recency — oldest (least recently used) first.
        self._memory: OrderedDict[str, list[ExtractionResult]] = OrderedDict()
        self._hits = 0
        self._misses = 0
        self._conn: sqlite3.Connection | None = None
        if self._cache_path is not None:
            self._conn = _connect(self._cache_path)
            # Warm the in-memory LRU with any rows already on disk so the
            # first call after reopening returns without a SQLite roundtrip.
            self._warm_from_disk()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def get_or_extract(
        self,
        text: str,
        extractor: BaseExtractor,
        *,
        source_id: str = "",
    ) -> list[ExtractionResult]:
        """Return cached extraction results or compute + cache them.

        Empty / whitespace-only text bypasses the cache and returns an empty
        list so callers don't waste a cache slot on useless hashes.
        """
        if not text or not text.strip():
            return []

        key = _hash_text(text)
        cached = self._get(key)
        if cached is not None:
            self._hits += 1
            return cached

        self._misses += 1
        kwargs: dict[str, Any] = {}
        if source_id:
            kwargs["source_id"] = source_id
        try:
            results = await extractor.aextract(text, **kwargs)
        except TypeError:
            # Pre-0.2 extractors without the ``source_id`` kwarg.
            kwargs.pop("source_id", None)
            results = await extractor.aextract(text, **kwargs)

        self._put(key, list(results))
        return results

    def stats(self) -> dict[str, int]:
        """Return ``{"hits": h, "misses": m, "total": h+m, "size": s}``."""
        return {
            "hits": self._hits,
            "misses": self._misses,
            "total": self._hits + self._misses,
            "size": len(self._memory),
        }

    def close(self) -> None:
        """Close the on-disk connection (if any). Idempotent."""
        if self._conn is not None:
            try:
                self._conn.close()
            finally:
                self._conn = None

    def __del__(self) -> None:  # pragma: no cover -- best-effort finaliser
        with contextlib.suppress(Exception):
            self.close()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _get(self, key: str) -> list[ExtractionResult] | None:
        """Return the LRU entry for ``key``, refreshing its recency."""
        if key in self._memory:
            self._memory.move_to_end(key)
            return self._memory[key]
        if self._conn is None:
            return None
        row = self._conn.execute(
            "SELECT results_json FROM jit_passage_cache WHERE content_hash = ?",
            (key,),
        ).fetchone()
        if row is None:
            return None
        results = _deserialize_results(row[0])
        # Warm-populate the in-memory LRU on a disk hit so repeat calls
        # from the same process don't re-decode the JSON.
        self._put(key, results, write_disk=False)
        return results

    def _put(
        self,
        key: str,
        value: list[ExtractionResult],
        *,
        write_disk: bool = True,
    ) -> None:
        """Insert or refresh ``key``; evict LRU tail when oversize."""
        if key in self._memory:
            self._memory.move_to_end(key)
            self._memory[key] = value
        else:
            self._memory[key] = value
            while len(self._memory) > self._max_entries:
                self._memory.popitem(last=False)
        if write_disk and self._conn is not None:
            payload = _serialize_results(value)
            self._conn.execute(
                "INSERT OR REPLACE INTO jit_passage_cache "
                "(content_hash, results_json, created_at) "
                "VALUES (?, ?, datetime('now'))",
                (key, payload),
            )

    def _warm_from_disk(self) -> None:
        """Best-effort warm-load up to ``max_entries`` rows, newest first."""
        if self._conn is None:
            return
        try:
            rows = self._conn.execute(
                "SELECT content_hash, results_json FROM jit_passage_cache "
                "ORDER BY created_at DESC LIMIT ?",
                (self._max_entries,),
            ).fetchall()
        except sqlite3.DatabaseError:
            return
        # Insert in reverse so the newest rows end up at the "most recent" tail
        # of the OrderedDict after the oldest are inserted first.
        for content_hash, results_json in reversed(rows):
            try:
                results = _deserialize_results(results_json)
            except Exception:  # pragma: no cover -- defensive on corrupt blobs
                continue
            self._memory[content_hash] = results


# ---------------------------------------------------------------------------
# Serialisation helpers (mirror CachedExtractor's shape, minus extractor_id)
# ---------------------------------------------------------------------------


def _serialize_results(results: list[ExtractionResult]) -> str:
    """Pydantic-JSON dump of a list of :class:`ExtractionResult`."""
    dumped = [r.model_dump(mode="json") for r in results]
    return json.dumps(dumped, sort_keys=False)


def _deserialize_results(payload: str) -> list[ExtractionResult]:
    """Reconstruct :class:`ExtractionResult` instances from JSON."""
    from nuggetindex.extractors.base import ExtractionResult

    raw = json.loads(payload)
    if not isinstance(raw, list):
        raise RuntimeError(
            "jit_passage_cache payload was not a JSON array; the cache file "
            "is corrupted — delete it and re-run."
        )
    return [ExtractionResult.model_validate(item) for item in raw]


def _connect(cache_path: Path) -> sqlite3.Connection:
    """Open (or create) the SQLite cache file and validate / create the table."""
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(cache_path), isolation_level=None)
    with contextlib.suppress(sqlite3.DatabaseError):  # pragma: no cover
        conn.execute("PRAGMA journal_mode=WAL")

    existing = conn.execute(
        "SELECT name FROM sqlite_master "
        "WHERE type='table' AND name='jit_passage_cache'"
    ).fetchone()
    if existing is None:
        conn.execute(
            "CREATE TABLE jit_passage_cache ("
            "content_hash TEXT PRIMARY KEY,"
            "results_json TEXT NOT NULL,"
            "created_at TEXT NOT NULL"
            ")"
        )
        return conn

    cols = {row[1] for row in conn.execute("PRAGMA table_info(jit_passage_cache)")}
    missing = [c for c in _EXPECTED_COLUMNS if c not in cols]
    if missing:
        conn.close()
        raise RuntimeError(
            "jit_passage_cache schema mismatch at "
            f"{cache_path}: missing column(s) {missing}. "
            "Delete the cache file and re-run (no migration path in this release)."
        )
    return conn


__all__ = ["JITPassageCache"]
