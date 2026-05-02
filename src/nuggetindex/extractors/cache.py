"""Content-hash memoizing wrapper around any :class:`BaseExtractor`.

For re-ingests, a cache hit costs $0 + ~1 ms instead of $0.0005 + ~2 s.

Design
------

* Cache lives in a single SQLite file at ``cache_path``. One table,
  ``extractor_cache``, keyed by a SHA-256 of ``text + '|' + extractor_id``.
* The ``extractor_id`` is auto-inferred from the wrapped extractor's type
  so upgrading the LLM model / bumping ``PROMPT_VERSION`` invalidates the
  cache without an explicit migration.
* ``context`` is folded into ``extractor_id`` (NOT into the text hash)
  because a changed context implies a whole different prompt surface;
  mixing it into the id means stale entries become unreachable and the
  cache never serves a result that disagrees with the current call.
* Schema is single-version. If the on-disk file already exists with a
  mismatching shape we raise :class:`RuntimeError` with a clear hint --
  users delete the cache file and re-run; no migrations in this release.

Concurrency: SQLite's default locking is sufficient for single-process
ingest. Multi-process ingest should wrap with an external lock (out of
scope for this release).
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from nuggetindex.extractors.base import BaseExtractor, ExtractionResult

_EXPECTED_COLUMNS: tuple[str, ...] = (
    "content_hash",
    "extractor_id",
    "results_json",
    "created_at",
)


class CachedExtractor(BaseExtractor):
    """Memoize another extractor's output by ``hash(text + extractor_id)``.

    The cache is a SQLite table at ``cache_path`` with schema::

        CREATE TABLE extractor_cache (
            content_hash TEXT PRIMARY KEY,
            extractor_id TEXT NOT NULL,
            results_json TEXT NOT NULL,
            created_at TEXT NOT NULL
        );

    Keys are ``sha256(text + '|' + extractor_id).hexdigest()``. Values are
    the inner extractor's ``ExtractionResult`` list serialised as a JSON
    array via ``pydantic.BaseModel.model_dump(mode='json')``. Hits are
    served in ~1 ms; misses delegate to the inner extractor and
    write-through on success.
    """

    # ``CachedExtractor`` forwards the inner extractor's placeholder-validity
    # posture so the pipeline's "missing source_date" warning fires exactly
    # as it would without the wrapper. Computed lazily in ``__init__``.
    emits_placeholder_validity: bool = False

    def __init__(
        self,
        inner: BaseExtractor,
        *,
        cache_path: Path | str = ".nuggetindex-extractor-cache.db",
        extractor_id: str | None = None,
    ) -> None:
        self._inner = inner
        self._cache_path = Path(cache_path)
        self._extractor_id_base = (
            extractor_id if extractor_id is not None else _infer_extractor_id(inner)
        )
        self._hits = 0
        self._misses = 0
        # Mirror the inner extractor's placeholder flag so the pipeline's
        # source-date warning stays intact when the inner extractor opts in.
        self.emits_placeholder_validity = bool(getattr(inner, "emits_placeholder_validity", False))
        self._conn = _connect(self._cache_path)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def cache_path(self) -> Path:
        return self._cache_path

    @property
    def extractor_id(self) -> str:
        """Base extractor-id (without the per-call context suffix)."""
        return self._extractor_id_base

    async def aextract(
        self,
        text: str,
        *,
        context: str = "",
        source_id: str = "",
    ) -> list[ExtractionResult]:
        if not text:
            return []
        effective_id = self._effective_extractor_id(context)
        key = _hash_key(text, effective_id)

        cached = self._read(key)
        if cached is not None:
            self._hits += 1
            return cached

        self._misses += 1
        kwargs: dict[str, Any] = {}
        if context:
            kwargs["context"] = context
        if source_id:
            kwargs["source_id"] = source_id
        try:
            results = await self._inner.aextract(text, **kwargs)
        except TypeError:
            # ``source_id`` support is optional (accepts_source_id()); retry
            # without it if the inner extractor predates 0.2's convention.
            kwargs.pop("source_id", None)
            results = await self._inner.aextract(text, **kwargs)

        self._write(key, effective_id, results)
        return results

    def stats(self) -> dict[str, int]:
        """Return ``{"hits": n, "misses": m, "total": n+m}``."""
        return {
            "hits": self._hits,
            "misses": self._misses,
            "total": self._hits + self._misses,
        }

    def close(self) -> None:
        """Close the underlying SQLite connection. Idempotent."""
        if self._conn is not None:
            try:
                self._conn.close()
            finally:
                self._conn = None  # type: ignore[assignment]

    def __del__(self) -> None:  # pragma: no cover -- best-effort finaliser
        with contextlib.suppress(Exception):
            self.close()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _effective_extractor_id(self, context: str) -> str:
        if not context:
            return self._extractor_id_base
        ctx_hash = hashlib.sha256(context.encode("utf-8")).hexdigest()[:16]
        return f"{self._extractor_id_base}|ctx={ctx_hash}"

    def _read(self, key: str) -> list[ExtractionResult] | None:
        row = self._conn.execute(
            "SELECT results_json FROM extractor_cache WHERE content_hash = ?",
            (key,),
        ).fetchone()
        if row is None:
            return None
        return _deserialize_results(row[0])

    def _write(self, key: str, extractor_id: str, results: list[ExtractionResult]) -> None:
        payload = _serialize_results(results)
        self._conn.execute(
            "INSERT OR REPLACE INTO extractor_cache "
            "(content_hash, extractor_id, results_json, created_at) "
            "VALUES (?, ?, ?, ?)",
            (
                key,
                extractor_id,
                payload,
                datetime.now(UTC).isoformat(),
            ),
        )


# ---------------------------------------------------------------------------
# Module-level helpers (kept free-standing so the cost-estimator can reuse
# ``_hash_key`` and ``_infer_extractor_id`` without instantiating a full
# ``CachedExtractor``).
# ---------------------------------------------------------------------------


def _infer_extractor_id(inner: BaseExtractor) -> str:
    """Build a stable id string for the wrapped extractor.

    Shape:

    * :class:`LLMExtractor` -> ``"llm:{provider}:{model}:v{PROMPT_VERSION}"``
    * :class:`TriggerExtractor` -> ``"trigger:v{TRIGGER_VERSION}"``
    * Anything else -> the fully-qualified class name. Users who subclass
      and want cache-stability across refactors should pass ``extractor_id``
      explicitly at wrap time.
    """
    cls = type(inner)
    full_name = f"{cls.__module__}.{cls.__qualname__}"

    # LLM extractor branch. Importing lazily keeps this module free of the
    # ``[openai]`` extra requirement.
    try:
        from nuggetindex.extractors.llm import LLMExtractor
    except Exception:  # pragma: no cover -- LLMExtractor is first-party
        LLMExtractor = None  # type: ignore[assignment]
    if LLMExtractor is not None and isinstance(inner, LLMExtractor):
        from nuggetindex.extractors.prompts import PROMPT_VERSION

        cfg = getattr(inner, "cfg", None)
        provider = getattr(cfg, "provider", "unknown") if cfg else "unknown"
        model = getattr(cfg, "model", "unknown") if cfg else "unknown"
        return f"llm:{provider}:{model}:{PROMPT_VERSION}"

    # TriggerExtractor branch -- cheap; no extra deps.
    try:
        from nuggetindex.extractors.trigger import TriggerExtractor
    except Exception:  # pragma: no cover -- first-party
        TriggerExtractor = None  # type: ignore[assignment]
    if TriggerExtractor is not None and isinstance(inner, TriggerExtractor):
        from nuggetindex.audit.heuristics.triggers import TRIGGER_VERSION

        return f"trigger:{TRIGGER_VERSION}"

    return full_name


def _hash_key(text: str, extractor_id: str) -> str:
    """SHA-256 of ``text + '|' + extractor_id``. Hex-encoded."""
    h = hashlib.sha256()
    h.update(text.encode("utf-8"))
    h.update(b"|")
    h.update(extractor_id.encode("utf-8"))
    return h.hexdigest()


def content_hash_for(text: str, extractor_id: str) -> str:
    """Public alias for :func:`_hash_key`, used by the cost estimator.

    Kept as a named public helper so callers don't have to import a
    leading-underscore name to probe cache-hit rates.
    """
    return _hash_key(text, extractor_id)


def _connect(cache_path: Path) -> sqlite3.Connection:
    """Open (or create) the SQLite cache file and validate / create the table."""
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(cache_path), isolation_level=None)
    # Autocommit mode (isolation_level=None) is what the spec asks for.
    # WAL for friendlier concurrent reads when the caller opens a second
    # instance to probe hit-rate (cost estimator).
    with contextlib.suppress(sqlite3.DatabaseError):  # pragma: no cover -- defensive on :memory:
        conn.execute("PRAGMA journal_mode=WAL")

    existing = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='extractor_cache'"
    ).fetchone()
    if existing is None:
        conn.execute(
            "CREATE TABLE extractor_cache ("
            "content_hash TEXT PRIMARY KEY,"
            "extractor_id TEXT NOT NULL,"
            "results_json TEXT NOT NULL,"
            "created_at TEXT NOT NULL"
            ")"
        )
        return conn

    # Validate column shape so we never silently serve from an
    # incompatible on-disk schema.
    cols = {row[1] for row in conn.execute("PRAGMA table_info(extractor_cache)")}
    missing = [c for c in _EXPECTED_COLUMNS if c not in cols]
    if missing:
        conn.close()
        raise RuntimeError(
            "extractor_cache schema mismatch at "
            f"{cache_path}: missing column(s) {missing}. "
            "Delete the cache file and re-run (no migration path in this release)."
        )
    return conn


def _serialize_results(results: list[ExtractionResult]) -> str:
    """JSON-encode a list of :class:`ExtractionResult` via Pydantic."""
    dumped = [r.model_dump(mode="json") for r in results]
    return json.dumps(dumped, sort_keys=False)


def _deserialize_results(payload: str) -> list[ExtractionResult]:
    """Reconstruct ``ExtractionResult`` instances from the cached JSON."""
    raw = json.loads(payload)
    if not isinstance(raw, list):
        raise RuntimeError(
            "extractor_cache payload was not a JSON array; the cache file is "
            "corrupted — delete it and re-run."
        )
    return [ExtractionResult.model_validate(item) for item in raw]
