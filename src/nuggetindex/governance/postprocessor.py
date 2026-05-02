"""Tier-1 governance postprocessor with session cache (Improvement A).

``GovernancePostProcessor`` is the framework-agnostic core that LangChain,
LlamaIndex, and Haystack adapters wrap to turn any retriever into a
governed retriever. It:

1. Maintains an on-disk ``NuggetStore`` (the "session cache") shared across
   queries so cross-document conflict detection improves as the session
   progresses. Cache path defaults to
   ``$NUGGETINDEX_CACHE_DIR / <config_hash>.db`` (see
   :func:`nuggetindex.governance.session_cache.default_cache_path`).
2. For each incoming batch of passages, hashes each passage's text and only
   ingests passages we haven't seen — the "content-addressed extraction
   cache." Known hashes are persisted in a sidecar file next to the cache DB.
3. After ingestion, looks up each passage's nuggets by ``source_id`` and
   decides whether to filter (all DEPRECATED), flag (``[DISPUTED]`` prefix
   for any CONTESTED), or pass through.

Crucially, ``__init__`` never runs ``asyncio.run`` — spec §7.2 requires the
constructor to be safe to call from inside a running event loop (the default
for LangChain / LlamaIndex). Use the async classmethod :meth:`acreate_warm`
to pre-populate the cache from documents before the first query.
"""
from __future__ import annotations

import asyncio
import warnings
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from nuggetindex.store.backends.sqlite import SQLiteBackend

from nuggetindex.core.enums import LifecycleStatus
from nuggetindex.core.models import Nugget
from nuggetindex.core.schema import RelationSchema
from nuggetindex.extractors.base import BaseExtractor
from nuggetindex.governance.session_cache import default_cache_path, passage_hash
from nuggetindex.pipeline.conflict import ConflictDetector
from nuggetindex.pipeline.constructor import Document, DocumentConstructor
from nuggetindex.pipeline.dedup import Deduplicator
from nuggetindex.store import NuggetStore


@dataclass
class RetrievedPassage:
    """Framework-agnostic retrieval result the postprocessor operates on.

    Adapters for LangChain ``Document``s, LlamaIndex ``NodeWithScore``s, and
    Haystack ``Document``s all translate to/from this shape.
    """

    source_id: str
    text: str
    score: float = 0.0
    metadata: dict[str, Any] | None = field(default=None)


class GovernancePostProcessor:
    """Lazy, session-cached postprocessor for cross-document conflict detection.

    Construction is cheap and loop-safe — no extraction or I/O beyond
    opening the cache SQLite file happens in ``__init__``. The first time
    :meth:`apostprocess` runs, any new passages are extracted + ingested into
    the session cache, then each passage's governance state is read back out
    and used to filter/flag.

    Use :meth:`acreate_warm` to pre-populate the cache with known documents
    before the first query — this is the eager mode that trades startup cost
    for immediate cross-doc conflict detection on query 1.
    """

    def __init__(
        self,
        *,
        cache_path: Path | str | None = None,
        extractor: str | BaseExtractor = "gpt-4o-mini",
        query_time: datetime | None = None,
        filter_deprecated: bool = True,
        flag_contested: bool = True,
        max_extraction_concurrency: int = 8,
        schema: RelationSchema | None = None,
        schema_hash: str = "default",
    ) -> None:
        self._extractor = self._build_extractor(extractor)
        extractor_cfg = (
            extractor if isinstance(extractor, str) else type(extractor).__name__
        )
        if cache_path is None:
            self._cache_path = default_cache_path(
                extractor_config=extractor_cfg, schema_hash=schema_hash,
            )
        else:
            self._cache_path = Path(cache_path)
            self._cache_path.parent.mkdir(parents=True, exist_ok=True)

        # The session cache is a full NuggetStore — the backend is what we
        # actually drive via DocumentConstructor + aupsert so we control
        # provenance attachment per passage.
        self._store = NuggetStore(
            db_path=self._cache_path,
            extractor=self._extractor,
            schema=schema,
        )
        self._constructor = DocumentConstructor(
            extractor=self._extractor,
            schema=self._store.schema,
            deduplicator=Deduplicator(encoder=self._store.encoder),
            conflict_detector=ConflictDetector(self._store.schema, judge=self._store.judge),
            quality_gate=self._store.quality_gate,
        )
        self.query_time = query_time
        self.filter_deprecated = filter_deprecated
        self.flag_contested = flag_contested
        self._sem = asyncio.Semaphore(max_extraction_concurrency)
        # A single-writer lock that serializes persistence. Cross-document
        # conflict detection requires each ingest's effects to be visible to
        # the next one, so we serialize *ingestion* even though extraction
        # itself is bounded by ``max_extraction_concurrency``.
        self._ingest_lock = asyncio.Lock()
        self._known_passage_hashes: set[str] = self._load_known_hashes()

    @classmethod
    async def acreate_warm(
        cls,
        *,
        warm_cache: Iterable[Document],
        **kw: Any,
    ) -> GovernancePostProcessor:
        """Construct + pre-populate the cache with ``warm_cache`` docs.

        This is the eager mode: pays the cost of ingesting warm docs up
        front so the very first query benefits from cross-document conflict
        detection against the pre-seeded corpus.
        """
        self = cls(**kw)
        for doc in warm_cache:
            passage = RetrievedPassage(source_id=doc.source_id, text=doc.text)
            await self._ingest_passage(passage, source_date=doc.source_date)
        self._persist_known_hashes()
        return self

    # --- extractor plumbing ------------------------------------------------

    @staticmethod
    def _build_extractor(extractor: str | BaseExtractor) -> BaseExtractor:
        if isinstance(extractor, BaseExtractor):
            return extractor
        if extractor == "rule_based":
            raise ValueError(
                "extractor='rule_based' is no longer supported: the rule-based "
                "extractor was removed from the public library in favour of the "
                "LLM extractor (full power) and the upcoming TriggerExtractor "
                "(LLM-free, pattern + NER). Pass a BaseExtractor instance or an "
                "LLM model id (e.g. 'gpt-4o-mini') instead."
            )
        # Assume an OpenAI-style LLM model name.
        from nuggetindex.extractors.clients.base import LLMConfig
        from nuggetindex.extractors.llm import LLMExtractor

        return LLMExtractor(LLMConfig(provider="openai", model=extractor))

    # --- sidecar hash persistence -----------------------------------------

    @property
    def _hashes_path(self) -> Path:
        # Use a suffixed name rather than with_suffix() so we keep the full
        # ".db" on the cache and just append ".hashes" — avoids collisions
        # with multi-suffix cache paths (e.g. "cache.sqlite.db").
        return self._cache_path.with_name(self._cache_path.name + ".hashes")

    def _load_known_hashes(self) -> set[str]:
        sidecar = self._hashes_path
        if sidecar.exists():
            return {line for line in sidecar.read_text().splitlines() if line}
        return set()

    def _persist_known_hashes(self) -> None:
        sidecar = self._hashes_path
        sidecar.write_text("\n".join(sorted(self._known_passage_hashes)))

    # --- ingest helpers ---------------------------------------------------

    async def _ingest_passage(
        self,
        passage: RetrievedPassage,
        *,
        source_date: datetime | None = None,
    ) -> None:
        """Extract, pipeline-process, persist.

        The backend's ``aupsert_passage`` + nugget upserts must run inside
        a single serial critical section — SQLite doesn't tolerate nested
        transactions, and cross-document conflict detection semantics also
        require sequential visibility.

        Provenance attachment is no longer needed here: the pipeline now
        forwards ``passage.source_id`` to any extractor that accepts it
        (Phase 2), so each emitted nugget already carries the passage's
        source_id in its provenance.
        """
        sd = source_date if source_date is not None else (
            self.query_time if self.query_time is not None else datetime.now(UTC)
        )
        doc = Document(source_id=passage.source_id, text=passage.text, source_date=sd)

        # 1. Run extraction (bounded concurrency) outside the ingest lock.
        async with self._sem:
            processed = await self._constructor.aprocess(
                doc, fetch_existing_by_key=self._store.backend.afind_by_key,
            )

        # 2. Persist under the serial lock so the next passage's conflict
        # detection sees this one's effects.
        async with self._ingest_lock:
            await self._store.backend.aupsert_passage(
                passage.source_id, None, passage.text,
            )
            for n in processed:
                await self._store.backend.aupsert(n)
            self._known_passage_hashes.add(passage_hash(passage.text))

    # --- main entry point --------------------------------------------------

    async def apostprocess(
        self, passages: list[RetrievedPassage]
    ) -> list[RetrievedPassage]:
        """Extract + ingest any new passages, then filter/flag based on cache state.

        Returns the (possibly reduced, possibly edited) list of passages.
        """
        if not passages:
            return []

        # 1. Ingest any passages we haven't seen (content-addressed).
        new_passages = [
            p for p in passages
            if passage_hash(p.text) not in self._known_passage_hashes
        ]
        if new_passages:
            await asyncio.gather(*(self._ingest_passage(p) for p in new_passages))
            self._persist_known_hashes()

        # 2. For each original passage, look up its nuggets and decide.
        keep: list[RetrievedPassage] = []
        for p in passages:
            nuggets = await self._find_nuggets_for_source(p.source_id)
            if not nuggets:
                # Nothing extracted from this passage -> nothing to govern.
                keep.append(p)
                continue

            if self.filter_deprecated and all(
                n.epistemic.status == LifecycleStatus.DEPRECATED for n in nuggets
            ):
                continue

            if self.flag_contested and any(
                n.epistemic.status == LifecycleStatus.CONTESTED for n in nuggets
            ):
                p = RetrievedPassage(
                    source_id=p.source_id,
                    text=f"[DISPUTED] {p.text}",
                    score=p.score,
                    metadata=p.metadata,
                )
            keep.append(p)

        return keep

    async def _find_nuggets_for_source(self, source_id: str) -> list[Nugget]:
        """Delegate to the backend's provenance-join helper."""
        return await self._store.backend.aget_nuggets_by_source(source_id)

    # --- introspection ----------------------------------------------------

    async def acount_cached_nuggets(
        self, status: LifecycleStatus | None = None
    ) -> int:
        """Total nuggets currently in the session cache (async-safe).

        Goes through the backend's writer queue / read executor, so it is
        safe to call concurrently with ingestion — the count reflects a
        consistent snapshot rather than a partially-applied ingest. Prefer
        this variant inside async contexts.
        """
        return await self._store.backend.acount(status)

    def count_cached_nuggets_unsafe(
        self, status: LifecycleStatus | None = None
    ) -> int:
        """Sync count that bypasses the backend writer queue.

        Reads a per-thread connection directly out of the backend's pool
        and runs ``COUNT(*)`` synchronously. This stays responsive from
        inside a running event loop (it never awaits) but may race with
        in-flight writes that have been dispatched to the writer task and
        not yet been persisted. Callers that need a consistent snapshot
        should use :meth:`acount_cached_nuggets` instead.
        """
        # ``_count_sync`` / ``_pool`` are concrete SQLiteBackend helpers;
        # the public ``backend`` attribute is typed against the narrower
        # ``MetadataBackend`` protocol so we cast for the typechecker.
        backend = cast("SQLiteBackend", self._store.backend)
        return backend._count_sync(backend._pool.get(), status)

    def count_cached_nuggets(self) -> int:
        """Deprecated alias for :meth:`count_cached_nuggets_unsafe`.

        The original sync-only method had ambiguous semantics under
        concurrent ingestion. In 0.2 the count API split into the
        async-safe :meth:`acount_cached_nuggets` and the explicitly-racy
        :meth:`count_cached_nuggets_unsafe`. This alias will be removed
        in nuggetindex 0.3.
        """
        warnings.warn(
            "count_cached_nuggets is ambiguous; use acount_cached_nuggets "
            "(async, safe) or count_cached_nuggets_unsafe (sync, racy). "
            "This alias will be removed in nuggetindex 0.3.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.count_cached_nuggets_unsafe()

    async def aclose(self) -> None:
        """Release the session-cache DB connection."""
        await self._store.aclose()
