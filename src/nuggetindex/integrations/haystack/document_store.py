"""Haystack 2.x ``DocumentStore`` implementation backed by a ``NuggetStore``.

``NuggetDocumentStore`` exposes the five-method Haystack 2.x DocumentStore
surface (``write_documents``, ``delete_documents``, ``count_documents``,
``filter_documents``, ``bm25_retrieval``) and delegates to an existing
:class:`~nuggetindex.NuggetStore` instance so users can plug nuggetindex
straight into a Haystack pipeline where a ``DocumentStore`` is expected.

Design notes
------------
- Haystack ``Document``  nuggetindex **source passage**. We do NOT pass
  user-supplied Haystack docs through the nugget extraction pipeline on
  ``write_documents``; that conflates two different ingestion models and
  forces every Haystack user to configure an extractor. Instead, we persist
  the Haystack ``Document`` verbatim in the ``passages`` table (id + content
  + ``Document.to_dict()`` JSON) and let ``NuggetConstructor`` (the
  ``@component``) or ``store.aingest`` remain the nugget-producing paths.

- ``bm25_retrieval`` delegates to ``store.aretrieve(..., view="all")`` which
  runs BM25 over the nugget index, then maps the matched nugget IDs back to
  the passage ``source_id`` via provenance and returns the stored Haystack
  ``Document`` for each unique source. Passages that were written through
  ``write_documents`` but have no extracted nuggets yet will not surface
  through BM25 (expected: no nuggets, no BM25 signal).

- ``filter_documents`` routes the Haystack filter dict into
  :meth:`MetadataBackend.afilter` with the v0.1 SQL allowlist
  (``_ALLOWED_FILTER_COLUMNS`` in :mod:`nuggetindex.store.backends.sqlite`).
  Disallowed fields raise ``ValueError``. Only the ``==`` operator is
  supported in v0.2 — richer comparison operators are deferred pending a
  proper filter-DSL design.

- Duplicate policy enforcement lives here (not in the backend). We look up
  each incoming doc's ``source_id`` via ``backend.apassage_exists`` and
  branch on ``DuplicatePolicy``. This keeps the backend surface
  policy-agnostic — the ``NuggetStore`` core continues to have
  "last-write-wins" semantics for direct users of ``aadd`` / ``aingest``.

Import hygiene
--------------
Per the integration-import-hygiene rules (see
``tests/integration/haystack/test_import_hygiene.py``) this module may only
import from the public ``nuggetindex`` top-level namespace. The Haystack
side imports go through ``_require_haystack()`` so callers missing the
``[haystack]`` extra get a useful ``pip install`` hint.
"""
from __future__ import annotations

import asyncio
import dataclasses
import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from nuggetindex import NuggetStore

if TYPE_CHECKING:
    # mypy-only: real Haystack types for annotations.
    from haystack import Document as HaystackDocument


def _require_haystack() -> tuple[Any, Any, Any]:
    try:
        from haystack import Document as _Document
        from haystack.document_stores.errors import (
            DuplicateDocumentError as _DupError,
        )
        from haystack.document_stores.types import (
            DuplicatePolicy as _DuplicatePolicy,
        )
    except ImportError as e:  # pragma: no cover - import guard
        raise ImportError(
            "nuggetindex[haystack] not installed. "
            "Run: pip install 'nuggetindex[haystack]'"
        ) from e
    return _Document, _DuplicatePolicy, _DupError


_HaystackDocument, DuplicatePolicy, DuplicateDocumentError = _require_haystack()

# Runtime binding under the same name (see retriever.py for the full rationale).
HaystackDocument = _HaystackDocument  # type: ignore[misc]


@dataclass
class _IngestDoc:
    """Minimal shape duck-typed by ``NuggetStore.aingest``.

    Mirrors the local ``_IngestDoc`` in ``constructor.py`` so we don't reach
    into ``nuggetindex.pipeline.constructor`` (blocked by the integration
    import-hygiene test).
    """

    source_id: str
    text: str
    uri: str | None = None
    source_date: Any = None


class NuggetDocumentStore:
    """Haystack 2.x ``DocumentStore`` backed by a :class:`NuggetStore`.

    Example::

        store = NuggetStore(db_path="my.db")
        ds = NuggetDocumentStore(store=store)
        ds.write_documents([Document(content="Sundar Pichai is CEO of Google.")])
        matches = ds.bm25_retrieval("Google CEO", top_k=5)

    Sync-only surface to match Haystack's protocol: internal async calls go
    through ``asyncio.run`` per method invocation (Haystack's own
    ``DocumentStore`` implementations are all sync).
    """

    def __init__(self, store: NuggetStore) -> None:
        self._store = store

    # ------------------------------------------------------------------
    # write_documents
    # ------------------------------------------------------------------

    def write_documents(
        self,
        documents: list[HaystackDocument],
        policy: Any = DuplicatePolicy.OVERWRITE,
    ) -> int:
        """Persist the given Haystack documents as source passages.

        Behaviour per policy:

        - ``OVERWRITE`` (default): every doc is upserted; returns the input
          count.
        - ``SKIP``: docs whose ``id`` already exists are skipped silently;
          returns the number actually written.
        - ``FAIL``: raises :class:`DuplicateDocumentError` on the first
          collision; leaves previously-written docs from the same batch in
          place.
        - ``NONE``: treated as ``OVERWRITE`` (Haystack's convention: "no
          policy specified" defers to the store; we pick overwrite for
          least-surprise parity with ``aingest``).

        Validates input: non-list inputs and lists containing non-``Document``
        items raise ``ValueError`` to satisfy Haystack's ``WriteDocumentsTest``.
        """
        if not isinstance(documents, list):
            raise ValueError(
                f"documents must be a list, got {type(documents).__name__}"
            )
        for d in documents:
            if not isinstance(d, _HaystackDocument):
                raise ValueError(
                    f"every entry must be a haystack.Document, "
                    f"got {type(d).__name__}"
                )

        return asyncio.run(self._awrite_documents(documents, policy))

    async def _awrite_documents(
        self,
        documents: list[HaystackDocument],
        policy: Any,
    ) -> int:
        written = 0
        for d in documents:
            content = d.content or ""
            exists = await self._store.backend.apassage_exists(d.id)
            if exists:
                if policy == DuplicatePolicy.SKIP:
                    continue
                if policy == DuplicatePolicy.FAIL:
                    raise DuplicateDocumentError(
                        f"Document with id {d.id!r} already exists"
                    )
            # OVERWRITE (explicit or default) + new-id paths land here. We
            # store the full ``to_dict()`` so ``filter_documents`` can
            # round-trip the original Document shape (meta, embeddings, etc.)
            # without re-deriving fields from the passage row.
            meta_json = json.dumps(d.to_dict(flatten=False))
            await self._store.backend.aupsert_passage_with_meta(
                source_id=d.id, uri=None, text=content, meta_json=meta_json,
            )
            written += 1
        return written

    # ------------------------------------------------------------------
    # delete_documents / count_documents
    # ------------------------------------------------------------------

    def delete_documents(self, document_ids: list[str]) -> None:
        """Delete passages and any nuggets sourced from them.

        Non-existing ids are silently ignored, matching Haystack's
        ``DeleteDocumentsTest.test_delete_documents_non_existing_document``.
        """
        if not document_ids:
            return
        asyncio.run(
            self._store.backend.adelete_by_source_ids(list(document_ids))
        )

    def count_documents(self) -> int:
        """Return the number of source-passage rows currently stored."""
        return asyncio.run(self._store.backend.acount_passages())

    # ------------------------------------------------------------------
    # filter_documents
    # ------------------------------------------------------------------

    def filter_documents(
        self, filters: dict[str, Any] | None = None
    ) -> list[HaystackDocument]:
        """Return stored documents matching ``filters``.

        ``filters=None`` returns every stored passage as a ``Document``
        (reconstructed from the stored ``to_dict()`` JSON, falling back to
        ``id`` + ``content`` for passages written via ``aingest``/other
        paths).

        A simple ``{"field": "...", "operator": "==", "value": "..."}`` dict
        is delegated to :meth:`NuggetStore.backend.afilter` with the v0.1
        allowlist. Only ``==`` is supported in v0.2 — other operators raise
        ``ValueError``. Disallowed fields likewise raise ``ValueError`` (via
        the backend's built-in allowlist check).
        """
        if filters is None:
            return asyncio.run(self._afilter_all())
        field = filters.get("field")
        op = filters.get("operator", "==")
        val = filters.get("value")
        if op != "==":
            raise ValueError(
                f"NuggetDocumentStore.filter_documents only supports '==' "
                f"in v0.2, got {op!r}"
            )
        if not isinstance(field, str):
            raise ValueError(
                "filter dict must contain a 'field' string key, "
                f"got {field!r}"
            )
        return asyncio.run(self._afilter_nuggets({field: val}))

    async def _afilter_all(self) -> list[HaystackDocument]:
        source_ids = await self._store.backend.alist_source_ids()
        records = await self._store.backend.aget_passage_records(source_ids)
        return [self._record_to_document(sid, records[sid]) for sid in source_ids]

    async def _afilter_nuggets(
        self, extra_filters: dict[str, Any]
    ) -> list[HaystackDocument]:
        # ``afilter`` wants a query_time; use "now" so the resulting validity
        # check is a no-op for most passages (ACTIVE nuggets with no
        # ``validity_end``). Callers who want point-in-time filtering should
        # use ``store.aretrieve`` directly.
        from datetime import UTC, datetime
        nugget_ids = await self._store.backend.afilter(
            query_time=datetime.now(UTC),
            view="all",
            extra_filters=extra_filters,
        )
        # Map nugget ids  source passages via provenance. We hydrate the
        # nuggets one-by-one for clarity; the result set is bounded by the
        # filter's selectivity, so this is fine for v0.2.
        source_ids: list[str] = []
        seen: set[str] = set()
        for nid in nugget_ids:
            n = await self._store.backend.aget(nid)
            if n is None:
                continue
            for p in n.provenance:
                if p.source_id not in seen:
                    seen.add(p.source_id)
                    source_ids.append(p.source_id)
        if not source_ids:
            return []
        records = await self._store.backend.aget_passage_records(source_ids)
        return [
            self._record_to_document(sid, records[sid])
            for sid in source_ids
            if sid in records
        ]

    # ------------------------------------------------------------------
    # bm25_retrieval
    # ------------------------------------------------------------------

    def bm25_retrieval(
        self,
        query: str,
        filters: dict[str, Any] | None = None,  # noqa: ARG002
        top_k: int = 10,
        scale_score: bool = False,  # noqa: ARG002
    ) -> list[HaystackDocument]:
        """Run BM25 over the nugget index and return unique source passages.

        Matches Haystack's ``InMemoryBM25Retriever``-style signature. The
        underlying search is over nugget fact sentences; we surface the
        *passage* each matched nugget was extracted from, deduplicating by
        ``source_id`` so the same passage never appears twice even when
        multiple nuggets from it matched.

        ``filters`` and ``scale_score`` are accepted for API parity but not
        honoured in v0.2 — use :meth:`filter_documents` or
        :meth:`NuggetStore.aretrieve` directly for filtered retrieval, and
        rely on the already-normalised fused ``score`` on each returned
        ``Document``.
        """
        return asyncio.run(self._abm25_retrieval(query, top_k=top_k))

    async def _abm25_retrieval(
        self, query: str, *, top_k: int,
    ) -> list[HaystackDocument]:
        # view="all" so CONTESTED/DEPRECATED matches still surface — the
        # caller is explicitly asking for BM25 over the corpus and Haystack
        # callers typically don't wire a lifecycle view.
        results = await self._store.aretrieve(
            query, view="all", top_k=top_k,
        )
        # Collect unique source_ids in fused-score order. ``RetrievalResult``
        # carries the underlying ``Nugget`` with its provenance list already
        # populated.
        ordered_sources: list[tuple[str, float]] = []
        seen: set[str] = set()
        for r in results:
            score = r.score
            for p in r.nugget.provenance:
                if p.source_id in seen:
                    continue
                seen.add(p.source_id)
                ordered_sources.append((p.source_id, score))
        if not ordered_sources:
            return []
        source_ids = [sid for sid, _ in ordered_sources]
        records = await self._store.backend.aget_passage_records(source_ids)
        docs: list[HaystackDocument] = []
        for sid, score in ordered_sources:
            rec = records.get(sid)
            if rec is None:
                continue
            doc = self._record_to_document(sid, rec)
            # Attach the fused score so downstream components can rank.
            # Use ``dataclasses.replace`` (not attribute assignment) per
            # Haystack's recommendation — mutating the dataclass in place
            # can leak state across pipeline steps that share the instance.
            docs.append(dataclasses.replace(doc, score=score))
        return docs

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _record_to_document(
        self, source_id: str, record: tuple[str, str | None],
    ) -> HaystackDocument:
        """Rebuild a Haystack ``Document`` from a stored passage row.

        Prefers the full ``to_dict()`` JSON when present (the
        ``write_documents`` path); falls back to ``id`` + ``content`` for
        passages written via other code paths (``store.aingest``, direct
        ``aupsert_passage``).
        """
        text, meta_json = record
        if meta_json:
            try:
                data = json.loads(meta_json)
            except json.JSONDecodeError:
                data = None
            if isinstance(data, dict):
                # ``Document.from_dict`` handles meta/embedding round-trip.
                doc: HaystackDocument = _HaystackDocument.from_dict(data)
                return doc
        fresh: HaystackDocument = _HaystackDocument(id=source_id, content=text)
        return fresh
