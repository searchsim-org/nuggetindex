"""Haystack 2.x ``DocumentStore`` -> :class:`CorpusSource` adapter.

Accepts any object that conforms to Haystack's ``DocumentStore`` Protocol
(e.g. ``InMemoryDocumentStore``, ``ElasticsearchDocumentStore``, ...). The
adapter is structural -- it doesn't import Haystack at module load time,
so ``nuggetindex`` keeps importing without the Haystack extra installed.

``topic_diverse`` mode requires the caller to bind a Haystack retriever
via ``retriever=...`` (since a DocumentStore alone doesn't do BM25).
Without a retriever, ``topic_diverse`` degrades to ``uniform`` with a
:class:`UserWarning`.
"""

from __future__ import annotations

import random
import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

from nuggetindex.adapters.base import _TOPIC_DIVERSE_QUERIES

if TYPE_CHECKING:  # pragma: no cover
    from nuggetindex.pipeline.constructor import Document


@dataclass
class HaystackCorpus:
    """Adapter bridging a Haystack ``DocumentStore`` to :class:`CorpusSource`.

    Parameters
    ----------
    document_store:
        Any Haystack 2.x ``DocumentStore`` (duck-typed: must expose
        ``filter_documents()``).
    retriever:
        Optional Haystack retriever (``InMemoryBM25Retriever``,
        ``ElasticsearchBM25Retriever``, ...) exposing a ``.run(query=...,
        top_k=...)`` method returning ``{"documents": [...]}``. Required
        for ``topic_diverse`` and :meth:`search`; without it those paths
        degrade (``topic_diverse`` -> ``uniform`` with a warning,
        :meth:`search` raises).
    rng_seed:
        Deterministic seed for :meth:`sample`'s uniform shuffle.
    """

    document_store: Any
    retriever: Any | None = None
    rng_seed: int = 0

    # -- CorpusSource API ---------------------------------------------------

    async def sample(
        self,
        *,
        mode: Literal["topic_diverse", "uniform", "random_ids"],
        n: int,
    ) -> list[Document]:
        if mode == "random_ids":
            raise NotImplementedError(
                "HaystackCorpus.sample(mode='random_ids') requires a "
                "document-ID enumeration API that Haystack doesn't expose "
                "uniformly. Use mode='uniform' or 'topic_diverse'."
            )
        if mode == "topic_diverse" and self.retriever is None:
            warnings.warn(
                "HaystackCorpus.sample(mode='topic_diverse') requires a "
                "retriever bound via `retriever=`; falling back to uniform "
                "sampling.",
                UserWarning,
                stacklevel=2,
            )
            mode = "uniform"

        if mode == "uniform":
            return self._sample_uniform(n)
        if mode == "topic_diverse":
            return await self._sample_topic_diverse(n)
        raise ValueError(f"unknown sample mode: {mode!r}")

    async def search(self, query: str, *, limit: int) -> list[Document]:
        if self.retriever is None:
            raise RuntimeError(
                "HaystackCorpus.search() requires a retriever bound via "
                "`retriever=`. Without one, the corpus is sample-only."
            )
        hits = self._run_retriever(query=query, top_k=limit)
        return [self._to_document(h) for h in hits]

    # -- internals ----------------------------------------------------------

    def _sample_uniform(self, n: int) -> list[Document]:
        all_docs = list(self.document_store.filter_documents())
        rng = random.Random(self.rng_seed)
        rng.shuffle(all_docs)
        return [self._to_document(d) for d in all_docs[:n]]

    async def _sample_topic_diverse(self, n: int) -> list[Document]:
        per_query = max(1, n // len(_TOPIC_DIVERSE_QUERIES)) + 1
        seen: set[str] = set()
        out: list[Document] = []
        for q in _TOPIC_DIVERSE_QUERIES:
            if len(out) >= n:
                break
            hits = self._run_retriever(query=q, top_k=per_query)
            for h in hits:
                doc = self._to_document(h)
                if not doc.source_id or doc.source_id in seen:
                    continue
                seen.add(doc.source_id)
                out.append(doc)
                if len(out) >= n:
                    break
        return out

    def _run_retriever(self, *, query: str, top_k: int) -> list[Any]:
        """Invoke the bound retriever and return the raw ``documents`` list.

        Haystack's retriever returns ``{"documents": [...]}``; this wrapper
        centralises the accessor so the rest of the module can treat the
        retriever as opaque.
        """
        result = self.retriever.run(query=query, top_k=top_k)
        if isinstance(result, dict):
            return list(result.get("documents") or [])
        # Some future shape could return objects directly; be permissive.
        return list(result) if result else []

    def _to_document(self, h: Any) -> Document:
        """Project a Haystack ``Document`` into a nuggetindex :class:`Document`."""
        from nuggetindex.pipeline.constructor import Document

        meta: dict[str, Any] = dict(getattr(h, "meta", None) or {})
        source_id = str(
            getattr(h, "id", None) or meta.get("source_id") or ""
        )
        text = getattr(h, "content", "") or ""
        uri = meta.get("url") or meta.get("uri")
        return Document(
            source_id=source_id,
            text=text,
            uri=uri,
            source_date=None,
        )


__all__ = ["HaystackCorpus"]
