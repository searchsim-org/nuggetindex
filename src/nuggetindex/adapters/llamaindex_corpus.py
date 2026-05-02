"""LlamaIndex ``VectorStoreIndex`` / ``BaseNode`` iterable -> :class:`CorpusSource`.

Accepts a LlamaIndex ``VectorStoreIndex`` (anything exposing
``.as_retriever(...)`` and a ``.docstore`` with a ``.docs`` dict) *or* a
plain iterable of ``BaseNode`` / ``Document`` / ``TextNode`` objects.

The adapter is structural -- it doesn't import LlamaIndex at module load
time, so ``nuggetindex`` keeps importing without the ``[llamaindex]``
extra. ``topic_diverse`` mode needs a retriever: either one bound
explicitly via ``retriever=`` or one derived from the source's
``.as_retriever()``. Without one (e.g. a raw iterable of nodes),
``topic_diverse`` degrades to ``uniform`` with a :class:`UserWarning`.
"""

from __future__ import annotations

import random
import warnings
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

from nuggetindex.adapters.base import _TOPIC_DIVERSE_QUERIES

if TYPE_CHECKING:  # pragma: no cover
    from nuggetindex.pipeline.constructor import Document


@dataclass
class LlamaIndexCorpus:
    """Adapter bridging LlamaIndex to :class:`CorpusSource`.

    Parameters
    ----------
    source:
        Either a LlamaIndex ``VectorStoreIndex`` (anything exposing
        ``.as_retriever()`` and ``.docstore.docs``) or a plain iterable of
        ``BaseNode`` / ``Document`` / ``TextNode`` objects.
    retriever:
        Optional retriever with a ``.retrieve(query_or_bundle)`` method
        returning a list of ``NodeWithScore``. If ``None`` and ``source``
        exposes ``.as_retriever(similarity_top_k=...)``, one is lazily
        derived for ``topic_diverse`` / :meth:`search`. A raw iterable of
        nodes has no retriever -- :meth:`search` raises, and
        ``topic_diverse`` degrades to ``uniform`` with a warning.
    rng_seed:
        Deterministic seed for :meth:`sample`'s uniform shuffle.
    """

    source: Any
    retriever: Any | None = None
    rng_seed: int = 0
    _retriever_cache: dict[int, Any] = field(default_factory=dict, repr=False)

    # -- CorpusSource API ---------------------------------------------------

    async def sample(
        self,
        *,
        mode: Literal["topic_diverse", "uniform", "random_ids"],
        n: int,
    ) -> list[Document]:
        if mode == "random_ids":
            raise NotImplementedError(
                "LlamaIndexCorpus.sample(mode='random_ids') requires a "
                "document-ID enumeration API that LlamaIndex doesn't "
                "expose uniformly. Use mode='uniform' or 'topic_diverse'."
            )
        if mode == "topic_diverse" and self._resolve_retriever(top_k=1) is None:
            warnings.warn(
                "LlamaIndexCorpus.sample(mode='topic_diverse') requires a "
                "retriever -- either bind one via `retriever=` or pass a "
                "VectorStoreIndex-shaped source. Falling back to uniform "
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
        retriever = self._resolve_retriever(top_k=limit)
        if retriever is None:
            raise RuntimeError(
                "LlamaIndexCorpus.search() requires a retriever -- either "
                "bind one via `retriever=` or pass a VectorStoreIndex-"
                "shaped source so one can be derived."
            )
        return self._retrieve_as_documents(retriever, query, limit)

    # -- internals ----------------------------------------------------------

    def _iter_nodes(self) -> list[Any]:
        """Materialise the source into a list of node-like objects.

        Accepts three shapes (mirrors
        :func:`nuggetindex.integrations.llamaindex.doctor._iter_nodes`):

        * A ``VectorStoreIndex`` with ``.docstore.docs`` (dict) -- iterate
          values.
        * A ``.docstore.get_nodes(node_ids=None)`` fallback for persisted
          stores that don't expose ``.docs``.
        * Any plain iterable of nodes.
        """
        docstore = getattr(self.source, "docstore", None)
        if docstore is not None:
            docs = getattr(docstore, "docs", None)
            if isinstance(docs, dict):
                return list(docs.values())
            get_nodes = getattr(docstore, "get_nodes", None)
            if callable(get_nodes):
                try:
                    return list(get_nodes(node_ids=None))
                except TypeError:
                    pass
        return list(self.source)

    def _sample_uniform(self, n: int) -> list[Document]:
        nodes = self._iter_nodes()
        rng = random.Random(self.rng_seed)
        rng.shuffle(nodes)
        return [self._to_document(node) for node in nodes[:n]]

    async def _sample_topic_diverse(self, n: int) -> list[Document]:
        per_query = max(1, n // len(_TOPIC_DIVERSE_QUERIES)) + 1
        retriever = self._resolve_retriever(top_k=per_query)
        assert retriever is not None  # sample() guards against this branch
        seen: set[str] = set()
        out: list[Document] = []
        for q in _TOPIC_DIVERSE_QUERIES:
            if len(out) >= n:
                break
            hits = self._retrieve_as_documents(retriever, q, per_query)
            for doc in hits:
                if not doc.source_id or doc.source_id in seen:
                    continue
                seen.add(doc.source_id)
                out.append(doc)
                if len(out) >= n:
                    break
        return out

    def _resolve_retriever(self, *, top_k: int) -> Any | None:
        """Return a usable retriever or ``None`` if none can be obtained.

        Precedence: an explicitly-bound ``self.retriever`` wins. Otherwise
        we call ``self.source.as_retriever(similarity_top_k=top_k)`` if
        available and cache the result by ``top_k``. A raw iterable of
        nodes returns ``None``.
        """
        if self.retriever is not None:
            return self.retriever
        as_retriever = getattr(self.source, "as_retriever", None)
        if not callable(as_retriever):
            return None
        if top_k in self._retriever_cache:
            return self._retriever_cache[top_k]
        try:
            retr = as_retriever(similarity_top_k=top_k)
        except TypeError:
            # Some retriever factories use a different kwarg name.
            retr = as_retriever()
        self._retriever_cache[top_k] = retr
        return retr

    def _retrieve_as_documents(
        self,
        retriever: Any,
        query: str,
        limit: int,
    ) -> list[Document]:
        """Call the retriever and project each ``NodeWithScore`` to a Document."""
        hits = retriever.retrieve(query)
        out: list[Document] = []
        for hit in hits[:limit]:
            # ``NodeWithScore`` wraps the real node under ``.node``.
            node = getattr(hit, "node", None) or hit
            out.append(self._to_document(node))
        return out

    def _to_document(self, node: Any) -> Document:
        """Project a LlamaIndex node into a nuggetindex :class:`Document`."""
        from nuggetindex.pipeline.constructor import Document

        node_id = (
            getattr(node, "node_id", None)
            or getattr(node, "id_", None)
            or ""
        )
        if hasattr(node, "get_content"):
            try:
                text = node.get_content() or ""
            except TypeError:
                # Some implementations require a MetadataMode arg.
                text = getattr(node, "text", "") or ""
        else:
            text = getattr(node, "text", "") or ""
        metadata: dict[str, Any] = dict(getattr(node, "metadata", None) or {})
        uri = metadata.get("file_path") or metadata.get("url")
        return Document(
            source_id=str(node_id),
            text=text,
            uri=uri,
            source_date=None,
        )


__all__ = ["LlamaIndexCorpus"]
