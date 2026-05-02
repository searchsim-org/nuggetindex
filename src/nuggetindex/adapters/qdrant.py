"""Qdrant adapter conforming to :class:`CorpusSource`.

Vector-DB. "Search" is a similarity query against an embedding: the adapter
accepts a caller-supplied ``embedder: Callable[[str], list[float]]`` that
converts queries to vectors. Does not hard-import ``qdrant-client`` -- the
core package imports without the ``[qdrant]`` extra.

Payload mapping:

    ``point.id``                                      -> :attr:`Document.source_id`
    ``payload[title_field]`` + ``"\\n"`` +
    ``payload[text_field]``                           -> :attr:`Document.text`
    ``payload[url_field]``                            -> :attr:`Document.uri`
    ``payload[date_field]``                           -> :attr:`Document.source_date`
                                                        (ISO-8601 parsed; ``None`` on
                                                        error)
"""
from __future__ import annotations

import random
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:  # pragma: no cover
    from nuggetindex.pipeline.constructor import Document


@dataclass
class QdrantCorpus:
    """Qdrant-backed :class:`CorpusSource`.

    ``client`` is duck-typed: any object exposing ``.search(...)`` /
    ``.scroll(...)`` works. The adapter awaits the return value iff it is
    awaitable, so both ``qdrant_client.AsyncQdrantClient`` (async) and
    ``qdrant_client.QdrantClient`` (sync) are supported without branching.

    ``embedder`` is the caller-supplied callable that converts a query
    string into a vector. It is invoked in :meth:`search` and in
    :meth:`_sample_topic_diverse` (to probe vector dimensionality).
    """

    client: Any
    collection: str
    embedder: Callable[[str], list[float]]
    text_field: str = "content"
    title_field: str = "title"
    url_field: str = "url"
    date_field: str = "source_date"
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
                "QdrantCorpus.sample(mode='random_ids') is not supported. "
                "Use mode='topic_diverse' or 'uniform'.",
            )
        if mode == "topic_diverse":
            return await self._sample_topic_diverse(n)
        return await self._sample_uniform(n)

    async def search(self, query: str, *, limit: int) -> list[Document]:
        vec = self.embedder(query)
        result = self.client.search(
            collection_name=self.collection,
            query_vector=vec,
            limit=limit,
        )
        if hasattr(result, "__await__"):
            result = await result
        return [self._point_to_document(p) for p in result]

    # -- internals ----------------------------------------------------------

    async def _sample_topic_diverse(self, n: int) -> list[Document]:
        # Without knowing the corpus's semantic layout we can't run genuine
        # topic-diverse queries. Approximate: probe the vector space with
        # multiple random unit vectors drawn from the same dimensionality as
        # ``embedder("probe")``, dedupe across probes.
        rng = random.Random(self.rng_seed)
        dim = len(self.embedder("probe"))
        seen: set[str] = set()
        out: list[Document] = []
        for _ in range(8):
            if len(out) >= n:
                break
            vec = [rng.uniform(-1.0, 1.0) for _ in range(dim)]
            result = self.client.search(
                collection_name=self.collection,
                query_vector=vec,
                limit=max(1, n // 4),
            )
            if hasattr(result, "__await__"):
                result = await result
            for p in result:
                doc = self._point_to_document(p)
                if doc.source_id not in seen:
                    seen.add(doc.source_id)
                    out.append(doc)
                    if len(out) >= n:
                        return out
        return out

    async def _sample_uniform(self, n: int) -> list[Document]:
        result = self.client.scroll(collection_name=self.collection, limit=n)
        if hasattr(result, "__await__"):
            result = await result
        points, _ = result
        return [self._point_to_document(p) for p in points]

    def _point_to_document(self, point: Any) -> Document:
        from nuggetindex.pipeline.constructor import Document

        payload = getattr(point, "payload", None) or {}
        title = payload.get(self.title_field, "") or ""
        content = payload.get(self.text_field, "") or ""
        text = (
            f"{title}\n{content}" if title and content else title or content
        ).strip()
        uri = payload.get(self.url_field) or None
        raw_date = payload.get(self.date_field)
        source_date: datetime | None = None
        if raw_date:
            try:
                source_date = datetime.fromisoformat(
                    str(raw_date).replace("Z", "+00:00"),
                )
                if source_date.tzinfo is None:
                    source_date = source_date.replace(tzinfo=UTC)
            except ValueError:
                source_date = None
        return Document(
            source_id=str(getattr(point, "id", "")),
            text=text,
            uri=uri,
            source_date=source_date,
        )


__all__ = ["QdrantCorpus"]
