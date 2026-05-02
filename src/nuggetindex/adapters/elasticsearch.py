"""Elasticsearch adapter conforming to :class:`CorpusSource`.

Accepts any object with an ``.search(index=..., body=..., size=...)`` method:
works with both ``elasticsearch.Elasticsearch`` (sync) and
``elasticsearch.AsyncElasticsearch`` (recommended). Does not import the
``elasticsearch`` package at module load -- the core nuggetindex import
stays lean even without the ``[elasticsearch]`` extra installed.

Result-dict mapping (override via field-name kwargs):

    ``_id``                              -> :attr:`Document.source_id`
    ``_source[title_field]`` + ``"\\n"`` +
    ``_source[text_field]``              -> :attr:`Document.text`
    ``_source[url_field]``               -> :attr:`Document.uri`
                                             (``None`` if absent)
    ``_source[date_field]``              -> :attr:`Document.source_date`
                                             (ISO-8601 parsed; ``None`` on
                                             error)
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, Literal

from nuggetindex.adapters.base import _TOPIC_DIVERSE_QUERIES

if TYPE_CHECKING:  # pragma: no cover
    from nuggetindex.pipeline.constructor import Document


@dataclass
class ElasticsearchCorpus:
    """Elasticsearch-backed :class:`CorpusSource`.

    ``client`` is duck-typed: any object exposing
    ``.search(index=..., body=..., size=...)`` works. The adapter awaits the
    return value iff it is awaitable, so both
    ``elasticsearch.AsyncElasticsearch`` (async) and
    ``elasticsearch.Elasticsearch`` (sync) are supported without branching.

    Elasticsearch 8.x wraps responses in ``ObjectApiResponse``; when the
    returned object exposes a ``.body`` attribute we unwrap it so downstream
    code always sees a plain ``dict``.
    """

    client: Any
    index: str
    text_field: str = "content"
    title_field: str = "title"
    url_field: str = "url"
    date_field: str = "source_date"

    # -- CorpusSource API ---------------------------------------------------

    async def sample(
        self,
        *,
        mode: Literal["topic_diverse", "uniform", "random_ids"],
        n: int,
    ) -> list[Document]:
        if mode == "random_ids":
            return await self._sample_random_ids(n)
        if mode == "uniform":
            return await self._sample_uniform(n)
        return await self._sample_topic_diverse(n)

    async def search(self, query: str, *, limit: int) -> list[Document]:
        resp = await self._call_search(
            {"query": {"multi_match": {"query": query}}},
            size=limit,
        )
        return [self._hit_to_document(h) for h in resp["hits"]["hits"]]

    # -- internals ----------------------------------------------------------

    async def _sample_topic_diverse(self, n: int) -> list[Document]:
        per = max(1, n // len(_TOPIC_DIVERSE_QUERIES)) + 1
        seen: set[str] = set()
        out: list[Document] = []
        for q in _TOPIC_DIVERSE_QUERIES:
            if len(out) >= n:
                break
            hits = await self.search(q, limit=per)
            for h in hits:
                if h.source_id not in seen:
                    seen.add(h.source_id)
                    out.append(h)
                    if len(out) >= n:
                        break
        return out

    async def _sample_uniform(self, n: int) -> list[Document]:
        resp = await self._call_search({"query": {"match_all": {}}}, size=n)
        return [self._hit_to_document(h) for h in resp["hits"]["hits"]]

    async def _sample_random_ids(self, n: int) -> list[Document]:
        resp = await self._call_search(
            {
                "query": {
                    "function_score": {
                        "functions": [{"random_score": {"seed": 0}}],
                    },
                },
            },
            size=n,
        )
        return [self._hit_to_document(h) for h in resp["hits"]["hits"]]

    async def _call_search(self, body: dict, *, size: int) -> dict:
        result = self.client.search(index=self.index, body=body, size=size)
        if hasattr(result, "__await__"):
            result = await result
        # Elasticsearch 8.x wraps responses in ObjectApiResponse; unwrap.
        if hasattr(result, "body"):
            result = result.body
        return result

    def _hit_to_document(self, hit: dict) -> Document:
        from nuggetindex.pipeline.constructor import Document

        source = hit.get("_source", {}) or {}
        title = source.get(self.title_field, "") or ""
        content = source.get(self.text_field, "") or ""
        text = (f"{title}\n{content}" if title and content else title or content).strip()
        uri = source.get(self.url_field) or None
        raw_date = source.get(self.date_field)
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
            source_id=str(hit.get("_id", "")),
            text=text,
            uri=uri,
            source_date=source_date,
        )


__all__ = ["ElasticsearchCorpus"]
