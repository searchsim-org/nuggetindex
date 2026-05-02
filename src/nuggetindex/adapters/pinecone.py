"""Pinecone adapter conforming to :class:`CorpusSource`.

Vector-DB. Accepts a Pinecone v5+ ``Index`` object (sync API; Pinecone's
async support is still limited in v5) and an embedder callable. Does not
hard-import the ``pinecone`` package at module load.

Payload mapping:

    ``match['id']``                                      -> :attr:`Document.source_id`
    ``metadata[title_field]`` + ``"\\n"`` +
    ``metadata[text_field]``                             -> :attr:`Document.text`
    ``metadata[url_field]``                              -> :attr:`Document.uri`
    ``metadata[date_field]``                             -> :attr:`Document.source_date`
                                                           (ISO-8601 parsed; ``None``
                                                           on error)
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
class PineconeCorpus:
    """Pinecone-backed :class:`CorpusSource`.

    ``index`` is duck-typed: any object exposing ``.query(...)`` /
    ``.list(...)`` / ``.fetch(...)`` works. The adapter awaits the return
    value iff it is awaitable, so both sync and async clients are supported
    without branching.

    ``embedder`` is the caller-supplied callable that converts a query
    string into a vector. It is invoked in :meth:`search` and in
    :meth:`_sample_topic_diverse` (to probe vector dimensionality).
    """

    index: Any
    embedder: Callable[[str], list[float]]
    text_field: str = "content"
    title_field: str = "title"
    url_field: str = "url"
    date_field: str = "source_date"
    namespace: str | None = None
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
                "PineconeCorpus.sample(mode='random_ids') is not supported. "
                "Use mode='topic_diverse' or 'uniform'.",
            )
        if mode == "topic_diverse":
            return await self._sample_topic_diverse(n)
        return await self._sample_uniform(n)

    async def search(self, query: str, *, limit: int) -> list[Document]:
        vec = self.embedder(query)
        kwargs: dict[str, Any] = {
            "vector": vec,
            "top_k": limit,
            "include_metadata": True,
        }
        if self.namespace is not None:
            kwargs["namespace"] = self.namespace
        result = self.index.query(**kwargs)
        if hasattr(result, "__await__"):
            result = await result
        matches = self._matches(result)
        return [self._match_to_document(m) for m in matches]

    # -- internals ----------------------------------------------------------

    async def _sample_topic_diverse(self, n: int) -> list[Document]:
        rng = random.Random(self.rng_seed)
        dim = len(self.embedder("probe"))
        seen: set[str] = set()
        out: list[Document] = []
        for _ in range(8):
            if len(out) >= n:
                break
            vec = [rng.uniform(-1.0, 1.0) for _ in range(dim)]
            kwargs: dict[str, Any] = {
                "vector": vec,
                "top_k": max(1, n // 4),
                "include_metadata": True,
            }
            if self.namespace is not None:
                kwargs["namespace"] = self.namespace
            result = self.index.query(**kwargs)
            if hasattr(result, "__await__"):
                result = await result
            for m in self._matches(result):
                doc = self._match_to_document(m)
                if doc.source_id not in seen:
                    seen.add(doc.source_id)
                    out.append(doc)
                    if len(out) >= n:
                        return out
        return out

    async def _sample_uniform(self, n: int) -> list[Document]:
        list_kwargs: dict[str, Any] = {"limit": n}
        if self.namespace is not None:
            list_kwargs["namespace"] = self.namespace
        listed = self.index.list(**list_kwargs)
        if hasattr(listed, "__await__"):
            listed = await listed
        ids = [entry["id"] if isinstance(entry, dict) else entry for entry in listed]
        fetch_kwargs: dict[str, Any] = {"ids": ids}
        if self.namespace is not None:
            fetch_kwargs["namespace"] = self.namespace
        fetched = self.index.fetch(**fetch_kwargs)
        if hasattr(fetched, "__await__"):
            fetched = await fetched
        vectors = fetched.get("vectors", {}) if isinstance(fetched, dict) else {}
        return [self._match_to_document(v) for v in vectors.values()]

    def _matches(self, result: Any) -> list[dict]:
        if isinstance(result, dict):
            return result.get("matches", [])
        return list(getattr(result, "matches", []) or [])

    def _match_to_document(self, match: Any) -> Document:
        from nuggetindex.pipeline.constructor import Document

        if isinstance(match, dict):
            id_ = match.get("id", "")
            metadata = match.get("metadata", {}) or {}
        else:
            id_ = getattr(match, "id", "")
            metadata = getattr(match, "metadata", None) or {}
        title = metadata.get(self.title_field, "") or ""
        content = metadata.get(self.text_field, "") or ""
        text = (
            f"{title}\n{content}" if title and content else title or content
        ).strip()
        uri = metadata.get(self.url_field) or None
        raw_date = metadata.get(self.date_field)
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
            source_id=str(id_),
            text=text,
            uri=uri,
            source_date=source_date,
        )


__all__ = ["PineconeCorpus"]
