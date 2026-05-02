"""Vespa-backed :class:`~nuggetindex.adapters.base.CorpusSource` adapter.

A narrow adapter for Vespa-style BM25 search clusters that expose a
JSON REST API. *Not* a generic HTTP search adapter -- the response
shape and URL layout assume the conventions documented below.

Endpoints used:

* ``POST /api/v1/search/docs/{corpus}/search``          BM25 search
* ``GET  /api/v1/cluster/corpora/{corpus}/statistics``  total doc count

Any Vespa-like BM25 HTTP endpoint that accepts
``{"query": str, "limit": int, "offset": int}`` POST bodies and returns
``{"hits": [{"_id", "title", "url", "description_snippet", ...}]}`` will
work unmodified.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

from nuggetindex.adapters.base import _TOPIC_DIVERSE_QUERIES

if TYPE_CHECKING:  # pragma: no cover
    import httpx

    from nuggetindex.pipeline.constructor import Document


def _strip_highlights(s: str) -> str:
    """Remove ``<hi>`` / ``</hi>`` / ``<sep/>`` markup from Vespa snippets."""
    return (
        s.replace("<hi>", "")
        .replace("</hi>", "")
        .replace("<sep />", " ")
        .replace("<sep/>", " ")
    )


def _hit_to_document(hit: dict[str, Any]) -> Document:
    """Project a Vespa search hit into a :class:`Document`.

    ``source_id`` ← ``hit["_id"]`` (Vespa fully-qualified id).
    ``text``      ← ``title`` + ``description_snippet`` + ``content_snippet``,
                     with ``<hi>`` markup stripped.
    ``uri``       ← ``hit.get("url")``.
    ``source_date`` ← ``None`` (the search response doesn't carry dates).
    """
    from nuggetindex.pipeline.constructor import Document

    parts: list[str] = []
    title = hit.get("title") or ""
    if title:
        parts.append(title)
    for key in ("description_snippet", "content_snippet", "description", "content"):
        val = hit.get(key)
        if val:
            parts.append(_strip_highlights(str(val)))
    text = "\n".join(parts).strip() or (hit.get("_id") or "")
    return Document(
        source_id=str(hit["_id"]),
        text=text,
        uri=hit.get("url"),
        source_date=None,
    )


@dataclass
class VespaCorpus:
    """Adapter for Vespa-style BM25 REST search clusters.

    Endpoints used:

    * ``POST /api/v1/search/docs/{corpus}/search``           BM25 search
    * ``GET  /api/v1/cluster/corpora/{corpus}/statistics``   total doc count

    Keep this narrow: any Vespa-like BM25 HTTP endpoint that accepts
    ``{"query": str, "limit": int, "offset": int}`` POST bodies and
    returns ``{"hits": [{"_id", "title", "url", "description_snippet",
    ...}]}`` works. A generic ``HttpCorpus`` is a separate future adapter.

    Use as an async context manager so the underlying ``httpx.AsyncClient``
    is closed:

    .. code-block:: python

        async with VespaCorpus(base_url="http://...", corpus="my-corpus") as vc:
            docs = await vc.sample(mode="topic_diverse", n=500)
    """

    base_url: str
    corpus: str
    timeout: float = 30.0
    # Lazy: built on first use. Tests can inject an ``httpx.AsyncClient``
    # wired to an ``httpx.MockTransport`` to avoid any real network.
    http_client: Any | None = field(default=None)
    # Pool of broad stopword-ish queries to try when paginating via
    # ``_sample_uniform``. Some clusters strip single-letter tokens (so the
    # original literal "a" silently returned zero hits); trying a handful of
    # common words and keeping the first that returns results makes the
    # adapter robust against that analyzer drift. Also used as a defensive
    # fallback when ``_sample_topic_diverse`` comes back empty.
    broad_query_pool: tuple[str, ...] = (
        "the",
        "a",
        "is",
        "and",
        "of",
        "to",
        "in",
        "that",
    )

    def _ensure_client(self) -> httpx.AsyncClient:
        import httpx

        if self.http_client is None:
            self.http_client = httpx.AsyncClient(
                base_url=self.base_url.rstrip("/"),
                timeout=self.timeout,
            )
        return self.http_client

    # -- CorpusSource API ---------------------------------------------------

    async def sample(
        self,
        *,
        mode: Literal["topic_diverse", "uniform", "random_ids"],
        n: int,
    ) -> list[Document]:
        if mode == "topic_diverse":
            return await self._sample_topic_diverse(n)
        if mode == "uniform":
            return await self._sample_uniform(n)
        if mode == "random_ids":
            raise NotImplementedError(
                "VespaCorpus does not currently support random_ids sampling. "
                "Use mode='topic_diverse' (recommended) or mode='uniform'."
            )
        raise ValueError(f"unknown sample mode: {mode!r}")

    async def search(self, query: str, *, limit: int) -> list[Document]:
        return await self._search(query, limit=limit, offset=0)

    # -- internals ----------------------------------------------------------

    async def _sample_topic_diverse(self, n: int) -> list[Document]:
        """Query each topic-diverse seed with a per-query budget. Dedup by ``_id``.

        If *none* of the topic-diverse queries return any hits (shouldn't
        happen on a populated corpus, but we've been burned by analyzer
        quirks before), fall back to :meth:`_sample_uniform` so a deep
        cluster misconfiguration surfaces as a ``RuntimeError`` with a
        helpful knob hint rather than a silent empty bootstrap.
        """
        per_query = max(1, n // len(_TOPIC_DIVERSE_QUERIES)) + 1
        seen: set[str] = set()
        out: list[Document] = []
        for q in _TOPIC_DIVERSE_QUERIES:
            if len(out) >= n:
                break
            hits = await self._search(q, limit=per_query)
            for h in hits:
                if h.source_id in seen:
                    continue
                seen.add(h.source_id)
                out.append(h)
                if len(out) >= n:
                    break
        if not out:
            # Defensive fallback -- the topic pack returned nothing at all.
            return await self._sample_uniform(n)
        return out

    async def _sample_uniform(self, n: int) -> list[Document]:
        """Page through via offset with a broad stopword-ish query.

        Vespa's BM25 requires a non-empty query, and some clusters strip
        single-letter tokens (the original literal ``"a"`` returned zero
        hits on those clusters). We try every entry in
        :attr:`broad_query_pool` in turn and keep the first one that
        produces a non-empty first page, paginating from there.

        If *all* pool entries come back empty we raise ``RuntimeError``
        with a pointer at the ``broad_query_pool=`` knob so callers can
        pass their cluster's actual non-stopword vocabulary.
        """
        page_size = max(1, min(50, n))
        if not self.broad_query_pool:
            raise RuntimeError(
                "VespaCorpus._sample_uniform: broad_query_pool is empty; "
                "pass broad_query_pool=(...) with at least one non-stopword "
                "term your cluster's analyzer retains."
            )

        # Try each pool entry until one returns hits on page 1. Remember the
        # tried queries so the error message is informative.
        seed_query: str | None = None
        first_page: list[Document] = []
        tried: list[str] = []
        for q in self.broad_query_pool:
            tried.append(q)
            hits = await self._search(q, limit=page_size, offset=0)
            if hits:
                seed_query = q
                first_page = hits
                break

        if seed_query is None:
            raise RuntimeError(
                "VespaCorpus._sample_uniform: none of the broad_query_pool "
                f"entries returned any hits (tried {tried!r}). The cluster "
                "is either empty or its analyzer strips every term in the "
                "pool -- pass broad_query_pool=(...) with terms your cluster "
                "retains."
            )

        seen: set[str] = set()
        out: list[Document] = []
        for h in first_page:
            if h.source_id in seen:
                continue
            seen.add(h.source_id)
            out.append(h)
            if len(out) >= n:
                return out

        offset = page_size
        while len(out) < n and offset < 50_000:
            hits = await self._search(seed_query, limit=page_size, offset=offset)
            if not hits:
                break
            added_this_page = 0
            for h in hits:
                if h.source_id in seen:
                    continue
                seen.add(h.source_id)
                out.append(h)
                added_this_page += 1
                if len(out) >= n:
                    break
            offset += page_size
            # Safety: a backend that keeps returning the same ids forever
            # would otherwise loop until the 50k cap. Bail on a full page
            # of duplicates.
            if added_this_page == 0:
                break
        return out

    async def _search(
        self,
        query: str,
        *,
        limit: int,
        offset: int = 0,
    ) -> list[Document]:
        """POST ``/api/v1/search/docs/{corpus}/search`` and project hits."""
        client = self._ensure_client()
        body: dict[str, Any] = {
            "query": query,
            "limit": int(limit),
            "offset": int(offset),
        }
        resp = await client.post(
            f"/api/v1/search/docs/{self.corpus}/search",
            json=body,
        )
        resp.raise_for_status()
        payload = resp.json()
        hits = payload.get("hits") or []
        return [_hit_to_document(h) for h in hits]

    # -- lifecycle ----------------------------------------------------------

    async def aclose(self) -> None:
        if self.http_client is not None:
            await self.http_client.aclose()
            self.http_client = None

    async def __aenter__(self) -> VespaCorpus:
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.aclose()


__all__ = ["VespaCorpus"]
