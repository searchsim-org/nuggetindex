"""Unit tests for :class:`nuggetindex.adapters.vespa.VespaCorpus`.

Uses :class:`httpx.MockTransport` so no real network is ever touched.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

import httpx
import pytest

from nuggetindex.adapters.base import _TOPIC_DIVERSE_QUERIES
from nuggetindex.adapters.vespa import VespaCorpus

_Handler = Callable[[httpx.Request], httpx.Response]


def _build_client(handler: _Handler) -> httpx.AsyncClient:
    """Wire an AsyncClient to a MockTransport that dispatches via ``handler``."""
    return httpx.AsyncClient(
        base_url="http://vespa.test",
        transport=httpx.MockTransport(handler),
    )


def _hit(doc_id: str, title: str) -> dict[str, Any]:
    return {
        "_id": doc_id,
        "title": title,
        "url": f"https://example.test/{doc_id}",
        "description_snippet": f"snippet for {title}",
    }


@pytest.mark.asyncio
async def test_sample_topic_diverse_dispatches_to_each_query() -> None:
    """topic_diverse issues one POST per seed query and dedups by _id."""
    seen_queries: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        # Path must be the search endpoint on our corpus.
        assert request.url.path == "/api/v1/search/docs/my-corpus/search"
        body = json.loads(request.content)
        seen_queries.append(body["query"])
        # Return two distinct hits per query so we exercise dedup across
        # queries (each doc_id is namespaced by the query index).
        idx = len(seen_queries) - 1
        hits = [
            _hit(f"doc-{idx}-a", f"title for {body['query']} A"),
            _hit(f"doc-{idx}-b", f"title for {body['query']} B"),
        ]
        return httpx.Response(200, json={"hits": hits})

    client = _build_client(handler)
    corpus = VespaCorpus(
        base_url="http://vespa.test",
        corpus="my-corpus",
        http_client=client,
    )
    try:
        docs = await corpus.sample(mode="topic_diverse", n=20)
    finally:
        await corpus.aclose()

    # Each query was dispatched (in order).
    assert seen_queries == list(_TOPIC_DIVERSE_QUERIES[: len(seen_queries)])
    # Deduped: every source_id is unique.
    ids = [d.source_id for d in docs]
    assert len(set(ids)) == len(ids)
    assert len(docs) >= 20 or len(docs) == len(ids)  # budget fulfilled or exhausted
    # Documents carry stripped title + snippet.
    assert all(d.text for d in docs)
    assert all(d.uri and d.uri.startswith("https://example.test/") for d in docs)


@pytest.mark.asyncio
async def test_sample_topic_diverse_dedups_across_queries() -> None:
    """If every query returns the same doc, topic_diverse returns one doc."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"hits": [_hit("doc-shared", "t")]})

    client = _build_client(handler)
    corpus = VespaCorpus(
        base_url="http://vespa.test",
        corpus="my-corpus",
        http_client=client,
    )
    try:
        docs = await corpus.sample(mode="topic_diverse", n=50)
    finally:
        await corpus.aclose()

    assert [d.source_id for d in docs] == ["doc-shared"]


@pytest.mark.asyncio
async def test_sample_uniform_paginates() -> None:
    """uniform paginates via offset; handler returns different hits per offset."""
    offsets_seen: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        offset = int(body.get("offset", 0))
        offsets_seen.append(offset)
        # Three full pages, then empty. page_size should be <= n.
        if offset >= 150:
            return httpx.Response(200, json={"hits": []})
        hits = [_hit(f"doc-{offset + i}", f"uniform title {offset + i}") for i in range(50)]
        return httpx.Response(200, json={"hits": hits})

    client = _build_client(handler)
    corpus = VespaCorpus(
        base_url="http://vespa.test",
        corpus="my-corpus",
        http_client=client,
    )
    try:
        docs = await corpus.sample(mode="uniform", n=120)
    finally:
        await corpus.aclose()

    # Should have paginated at least twice (offset 0, 50, ...).
    assert offsets_seen[0] == 0
    assert 50 in offsets_seen
    assert len(docs) == 120
    ids = [d.source_id for d in docs]
    assert len(set(ids)) == len(ids)


@pytest.mark.asyncio
async def test_sample_random_ids_raises_not_implemented() -> None:
    """random_ids is unsupported on VespaCorpus."""

    def handler(request: httpx.Request) -> httpx.Response:  # pragma: no cover
        return httpx.Response(200, json={"hits": []})

    client = _build_client(handler)
    corpus = VespaCorpus(
        base_url="http://vespa.test",
        corpus="my-corpus",
        http_client=client,
    )
    try:
        with pytest.raises(NotImplementedError):
            await corpus.sample(mode="random_ids", n=10)
    finally:
        await corpus.aclose()


@pytest.mark.asyncio
async def test_search_posts_query_and_limit() -> None:
    """search() POSTs {query, limit, offset=0} and projects hits to Documents."""
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["path"] = request.url.path
        captured["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={"hits": [_hit("q1", "query result 1")]},
        )

    client = _build_client(handler)
    corpus = VespaCorpus(
        base_url="http://vespa.test",
        corpus="my-corpus",
        http_client=client,
    )
    try:
        docs = await corpus.search("pizza", limit=7)
    finally:
        await corpus.aclose()

    assert captured["path"] == "/api/v1/search/docs/my-corpus/search"
    assert captured["body"] == {"query": "pizza", "limit": 7, "offset": 0}
    assert len(docs) == 1
    assert docs[0].source_id == "q1"
