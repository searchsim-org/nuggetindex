"""Unit tests for :class:`nuggetindex.adapters.elasticsearch.ElasticsearchCorpus`.

The adapter duck-types on the client's ``.search(index=..., body=..., size=...)``
method -- no real ``elasticsearch`` package is required. Tests use
:class:`MagicMock` / :class:`AsyncMock` to simulate both sync and async clients.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from nuggetindex.adapters import ElasticsearchCorpus


@pytest.mark.asyncio
async def test_search_translates_hits_to_documents():
    mock_client = MagicMock()
    mock_client.search = AsyncMock(
        return_value={
            "hits": {
                "hits": [
                    {"_id": "1", "_source": {"title": "t", "content": "c", "url": "u"}},
                    {"_id": "2", "_source": {"title": "t2", "content": "c2"}},
                ]
            },
        }
    )
    corpus = ElasticsearchCorpus(client=mock_client, index="docs")
    docs = await corpus.search("query", limit=10)
    assert len(docs) == 2
    assert docs[0].source_id == "1"
    assert docs[0].text == "t\nc"
    assert docs[0].uri == "u"
    assert docs[1].uri is None


@pytest.mark.asyncio
async def test_sample_topic_diverse_dedups_across_queries():
    call_count = {"n": 0}

    async def fake_search(**kwargs):
        call_count["n"] += 1
        return {
            "hits": {
                "hits": [
                    {"_id": "shared", "_source": {"title": "t", "content": "c"}},
                ]
            },
        }

    mock_client = MagicMock()
    mock_client.search = fake_search
    corpus = ElasticsearchCorpus(client=mock_client, index="docs")
    docs = await corpus.sample(mode="topic_diverse", n=5)
    assert len(docs) == 1
    assert call_count["n"] >= 1


@pytest.mark.asyncio
async def test_sample_uniform_uses_match_all():
    captured = {}

    async def fake_search(**kwargs):
        captured.update(kwargs)
        return {
            "hits": {
                "hits": [
                    {"_id": "u1", "_source": {"title": "t", "content": "c"}},
                ]
            }
        }

    mock_client = MagicMock()
    mock_client.search = fake_search
    corpus = ElasticsearchCorpus(client=mock_client, index="docs")
    await corpus.sample(mode="uniform", n=3)
    body = captured["body"]
    assert body == {"query": {"match_all": {}}}


@pytest.mark.asyncio
async def test_sample_random_ids_uses_function_score():
    captured = {}

    async def fake_search(**kwargs):
        captured.update(kwargs)
        return {"hits": {"hits": []}}

    mock_client = MagicMock()
    mock_client.search = fake_search
    corpus = ElasticsearchCorpus(client=mock_client, index="docs")
    await corpus.sample(mode="random_ids", n=3)
    body = captured["body"]
    assert "function_score" in body["query"]


@pytest.mark.asyncio
async def test_sync_client_also_works():
    def sync_search(**kwargs):
        return {
            "hits": {
                "hits": [
                    {"_id": "s1", "_source": {"title": "t", "content": "c"}},
                ]
            }
        }

    mock_client = MagicMock()
    mock_client.search = sync_search
    corpus = ElasticsearchCorpus(client=mock_client, index="docs")
    docs = await corpus.search("q", limit=5)
    assert len(docs) == 1


@pytest.mark.asyncio
async def test_custom_field_names():
    async def fake_search(**kwargs):
        return {
            "hits": {
                "hits": [
                    {
                        "_id": "1",
                        "_source": {
                            "heading": "mytitle",
                            "body": "mybody",
                            "link": "myurl",
                            "published": "2023-01-15T00:00:00Z",
                        },
                    },
                ]
            }
        }

    mock_client = MagicMock()
    mock_client.search = fake_search
    corpus = ElasticsearchCorpus(
        client=mock_client,
        index="docs",
        title_field="heading",
        text_field="body",
        url_field="link",
        date_field="published",
    )
    docs = await corpus.search("q", limit=1)
    assert docs[0].text == "mytitle\nmybody"
    assert docs[0].uri == "myurl"
    assert docs[0].source_date is not None
    assert docs[0].source_date.year == 2023
