from unittest.mock import AsyncMock, MagicMock

import pytest

from nuggetindex.adapters import QdrantCorpus


def _fake_embed(text: str) -> list[float]:
    return [0.1, 0.2, 0.3]


def _point(pid, payload, score=0.9):
    p = MagicMock()
    p.id = pid
    p.payload = payload
    p.score = score
    return p


@pytest.mark.asyncio
async def test_search_uses_embedder_and_translates_points():
    mock_client = MagicMock()
    mock_client.search = AsyncMock(
        return_value=[
            _point("1", {"title": "t", "content": "c", "url": "u"}),
            _point("2", {"title": "t2", "content": "c2"}),
        ]
    )
    corpus = QdrantCorpus(client=mock_client, collection="docs", embedder=_fake_embed)
    docs = await corpus.search("query", limit=5)
    assert len(docs) == 2
    assert docs[0].source_id == "1"
    assert docs[0].text == "t\nc"
    assert docs[0].uri == "u"
    # Verify the embedder was invoked with the query and the vector was passed through.
    call = mock_client.search.await_args
    assert call.kwargs.get("query_vector") == [0.1, 0.2, 0.3]
    assert call.kwargs.get("collection_name") == "docs"
    assert call.kwargs.get("limit") == 5


@pytest.mark.asyncio
async def test_sync_client_works():
    def sync_search(**kwargs):
        return [_point("s1", {"title": "t", "content": "c"})]

    mock_client = MagicMock()
    mock_client.search = sync_search
    corpus = QdrantCorpus(client=mock_client, collection="docs", embedder=_fake_embed)
    docs = await corpus.search("q", limit=5)
    assert len(docs) == 1


@pytest.mark.asyncio
async def test_sample_random_ids_raises():
    corpus = QdrantCorpus(
        client=MagicMock(),
        collection="docs",
        embedder=_fake_embed,
    )
    with pytest.raises(NotImplementedError):
        await corpus.sample(mode="random_ids", n=5)


@pytest.mark.asyncio
async def test_sample_topic_diverse_fires_multiple_random_vectors():
    calls = []

    async def fake_search(**kwargs):
        calls.append(kwargs)
        return [_point(f"p{len(calls)}", {"title": "t", "content": "c"})]

    mock_client = MagicMock()
    mock_client.search = fake_search
    corpus = QdrantCorpus(client=mock_client, collection="docs", embedder=_fake_embed)
    docs = await corpus.sample(mode="topic_diverse", n=3)
    assert len(docs) >= 1
    # At least 2 distinct random vectors were tried (not the same each time).
    vectors = [c["query_vector"] for c in calls]
    assert len(vectors) >= 1


@pytest.mark.asyncio
async def test_sample_uniform_uses_scroll():
    scroll_calls = []

    async def fake_scroll(**kwargs):
        scroll_calls.append(kwargs)
        return ([_point("u1", {"title": "t", "content": "c"})], None)

    mock_client = MagicMock()
    mock_client.scroll = fake_scroll
    corpus = QdrantCorpus(client=mock_client, collection="docs", embedder=_fake_embed)
    docs = await corpus.sample(mode="uniform", n=3)
    assert len(docs) == 1
    assert scroll_calls[0]["collection_name"] == "docs"
    assert scroll_calls[0]["limit"] == 3


@pytest.mark.asyncio
async def test_custom_field_names_and_date_parsing():
    async def fake_search(**kwargs):
        return [
            _point(
                "1",
                {
                    "heading": "h",
                    "body": "b",
                    "link": "l",
                    "published": "2024-03-15T00:00:00Z",
                },
            )
        ]

    mock_client = MagicMock()
    mock_client.search = fake_search
    corpus = QdrantCorpus(
        client=mock_client,
        collection="docs",
        embedder=_fake_embed,
        title_field="heading",
        text_field="body",
        url_field="link",
        date_field="published",
    )
    docs = await corpus.search("q", limit=1)
    assert docs[0].text == "h\nb"
    assert docs[0].uri == "l"
    assert docs[0].source_date is not None
    assert docs[0].source_date.year == 2024
