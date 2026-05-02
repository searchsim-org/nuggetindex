from unittest.mock import MagicMock

import pytest

from nuggetindex.adapters import PineconeCorpus


def _fake_embed(text: str) -> list[float]:
    return [0.1, 0.2, 0.3]


def _match(id_, metadata, score=0.9):
    return {"id": id_, "metadata": metadata, "score": score}


@pytest.mark.asyncio
async def test_search_translates_matches_to_documents():
    mock_index = MagicMock()
    mock_index.query = MagicMock(
        return_value={
            "matches": [
                _match("1", {"title": "t", "content": "c", "url": "u"}),
                _match("2", {"title": "t2", "content": "c2"}),
            ],
        }
    )
    corpus = PineconeCorpus(index=mock_index, embedder=_fake_embed)
    docs = await corpus.search("query", limit=5)
    assert len(docs) == 2
    assert docs[0].source_id == "1"
    assert docs[0].text == "t\nc"
    assert docs[0].uri == "u"
    call = mock_index.query.call_args
    assert call.kwargs["vector"] == [0.1, 0.2, 0.3]
    assert call.kwargs["top_k"] == 5
    assert call.kwargs["include_metadata"] is True


@pytest.mark.asyncio
async def test_sample_random_ids_raises():
    corpus = PineconeCorpus(index=MagicMock(), embedder=_fake_embed)
    with pytest.raises(NotImplementedError):
        await corpus.sample(mode="random_ids", n=5)


@pytest.mark.asyncio
async def test_sample_topic_diverse_uses_random_vectors():
    calls = []

    def fake_query(**kwargs):
        calls.append(kwargs)
        return {"matches": [_match(f"p{len(calls)}", {"title": "t", "content": "c"})]}

    mock_index = MagicMock()
    mock_index.query = fake_query
    corpus = PineconeCorpus(index=mock_index, embedder=_fake_embed)
    docs = await corpus.sample(mode="topic_diverse", n=3)
    assert len(docs) >= 1
    assert len(calls) >= 1


@pytest.mark.asyncio
async def test_sample_uniform_lists_ids_then_fetches():
    """Pinecone has no 'match all'. `list()` gives IDs, `fetch()` returns records."""
    list_calls = []
    fetch_calls = []

    def fake_list(**kwargs):
        list_calls.append(kwargs)
        return [{"id": "a"}, {"id": "b"}]

    def fake_fetch(**kwargs):
        fetch_calls.append(kwargs)
        return {
            "vectors": {
                "a": {"id": "a", "metadata": {"title": "ta", "content": "ca"}},
                "b": {"id": "b", "metadata": {"title": "tb", "content": "cb"}},
            },
        }

    mock_index = MagicMock()
    mock_index.list = fake_list
    mock_index.fetch = fake_fetch
    corpus = PineconeCorpus(index=mock_index, embedder=_fake_embed)
    docs = await corpus.sample(mode="uniform", n=2)
    assert len(docs) == 2
    assert {d.source_id for d in docs} == {"a", "b"}


@pytest.mark.asyncio
async def test_custom_field_names():
    def fake_query(**kwargs):
        return {
            "matches": [
                _match(
                    "1",
                    {
                        "heading": "h",
                        "body": "b",
                        "link": "l",
                        "published": "2023-05-10T00:00:00Z",
                    },
                )
            ]
        }

    mock_index = MagicMock()
    mock_index.query = fake_query
    corpus = PineconeCorpus(
        index=mock_index,
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
    assert docs[0].source_date.year == 2023
