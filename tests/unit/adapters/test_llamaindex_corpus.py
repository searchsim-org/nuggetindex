"""Unit tests for :class:`nuggetindex.adapters.llamaindex_corpus.LlamaIndexCorpus`.

Uses LlamaIndex's in-memory ``SimpleVectorStore`` + ``MockEmbedding`` so no
network and no real embedding model is touched.
"""

from __future__ import annotations

import pytest

pytest.importorskip("llama_index")

from llama_index.core import VectorStoreIndex  # noqa: E402
from llama_index.core.embeddings import MockEmbedding  # noqa: E402
from llama_index.core.schema import TextNode  # noqa: E402

from nuggetindex.adapters.llamaindex_corpus import LlamaIndexCorpus  # noqa: E402


def _make_nodes() -> list[TextNode]:
    texts = [
        "apple pie recipe with cinnamon",
        "pizza dough tutorial for beginners",
        "travel guide to Kyoto neighbourhoods",
        "symptoms of early-onset arthritis",
        "research study on climate resilience",
        "how to fix a leaky kitchen faucet",
        "interview with a startup founder",
        "workout routine for marathon training",
        "book review of 'The Mirror & the Light'",
        "vaccine development timeline",
    ]
    return [
        TextNode(
            id_=f"n-{i}",
            text=t,
            metadata={"file_path": f"/corpus/n-{i}.md"},
        )
        for i, t in enumerate(texts)
    ]


def _make_index() -> VectorStoreIndex:
    emb = MockEmbedding(embed_dim=8)
    return VectorStoreIndex(nodes=_make_nodes(), embed_model=emb)


@pytest.mark.asyncio
async def test_uniform_sampling_from_vector_store_index() -> None:
    """uniform pulls from index.docstore.docs and slices to n."""
    idx = _make_index()
    corpus = LlamaIndexCorpus(source=idx, rng_seed=42)
    docs = await corpus.sample(mode="uniform", n=4)
    assert len(docs) == 4
    seeded_ids = {f"n-{i}" for i in range(10)}
    assert {d.source_id for d in docs}.issubset(seeded_ids)
    # Metadata -> uri mapping (file_path wins over url).
    assert all(d.uri and d.uri.startswith("/corpus/") for d in docs)


@pytest.mark.asyncio
async def test_uniform_sampling_from_node_iterable() -> None:
    """A plain iterable of TextNode is also acceptable as ``source``."""
    nodes = _make_nodes()
    corpus = LlamaIndexCorpus(source=nodes, rng_seed=0)
    docs = await corpus.sample(mode="uniform", n=3)
    assert len(docs) == 3
    assert all(d.source_id.startswith("n-") for d in docs)


@pytest.mark.asyncio
async def test_topic_diverse_without_retriever_falls_back_with_warning() -> None:
    """Raw node iterable + topic_diverse -> warn + degrade to uniform."""
    nodes = _make_nodes()
    corpus = LlamaIndexCorpus(source=nodes, retriever=None, rng_seed=1)
    with pytest.warns(UserWarning, match="retriever"):
        docs = await corpus.sample(mode="topic_diverse", n=3)
    assert 0 < len(docs) <= 3


@pytest.mark.asyncio
async def test_topic_diverse_with_index_dispatches_queries() -> None:
    """A VectorStoreIndex auto-derives a retriever for topic_diverse."""
    idx = _make_index()
    corpus = LlamaIndexCorpus(source=idx)
    docs = await corpus.sample(mode="topic_diverse", n=5)
    # Dedup by source_id holds.
    ids = [d.source_id for d in docs]
    assert len(set(ids)) == len(ids)
    assert 0 < len(docs) <= 5


@pytest.mark.asyncio
async def test_search_requires_retriever_for_raw_iterable() -> None:
    """A raw iterable of nodes can't serve search() -- raises."""
    nodes = _make_nodes()
    corpus = LlamaIndexCorpus(source=nodes, retriever=None)
    with pytest.raises(RuntimeError, match="retriever"):
        await corpus.search("apple", limit=3)


@pytest.mark.asyncio
async def test_search_with_index_projects_hits() -> None:
    """search() delegates to as_retriever() and maps NodeWithScore -> Document."""
    idx = _make_index()
    corpus = LlamaIndexCorpus(source=idx)
    hits = await corpus.search("apple pie", limit=3)
    assert 0 < len(hits) <= 3
    assert all(h.source_id.startswith("n-") for h in hits)
    assert all(h.text for h in hits)


@pytest.mark.asyncio
async def test_random_ids_raises() -> None:
    """random_ids is not exposed."""
    idx = _make_index()
    corpus = LlamaIndexCorpus(source=idx)
    with pytest.raises(NotImplementedError, match="random_ids"):
        await corpus.sample(mode="random_ids", n=5)
