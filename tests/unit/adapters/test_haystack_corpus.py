"""Unit tests for :class:`nuggetindex.adapters.haystack_corpus.HaystackCorpus`.

Uses an in-memory Haystack :class:`InMemoryDocumentStore` + in-memory BM25
retriever seeded with inline documents so no network is touched.
"""

from __future__ import annotations

import pytest

pytest.importorskip("haystack")

from haystack import Document as HaystackDocument  # noqa: E402
from haystack.components.retrievers.in_memory import (  # noqa: E402
    InMemoryBM25Retriever,
)
from haystack.document_stores.in_memory import InMemoryDocumentStore  # noqa: E402

from nuggetindex.adapters.haystack_corpus import HaystackCorpus  # noqa: E402


def _seed_store() -> InMemoryDocumentStore:
    store = InMemoryDocumentStore()
    store.write_documents(
        [
            HaystackDocument(
                id=f"h-{i}",
                content=text,
                meta={"url": f"https://haystack.test/{i}"},
            )
            for i, text in enumerate(
                [
                    "apple pie recipe with cinnamon",
                    "pizza dough tutorial for beginners",
                    "travel guide to Kyoto neighbourhoods",
                    "symptoms of early-onset arthritis",
                    "research study on climate resilience",
                    "how to fix a leaky kitchen faucet",
                    "interview with a startup founder",
                    "workout routine for marathon training",
                    "book review: 'The Mirror & the Light'",
                    "vaccine development timeline",
                ]
            )
        ]
    )
    return store


@pytest.mark.asyncio
async def test_uniform_sampling_returns_up_to_n_docs() -> None:
    """Uniform mode shuffles filter_documents() and slices to n."""
    store = _seed_store()
    corpus = HaystackCorpus(document_store=store, rng_seed=42)
    docs = await corpus.sample(mode="uniform", n=5)
    assert len(docs) == 5
    # All source_ids come from the seeded store.
    seeded_ids = {f"h-{i}" for i in range(10)}
    assert {d.source_id for d in docs}.issubset(seeded_ids)
    # The text round-trips from the Haystack content field.
    assert all(d.text for d in docs)
    # URI comes from meta['url'].
    assert all(d.uri and d.uri.startswith("https://haystack.test/") for d in docs)


@pytest.mark.asyncio
async def test_topic_diverse_without_retriever_falls_back_to_uniform() -> None:
    """Without a retriever, topic_diverse warns and degrades to uniform."""
    store = _seed_store()
    corpus = HaystackCorpus(document_store=store, retriever=None, rng_seed=7)
    with pytest.warns(UserWarning, match="retriever"):
        docs = await corpus.sample(mode="topic_diverse", n=4)
    assert 0 < len(docs) <= 4


@pytest.mark.asyncio
async def test_topic_diverse_with_retriever_dispatches_queries() -> None:
    """With a retriever, topic_diverse fans out and dedups by source_id."""
    store = _seed_store()
    retr = InMemoryBM25Retriever(document_store=store)

    call_queries: list[str] = []
    original_run = retr.run

    def _spy_run(query: str, top_k: int = 10):  # type: ignore[no-untyped-def]
        call_queries.append(query)
        return original_run(query=query, top_k=top_k)

    retr.run = _spy_run  # type: ignore[method-assign]

    corpus = HaystackCorpus(document_store=store, retriever=retr)
    docs = await corpus.sample(mode="topic_diverse", n=6)
    # Multiple queries were dispatched (at least 1, typically many).
    assert len(call_queries) >= 1
    # Dedup by source_id holds.
    ids = [d.source_id for d in docs]
    assert len(set(ids)) == len(ids)
    assert len(docs) <= 6


@pytest.mark.asyncio
async def test_search_requires_retriever() -> None:
    """Calling search() without a retriever raises RuntimeError."""
    store = _seed_store()
    corpus = HaystackCorpus(document_store=store, retriever=None)
    with pytest.raises(RuntimeError, match="retriever"):
        await corpus.search("recipe", limit=3)


@pytest.mark.asyncio
async def test_search_with_retriever_projects_hits() -> None:
    """search() delegates to the retriever and maps hits into Document."""
    store = _seed_store()
    retr = InMemoryBM25Retriever(document_store=store)
    corpus = HaystackCorpus(document_store=store, retriever=retr)

    hits = await corpus.search("apple pie", limit=2)
    assert 0 < len(hits) <= 2
    # The top hit is the apple-pie document.
    assert any("apple pie" in h.text.lower() for h in hits)


@pytest.mark.asyncio
async def test_random_ids_raises() -> None:
    """random_ids is not a Haystack capability we expose."""
    store = _seed_store()
    corpus = HaystackCorpus(document_store=store)
    with pytest.raises(NotImplementedError, match="random_ids"):
        await corpus.sample(mode="random_ids", n=5)
