"""Tests for ``NuggetIndexRetriever`` (BaseRetriever-based LlamaIndex adapter)."""
from __future__ import annotations

from datetime import UTC, datetime

import pytest

pytest.importorskip("llama_index.core")


def test_retriever_is_base_retriever() -> None:
    from llama_index.core.retrievers import BaseRetriever

    from nuggetindex.integrations.llamaindex import NuggetIndexRetriever

    assert issubclass(NuggetIndexRetriever, BaseRetriever)


@pytest.mark.asyncio
async def test_aretrieve_returns_nodes_with_score(populated_store) -> None:
    from llama_index.core.schema import NodeWithScore, QueryBundle, TextNode

    from nuggetindex.integrations.llamaindex import NuggetIndexRetriever

    retriever = NuggetIndexRetriever(
        store=populated_store,
        query_time=datetime(2020, 1, 1, tzinfo=UTC),
        top_k=5,
    )
    nodes = await retriever._aretrieve(QueryBundle(query_str="Google CEO"))
    assert len(nodes) >= 1
    assert all(isinstance(n, NodeWithScore) for n in nodes)
    assert all(isinstance(n.node, TextNode) for n in nodes)
    # Content is the nugget's fact text.
    assert any(
        "Google" in n.node.get_content() or "CEO" in n.node.get_content() for n in nodes
    )
    # Required metadata fields are present.
    for n in nodes:
        for key in (
            "nugget_id",
            "subject",
            "predicate",
            "object",
            "valid_from",
            "valid_until",
            "status",
            "confidence",
            "source",
            "evidence",
            "retrieval_score",
        ):
            assert key in n.node.metadata


@pytest.mark.asyncio
async def test_contested_nodes_get_disputed_prefix(populated_store) -> None:
    from llama_index.core.schema import QueryBundle

    from nuggetindex.integrations.llamaindex import NuggetIndexRetriever

    retriever = NuggetIndexRetriever(
        store=populated_store,
        query_time=datetime(2020, 1, 1, tzinfo=UTC),
        view="active_contested",
        top_k=10,
        flag_contested=True,
    )
    nodes = await retriever._aretrieve(QueryBundle(query_str="Foo"))
    contested = [n for n in nodes if n.node.metadata["status"] == "contested"]
    assert contested, "expected at least one contested node in fixture"
    for n in contested:
        assert n.node.get_content().startswith("[DISPUTED] ")


@pytest.mark.asyncio
async def test_flag_contested_false_omits_prefix(populated_store) -> None:
    from llama_index.core.schema import QueryBundle

    from nuggetindex.integrations.llamaindex import NuggetIndexRetriever

    retriever = NuggetIndexRetriever(
        store=populated_store,
        query_time=datetime(2020, 1, 1, tzinfo=UTC),
        view="active_contested",
        top_k=10,
        flag_contested=False,
    )
    nodes = await retriever._aretrieve(QueryBundle(query_str="Foo"))
    for n in nodes:
        assert not n.node.get_content().startswith("[DISPUTED] ")


@pytest.mark.asyncio
async def test_aretrieve_public_entry_point(populated_store) -> None:
    """``retriever.aretrieve(...)`` (the public async API) should work end-to-end."""
    from nuggetindex.integrations.llamaindex import NuggetIndexRetriever

    retriever = NuggetIndexRetriever(
        store=populated_store,
        query_time=datetime(2020, 1, 1, tzinfo=UTC),
        top_k=5,
    )
    nodes = await retriever.aretrieve("Google CEO")
    assert len(nodes) >= 1
    for n in nodes:
        assert n.score is not None


def test_sync_retrieve_wrapper(tmp_path) -> None:
    """``retriever.retrieve(...)`` should work from sync context."""
    import asyncio
    from datetime import UTC, datetime

    from nuggetindex.core.enums import LifecycleStatus, NuggetKind
    from nuggetindex.core.models import (
        EpistemicState,
        FactTriple,
        Nugget,
        ProvenanceRecord,
        ValidityInterval,
    )
    from nuggetindex.integrations.llamaindex import NuggetIndexRetriever
    from nuggetindex.store.base import NuggetStore

    # Build the store synchronously (outside any running loop) so the sync
    # wrapper's ``asyncio.run`` call doesn't collide with an active loop.
    async def _seed() -> NuggetStore:
        s = NuggetStore(tmp_path / "sync.db")
        await s.backend.aupsert_passage("d1", None, "Sundar Pichai is CEO of Google.")
        await s.aadd(
            Nugget.new(
                kind=NuggetKind.SEMANTIC_FACT,
                fact=FactTriple(
                    subject="Google",
                    predicate="ceo",
                    object="Sundar Pichai",
                    text="Sundar Pichai is CEO of Google.",
                ),
                validity=ValidityInterval(start=datetime(2019, 1, 1, tzinfo=UTC)),
                epistemic=EpistemicState(
                    status=LifecycleStatus.ACTIVE, confidence=0.9
                ),
                provenance=(
                    ProvenanceRecord(
                        source_id="d1",
                        evidence_span="Sundar Pichai is CEO of Google.",
                    ),
                ),
            )
        )
        return s

    store = asyncio.run(_seed())
    retriever = NuggetIndexRetriever(
        store=store,
        query_time=datetime(2020, 1, 1, tzinfo=UTC),
        top_k=3,
    )
    nodes = retriever.retrieve("Google")
    assert len(nodes) >= 1

    asyncio.run(store.aclose())
