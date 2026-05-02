"""Tests for LlamaIndex ``GovernancePostProcessor`` — the Tier-1 wedge."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

pytest.importorskip("llama_index.core")

from llama_index.core.schema import NodeWithScore, TextNode  # noqa: E402

from nuggetindex.integrations.llamaindex import (  # noqa: E402
    GovernancePostProcessor,
)
from tests.fixtures import RuleBasedExtractor  # noqa: E402


@pytest.fixture
def cache_path(tmp_path: Path) -> Path:
    return tmp_path / "gov_cache.db"


def _node(text: str, node_id: str | None = None) -> NodeWithScore:
    kwargs: dict[str, object] = {"text": text}
    if node_id is not None:
        kwargs["id_"] = node_id
    n = TextNode(**kwargs)
    return NodeWithScore(node=n, score=1.0)


def test_postprocessor_is_base_node_postprocessor() -> None:
    from llama_index.core.postprocessor.types import BaseNodePostprocessor

    assert issubclass(GovernancePostProcessor, BaseNodePostprocessor)


@pytest.mark.asyncio
async def test_postprocessor_passes_active_nodes(cache_path: Path) -> None:
    """With a rule-based extractor the fact extracts as ACTIVE — filter keeps it."""
    pp = GovernancePostProcessor(
        cache_path=cache_path,
        extractor=RuleBasedExtractor(),
        query_time=datetime(2020, 1, 1, tzinfo=UTC),
    )
    nodes = [_node("Sundar Pichai is CEO of Google.", node_id="d1")]
    out = await pp._apostprocess_nodes(nodes)
    assert len(out) == 1
    assert "Sundar Pichai" in out[0].node.get_content()
    await pp.aclose()


@pytest.mark.asyncio
async def test_postprocessor_empty_input(cache_path: Path) -> None:
    pp = GovernancePostProcessor(
        cache_path=cache_path,
        extractor=RuleBasedExtractor(),
        query_time=datetime(2020, 1, 1, tzinfo=UTC),
    )
    out = await pp._apostprocess_nodes([])
    assert out == []
    await pp.aclose()


@pytest.mark.asyncio
async def test_postprocessor_preserves_node_metadata(cache_path: Path) -> None:
    pp = GovernancePostProcessor(
        cache_path=cache_path,
        extractor=RuleBasedExtractor(),
        query_time=datetime(2020, 1, 1, tzinfo=UTC),
    )
    n = TextNode(text="Sundar Pichai is CEO of Google.", metadata={"custom": "x"})
    nws = NodeWithScore(node=n, score=0.5)
    out = await pp._apostprocess_nodes([nws])
    assert len(out) == 1
    assert out[0].node.metadata.get("custom") == "x"
    await pp.aclose()


def test_postprocess_nodes_sync_wrapper(cache_path: Path) -> None:
    """The sync ``_postprocess_nodes`` should work from outside a loop."""
    pp = GovernancePostProcessor(
        cache_path=cache_path,
        extractor=RuleBasedExtractor(),
        query_time=datetime(2020, 1, 1, tzinfo=UTC),
    )
    nodes = [_node("Sundar Pichai is CEO of Google.", node_id="d1")]
    out = pp._postprocess_nodes(nodes)
    assert len(out) == 1


@pytest.mark.asyncio
async def test_postprocessor_with_vector_store_index(cache_path: Path) -> None:
    """Smoke-test: wrap a VectorStoreIndex retriever's output with governance.

    We drive the retriever manually (no LLM) to keep the test offline,
    then hand the ``NodeWithScore`` list to the postprocessor directly —
    exactly what LlamaIndex's QueryEngine would do.
    """
    from llama_index.core import VectorStoreIndex
    from llama_index.core.embeddings import MockEmbedding
    from llama_index.core.schema import TextNode as LINode

    nodes = [
        LINode(
            text="Sundar Pichai is CEO of Google.",
            id_="v1",
        ),
        LINode(
            text="Larry Page was a founder of Google.",
            id_="v2",
        ),
    ]
    embed = MockEmbedding(embed_dim=8)
    index = VectorStoreIndex(nodes, embed_model=embed)
    retriever = index.as_retriever(similarity_top_k=2)
    retrieved = retriever.retrieve("Google")
    assert len(retrieved) > 0

    pp = GovernancePostProcessor(
        cache_path=cache_path,
        extractor=RuleBasedExtractor(),
        query_time=datetime(2020, 1, 1, tzinfo=UTC),
    )
    out = await pp._apostprocess_nodes(list(retrieved))
    # Rule-based extractor yields ACTIVE nuggets for these — nothing filtered.
    assert len(out) == len(retrieved)
    await pp.aclose()
