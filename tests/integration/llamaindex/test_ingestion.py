"""Tests for ``NuggetTransformation`` (LlamaIndex IngestionPipeline component)."""
from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("llama_index.core")

from llama_index.core.schema import TextNode  # noqa: E402

from nuggetindex.integrations.llamaindex import NuggetTransformation  # noqa: E402
from nuggetindex.store.base import NuggetStore  # noqa: E402
from tests.fixtures import RuleBasedExtractor  # noqa: E402


@pytest.fixture
async def store(tmp_path: Path):
    s = NuggetStore(tmp_path / "ingest.db", extractor=RuleBasedExtractor())
    try:
        yield s
    finally:
        await s.aclose()


def test_is_transform_component() -> None:
    from llama_index.core.schema import TransformComponent

    assert issubclass(NuggetTransformation, TransformComponent)


@pytest.mark.asyncio
async def test_acall_ingests_nodes(store: NuggetStore) -> None:
    transform = NuggetTransformation(store=store)
    nodes = [
        TextNode(text="Sundar Pichai is CEO of Google.", id_="n1"),
        TextNode(text="Larry Page was a founder of Google.", id_="n2"),
    ]
    out = await transform.acall(nodes)
    # Nodes returned unchanged for downstream composition.
    assert list(out) == nodes
    # The store now has rule-based-extracted nuggets persisted.
    count = await store.acount()
    assert count >= 1


@pytest.mark.asyncio
async def test_acall_skips_blank_nodes(store: NuggetStore) -> None:
    transform = NuggetTransformation(store=store)
    nodes = [
        TextNode(text="   ", id_="empty"),
        TextNode(text="Sundar Pichai is CEO of Google.", id_="ok"),
    ]
    out = await transform.acall(nodes)
    assert list(out) == nodes
    # At most one passage's worth of nuggets — the blank node was skipped.
    passages = await store.backend.aget_passages({"empty", "ok"})
    # Only the non-blank id round-trips as a passage in the store.
    assert "ok" in passages
    assert "empty" not in passages


@pytest.mark.asyncio
async def test_acall_empty_list(store: NuggetStore) -> None:
    transform = NuggetTransformation(store=store)
    out = await transform.acall([])
    assert list(out) == []


@pytest.mark.asyncio
async def test_in_ingestion_pipeline(tmp_path: Path) -> None:
    """Compose the transformation inside a real ``IngestionPipeline``."""
    from llama_index.core.ingestion import IngestionPipeline

    s = NuggetStore(tmp_path / "pipeline.db", extractor=RuleBasedExtractor())
    try:
        transform = NuggetTransformation(store=s)
        pipeline = IngestionPipeline(transformations=[transform])
        in_nodes = [
            TextNode(text="Sundar Pichai is CEO of Google.", id_="p1"),
        ]
        out_nodes = await pipeline.arun(nodes=in_nodes)
        # Pipeline returns nodes unchanged.
        assert len(list(out_nodes)) == 1
        # And the ingest side-effect hit the store.
        assert await s.acount() >= 1
    finally:
        await s.aclose()
