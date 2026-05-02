"""Tests for ``NuggetConstructionLoader`` (LangChain loader wrapper)."""
from __future__ import annotations

from collections.abc import AsyncIterator, Iterator

import pytest

langchain = pytest.importorskip("langchain_core")

from langchain_core.documents import Document  # noqa: E402

from nuggetindex.integrations.langchain import NuggetConstructionLoader  # noqa: E402


class _StubLoader:
    """Minimal LangChain-shaped loader with ``lazy_load``."""

    def __init__(self, docs: list[Document]) -> None:
        self._docs = docs

    def lazy_load(self) -> Iterator[Document]:
        yield from self._docs


class _StubLoaderSyncOnly:
    """Loader that only implements ``load()``."""

    def __init__(self, docs: list[Document]) -> None:
        self._docs = docs

    def load(self) -> list[Document]:
        return list(self._docs)


class _StubAsyncLoader:
    def __init__(self, docs: list[Document]) -> None:
        self._docs = docs

    def lazy_load(self) -> Iterator[Document]:
        yield from self._docs

    async def alazy_load(self) -> AsyncIterator[Document]:
        for d in self._docs:
            yield d


def _docs() -> list[Document]:
    return [
        Document(page_content="first doc", metadata={"id": "a"}),
        Document(page_content="second doc", metadata={"id": "b"}),
    ]


def test_lazy_load_yields_marked_documents() -> None:
    loader = NuggetConstructionLoader(base_loader=_StubLoader(_docs()))
    out = list(loader.lazy_load())
    assert len(out) == 2
    assert all(isinstance(d, Document) for d in out)
    assert all(d.metadata.get("nuggetindex_ingested") is True for d in out)
    # Original metadata survives.
    assert [d.metadata.get("id") for d in out] == ["a", "b"]


def test_load_returns_marked_documents() -> None:
    loader = NuggetConstructionLoader(base_loader=_StubLoader(_docs()))
    out = loader.load()
    assert len(out) == 2
    assert all(d.metadata["nuggetindex_ingested"] is True for d in out)


def test_load_fallback_for_sync_only_loader() -> None:
    loader = NuggetConstructionLoader(base_loader=_StubLoaderSyncOnly(_docs()))
    out = loader.load()
    assert len(out) == 2
    assert all(d.metadata["nuggetindex_ingested"] is True for d in out)


@pytest.mark.asyncio
async def test_alazy_load_uses_native_async_when_available() -> None:
    loader = NuggetConstructionLoader(base_loader=_StubAsyncLoader(_docs()))
    out = [d async for d in loader.alazy_load()]
    assert len(out) == 2
    assert all(d.metadata["nuggetindex_ingested"] is True for d in out)


@pytest.mark.asyncio
async def test_alazy_load_falls_back_to_sync() -> None:
    loader = NuggetConstructionLoader(base_loader=_StubLoader(_docs()))
    out = [d async for d in loader.alazy_load()]
    assert len(out) == 2


def test_does_not_overwrite_existing_flag() -> None:
    """If caller already marked a doc, we preserve their choice."""
    doc = Document(
        page_content="already", metadata={"nuggetindex_ingested": False}
    )
    loader = NuggetConstructionLoader(base_loader=_StubLoader([doc]))
    out = loader.load()
    assert out[0].metadata["nuggetindex_ingested"] is False


def test_preserves_empty_metadata() -> None:
    doc = Document(page_content="no-meta")
    loader = NuggetConstructionLoader(base_loader=_StubLoader([doc]))
    out = loader.load()
    assert out[0].metadata["nuggetindex_ingested"] is True
