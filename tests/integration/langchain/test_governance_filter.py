"""Tests for ``GovernanceFilter`` (wraps any LangChain retriever)."""
from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

langchain = pytest.importorskip("langchain_core")

from langchain_core.documents import Document  # noqa: E402
from langchain_core.runnables import RunnableLambda  # noqa: E402

from nuggetindex.governance import GovernancePostProcessor  # noqa: E402
from nuggetindex.integrations.langchain import GovernanceFilter  # noqa: E402
from tests.fixtures import RuleBasedExtractor  # noqa: E402


class _StubRetriever:
    """A minimal LangChain-shaped retriever yielding canned docs."""

    def __init__(self, docs: list[Document]) -> None:
        self._docs = docs

    async def ainvoke(
        self, input: Any, config: Any = None, **kwargs: Any
    ) -> list[Document]:
        return list(self._docs)


@pytest.fixture
def cache_path(tmp_path: Path) -> Path:
    return tmp_path / "gov_cache.db"


@pytest.mark.asyncio
async def test_governance_filter_passes_through_active_docs(cache_path: Path) -> None:
    """With a rule-based extractor the fact extracts as ACTIVE — filter keeps it."""
    postprocessor = GovernancePostProcessor(
        cache_path=cache_path,
        extractor=RuleBasedExtractor(),
        query_time=datetime(2020, 1, 1, tzinfo=UTC),
    )
    base = _StubRetriever(
        [
            Document(
                page_content="Sundar Pichai is CEO of Google.",
                metadata={"source": "d1"},
            ),
        ]
    )
    filt = GovernanceFilter(base_retriever=base, postprocessor=postprocessor)
    out = await filt.ainvoke("Who is CEO?")
    assert len(out) == 1
    assert "Sundar Pichai" in out[0].page_content


@pytest.mark.asyncio
async def test_governance_filter_handles_missing_source_metadata(
    cache_path: Path,
) -> None:
    """When a base doc has no ``source``, we synthesize an id and still filter."""
    postprocessor = GovernancePostProcessor(
        cache_path=cache_path,
        extractor=RuleBasedExtractor(),
        query_time=datetime(2020, 1, 1, tzinfo=UTC),
    )
    base = _StubRetriever(
        [
            Document(page_content="Sundar Pichai is CEO of Google.", metadata={}),
        ]
    )
    filt = GovernanceFilter(base_retriever=base, postprocessor=postprocessor)
    out = await filt.ainvoke("anything")
    # No source metadata, but governance still lets the (ACTIVE) passage through.
    assert len(out) == 1
    assert "Pichai" in out[0].page_content


@pytest.mark.asyncio
async def test_governance_filter_empty_input(cache_path: Path) -> None:
    postprocessor = GovernancePostProcessor(
        cache_path=cache_path,
        extractor=RuleBasedExtractor(),
        query_time=datetime(2020, 1, 1, tzinfo=UTC),
    )
    base = _StubRetriever([])
    filt = GovernanceFilter(base_retriever=base, postprocessor=postprocessor)
    assert await filt.ainvoke("q") == []


@pytest.mark.asyncio
async def test_governance_filter_is_runnable_serializable() -> None:
    from langchain_core.runnables import RunnableSerializable

    assert issubclass(GovernanceFilter, RunnableSerializable)


@pytest.mark.asyncio
async def test_governance_filter_composes_with_lambda(cache_path: Path) -> None:
    """`retriever | governance | format` style composition."""
    postprocessor = GovernancePostProcessor(
        cache_path=cache_path,
        extractor=RuleBasedExtractor(),
        query_time=datetime(2020, 1, 1, tzinfo=UTC),
    )
    base = _StubRetriever(
        [Document(page_content="Sundar Pichai is CEO of Google.", metadata={"source": "d1"})]
    )
    filt = GovernanceFilter(base_retriever=base, postprocessor=postprocessor)
    chain = filt | RunnableLambda(lambda docs: [d.page_content for d in docs])
    out = await chain.ainvoke("q")
    assert out == ["Sundar Pichai is CEO of Google."]
