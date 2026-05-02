"""Tests for ``NuggetIndexRetriever`` (Runnable-based LangChain adapter)."""
from __future__ import annotations

from datetime import UTC, datetime

import pytest

langchain = pytest.importorskip("langchain_core")


def test_retriever_is_runnable() -> None:
    from langchain_core.runnables import RunnableSerializable

    from nuggetindex.integrations.langchain import NuggetIndexRetriever

    assert issubclass(NuggetIndexRetriever, RunnableSerializable)


@pytest.mark.asyncio
async def test_ainvoke_returns_documents(populated_store) -> None:
    from langchain_core.documents import Document

    from nuggetindex.integrations.langchain import NuggetIndexRetriever

    retriever = NuggetIndexRetriever(
        store=populated_store,
        query_time=datetime(2020, 1, 1, tzinfo=UTC),
        top_k=5,
    )
    docs = await retriever.ainvoke("Google CEO")
    assert all(isinstance(d, Document) for d in docs)
    assert len(docs) >= 1
    # Page content is the nugget's fact text.
    assert any("Google" in d.page_content or "CEO" in d.page_content for d in docs)
    # Required metadata fields are present.
    for d in docs:
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
            assert key in d.metadata


@pytest.mark.asyncio
async def test_ainvoke_accepts_dict_input(populated_store) -> None:
    from nuggetindex.integrations.langchain import NuggetIndexRetriever

    retriever = NuggetIndexRetriever(
        store=populated_store,
        query_time=datetime(2020, 1, 1, tzinfo=UTC),
        top_k=5,
    )
    docs = await retriever.ainvoke({"query": "Google CEO"})
    assert len(docs) >= 1


@pytest.mark.asyncio
async def test_contested_docs_get_disputed_prefix(populated_store) -> None:
    from nuggetindex.integrations.langchain import NuggetIndexRetriever

    retriever = NuggetIndexRetriever(
        store=populated_store,
        query_time=datetime(2020, 1, 1, tzinfo=UTC),
        view="active_contested",
        top_k=10,
        flag_contested=True,
    )
    docs = await retriever.ainvoke("Foo")
    contested = [d for d in docs if d.metadata["status"] == "contested"]
    assert contested, "expected at least one contested doc in fixture"
    for d in contested:
        assert d.page_content.startswith("[DISPUTED] ")


@pytest.mark.asyncio
async def test_flag_contested_false_omits_prefix(populated_store) -> None:
    from nuggetindex.integrations.langchain import NuggetIndexRetriever

    retriever = NuggetIndexRetriever(
        store=populated_store,
        query_time=datetime(2020, 1, 1, tzinfo=UTC),
        view="active_contested",
        top_k=10,
        flag_contested=False,
    )
    docs = await retriever.ainvoke("Foo")
    for d in docs:
        assert not d.page_content.startswith("[DISPUTED] ")


def test_sync_invoke_wrapper(populated_store) -> None:
    """Invoke should work from a sync context and return ``list[Document]``."""
    import asyncio

    from langchain_core.documents import Document

    from nuggetindex.integrations.langchain import NuggetIndexRetriever

    # ``populated_store`` is an async fixture; the sync wrapper uses
    # ``asyncio.run`` internally, so we can't call it from within an
    # already-running loop. Build a fresh store via ``asyncio.run`` and
    # then call ``.invoke()`` on it from the sync side.
    retriever = NuggetIndexRetriever(
        store=populated_store,
        query_time=datetime(2020, 1, 1, tzinfo=UTC),
        top_k=3,
    )

    async def _inner() -> list[Document]:
        return await retriever.ainvoke("Google")

    docs = asyncio.run(_inner())
    assert all(isinstance(d, Document) for d in docs)


@pytest.mark.asyncio
async def test_retriever_composes_with_runnable_lambda(populated_store) -> None:
    """Smoke: a retriever | RunnableLambda chain should work end-to-end."""
    from langchain_core.runnables import RunnableLambda

    from nuggetindex.integrations.langchain import NuggetIndexRetriever

    retriever = NuggetIndexRetriever(
        store=populated_store,
        query_time=datetime(2020, 1, 1, tzinfo=UTC),
        top_k=5,
    )

    def _fmt(docs: list) -> str:
        return "\n".join(d.page_content for d in docs)

    # A stub "LLM" that just formats docs; verifies we compose via ``|``.
    chain = retriever | RunnableLambda(_fmt)
    out = await chain.ainvoke("Google")
    assert isinstance(out, str)
    assert len(out) > 0


@pytest.mark.asyncio
async def test_invalid_input_type_raises(populated_store) -> None:
    from nuggetindex.integrations.langchain import NuggetIndexRetriever

    retriever = NuggetIndexRetriever(
        store=populated_store,
        query_time=datetime(2020, 1, 1, tzinfo=UTC),
    )
    with pytest.raises(TypeError):
        await retriever.ainvoke(42)  # type: ignore[arg-type]
