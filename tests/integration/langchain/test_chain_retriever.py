"""Tests for ``NuggetChainRetriever`` (Runnable-based LangChain chain adapter).

Mirrors the structure of ``test_retriever.py`` but exercises the three
chain kinds (succession / rename / joined) against a purpose-built chain
fixture store. Each returned ``Document`` carries chain-specific metadata
(``chain_position``, ``chain_type``, ``gap_seconds_to_prev``,
``edge_type_to_prev``) on top of the governance fields the regular
retriever emits.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

pytest.importorskip("langchain_core")

from nuggetindex.core.enums import LifecycleStatus, NuggetKind  # noqa: E402
from nuggetindex.core.models import (  # noqa: E402
    EpistemicState,
    FactTriple,
    Nugget,
    ProvenanceRecord,
    ValidityInterval,
)
from nuggetindex.store.base import NuggetStore  # noqa: E402


def _n(
    subject: str,
    predicate: str,
    object_: str,
    start_year: int,
    end_year: int | None = None,
    *,
    status: LifecycleStatus = LifecycleStatus.ACTIVE,
) -> Nugget:
    vi = ValidityInterval(
        start=datetime(start_year, 1, 1, tzinfo=UTC),
        end=datetime(end_year, 1, 1, tzinfo=UTC) if end_year else None,
    )
    return Nugget.new(
        kind=NuggetKind.SEMANTIC_FACT,
        fact=FactTriple(
            subject=subject,
            predicate=predicate,
            object=object_,
            text=f"{subject} {predicate} {object_}",
        ),
        validity=vi,
        epistemic=EpistemicState(status=status),
        provenance=(
            ProvenanceRecord(
                source_id=f"doc-{subject}-{predicate}-{object_}",
                evidence_span=f"{subject} {predicate} {object_}",
            ),
        ),
    )


@pytest.fixture
async def chain_store(tmp_path: Path):
    """Seed the same corpus used by ``test_chains_end_to_end.py``."""
    store = NuggetStore(tmp_path / "lc_chain.db")
    corpus = [
        _n("Google", "chiefExecutiveOfficer", "Schmidt", 2001, 2011),
        _n("Google", "chiefExecutiveOfficer", "Page", 2011, 2015),
        _n("Google", "chiefExecutiveOfficer", "Pichai", 2015, None),
        _n("Twitter Inc", "renamedTo", "X Corp", 2023, None),
        _n("X Corp", "chiefExecutiveOfficer", "Yaccarino", 2023, None),
        _n("Google", "parentCompany", "Alphabet", 2015, None),
        _n("Alphabet", "chiefExecutiveOfficer", "Pichai", 2019, None),
    ]
    for n in corpus:
        await store.aadd(n)
    try:
        yield store
    finally:
        await store.aclose()


def test_chain_retriever_is_runnable() -> None:
    from langchain_core.runnables import RunnableSerializable

    from nuggetindex.integrations.langchain import NuggetChainRetriever

    assert issubclass(NuggetChainRetriever, RunnableSerializable)


@pytest.mark.asyncio
async def test_ainvoke_succession_returns_ordered_documents(chain_store) -> None:
    from langchain_core.documents import Document

    from nuggetindex.integrations.langchain import NuggetChainRetriever

    retriever = NuggetChainRetriever(store=chain_store)
    docs = await retriever.ainvoke({"type": "succession", "subject": "Google", "predicate": "ceo"})
    assert all(isinstance(d, Document) for d in docs)
    assert [d.metadata["object"] for d in docs] == ["Schmidt", "Page", "Pichai"]
    # Chain-position and chain-type set on every doc.
    for i, d in enumerate(docs):
        assert d.metadata["chain_position"] == i
        assert d.metadata["chain_type"] == "succession"
    # First doc has no predecessor.
    assert docs[0].metadata["gap_seconds_to_prev"] is None
    assert docs[0].metadata["edge_type_to_prev"] is None
    # Subsequent docs surface edge metadata.
    assert docs[1].metadata["edge_type_to_prev"] == "succeeds"
    # Page started exactly when Schmidt's interval ended so gap is 0s.
    assert docs[1].metadata["gap_seconds_to_prev"] == 0.0
    # Governance fields also present (parity with regular retriever).
    for d in docs:
        for key in (
            "nugget_id",
            "subject",
            "predicate",
            "valid_from",
            "valid_until",
            "status",
        ):
            assert key in d.metadata


@pytest.mark.asyncio
async def test_ainvoke_succession_with_as_of(chain_store) -> None:
    from nuggetindex.integrations.langchain import NuggetChainRetriever

    retriever = NuggetChainRetriever(store=chain_store)
    docs = await retriever.ainvoke(
        {
            "type": "succession",
            "subject": "Google",
            "predicate": "ceo",
            "as_of": datetime(2013, 1, 1, tzinfo=UTC),
        }
    )
    assert [d.metadata["object"] for d in docs] == ["Schmidt", "Page"]


@pytest.mark.asyncio
async def test_ainvoke_rename_returns_forward_walk(chain_store) -> None:
    from nuggetindex.integrations.langchain import NuggetChainRetriever

    retriever = NuggetChainRetriever(store=chain_store)
    docs = await retriever.ainvoke({"type": "rename", "subject": "Twitter Inc"})
    assert [d.metadata["object"] for d in docs] == ["X Corp"]
    assert docs[0].metadata["chain_type"] == "rename"


@pytest.mark.asyncio
async def test_ainvoke_joined_chain(chain_store) -> None:
    from nuggetindex.integrations.langchain import NuggetChainRetriever

    retriever = NuggetChainRetriever(store=chain_store)
    docs = await retriever.ainvoke(
        {
            "type": "joined",
            "start": ("Google", "parentCompany"),
            "then": ["ceo"],
            "as_of": datetime(2020, 1, 1, tzinfo=UTC),
        }
    )
    assert len(docs) == 2
    assert docs[0].metadata["object"] == "Alphabet"
    assert docs[1].metadata["object"] == "Pichai"
    assert docs[1].metadata["edge_type_to_prev"] == "object_is_subject"
    for d in docs:
        assert d.metadata["chain_type"] == "joined"


@pytest.mark.asyncio
async def test_ainvoke_unknown_type_raises(chain_store) -> None:
    from nuggetindex.integrations.langchain import NuggetChainRetriever

    retriever = NuggetChainRetriever(store=chain_store)
    with pytest.raises(ValueError, match="unknown chain type"):
        await retriever.ainvoke({"type": "nonsense"})


@pytest.mark.asyncio
async def test_chain_retriever_composes_with_runnable_lambda(chain_store) -> None:
    """Smoke: ``retriever | RunnableLambda`` composes end-to-end."""
    from langchain_core.runnables import RunnableLambda

    from nuggetindex.integrations.langchain import NuggetChainRetriever

    retriever = NuggetChainRetriever(store=chain_store)
    chain = retriever | RunnableLambda(lambda docs: len(docs))
    count = await chain.ainvoke({"type": "succession", "subject": "Google", "predicate": "ceo"})
    assert count == 3
