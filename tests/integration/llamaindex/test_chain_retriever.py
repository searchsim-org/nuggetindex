"""Tests for ``NuggetChainRetriever`` (BaseRetriever-based LlamaIndex adapter).

Mirrors the LangChain chain-retriever tests: exercises succession / rename /
joined chains against a purpose-built store fixture. Supports both the
primary passthrough API (``await retriever.achain_succession(...)``) and
the ``QueryBundle`` fallback (``_aretrieve(QueryBundle(query_str=json_spec))``).
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

pytest.importorskip("llama_index.core")

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
    store = NuggetStore(tmp_path / "li_chain.db")
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


def test_chain_retriever_is_base_retriever() -> None:
    from llama_index.core.retrievers import BaseRetriever

    from nuggetindex.integrations.llamaindex import NuggetChainRetriever

    assert issubclass(NuggetChainRetriever, BaseRetriever)


@pytest.mark.asyncio
async def test_achain_succession_passthrough(chain_store) -> None:
    from llama_index.core.schema import NodeWithScore, TextNode

    from nuggetindex.integrations.llamaindex import NuggetChainRetriever

    retriever = NuggetChainRetriever(store=chain_store)
    nodes = await retriever.achain_succession(subject="Google", predicate="ceo")
    assert len(nodes) == 3
    assert all(isinstance(n, NodeWithScore) for n in nodes)
    assert all(isinstance(n.node, TextNode) for n in nodes)
    assert [n.node.metadata["object"] for n in nodes] == [
        "Schmidt",
        "Page",
        "Pichai",
    ]
    for i, n in enumerate(nodes):
        assert n.node.metadata["chain_position"] == i
        assert n.node.metadata["chain_type"] == "succession"
    assert nodes[0].node.metadata["edge_type_to_prev"] is None
    assert nodes[1].node.metadata["edge_type_to_prev"] == "succeeds"


@pytest.mark.asyncio
async def test_achain_rename_passthrough(chain_store) -> None:
    from nuggetindex.integrations.llamaindex import NuggetChainRetriever

    retriever = NuggetChainRetriever(store=chain_store)
    nodes = await retriever.achain_rename(subject="Twitter Inc")
    assert [n.node.metadata["object"] for n in nodes] == ["X Corp"]
    assert nodes[0].node.metadata["chain_type"] == "rename"


@pytest.mark.asyncio
async def test_achain_join_passthrough(chain_store) -> None:
    from nuggetindex.integrations.llamaindex import NuggetChainRetriever

    retriever = NuggetChainRetriever(store=chain_store)
    nodes = await retriever.achain_join(
        start=("Google", "parentCompany"),
        then=["ceo"],
        as_of=datetime(2020, 1, 1, tzinfo=UTC),
    )
    assert len(nodes) == 2
    assert nodes[0].node.metadata["object"] == "Alphabet"
    assert nodes[1].node.metadata["object"] == "Pichai"


@pytest.mark.asyncio
async def test_aretrieve_json_query_bundle(chain_store) -> None:
    """``_aretrieve`` accepts a JSON-encoded chain spec in ``query_str``."""
    from llama_index.core.schema import QueryBundle

    from nuggetindex.integrations.llamaindex import NuggetChainRetriever

    retriever = NuggetChainRetriever(store=chain_store)
    spec = json.dumps({"type": "succession", "subject": "Google", "predicate": "ceo"})
    nodes = await retriever._aretrieve(QueryBundle(query_str=spec))
    assert [n.node.metadata["object"] for n in nodes] == [
        "Schmidt",
        "Page",
        "Pichai",
    ]


@pytest.mark.asyncio
async def test_aretrieve_unknown_type_raises(chain_store) -> None:
    from llama_index.core.schema import QueryBundle

    from nuggetindex.integrations.llamaindex import NuggetChainRetriever

    retriever = NuggetChainRetriever(store=chain_store)
    spec = json.dumps({"type": "nonsense"})
    with pytest.raises(ValueError, match="unknown chain type"):
        await retriever._aretrieve(QueryBundle(query_str=spec))
