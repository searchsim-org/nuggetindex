"""Tests for ``NuggetChainRetriever`` (Haystack 2.x ``@component`` chain adapter).

Haystack components are inherently sync -- ``run()`` calls ``asyncio.run``
internally -- so these tests use the same sync-seed pattern as
``test_retriever.py``.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from pathlib import Path

import pytest

pytest.importorskip("haystack")

from haystack import Document  # noqa: E402

from nuggetindex.core.enums import LifecycleStatus, NuggetKind  # noqa: E402
from nuggetindex.core.models import (  # noqa: E402
    EpistemicState,
    FactTriple,
    Nugget,
    ProvenanceRecord,
    ValidityInterval,
)
from nuggetindex.integrations.haystack import NuggetChainRetriever  # noqa: E402
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


def _seed_store(db_path: Path) -> NuggetStore:
    """Build a pre-seeded store synchronously (Haystack ``run()`` is sync)."""

    async def _inner() -> NuggetStore:
        store = NuggetStore(db_path)
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
        return store

    return asyncio.run(_inner())


def _close_store(store: NuggetStore) -> None:
    asyncio.run(store.aclose())


def test_chain_retriever_is_haystack_component(tmp_path: Path) -> None:
    store = _seed_store(tmp_path / "hs_chain.db")
    try:
        inst = NuggetChainRetriever(store=store)
        assert hasattr(inst, "__haystack_input__")
        assert hasattr(inst, "__haystack_output__")
    finally:
        _close_store(store)


def test_run_succession_returns_documents_and_metadata(tmp_path: Path) -> None:
    store = _seed_store(tmp_path / "hs_chain.db")
    try:
        retriever = NuggetChainRetriever(store=store)
        out = retriever.run(
            chain_spec={
                "type": "succession",
                "subject": "Google",
                "predicate": "ceo",
            }
        )
        assert set(out.keys()) == {"documents", "chain_metadata"}
        docs = out["documents"]
        assert all(isinstance(d, Document) for d in docs)
        assert [d.meta["object"] for d in docs] == [
            "Schmidt",
            "Page",
            "Pichai",
        ]
        # Chain-specific per-doc metadata.
        for i, d in enumerate(docs):
            assert d.meta["chain_position"] == i
            assert d.meta["chain_type"] == "succession"
        assert docs[0].meta["edge_type_to_prev"] is None
        assert docs[1].meta["edge_type_to_prev"] == "succeeds"
        # Top-level ``chain_metadata`` aggregate payload.
        meta = out["chain_metadata"]
        assert meta["chain_type"] == "succession"
        assert meta["length"] == 3
        assert meta["truncated"] is False
    finally:
        _close_store(store)


def test_run_rename_chain(tmp_path: Path) -> None:
    store = _seed_store(tmp_path / "hs_chain.db")
    try:
        retriever = NuggetChainRetriever(store=store)
        out = retriever.run(chain_spec={"type": "rename", "subject": "Twitter Inc"})
        docs = out["documents"]
        assert [d.meta["object"] for d in docs] == ["X Corp"]
        assert out["chain_metadata"]["chain_type"] == "rename"
    finally:
        _close_store(store)


def test_run_joined_chain(tmp_path: Path) -> None:
    store = _seed_store(tmp_path / "hs_chain.db")
    try:
        retriever = NuggetChainRetriever(store=store)
        out = retriever.run(
            chain_spec={
                "type": "joined",
                "start": ("Google", "parentCompany"),
                "then": ["ceo"],
                "as_of": datetime(2020, 1, 1, tzinfo=UTC),
            }
        )
        docs = out["documents"]
        assert [d.meta["object"] for d in docs] == ["Alphabet", "Pichai"]
        assert out["chain_metadata"]["chain_type"] == "joined"
    finally:
        _close_store(store)


def test_run_unknown_type_raises(tmp_path: Path) -> None:
    store = _seed_store(tmp_path / "hs_chain.db")
    try:
        retriever = NuggetChainRetriever(store=store)
        with pytest.raises(ValueError, match="unknown chain type"):
            retriever.run(chain_spec={"type": "nonsense"})
    finally:
        _close_store(store)
