"""Tests for :meth:`NuggetStore.achain_succession`."""

from datetime import UTC, datetime
from pathlib import Path

import pytest

from nuggetindex import NuggetStore
from nuggetindex.core.enums import LifecycleStatus, NuggetKind
from nuggetindex.core.models import (
    EpistemicState,
    FactTriple,
    Nugget,
    ProvenanceRecord,
    ValidityInterval,
)


def _contested_ceo(obj: str, start_year: int, end_year: int | None) -> Nugget:
    vi = ValidityInterval(
        start=datetime(start_year, 1, 1, tzinfo=UTC),
        end=datetime(end_year, 1, 1, tzinfo=UTC) if end_year else None,
    )
    return Nugget.new(
        kind=NuggetKind.SEMANTIC_FACT,
        fact=FactTriple(
            subject="Google",
            predicate="chiefExecutiveOfficer",
            object=obj,
            text=f"{obj} is CEO",
        ),
        validity=vi,
        epistemic=EpistemicState(status=LifecycleStatus.CONTESTED),
        provenance=(
            ProvenanceRecord(
                source_id=f"doc-{obj}-contested",
                evidence_span=f"{obj} is CEO",
            ),
        ),
    )


async def _populate(tmp_db_path: Path, nuggets: list[Nugget]) -> NuggetStore:
    store = NuggetStore(db_path=tmp_db_path)
    for n in nuggets:
        await store.aadd(n)
    return store


@pytest.mark.asyncio
async def test_succession_returns_ordered_chain(
    tmp_db_path: Path, sample_nuggets: list[Nugget]
) -> None:
    store = await _populate(tmp_db_path, sample_nuggets)
    try:
        chain = await store.achain_succession(subject="Google", predicate="ceo")
        assert chain.chain_type == "succession"
        assert [n.fact.object for n in chain.nuggets] == [
            "Schmidt",
            "Page",
            "Pichai",
        ]
        assert len(chain.edges) == 2
        assert all(e.edge_type == "succeeds" for e in chain.edges)
    finally:
        await store.aclose()


@pytest.mark.asyncio
async def test_succession_as_of_cutoff(
    tmp_db_path: Path, sample_nuggets: list[Nugget]
) -> None:
    store = await _populate(tmp_db_path, sample_nuggets)
    try:
        chain = await store.achain_succession(
            subject="Google",
            predicate="ceo",
            as_of=datetime(2013, 1, 1, tzinfo=UTC),
        )
        assert [n.fact.object for n in chain.nuggets] == ["Schmidt", "Page"]
        assert chain.as_of == datetime(2013, 1, 1, tzinfo=UTC)
    finally:
        await store.aclose()


@pytest.mark.asyncio
async def test_succession_respects_include_contested(
    tmp_db_path: Path, sample_nuggets: list[Nugget]
) -> None:
    # Add a CONTESTED entry
    contested = _contested_ceo("Other", 2005, 2007)
    store = await _populate(tmp_db_path, [*sample_nuggets, contested])
    try:
        default_chain = await store.achain_succession(
            subject="Google", predicate="ceo"
        )
        assert "Other" not in [n.fact.object for n in default_chain.nuggets]

        inclusive = await store.achain_succession(
            subject="Google", predicate="ceo", include_contested=True
        )
        assert "Other" in [n.fact.object for n in inclusive.nuggets]
    finally:
        await store.aclose()


@pytest.mark.asyncio
async def test_succession_empty_when_subject_unknown(tmp_db_path: Path) -> None:
    store = NuggetStore(db_path=tmp_db_path)
    try:
        chain = await store.achain_succession(
            subject="Nonexistent", predicate="ceo"
        )
        assert chain.nuggets == ()
        assert chain.edges == ()
        assert chain.chain_type == "succession"
    finally:
        await store.aclose()


@pytest.mark.asyncio
async def test_succession_includes_deprecated_by_default(
    tmp_db_path: Path, sample_nuggets: list[Nugget]
) -> None:
    store = NuggetStore(db_path=tmp_db_path)
    try:
        # Mark Schmidt DEPRECATED before adding
        schmidt = sample_nuggets[0]
        deprecated = schmidt.model_copy(
            update={
                "epistemic": EpistemicState(status=LifecycleStatus.DEPRECATED),
            }
        )
        await store.aadd(deprecated)
        for n in sample_nuggets[1:]:
            await store.aadd(n)
        chain = await store.achain_succession(
            subject="Google", predicate="ceo"
        )
        assert "Schmidt" in [n.fact.object for n in chain.nuggets]
    finally:
        await store.aclose()


@pytest.mark.asyncio
async def test_succession_edges_have_gap(
    tmp_db_path: Path, sample_nuggets: list[Nugget]
) -> None:
    store = await _populate(tmp_db_path, sample_nuggets)
    try:
        chain = await store.achain_succession(
            subject="Google", predicate="ceo"
        )
        # Schmidt ends 2011-01-01, Page starts 2011-01-01 -> gap == zero
        # but validity is non-overlapping, so gap should be zero.
        assert chain.edges[0].gap is not None
        assert chain.edges[0].gap.total_seconds() == 0
    finally:
        await store.aclose()


@pytest.mark.asyncio
async def test_succession_truncated_by_max_depth(tmp_db_path: Path) -> None:
    # Build 5 nuggets and truncate to 3
    nuggets = []
    for year in (2000, 2002, 2004, 2006, 2008):
        nuggets.append(
            Nugget.new(
                kind=NuggetKind.SEMANTIC_FACT,
                fact=FactTriple(
                    subject="Google",
                    predicate="chiefExecutiveOfficer",
                    object=f"P{year}",
                    text=f"CEO at {year}",
                ),
                validity=ValidityInterval(
                    start=datetime(year, 1, 1, tzinfo=UTC),
                    end=datetime(year + 2, 1, 1, tzinfo=UTC),
                ),
                epistemic=EpistemicState(),
                provenance=(
                    ProvenanceRecord(
                        source_id=f"doc-{year}",
                        evidence_span="",
                    ),
                ),
            )
        )
    store = await _populate(tmp_db_path, nuggets)
    try:
        chain = await store.achain_succession(
            subject="Google", predicate="ceo", max_depth=3
        )
        assert len(chain.nuggets) == 3
        assert chain.truncated is True
    finally:
        await store.aclose()


def test_succession_sync_wrapper(
    tmp_db_path: Path, sample_nuggets: list[Nugget]
) -> None:
    store = NuggetStore(db_path=tmp_db_path)
    try:
        for n in sample_nuggets:
            store.add(n)
        chain = store.chain_succession(subject="Google", predicate="ceo")
        assert [n.fact.object for n in chain.nuggets] == [
            "Schmidt",
            "Page",
            "Pichai",
        ]
    finally:
        store.close()
