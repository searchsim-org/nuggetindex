"""Tests for :meth:`NuggetStore.achain_rename`."""

from datetime import UTC, datetime
from pathlib import Path

import pytest

from nuggetindex import NuggetStore
from nuggetindex.chains.models import NuggetChain
from nuggetindex.core.enums import LifecycleStatus, NuggetKind
from nuggetindex.core.errors import ChainAmbiguousError
from nuggetindex.core.models import (
    EpistemicState,
    FactTriple,
    Nugget,
    ProvenanceRecord,
    ValidityInterval,
)


def _rename(
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
                source_id=f"doc-{subject}-{object_}",
                evidence_span=f"{subject} {predicate} {object_}",
            ),
        ),
    )


async def _populate(path: Path, nuggets: list[Nugget]) -> NuggetStore:
    store = NuggetStore(db_path=path)
    for n in nuggets:
        await store.aadd(n)
    return store


# --- Twitter -> X rename fixture ---


def _twitter_chain() -> list[Nugget]:
    return [
        _rename("Twitter", "renamedTo", "X Corp", 2023, None),
    ]


def _twitter_multihop() -> list[Nugget]:
    # Twitter Inc. -> X Corp. -> X (pretend two-step rename)
    return [
        _rename("Twitter Inc", "renamedTo", "X Corp", 2023, 2024),
        _rename("X Corp", "renamedTo", "X", 2024, None),
    ]


@pytest.mark.asyncio
async def test_rename_forward_single_hop(tmp_db_path: Path) -> None:
    store = await _populate(tmp_db_path, _twitter_chain())
    try:
        chain = await store.achain_rename(subject="Twitter")
        assert chain.chain_type == "rename"
        assert len(chain.nuggets) == 1
        assert chain.nuggets[0].fact.object == "X Corp"
    finally:
        await store.aclose()


@pytest.mark.asyncio
async def test_rename_forward_multi_hop(tmp_db_path: Path) -> None:
    store = await _populate(tmp_db_path, _twitter_multihop())
    try:
        chain = await store.achain_rename(subject="Twitter Inc")
        assert [n.fact.object for n in chain.nuggets] == ["X Corp", "X"]
        # Edges are RENAMES_TO
        assert all(e.edge_type == "renames_to" for e in chain.edges)
    finally:
        await store.aclose()


@pytest.mark.asyncio
async def test_rename_backward(tmp_db_path: Path) -> None:
    store = await _populate(tmp_db_path, _twitter_multihop())
    try:
        chain = await store.achain_rename(subject="X", direction="backward")
        # Backward walk: X was renamed *from* X Corp, which was *from*
        # Twitter Inc. Result should be chronological.
        subjects = [n.fact.subject for n in chain.nuggets]
        assert subjects == ["Twitter Inc", "X Corp"]
    finally:
        await store.aclose()


@pytest.mark.asyncio
async def test_rename_both(tmp_db_path: Path) -> None:
    store = await _populate(tmp_db_path, _twitter_multihop())
    try:
        chain = await store.achain_rename(subject="X Corp", direction="both")
        # Both directions should collectively visit both edges.
        # Backward finds Twitter Inc -> X Corp (2023); forward finds
        # X Corp -> X (2024). Merged chronologically: 2 unique nuggets.
        assert len(chain.nuggets) == 2
        texts = {(n.fact.subject, n.fact.object) for n in chain.nuggets}
        assert ("Twitter Inc", "X Corp") in texts
        assert ("X Corp", "X") in texts
    finally:
        await store.aclose()


@pytest.mark.asyncio
async def test_rename_empty_chain_for_unknown_subject(tmp_db_path: Path) -> None:
    store = NuggetStore(db_path=tmp_db_path)
    try:
        chain = await store.achain_rename(subject="NonExistent")
        assert chain.nuggets == ()
        assert chain.chain_type == "rename"
    finally:
        await store.aclose()


@pytest.mark.asyncio
async def test_rename_max_depth_truncates(tmp_db_path: Path) -> None:
    nuggets = [_rename(f"E{i}", "renamedTo", f"E{i + 1}", 2000 + i, 2001 + i) for i in range(6)]
    store = await _populate(tmp_db_path, nuggets)
    try:
        chain = await store.achain_rename(subject="E0", max_depth=3)
        assert len(chain.nuggets) == 3
        assert chain.truncated is True
    finally:
        await store.aclose()


@pytest.mark.asyncio
async def test_rename_cycle_terminates(tmp_db_path: Path) -> None:
    # A -> B -> A creates a cycle.
    nuggets = [
        _rename("A", "renamedTo", "B", 2000, 2001),
        _rename("B", "renamedTo", "A", 2001, 2002),
    ]
    store = await _populate(tmp_db_path, nuggets)
    try:
        chain = await store.achain_rename(subject="A", max_depth=10)
        # Walker must terminate without infinite looping.
        assert len(chain.nuggets) <= 2
    finally:
        await store.aclose()


@pytest.mark.asyncio
async def test_rename_ambiguous_without_resolver_raises(tmp_db_path: Path) -> None:
    # Same subject, two possible rename targets in overlapping time.
    nuggets = [
        _rename("A", "renamedTo", "B", 2000),
        _rename("A", "renamedTo", "C", 2001),
    ]
    store = await _populate(tmp_db_path, nuggets)
    try:
        with pytest.raises(ChainAmbiguousError) as ei:
            await store.achain_rename(subject="A")
        assert ei.value.subject == "A"
        assert len(ei.value.candidates) == 2
        assert ei.value.step == 0
    finally:
        await store.aclose()


@pytest.mark.asyncio
async def test_rename_ambiguous_with_resolver_delegates(tmp_db_path: Path) -> None:
    class StubResolver:
        def __init__(self) -> None:
            self.calls = 0

        async def adisambiguate(self, *, candidates: list, context: str) -> object:
            self.calls += 1
            # Mimic the ChainResolution return shape
            from types import SimpleNamespace

            return SimpleNamespace(picked=candidates[0], rationale="test pick first")

    nuggets = [
        _rename("A", "renamedTo", "B", 2000),
        _rename("A", "renamedTo", "C", 2001),
    ]
    store = await _populate(tmp_db_path, nuggets)
    resolver = StubResolver()
    try:
        chain = await store.achain_rename(subject="A", resolver=resolver)
        assert resolver.calls >= 1
        assert isinstance(chain, NuggetChain)
    finally:
        await store.aclose()


@pytest.mark.asyncio
async def test_rename_as_of_cutoff(tmp_db_path: Path) -> None:
    store = await _populate(tmp_db_path, _twitter_multihop())
    try:
        # as_of before the second rename -> only Twitter Inc -> X Corp visible
        chain = await store.achain_rename(
            subject="Twitter Inc", as_of=datetime(2023, 6, 1, tzinfo=UTC)
        )
        assert len(chain.nuggets) == 1
        assert chain.nuggets[0].fact.object == "X Corp"
    finally:
        await store.aclose()


@pytest.mark.asyncio
async def test_rename_include_contested(tmp_db_path: Path) -> None:
    nuggets = [
        _rename(
            "A",
            "renamedTo",
            "Disputed",
            2000,
            status=LifecycleStatus.CONTESTED,
        ),
    ]
    store = await _populate(tmp_db_path, nuggets)
    try:
        default_chain = await store.achain_rename(subject="A")
        assert len(default_chain.nuggets) == 0

        included = await store.achain_rename(subject="A", include_contested=True)
        assert len(included.nuggets) == 1
        assert included.nuggets[0].fact.object == "Disputed"
    finally:
        await store.aclose()


@pytest.mark.asyncio
async def test_rename_skips_non_renaming_predicates(tmp_db_path: Path) -> None:
    # ``parentCompany`` is not a renaming predicate.
    nugget = _rename("Google", "parentCompany", "Alphabet", 2015)
    store = await _populate(tmp_db_path, [nugget])
    try:
        chain = await store.achain_rename(subject="Google")
        # parentCompany is not in renaming_predicates, so no hops happen.
        assert chain.nuggets == ()
    finally:
        await store.aclose()


def test_rename_sync_wrapper(tmp_db_path: Path) -> None:
    store = NuggetStore(db_path=tmp_db_path)
    try:
        for n in _twitter_chain():
            store.add(n)
        chain = store.chain_rename(subject="Twitter")
        assert chain.nuggets[0].fact.object == "X Corp"
    finally:
        store.close()


@pytest.mark.asyncio
async def test_rename_strict_excludes_succession_predicate(
    tmp_db_path: Path,
    tmp_path: Path,
) -> None:
    """``strict=True`` consumes ``entity_rename_predicates``, which never
    includes ``succeededBy`` even when a user-supplied schema flags it as
    ``renaming: true``.
    """
    from nuggetindex.core.schema import RelationSchema

    # Build a permissive schema that (incorrectly) marks succeededBy as
    # renaming. With strict=False, this would drive the rename walk;
    # with strict=True, the whitelist-filtered set excludes it, so the
    # walk produces nothing.
    yaml = tmp_path / "permissive.yaml"
    yaml.write_text(
        """
version: 1
predicates:
  succeededBy:
    functional: true
    renaming: true
    aliases: [successor]
  renamedTo:
    functional: true
    renaming: true
    aliases: []
"""
    )
    schema = RelationSchema.from_yaml(yaml)
    store = NuggetStore(db_path=tmp_db_path, schema=schema)
    try:
        await store.aadd(_rename("Apple", "succeededBy", "Tim Cook", 2011))

        # Default (strict=False) uses renaming_predicates which includes
        # succeededBy in this permissive schema: the walk finds Tim Cook.
        lenient = await store.achain_rename(subject="Apple")
        assert len(lenient.nuggets) == 1
        assert lenient.nuggets[0].fact.object == "Tim Cook"

        # strict=True uses entity_rename_predicates which enforces the
        # library-level whitelist and therefore excludes succeededBy.
        strict_chain = await store.achain_rename(subject="Apple", strict=True)
        assert strict_chain.nuggets == ()
    finally:
        await store.aclose()
