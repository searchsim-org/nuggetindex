"""Integration tests for Retriever + NuggetStore.aretrieve."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from nuggetindex.core.enums import LifecycleStatus, NuggetKind
from nuggetindex.core.models import (
    EpistemicState,
    FactTriple,
    Nugget,
    ProvenanceRecord,
    ValidityInterval,
)
from nuggetindex.retrieve import RetrievalResult, Retriever
from nuggetindex.store import NuggetStore


def _n(
    *,
    subject: str,
    predicate: str,
    obj: str,
    text: str,
    start: datetime,
    end: datetime | None = None,
    status: LifecycleStatus = LifecycleStatus.ACTIVE,
    source_id: str = "doc-1",
) -> Nugget:
    return Nugget.new(
        kind=NuggetKind.SEMANTIC_FACT,
        fact=FactTriple(subject=subject, predicate=predicate, object=obj, text=text),
        validity=ValidityInterval(start=start, end=end),
        epistemic=EpistemicState(status=status),
        provenance=(ProvenanceRecord(source_id=source_id, evidence_span=text),),
    )


@pytest.mark.asyncio
async def test_aretrieve_returns_ranked_results(tmp_db_path):
    store = NuggetStore(db_path=tmp_db_path)
    nuggets = [
        _n(
            subject="Google",
            predicate="ceo",
            obj="Pichai",
            text="Sundar Pichai is CEO of Google",
            start=datetime(2015, 10, 1, tzinfo=UTC),
        ),
        _n(
            subject="Apple",
            predicate="hq",
            obj="Cupertino",
            text="Apple is headquartered in Cupertino",
            start=datetime(2015, 10, 1, tzinfo=UTC),
        ),
        _n(
            subject="Google",
            predicate="founded",
            obj="1998",
            text="Google was founded in 1998",
            start=datetime(2015, 10, 1, tzinfo=UTC),
        ),
    ]
    for n in nuggets:
        await store.aadd(n)

    results = await store.aretrieve("CEO of Google", top_k=5)
    assert isinstance(results, list)
    assert all(isinstance(r, RetrievalResult) for r in results)
    assert len(results) >= 1
    # Ranks are 1-indexed and contiguous
    assert [r.rank for r in results] == list(range(1, len(results) + 1))
    # First result should be the Pichai/CEO one
    assert "CEO" in results[0].nugget.fact.text
    # Only sparse present (no dense backend) -> sparse_score populated, dense None
    assert results[0].sparse_score is not None
    assert results[0].dense_score is None
    assert "sparse" in results[0].component_ranks
    assert "dense" not in results[0].component_ranks
    await store.aclose()


@pytest.mark.asyncio
async def test_aretrieve_empty_when_no_candidates(tmp_db_path):
    store = NuggetStore(db_path=tmp_db_path)
    # No nuggets added.
    results = await store.aretrieve("anything", top_k=5)
    assert results == []
    await store.aclose()


@pytest.mark.asyncio
async def test_aretrieve_query_time_excludes_expired(tmp_db_path):
    store = NuggetStore(db_path=tmp_db_path)
    # An expired fact + a currently-valid one with similar text.
    expired = _n(
        subject="Google",
        predicate="ceo",
        obj="Schmidt",
        text="Eric Schmidt is CEO of Google",
        start=datetime(2001, 1, 1, tzinfo=UTC),
        end=datetime(2015, 10, 2, tzinfo=UTC),
    )
    current = _n(
        subject="Google",
        predicate="ceo",
        obj="Pichai",
        text="Sundar Pichai is CEO of Google",
        start=datetime(2015, 10, 2, tzinfo=UTC),
    )
    await store.aadd(expired)
    await store.aadd(current)

    # Query at 2024: Schmidt is out of validity, Pichai is in.
    results = await store.aretrieve(
        "CEO of Google",
        query_time=datetime(2024, 1, 1, tzinfo=UTC),
        top_k=5,
    )
    ids = [r.nugget.id for r in results]
    assert current.id in ids
    assert expired.id not in ids
    await store.aclose()


@pytest.mark.asyncio
async def test_aretrieve_view_filters_deprecated(tmp_db_path):
    store = NuggetStore(db_path=tmp_db_path)
    deprecated = _n(
        subject="Google",
        predicate="ceo",
        obj="Page",
        text="Larry Page CEO Google",
        start=datetime(2011, 1, 1, tzinfo=UTC),
        status=LifecycleStatus.DEPRECATED,
    )
    active = _n(
        subject="Google",
        predicate="ceo",
        obj="Pichai",
        text="Sundar Pichai CEO Google",
        start=datetime(2015, 10, 1, tzinfo=UTC),
        status=LifecycleStatus.ACTIVE,
    )
    await store.aadd(deprecated)
    await store.aadd(active)

    # Default view=active -> deprecated excluded.
    results_active = await store.aretrieve("CEO Google", top_k=5)
    ids_active = {r.nugget.id for r in results_active}
    assert active.id in ids_active
    assert deprecated.id not in ids_active

    # view=all -> both included.
    results_all = await store.aretrieve("CEO Google", view="all", top_k=5)
    ids_all = {r.nugget.id for r in results_all}
    assert active.id in ids_all
    assert deprecated.id in ids_all
    await store.aclose()


@pytest.mark.asyncio
async def test_aretrieve_weighted_minmax_mode(tmp_db_path):
    """fusion='weighted_minmax' should work even without dense backend."""
    store = NuggetStore(db_path=tmp_db_path)
    await store.aadd(
        _n(
            subject="Google",
            predicate="ceo",
            obj="Pichai",
            text="Pichai is CEO of Google",
            start=datetime(2015, 10, 1, tzinfo=UTC),
        )
    )
    results = await store.aretrieve("CEO Google", top_k=5, fusion="weighted_minmax")
    assert len(results) >= 1
    await store.aclose()


class _FakeDense:
    """Minimal dense backend stub for fusion-path testing."""

    def __init__(self, scores: dict[str, float]) -> None:
        self.scores = scores

    async def asearch(
        self,
        query: str,  # noqa: ARG002
        *,
        candidate_ids: list[str] | None = None,
        top_k: int = 20,  # noqa: ARG002
    ) -> list[tuple[str, float]]:
        pairs = [
            (nid, score)
            for nid, score in self.scores.items()
            if candidate_ids is None or nid in candidate_ids
        ]
        return sorted(pairs, key=lambda kv: -kv[1])


@pytest.mark.asyncio
async def test_retriever_with_dense_backend_populates_both_scores(tmp_db_path):
    from nuggetindex.store.backends.sqlite import SQLiteBackend

    backend = SQLiteBackend(tmp_db_path)
    n1 = _n(
        subject="Google",
        predicate="ceo",
        obj="Pichai",
        text="Pichai CEO Google",
        start=datetime(2015, 10, 1, tzinfo=UTC),
    )
    n2 = _n(
        subject="Apple",
        predicate="hq",
        obj="Cupertino",
        text="Apple Cupertino",
        start=datetime(2015, 10, 1, tzinfo=UTC),
    )
    await backend.aupsert(n1)
    await backend.aupsert(n2)

    dense = _FakeDense({n1.id: 0.9, n2.id: 0.3})
    retriever = Retriever(backend=backend, dense_backend=dense)
    results = await retriever.aretrieve("CEO Google", top_k=5)
    assert len(results) >= 1
    top = results[0]
    # n1 should dominate: top in both sparse + dense.
    assert top.nugget.id == n1.id
    assert top.sparse_score is not None
    assert top.dense_score is not None
    assert "sparse" in top.component_ranks
    assert "dense" in top.component_ranks
    await backend.aclose()


def test_sync_retrieve_wrapper(tmp_db_path):
    """Pure-sync entry point: store.retrieve() wraps asyncio.run correctly."""
    store = NuggetStore(db_path=tmp_db_path)
    try:
        store.add(
            _n(
                subject="Google",
                predicate="ceo",
                obj="Pichai",
                text="Pichai CEO Google",
                start=datetime(2015, 10, 1, tzinfo=UTC),
            )
        )
        results = store.retrieve("CEO Google", top_k=5)
        assert len(results) >= 1
    finally:
        store.close()
