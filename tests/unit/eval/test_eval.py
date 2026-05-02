"""Unit tests for :mod:`nuggetindex.eval`.

Cover the sanity-benchmark loader, the exact-match plumbing, and the
"fixed by sidecar" diff list using a hand-crafted scenario. No external
datasets are downloaded.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from nuggetindex import NuggetStore
from nuggetindex.core.enums import EpistemicRank, LifecycleStatus, NuggetKind
from nuggetindex.core.models import (
    EpistemicState,
    FactTriple,
    Nugget,
    ProvenanceRecord,
    ValidityInterval,
)
from nuggetindex.eval import BenchmarkQuery, EvalReport, run_eval
from nuggetindex.sidecar import Sidecar


def _empty_retriever(_query: str, _top_k: int) -> list[Any]:
    """Baseline that always returns nothing — forces the sidecar to carry the load."""
    return []


async def _make_sidecar_with_google_ceo(tmp_path: Path) -> tuple[Sidecar, NuggetStore]:
    """Return a sidecar whose store knows Larry Page was Google CEO in 2013."""
    store = NuggetStore(db_path=tmp_path / "s.db")
    now = datetime.now(tz=UTC)
    n = Nugget.new(
        kind=NuggetKind.SEMANTIC_FACT,
        fact=FactTriple(
            subject="Google",
            predicate="chiefExecutiveOfficer",
            object="Larry Page",
            text="Larry Page served as Google CEO from 2011 through 2015",
        ),
        validity=ValidityInterval(
            start=datetime(2011, 4, 4, tzinfo=UTC),
            end=datetime(2015, 10, 2, tzinfo=UTC),
        ),
        epistemic=EpistemicState(
            status=LifecycleStatus.ACTIVE,
            rank=EpistemicRank.NORMAL,
            confidence=0.9,
        ),
        provenance=(
            ProvenanceRecord(
                source_id="wiki",
                evidence_span="Larry Page served as CEO from 2011 to 2015.",
                char_start=0,
                char_end=50,
                created_at=now,
            ),
        ),
        extraction_confidence=0.9,
    )
    await store.aadd(n)
    return Sidecar(store=store, mode="offline-curated"), store


@pytest.mark.asyncio
async def test_sanity_benchmark_loads(tmp_path: Path) -> None:
    """``run_eval(benchmark='sanity', ...)`` returns a 10-query report."""
    sidecar, store = await _make_sidecar_with_google_ceo(tmp_path)
    try:
        report = await run_eval(
            benchmark="sanity",
            sidecar=sidecar,
            baseline_retriever=_empty_retriever,
        )
        assert isinstance(report, EvalReport)
        assert report.n_queries == 10
        assert report.benchmark == "sanity"
    finally:
        await store.backend.aclose()


@pytest.mark.asyncio
async def test_em_computed_correctly(tmp_path: Path) -> None:
    """A fake answerer that echoes the expected answer yields EM=1.0."""
    sidecar, store = await _make_sidecar_with_google_ceo(tmp_path)
    try:
        queries = [
            BenchmarkQuery(query="q1", expected_answer="Larry Page"),
            BenchmarkQuery(query="q2", expected_answer="Sundar Pichai"),
        ]

        def perfect_answerer(_ctx: str, query: str) -> str:
            # Produce the expected answer from the BenchmarkQuery list above.
            mapping = {"q1": "Larry Page", "q2": "Sundar Pichai"}
            return mapping[query]

        # Pass a retriever that returns non-empty context so the answerer is
        # consulted (the default oracle short-circuits on empty context).
        def fake_retriever(_query: str, _top_k: int) -> list[Any]:
            return [{"id": "x", "content": "filler"}]

        report = await run_eval(
            benchmark=queries,
            sidecar=sidecar,
            baseline_retriever=fake_retriever,
            answerer=perfect_answerer,
        )
        assert report.n_queries == 2
        assert report.sidecar_em == pytest.approx(1.0)
        assert report.baseline_em == pytest.approx(1.0)
    finally:
        await store.backend.aclose()


@pytest.mark.asyncio
async def test_delta_em_positive_when_sidecar_helps(tmp_path: Path) -> None:
    """Baseline retriever empty + sidecar has the answer → positive delta."""
    sidecar, store = await _make_sidecar_with_google_ceo(tmp_path)
    try:
        queries = [
            BenchmarkQuery(
                query="Who was Google's CEO in 2013?",
                expected_answer="Larry Page",
                query_time=datetime(2013, 6, 15, tzinfo=UTC),
            ),
        ]
        # The default oracle answers "expected" iff context contains the
        # expected string. Empty baseline context forces EM=0 for baseline.
        report = await run_eval(
            benchmark=queries,
            sidecar=sidecar,
            baseline_retriever=_empty_retriever,
        )
        assert report.baseline_em == pytest.approx(0.0)
        assert report.sidecar_em > 0.0
        assert report.delta_em > 0.0
    finally:
        await store.backend.aclose()


@pytest.mark.asyncio
async def test_fixed_by_sidecar_list_nonempty(tmp_path: Path) -> None:
    """The sidecar-helps scenario populates ``fixed_by_sidecar``."""
    sidecar, store = await _make_sidecar_with_google_ceo(tmp_path)
    try:
        queries = [
            BenchmarkQuery(
                query="Who was Google's CEO in 2013?",
                expected_answer="Larry Page",
                query_time=datetime(2013, 6, 15, tzinfo=UTC),
            ),
        ]
        report = await run_eval(
            benchmark=queries,
            sidecar=sidecar,
            baseline_retriever=_empty_retriever,
        )
        assert len(report.fixed_by_sidecar) >= 1
        assert len(report.broken_by_sidecar) == 0
    finally:
        await store.backend.aclose()


@pytest.mark.asyncio
async def test_benchmark_list_of_queries(tmp_path: Path) -> None:
    """Passing an inline list works without a loader."""
    sidecar, store = await _make_sidecar_with_google_ceo(tmp_path)
    try:
        inline = [
            BenchmarkQuery(
                query="who was Google's CEO in 2013?",
                expected_answer="Larry Page",
                query_time=datetime(2013, 6, 15, tzinfo=UTC),
            ),
            BenchmarkQuery(query="unrelated", expected_answer="no-answer"),
        ]
        report = await run_eval(
            benchmark=inline,
            sidecar=sidecar,
            baseline_retriever=_empty_retriever,
        )
        assert report.n_queries == 2
        assert report.benchmark == "inline"
    finally:
        await store.backend.aclose()
