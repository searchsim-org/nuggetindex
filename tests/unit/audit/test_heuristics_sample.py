"""Tests for ``nuggetindex.audit.heuristics.sample`` (Task 2.2)."""

from __future__ import annotations

from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta

import pytest

from nuggetindex.audit.heuristics import stratified_sample
from nuggetindex.pipeline.constructor import Document


def _mk_docs(n: int, *, with_dates: bool = True) -> list[Document]:
    """Build ``n`` documents spread evenly between 2010-01-01 and 2024-01-01."""
    start = datetime(2010, 1, 1, tzinfo=UTC)
    end = datetime(2024, 1, 1, tzinfo=UTC)
    span = timedelta(0) if n <= 1 else (end - start) / (n - 1)
    out: list[Document] = []
    for i in range(n):
        d = start + span * i if with_dates else None
        out.append(Document(source_id=f"d{i:04d}", text=f"doc {i}", source_date=d))
    return out


@pytest.mark.asyncio
async def test_sample_from_list_returns_requested_size() -> None:
    docs = _mk_docs(100)
    sampled, n_total = await stratified_sample(docs, sample_size=20, stratify_by="none")
    assert len(sampled) == 20
    assert n_total == 100


@pytest.mark.asyncio
async def test_sample_from_list_sample_size_ge_total() -> None:
    docs = _mk_docs(10)
    sampled, n_total = await stratified_sample(docs, sample_size=20, stratify_by="none")
    # All 10 docs returned (shuffled deterministically).
    assert len(sampled) == 10
    assert n_total == 10
    assert {d.source_id for d in sampled} == {d.source_id for d in docs}


@pytest.mark.asyncio
async def test_sample_deterministic() -> None:
    docs = _mk_docs(200)
    s1, _ = await stratified_sample(docs, sample_size=30, stratify_by="none", rng_seed=42)
    s2, _ = await stratified_sample(docs, sample_size=30, stratify_by="none", rng_seed=42)
    assert [d.source_id for d in s1] == [d.source_id for d in s2]


@pytest.mark.asyncio
async def test_sample_deterministic_stratified() -> None:
    docs = _mk_docs(200)
    s1, _ = await stratified_sample(docs, sample_size=30, stratify_by="source_date", rng_seed=7)
    s2, _ = await stratified_sample(docs, sample_size=30, stratify_by="source_date", rng_seed=7)
    assert [d.source_id for d in s1] == [d.source_id for d in s2]


@pytest.mark.asyncio
async def test_sample_stratified_covers_deciles() -> None:
    docs = _mk_docs(100)  # spread evenly across 2010->2024
    sampled, n_total = await stratified_sample(
        docs, sample_size=20, stratify_by="source_date", rng_seed=0
    )
    assert n_total == 100
    assert len(sampled) == 20

    # Bucket each sampled doc back into its decile using rank within the
    # original population (rank-based, matching the sampler's bucketing).
    sorted_by_date = sorted(docs, key=lambda d: d.source_date)  # type: ignore[arg-type,return-value]
    rank_by_id = {d.source_id: i for i, d in enumerate(sorted_by_date)}
    deciles_hit = {min(9, (rank_by_id[d.source_id] * 10) // len(docs)) for d in sampled}
    assert deciles_hit == set(range(10)), f"expected all 10 deciles, got {sorted(deciles_hit)}"


@pytest.mark.asyncio
async def test_sample_stratified_handles_unknown_bucket() -> None:
    dated = _mk_docs(80)
    undated = [
        Document(source_id=f"u{i:04d}", text=f"undated {i}", source_date=None) for i in range(20)
    ]
    docs = dated + undated

    sampled, n_total = await stratified_sample(
        docs, sample_size=22, stratify_by="source_date", rng_seed=0
    )
    assert n_total == 100
    unknown_hits = [d for d in sampled if d.source_date is None]
    assert len(unknown_hits) >= 1


@pytest.mark.asyncio
async def test_sample_reservoir_for_streaming() -> None:
    async def _stream() -> AsyncIterator[Document]:
        for i in range(1000):
            yield Document(
                source_id=f"s{i:05d}",
                text=f"stream doc {i}",
                source_date=None,
            )

    sampled, n_total = await stratified_sample(
        _stream(), sample_size=50, stratify_by="source_date", rng_seed=0
    )
    assert n_total is None
    assert len(sampled) == 50
    # All items unique (reservoir should never duplicate).
    assert len({d.source_id for d in sampled}) == 50


@pytest.mark.asyncio
async def test_sample_reservoir_sync_iterable() -> None:
    # A generator (Iterable but not Sequence) should also go through reservoir.
    def _gen():
        for i in range(500):
            yield Document(source_id=f"g{i}", text="", source_date=None)

    sampled, n_total = await stratified_sample(
        _gen(), sample_size=25, stratify_by="none", rng_seed=0
    )
    assert n_total is None
    assert len(sampled) == 25
