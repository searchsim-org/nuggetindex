"""Tests for :func:`nuggetindex.auto.auto`'s optional two-pass deep ingest."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from nuggetindex.auto import auto
from nuggetindex.pipeline.constructor import Document


def _synthetic_docs(prefix: str, n: int, start: int = 0) -> list[Document]:
    """Produce ``n`` synthetic docs whose text contains trigger-able content.

    The ``Satya Nadella`` / ``Microsoft`` surface gives the trigger extractor
    enough NER + verb signal to propose at least one seed via
    :func:`propose_seeds`, which is what the Pass-2 pipeline iterates over.
    """
    return [
        Document(
            source_id=f"{prefix}-{i}",
            text=(
                f"doc {i}: Satya Nadella became CEO of Microsoft in 2014. "
                "Acquisitions, renamings, and executive transitions followed."
            ),
        )
        for i in range(start, start + n)
    ]


def _stub_corpus(
    sample_docs: list[Document],
    search_docs: list[Document],
) -> MagicMock:
    """Build a MagicMock satisfying the CorpusSource protocol shape."""
    stub = MagicMock()
    stub.sample = AsyncMock(return_value=sample_docs)
    # Every search() call returns the same list -- the auto() dedup by
    # source_id ensures we don't double-count across seeds.
    stub.search = AsyncMock(return_value=search_docs)
    return stub


@pytest.mark.asyncio
async def test_two_pass_doubles_ingest_count(tmp_path: Path) -> None:
    """Pass 1 ingests N docs; Pass 2 ingests (seeds * M) new docs, deduped."""
    sample_docs = _synthetic_docs("boot", 5)
    # Pass-2 search docs -- distinct source_ids from the bootstrap.
    search_docs = _synthetic_docs("deep", 4, start=100)
    corpus = _stub_corpus(sample_docs, search_docs)

    _sidecar, report = await auto(
        corpus=corpus,
        bootstrap="topic_diverse",
        sample_size=5,
        budget=3,
        two_pass=True,
        deep_docs_per_seed=4,
        store_path=tmp_path / "store.db",
        cache_path=tmp_path / "cache.db",
    )
    try:
        assert report.two_pass_enabled is True
        assert report.bootstrap_docs == 5
        # corpus.search was called seeds_accepted times; each call returned
        # 4 unique search_docs, so after the first seed we've already added
        # all 4, and subsequent seeds contribute nothing (dedup).
        assert report.seeds_accepted > 0
        assert corpus.search.await_count == report.seeds_accepted
        # Deep-pass doc count <= len(search_docs) because of dedup across
        # seeds, and > 0 because the seed list was non-empty.
        assert 0 < report.deep_pass_docs <= len(search_docs)
        assert report.n_docs_processed == (
            report.bootstrap_docs + report.deep_pass_docs
        )
    finally:
        await _sidecar.store.backend.aclose()


@pytest.mark.asyncio
async def test_two_pass_dedup_by_source_id(tmp_path: Path) -> None:
    """Docs already seen in Pass 1 are NOT re-ingested in Pass 2."""
    sample_docs = _synthetic_docs("shared", 3)
    # Pass 2 returns the *same* docs already ingested in Pass 1 plus one
    # new doc -- only the new one should count towards deep_pass_docs.
    overlap_plus_new = sample_docs + _synthetic_docs("extra", 1, start=999)
    corpus = _stub_corpus(sample_docs, overlap_plus_new)

    _sidecar, report = await auto(
        corpus=corpus,
        bootstrap="topic_diverse",
        sample_size=3,
        budget=2,
        two_pass=True,
        deep_docs_per_seed=len(overlap_plus_new),
        store_path=tmp_path / "store.db",
        cache_path=tmp_path / "cache.db",
    )
    try:
        assert report.two_pass_enabled is True
        assert report.bootstrap_docs == 3
        # Only the "extra-999" doc slipped past the dedup filter.
        assert report.deep_pass_docs == 1
        assert report.n_docs_processed == 4
    finally:
        await _sidecar.store.backend.aclose()


@pytest.mark.asyncio
async def test_two_pass_off_by_default(tmp_path: Path) -> None:
    """Without ``two_pass=True``, behaviour is exactly the legacy path."""
    sample_docs = _synthetic_docs("boot", 4)
    search_docs = _synthetic_docs("deep", 3, start=50)
    corpus = _stub_corpus(sample_docs, search_docs)

    _sidecar, report = await auto(
        corpus=corpus,
        bootstrap="topic_diverse",
        sample_size=4,
        budget=2,
        store_path=tmp_path / "store.db",
        cache_path=tmp_path / "cache.db",
    )
    try:
        # Pass 2 never ran -> search() never invoked.
        corpus.search.assert_not_awaited()
        assert report.two_pass_enabled is False
        assert report.bootstrap_docs == 4
        assert report.deep_pass_docs == 0
        assert report.nuggets_deep_pass == 0
        assert report.n_docs_processed == 4
    finally:
        await _sidecar.store.backend.aclose()


@pytest.mark.asyncio
async def test_two_pass_requires_corpus(tmp_path: Path) -> None:
    """``two_pass=True`` without ``corpus=`` must raise ValueError."""
    with pytest.raises(ValueError, match="two_pass"):
        await auto(
            docs=_synthetic_docs("local", 2),
            two_pass=True,
            store_path=tmp_path / "store.db",
            cache_path=tmp_path / "cache.db",
        )
