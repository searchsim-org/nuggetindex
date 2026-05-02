"""Tests for :func:`nuggetindex.auto.auto`'s corpus-bootstrap wiring."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from nuggetindex.auto import AutoReport, auto
from nuggetindex.pipeline.constructor import Document
from nuggetindex.sidecar import Sidecar


def _stub_corpus(docs: list[Document]) -> MagicMock:
    """Build a MagicMock that satisfies the CorpusSource protocol shape."""
    stub = MagicMock()
    stub.sample = AsyncMock(return_value=docs)
    stub.search = AsyncMock(return_value=docs[: min(5, len(docs))])
    return stub


def _synthetic_docs(n: int) -> list[Document]:
    return [
        Document(
            source_id=f"d{i}",
            text=f"doc {i}: Satya Nadella became CEO of Microsoft in 2014.",
        )
        for i in range(n)
    ]


@pytest.mark.asyncio
async def test_auto_with_corpus_and_bootstrap(tmp_path: Path) -> None:
    """Corpus+bootstrap path: sample() is called and its docs are ingested."""
    corpus = _stub_corpus(_synthetic_docs(20))

    sidecar, report = await auto(
        corpus=corpus,
        bootstrap="topic_diverse",
        sample_size=20,
        budget=5,
        store_path=tmp_path / "store.db",
        cache_path=tmp_path / "cache.db",
    )
    try:
        assert isinstance(sidecar, Sidecar)
        assert isinstance(report, AutoReport)
        assert report.n_docs_processed == 20
        # The stub's sample() was awaited once with the requested mode.
        corpus.sample.assert_awaited_once()
        call = corpus.sample.await_args
        assert call.kwargs == {"mode": "topic_diverse", "n": 20}
    finally:
        await sidecar.store.backend.aclose()


@pytest.mark.asyncio
async def test_auto_corpus_without_bootstrap_raises(tmp_path: Path) -> None:
    """corpus=stub + bootstrap='caller' must raise ValueError."""
    corpus = _stub_corpus(_synthetic_docs(5))
    with pytest.raises(ValueError, match="bootstrap"):
        await auto(
            corpus=corpus,
            bootstrap="caller",
            store_path=tmp_path / "store.db",
            cache_path=tmp_path / "cache.db",
        )
    corpus.sample.assert_not_awaited()


@pytest.mark.asyncio
async def test_auto_no_docs_and_no_corpus_raises(tmp_path: Path) -> None:
    """Omitting both docs and corpus is an error."""
    with pytest.raises(ValueError, match="docs"):
        await auto(
            store_path=tmp_path / "store.db",
            cache_path=tmp_path / "cache.db",
        )


@pytest.mark.asyncio
async def test_auto_backward_compat_docs_arg_still_works(tmp_path: Path) -> None:
    """Passing docs= with no corpus= behaves exactly like before."""
    docs = _synthetic_docs(5)
    sidecar, report = await auto(
        docs=docs,
        budget=3,
        store_path=tmp_path / "store.db",
        cache_path=tmp_path / "cache.db",
    )
    try:
        assert report.n_docs_processed == 5
        # The default bootstrap ("caller") preserves the legacy contract.
        assert report.sidecar_mode == "offline-curated"
    finally:
        await sidecar.store.backend.aclose()
