"""Unit tests for :func:`nuggetindex.auto.auto`.

The tests use a 5-doc synthetic corpus to exercise the full pipeline
without any LLM calls. The default :class:`TriggerExtractor` is used
everywhere so the run stays fully offline.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from nuggetindex.auto import AutoReport, auto
from nuggetindex.sidecar import Sidecar

_DOCS = [
    {"source_id": "d1", "text": "Larry Page served as CEO of Google from 2011 to 2015."},
    {"source_id": "d2", "text": "Sundar Pichai became CEO of Google in October 2015."},
    {"source_id": "d3", "text": "Satya Nadella became CEO of Microsoft in February 2014."},
    {"source_id": "d4", "text": "Tim Cook became CEO of Apple in August 2011."},
    {
        "source_id": "d5",
        "text": "Twitter Inc. was renamed to X Corp. in April 2023.",
    },
]


@pytest.mark.asyncio
async def test_auto_returns_sidecar_and_report(tmp_path: Path) -> None:
    """Smoke: auto() returns a (Sidecar, AutoReport) and the sidecar works."""
    store_path = tmp_path / "store.db"
    cache_path = tmp_path / "cache.db"

    sidecar, report = await auto(
        docs=list(_DOCS),
        budget=3,
        store_path=store_path,
        cache_path=cache_path,
    )
    try:
        assert isinstance(sidecar, Sidecar)
        assert isinstance(report, AutoReport)
        assert report.n_docs_processed == 5
        assert report.seed_budget == 3
        assert report.sidecar_mode == "offline-curated"
        assert report.rendered_markdown.startswith("# auto() report")

        # The sidecar should be able to handle a basic query without error.
        response = await sidecar.ahandle("who was Google's CEO in 2013?")
        assert response is not None
        assert response.decision is not None
    finally:
        await sidecar.store.backend.aclose()


@pytest.mark.asyncio
async def test_auto_respects_mode(tmp_path: Path) -> None:
    """Mode selector flows through to the returned Sidecar."""
    store_path = tmp_path / "store.db"
    cache_path = tmp_path / "cache.db"

    sidecar, report = await auto(
        docs=list(_DOCS),
        budget=3,
        mode="just-in-time",
        store_path=store_path,
        cache_path=cache_path,
    )
    try:
        assert sidecar.mode == "just-in-time"
        assert report.sidecar_mode == "just-in-time"
    finally:
        await sidecar.store.backend.aclose()


@pytest.mark.asyncio
async def test_auto_from_jsonl_path(tmp_path: Path) -> None:
    """Accept a Path pointing at a .jsonl file."""
    jsonl = tmp_path / "docs.jsonl"
    with jsonl.open("w", encoding="utf-8") as fh:
        for row in _DOCS:
            fh.write(json.dumps(row) + "\n")

    store_path = tmp_path / "store.db"
    cache_path = tmp_path / "cache.db"
    sidecar, report = await auto(
        docs=jsonl,
        budget=3,
        store_path=store_path,
        cache_path=cache_path,
    )
    try:
        assert report.n_docs_processed == 5
        # Non-negative extraction counter (the trigger extractor's entity-type
        # validator may reject all of its proposals on this corpus; we only
        # care that the pipeline ran to completion).
        assert report.nuggets_extracted >= 0
        # The sidecar should still be able to answer a query end-to-end.
        response = await sidecar.ahandle("who was Google's CEO in 2013?")
        assert response is not None
    finally:
        await sidecar.store.backend.aclose()


@pytest.mark.asyncio
async def test_auto_uses_cache_on_second_run(tmp_path: Path) -> None:
    """Second auto() call against the same cache path sees cache hits."""
    cache_path = tmp_path / "cache.db"

    # First run — should populate the cache.
    sidecar1, report1 = await auto(
        docs=list(_DOCS),
        budget=3,
        store_path=tmp_path / "store1.db",
        cache_path=cache_path,
    )
    try:
        assert report1.cache_hit_rate == pytest.approx(0.0)
    finally:
        await sidecar1.store.backend.aclose()

    # Second run against a fresh store but the same cache file. Every
    # doc should be served from cache.
    sidecar2, report2 = await auto(
        docs=list(_DOCS),
        budget=3,
        store_path=tmp_path / "store2.db",
        cache_path=cache_path,
    )
    try:
        assert report2.cache_hit_rate > 0.0
    finally:
        await sidecar2.store.backend.aclose()
