"""Behavioural tests for the fast-mode ``scan_index`` implementation (Task 2.4).

These tests exercise the stratified-sample + heuristic pipeline end-to-end on
a small synthetic corpus that contains:

* a role-succession pair (Larry Page / Sundar Pichai at Google),
* an entity-rename statement (Twitter -> X Corp.),
* a pair of documents disagreeing on a functional predicate
  (conflicting headquarters for Microsoft),
* and a control document with no interesting triggers.

The heuristic layer (spaCy-based TIMEX) is optional, so the tests only assert
on signals that can be produced by the stdlib-only trigger scanner. The
temporal dimensions are covered by shape assertions and gracefully degrade
when the ``[doctor]`` extra is not installed.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from nuggetindex.audit.doctor import DoctorReport, scan_index
from nuggetindex.pipeline.constructor import Document


@pytest.fixture
def synthetic_corpus() -> list[Document]:
    return [
        Document(
            source_id="d1",
            text="Larry Page became CEO of Google in 2011.",
            source_date=datetime(2011, 4, 5, tzinfo=UTC),
        ),
        Document(
            source_id="d2",
            text="Sundar Pichai became CEO of Google in 2015.",
            source_date=datetime(2015, 10, 3, tzinfo=UTC),
        ),
        Document(
            source_id="d3",
            text="Twitter Inc. was renamed to X Corp. in 2023.",
            source_date=datetime(2023, 4, 12, tzinfo=UTC),
        ),
        # Two documents disagreeing on a TRULY functional predicate.
        # ``headquarteredIn`` stays ``functional: true`` in the default schema
        # and is emitted by the trigger scanner; the conflict detector should
        # still flag competing single-valued claims.
        Document(
            source_id="d4",
            text="Microsoft is headquartered in Redmond.",
            source_date=datetime(2016, 6, 14, tzinfo=UTC),
        ),
        Document(
            source_id="d5",
            text="Microsoft is headquartered in Seattle.",
            source_date=datetime(2016, 6, 14, tzinfo=UTC),
        ),
        Document(source_id="d6", text="The sky is blue.", source_date=None),
    ]


@pytest.mark.asyncio
async def test_fast_mode_shape(synthetic_corpus: list[Document]) -> None:
    report = await scan_index(docs=synthetic_corpus, mode="fast", sample_size=10)
    assert isinstance(report, DoctorReport)
    assert report.sample_mode == "fast"
    assert len(report.scores) == 4
    assert {s.dimension for s in report.scores} == {
        "temporal_depth",
        "temporal_drift",
        "conflict_surface",
        "rename_events",
    }


@pytest.mark.asyncio
async def test_fast_mode_detects_rename(synthetic_corpus: list[Document]) -> None:
    report = await scan_index(docs=synthetic_corpus, mode="fast", sample_size=10)
    rename = next(s for s in report.scores if s.dimension == "rename_events")
    assert rename.percentage > 0.0


@pytest.mark.asyncio
async def test_fast_mode_detects_conflict(synthetic_corpus: list[Document]) -> None:
    report = await scan_index(docs=synthetic_corpus, mode="fast", sample_size=10)
    conflict = next(s for s in report.scores if s.dimension == "conflict_surface")
    assert conflict.percentage > 0.0


@pytest.mark.asyncio
async def test_fast_mode_ci_bounds_valid(synthetic_corpus: list[Document]) -> None:
    report = await scan_index(docs=synthetic_corpus, mode="fast", sample_size=10)
    for s in report.scores:
        assert 0.0 <= s.percentage <= 100.0
        low, high = s.ci95
        assert 0.0 <= low <= s.percentage + 0.001  # float slack
        assert s.percentage - 0.001 <= high <= 100.0


@pytest.mark.asyncio
async def test_fast_mode_verdict_nonlow_on_synthetic(
    synthetic_corpus: list[Document],
) -> None:
    report = await scan_index(docs=synthetic_corpus, mode="fast", sample_size=10)
    assert report.verdict in {"medium", "high"}


@pytest.mark.asyncio
async def test_fast_mode_markdown_contains_verdict(
    synthetic_corpus: list[Document],
) -> None:
    report = await scan_index(docs=synthetic_corpus, mode="fast", sample_size=10)
    md_lower = report.rendered_markdown.lower()
    assert "verdict" in md_lower
    assert report.verdict in md_lower


@pytest.mark.asyncio
async def test_fast_mode_empty_docs() -> None:
    report = await scan_index(docs=[], mode="fast", sample_size=10)
    assert report.sample_mode == "fast"
    assert report.verdict == "low"
    assert all(s.percentage == 0.0 for s in report.scores)


@pytest.mark.asyncio
async def test_deep_mode_requires_extractor() -> None:
    # Task 2.5 replaced the NotImplementedError placeholder with a real
    # implementation that requires an ``extractor=`` argument; without one,
    # deep mode raises ValueError early.
    with pytest.raises(ValueError, match="extractor"):
        await scan_index(docs=[], mode="deep", sample_size=10)
