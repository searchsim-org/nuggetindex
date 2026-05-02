"""Tests for the QualityGate confidence-based partitioning + JSONL queue."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from nuggetindex.core.enums import NuggetKind
from nuggetindex.core.models import (
    EpistemicState,
    FactTriple,
    Nugget,
    ProvenanceRecord,
    ValidityInterval,
)
from nuggetindex.extractors.base import BaseExtractor, ExtractionResult
from nuggetindex.extractors.quality import QualityGate, QualityGateResult


def _make_result(conf: float, *, subject: str = "s") -> ExtractionResult:
    return ExtractionResult(
        nugget=Nugget.new(
            kind=NuggetKind.SEMANTIC_FACT,
            fact=FactTriple(subject=subject, predicate="p", object="o", text="s p o"),
            validity=ValidityInterval(start=datetime(2020, 1, 1, tzinfo=UTC)),
            epistemic=EpistemicState(),
            provenance=(ProvenanceRecord(source_id="d", evidence_span="x"),),
        ),
        confidence=conf,
        rationale=f"test-{conf}",
    )


class _Stub(BaseExtractor):
    def __init__(self, confidences: list[float]) -> None:
        self._confs = confidences

    async def aextract(
        self, text: str, *, context: str = ""
    ) -> list[ExtractionResult]:
        return [_make_result(c, subject=f"s-{i}") for i, c in enumerate(self._confs)]


@pytest.mark.asyncio
async def test_partitions_by_thresholds(tmp_path: Path) -> None:
    queue = tmp_path / "review_queue.jsonl"
    gate = QualityGate(
        _Stub([0.9, 0.7, 0.3]),
        accept_threshold=0.85,
        review_threshold=0.6,
        review_queue_path=queue,
    )
    result = await gate.aextract("some text")
    assert isinstance(result, QualityGateResult)
    assert len(result.accepted) == 1
    assert len(result.deferred) == 1
    assert len(result.rejected) == 1
    assert result.accepted[0].confidence == 0.9
    assert result.deferred[0].confidence == 0.7
    assert result.rejected[0].confidence == 0.3


@pytest.mark.asyncio
async def test_deferred_written_to_jsonl(tmp_path: Path) -> None:
    queue = tmp_path / "review_queue.jsonl"
    gate = QualityGate(
        _Stub([0.9, 0.7, 0.3]),
        accept_threshold=0.85,
        review_threshold=0.6,
        review_queue_path=queue,
    )
    await gate.aextract("some text", context="doc-42")
    lines = queue.read_text().strip().splitlines()
    assert len(lines) == 1
    row = json.loads(lines[0])
    assert row["confidence"] == 0.7
    assert row["source_text"] == "some text"
    assert row["context"] == "doc-42"
    assert row["extractor"] == "_Stub"
    assert row["rationale"] == "test-0.7"
    assert "timestamp" in row and row["timestamp"].endswith("+00:00")
    # The nugget payload is a dict (not a JSON string), so it's re-usable.
    assert isinstance(row["nugget"], dict)
    assert row["nugget"]["fact"]["predicate"] == "p"


@pytest.mark.asyncio
async def test_nothing_deferred_means_no_queue_file(tmp_path: Path) -> None:
    queue = tmp_path / "nested" / "review_queue.jsonl"
    gate = QualityGate(
        _Stub([0.9, 0.2]),
        accept_threshold=0.85,
        review_threshold=0.6,
        review_queue_path=queue,
    )
    await gate.aextract("x")
    assert not queue.exists()


@pytest.mark.asyncio
async def test_queue_file_appended_across_calls(tmp_path: Path) -> None:
    queue = tmp_path / "review_queue.jsonl"
    gate = QualityGate(
        _Stub([0.7]),
        accept_threshold=0.85,
        review_threshold=0.6,
        review_queue_path=queue,
    )
    await gate.aextract("first")
    await gate.aextract("second")
    lines = queue.read_text().strip().splitlines()
    assert len(lines) == 2
    row0 = json.loads(lines[0])
    row1 = json.loads(lines[1])
    assert row0["source_text"] == "first"
    assert row1["source_text"] == "second"


@pytest.mark.asyncio
async def test_boundary_confidence_goes_to_accepted(tmp_path: Path) -> None:
    # Exactly at accept_threshold should accept.
    queue = tmp_path / "review_queue.jsonl"
    gate = QualityGate(
        _Stub([0.85]),
        accept_threshold=0.85,
        review_threshold=0.6,
        review_queue_path=queue,
    )
    result = await gate.aextract("x")
    assert len(result.accepted) == 1
    assert len(result.deferred) == 0


@pytest.mark.asyncio
async def test_boundary_confidence_goes_to_deferred(tmp_path: Path) -> None:
    # Exactly at review_threshold should defer.
    queue = tmp_path / "review_queue.jsonl"
    gate = QualityGate(
        _Stub([0.6]),
        accept_threshold=0.85,
        review_threshold=0.6,
        review_queue_path=queue,
    )
    result = await gate.aextract("x")
    assert len(result.deferred) == 1
    assert len(result.rejected) == 0


def test_invalid_thresholds_rejected() -> None:
    with pytest.raises(ValueError, match="review_threshold"):
        QualityGate(_Stub([]), accept_threshold=0.5, review_threshold=0.9)


@pytest.mark.asyncio
async def test_parent_dirs_created_on_demand(tmp_path: Path) -> None:
    queue = tmp_path / "deeper" / "nested" / "review_queue.jsonl"
    gate = QualityGate(
        _Stub([0.7]),
        accept_threshold=0.85,
        review_threshold=0.6,
        review_queue_path=queue,
    )
    await gate.aextract("x")
    assert queue.exists()
