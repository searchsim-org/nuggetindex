"""Behavioural tests for the deep-mode ``scan_index`` implementation (Task 2.5).

Deep mode runs the full ``NuggetStore.aingest`` pipeline -- extractor,
canonicalize, temporal inference, dedup, and conflict resolution -- on a
stratified sample against a transient in-memory store. To keep these tests
hermetic (no LLM calls) we inject a :class:`_ScriptedExtractor` that returns
pre-constructed :class:`Nugget` objects keyed by ``source_id``.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from nuggetindex.audit.doctor import DoctorReport, scan_index
from nuggetindex.core.enums import EpistemicRank, LifecycleStatus, NuggetKind
from nuggetindex.core.models import (
    EpistemicState,
    FactTriple,
    Nugget,
    ProvenanceRecord,
    ValidityInterval,
)
from nuggetindex.extractors.base import BaseExtractor, ExtractionResult
from nuggetindex.pipeline.constructor import Document


class _ScriptedExtractor(BaseExtractor):
    """Deterministic extractor used only by these tests.

    The pipeline probes for a ``source_id`` kwarg via
    :func:`nuggetindex.extractors.base.accepts_source_id`, so declaring it
    here is what lets each scripted row be dispatched to the right document.
    """

    def __init__(
        self,
        script: dict[
            str,
            list[tuple[str, str, str, datetime | None, datetime | None, str]],
        ],
    ) -> None:
        # script maps source_id -> list of
        # (subject, predicate, object, vstart, vend, status).
        self._script = script

    async def aextract(
        self,
        text: str,
        *,
        context: str = "",
        source_id: str = "",
    ) -> list[ExtractionResult]:
        rows = self._script.get(source_id, [])
        results: list[ExtractionResult] = []
        for subj, pred, obj, vstart, vend, status in rows:
            now = datetime.now(tz=UTC)
            evidence = text[:200] if text else "(no text)"
            nugget = Nugget.new(
                kind=NuggetKind.SEMANTIC_FACT,
                fact=FactTriple(
                    subject=subj, predicate=pred, object=obj, text=evidence
                ),
                validity=ValidityInterval(
                    start=vstart or now,
                    end=vend,
                    validity_known=vstart is not None,
                ),
                epistemic=EpistemicState(
                    status=LifecycleStatus(status),
                    rank=EpistemicRank.NORMAL,
                    confidence=0.9,
                ),
                provenance=(
                    ProvenanceRecord(
                        source_id=source_id,
                        evidence_span=evidence,
                        char_start=0,
                        char_end=len(evidence),
                        created_at=now,
                    ),
                ),
                extraction_confidence=0.9,
            )
            results.append(ExtractionResult(nugget=nugget, confidence=0.9))
        return results


@pytest.mark.asyncio
async def test_deep_mode_requires_extractor() -> None:
    with pytest.raises(ValueError, match="extractor"):
        await scan_index(docs=[], mode="deep", sample_size=1)


@pytest.mark.asyncio
async def test_deep_mode_shape() -> None:
    extractor = _ScriptedExtractor(script={})
    docs = [
        Document(
            source_id="d1",
            text="foo",
            source_date=datetime(2020, 1, 1, tzinfo=UTC),
        )
    ]
    report = await scan_index(
        docs=docs, mode="deep", sample_size=10, extractor=extractor
    )
    assert isinstance(report, DoctorReport)
    assert report.sample_mode == "deep"
    assert len(report.scores) == 4
    assert {s.dimension for s in report.scores} == {
        "temporal_depth",
        "temporal_drift",
        "conflict_surface",
        "rename_events",
    }


@pytest.mark.asyncio
async def test_deep_mode_surfaces_contested() -> None:
    """Two documents produce conflicting claims on a functional key; deep-mode flags CONTESTED."""
    # Uses ``chiefExecutiveOfficer`` — still genuinely functional in the
    # default schema.  (Prior versions of this test used ``acquiredFor``, an
    # unknown predicate that the 0.3 cardinality fix now treats as
    # multi-valued; that's semantically correct but no longer triggers
    # CONTESTED, so the test was retargeted onto a real functional key.)
    script: dict[
        str,
        list[tuple[str, str, str, datetime | None, datetime | None, str]],
    ] = {
        "d1": [
            (
                "Apple",
                "chiefExecutiveOfficer",
                "Tim Cook",
                datetime(2018, 1, 1, tzinfo=UTC),
                None,
                "active",
            )
        ],
        "d2": [
            (
                "Apple",
                "chiefExecutiveOfficer",
                "Steve Jobs",
                datetime(2018, 1, 1, tzinfo=UTC),
                None,
                "active",
            )
        ],
    }
    docs = [
        Document(
            source_id="d1",
            text="Tim Cook is CEO of Apple.",
            source_date=datetime(2018, 6, 14, tzinfo=UTC),
        ),
        Document(
            source_id="d2",
            text="Steve Jobs is CEO of Apple.",
            source_date=datetime(2018, 6, 14, tzinfo=UTC),
        ),
    ]
    extractor = _ScriptedExtractor(script=script)
    report = await scan_index(
        docs=docs, mode="deep", sample_size=10, extractor=extractor
    )
    conflict = next(s for s in report.scores if s.dimension == "conflict_surface")
    assert conflict.percentage > 0.0


@pytest.mark.asyncio
async def test_deep_mode_rename_event_surfaces() -> None:
    script: dict[
        str,
        list[tuple[str, str, str, datetime | None, datetime | None, str]],
    ] = {
        "d1": [
            (
                "Twitter Inc.",
                "renamedTo",
                "X Corp.",
                datetime(2023, 4, 11, tzinfo=UTC),
                None,
                "active",
            )
        ],
    }
    docs = [
        Document(
            source_id="d1",
            text="Twitter Inc. was renamed to X Corp. in 2023.",
            source_date=datetime(2023, 4, 12, tzinfo=UTC),
        )
    ]
    extractor = _ScriptedExtractor(script=script)
    report = await scan_index(
        docs=docs, mode="deep", sample_size=10, extractor=extractor
    )
    rename = next(s for s in report.scores if s.dimension == "rename_events")
    assert rename.percentage > 0.0


@pytest.mark.asyncio
async def test_deep_mode_markdown_has_deep_label() -> None:
    extractor = _ScriptedExtractor(script={})
    docs = [Document(source_id="d1", text="foo", source_date=None)]
    report = await scan_index(
        docs=docs, mode="deep", sample_size=10, extractor=extractor
    )
    assert "deep" in report.rendered_markdown.lower()
