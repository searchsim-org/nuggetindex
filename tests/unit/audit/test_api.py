"""Tests for the Tier-0 audit API (Phase 7).

Covers:
* Report dataclasses and their three serializers (JSON / Markdown / Rich).
* `audit()` integration with a test-local rule-based extractor and the
  conflict detector.
* Stale-record heuristic (open-ended validity => flagged).
* Batch mode over a JSONL fixture.
* Top-level re-export.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from nuggetindex import audit, audit_batch
from nuggetindex.audit.api import AuditReport, ConflictRecord, StaleRecord
from nuggetindex.core.enums import NuggetKind
from nuggetindex.core.models import (
    EpistemicState,
    FactTriple,
    Nugget,
    ProvenanceRecord,
    ValidityInterval,
)
from tests.fixtures import RuleBasedExtractor


def _make_nugget(
    *,
    subject: str = "Google",
    predicate: str = "ceo",
    obj: str = "Pichai",
    start: datetime | None = None,
    end: datetime | None = None,
) -> Nugget:
    start = start or datetime(2020, 1, 1, tzinfo=UTC)
    return Nugget.new(
        kind=NuggetKind.SEMANTIC_FACT,
        fact=FactTriple(subject=subject, predicate=predicate, object=obj, text="x"),
        validity=ValidityInterval(start=start, end=end),
        epistemic=EpistemicState(),
        provenance=(ProvenanceRecord(source_id="doc-1", evidence_span="x"),),
    )


# --- Task 7.1: dataclasses + serializers ----------------------------------


def test_audit_report_empty_defaults() -> None:
    report = AuditReport()
    assert report.conflicts == []
    assert report.potentially_stale == []
    assert report.consistent == 0


def test_audit_report_to_json_roundtrips() -> None:
    n_a = _make_nugget(obj="Pichai")
    n_b = _make_nugget(obj="Page")
    report = AuditReport(
        conflicts=[
            ConflictRecord(
                key=n_a.key,
                nugget_a=n_a,
                nugget_b=n_b,
                reason="functional predicate + overlapping validity",
            )
        ],
        potentially_stale=[
            StaleRecord(nugget=n_a, source_date=None, reason="no explicit end-time")
        ],
        consistent=3,
    )
    s = report.to_json()
    parsed = json.loads(s)
    assert parsed["consistent"] == 3
    assert len(parsed["conflicts"]) == 1
    assert parsed["conflicts"][0]["reason"].startswith("functional")
    assert parsed["conflicts"][0]["nugget_a"]["fact"]["object"] == "Pichai"
    assert parsed["conflicts"][0]["nugget_b"]["fact"]["object"] == "Page"
    assert parsed["potentially_stale"][0]["reason"] == "no explicit end-time"


def test_audit_report_to_markdown_contains_sections() -> None:
    n_a = _make_nugget(obj="Pichai")
    n_b = _make_nugget(obj="Page")
    report = AuditReport(
        conflicts=[
            ConflictRecord(
                key=n_a.key,
                nugget_a=n_a,
                nugget_b=n_b,
                reason="functional predicate + overlapping validity",
            )
        ],
        potentially_stale=[
            StaleRecord(nugget=n_a, source_date=None, reason="no explicit end-time")
        ],
        consistent=2,
    )
    md = report.to_markdown()
    assert "# Audit Report" in md
    assert "## Conflicts" in md
    assert "## Potentially Stale" in md
    assert "Consistent" in md
    assert "Pichai" in md
    assert "Page" in md


def test_audit_report_to_rich_console_is_renderable() -> None:
    from io import StringIO

    from rich.console import Console

    n_a = _make_nugget(obj="Pichai")
    n_b = _make_nugget(obj="Page")
    report = AuditReport(
        conflicts=[
            ConflictRecord(
                key=n_a.key,
                nugget_a=n_a,
                nugget_b=n_b,
                reason="functional predicate + overlapping validity",
            )
        ],
        consistent=1,
    )
    renderable = report.to_rich_console()
    buf = StringIO()
    Console(file=buf, width=120, record=True).print(renderable)
    out = buf.getvalue()
    assert "Pichai" in out
    assert "Page" in out
    # Panel summary mentions the consistent count.
    assert "1" in out


# --- Task 7.2: audit() integration ----------------------------------------


@pytest.mark.asyncio
async def test_audit_with_rule_based_fixture_extractor() -> None:
    """Passing the test-local rule-based fixture extractor. No API keys, no network."""
    # stale_threshold_days=None restores the v0.2.0 rule (any open-ended
    # validity => flagged) so this assertion stays meaningful under the
    # v0.2.1 age-aware heuristic; see test_staleness_heuristic.py for the
    # default-threshold behaviour.
    report = await audit(
        query="Who is the CEO of Google?",
        passages=["Sundar Pichai is CEO of Google."],
        query_time=datetime(2020, 1, 1, tzinfo=UTC),
        stale_threshold_days=None,
        extractor=RuleBasedExtractor(),
    )
    assert isinstance(report, AuditReport)
    # One extracted nugget; open-ended => goes into potentially_stale.
    assert len(report.potentially_stale) >= 1
    assert report.conflicts == []


@pytest.mark.asyncio
async def test_audit_detects_conflicting_ceos() -> None:
    """Two passages asserting different CEOs for the same company => conflict."""
    report = await audit(
        query="Who is the CEO of Google?",
        passages=[
            "Sundar Pichai is CEO of Google.",
            "Larry Page is CEO of Google.",
        ],
        query_time=datetime(2020, 1, 1, tzinfo=UTC),
        extractor=RuleBasedExtractor(),
    )
    assert len(report.conflicts) >= 1
    conflict = report.conflicts[0]
    objects = {conflict.nugget_a.fact.object, conflict.nugget_b.fact.object}
    assert objects == {"Sundar Pichai", "Larry Page"}
    assert "functional" in conflict.reason


@pytest.mark.asyncio
async def test_audit_empty_passages_returns_empty_report() -> None:
    report = await audit(
        query="anything",
        passages=[],
        query_time=datetime(2020, 1, 1, tzinfo=UTC),
        extractor=RuleBasedExtractor(),
    )
    assert report.conflicts == []
    assert report.potentially_stale == []
    assert report.consistent == 0


@pytest.mark.asyncio
async def test_audit_accepts_custom_extractor_instance() -> None:
    """Passing a BaseExtractor instance bypasses string-based lookup."""
    ex = RuleBasedExtractor()
    report = await audit(
        query="q",
        passages=["Sundar Pichai is CEO of Google."],
        query_time=datetime(2020, 1, 1, tzinfo=UTC),
        extractor=ex,
    )
    assert isinstance(report, AuditReport)


# --- Task 7.3: batch mode -------------------------------------------------


@pytest.mark.asyncio
async def test_audit_batch_over_jsonl(tmp_path: Path) -> None:
    rows = [
        {
            "query": "Who is the CEO of Google?",
            "passages": [
                "Sundar Pichai is CEO of Google.",
                "Larry Page is CEO of Google.",
            ],
        },
        {
            "query": "Who is the CEO of OpenAI?",
            "passages": ["Sam Altman is CEO of OpenAI."],
        },
    ]
    jsonl_path = tmp_path / "batch.jsonl"
    with jsonl_path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")

    reports = await audit_batch(
        jsonl_path=jsonl_path,
        query_time=datetime(2020, 1, 1, tzinfo=UTC),
        extractor=RuleBasedExtractor(),
    )
    assert len(reports) == 2
    assert len(reports[0].conflicts) >= 1
    assert reports[1].conflicts == []


@pytest.mark.asyncio
async def test_audit_batch_skips_blank_lines(tmp_path: Path) -> None:
    jsonl_path = tmp_path / "batch.jsonl"
    jsonl_path.write_text(
        json.dumps({"query": "q", "passages": ["Sundar Pichai is CEO of Google."]})
        + "\n\n"
    )
    reports = await audit_batch(
        jsonl_path=jsonl_path,
        query_time=datetime(2020, 1, 1, tzinfo=UTC),
        extractor=RuleBasedExtractor(),
    )
    assert len(reports) == 1


# --- Task 7.4: top-level re-export ----------------------------------------


def test_public_reexports() -> None:
    import nuggetindex as ni

    assert ni.audit is audit
    assert ni.audit_batch is audit_batch
    assert ni.AuditReport is AuditReport
    assert ni.ConflictRecord is ConflictRecord
    assert ni.StaleRecord is StaleRecord
