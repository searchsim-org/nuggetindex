from datetime import UTC, datetime

from nuggetindex.core.enums import EpistemicRank, LifecycleStatus, NuggetKind
from nuggetindex.core.models import (
    EpistemicState,
    FactTriple,
    Nugget,
    ProvenanceRecord,
    ValidityInterval,
)
from nuggetindex.sidecar.context import ContextFormatter


def _make_nugget(obj: str, status: str, src: str, evidence: str) -> Nugget:
    return Nugget.new(
        kind=NuggetKind.SEMANTIC_FACT,
        fact=FactTriple(
            subject="Google",
            predicate="chiefExecutiveOfficer",
            object=obj,
            text=evidence,
        ),
        validity=ValidityInterval(start=datetime(2011, 1, 1, tzinfo=UTC), end=None),
        epistemic=EpistemicState(
            status=LifecycleStatus(status),
            rank=EpistemicRank.NORMAL,
            confidence=0.9,
        ),
        provenance=(
            ProvenanceRecord(
                source_id=src,
                evidence_span=evidence,
                char_start=0,
                char_end=len(evidence),
                created_at=datetime(2011, 1, 1, tzinfo=UTC),
            ),
        ),
        extraction_confidence=0.9,
    )


def test_format_active_fact():
    f = ContextFormatter()
    nugget = _make_nugget("Larry Page", "active", "wiki", "Page is CEO")
    out = f.format(nuggets=[nugget])
    assert "KNOWN FACTS" in out
    assert "Larry Page" in out
    assert "wiki" in out


def test_format_contested_groups_by_key():
    f = ContextFormatter()
    a = _make_nugget("$26.2B", "contested", "reuters", "Microsoft paid $26.2B")
    b = _make_nugget("$26.4B", "contested", "bloomberg", "Microsoft paid $26.4B")
    out = f.format(nuggets=[a, b])
    assert "DISPUTED" in out
    assert "$26.2B" in out and "$26.4B" in out


def test_format_empty():
    f = ContextFormatter()
    assert f.format(nuggets=[]) == ""


def test_format_validity_known_false_annotated():
    f = ContextFormatter()
    n = Nugget.new(
        kind=NuggetKind.SEMANTIC_FACT,
        fact=FactTriple(
            subject="Google",
            predicate="chiefExecutiveOfficer",
            object="Sundar Pichai",
            text="Pichai is CEO",
        ),
        validity=ValidityInterval(
            start=datetime(2026, 4, 19, tzinfo=UTC),
            end=None,
            validity_known=False,
        ),
        epistemic=EpistemicState(
            status=LifecycleStatus.ACTIVE,
            rank=EpistemicRank.NORMAL,
            confidence=0.9,
        ),
        provenance=(
            ProvenanceRecord(
                source_id="doc1",
                evidence_span="Pichai is CEO",
                char_start=0,
                char_end=13,
                created_at=datetime(2026, 4, 19, tzinfo=UTC),
            ),
        ),
        extraction_confidence=0.9,
    )
    out = f.format(nuggets=[n])
    lowered = out.lower()
    assert "calendar date unknown" in lowered or "source-date" in lowered
