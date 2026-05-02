"""Tests for Stage 2 canonicalization (subject + predicate)."""

from datetime import UTC, datetime

from nuggetindex.core.enums import NuggetKind
from nuggetindex.core.models import (
    EpistemicState,
    FactTriple,
    Nugget,
    ProvenanceRecord,
    ValidityInterval,
)
from nuggetindex.core.schema import RelationSchema
from nuggetindex.pipeline.canonicalize import canonicalize


def _n(*, subject: str = "  google  ", predicate: str = "ceo", obj: str = "Pichai") -> Nugget:
    return Nugget.new(
        kind=NuggetKind.SEMANTIC_FACT,
        fact=FactTriple(subject=subject, predicate=predicate, object=obj, text="x"),
        validity=ValidityInterval(start=datetime(2015, 10, 1, tzinfo=UTC)),
        epistemic=EpistemicState(),
        provenance=(ProvenanceRecord(source_id="d", evidence_span="x"),),
    )


def test_subject_whitespace_normalized() -> None:
    schema = RelationSchema.default()
    result = canonicalize(_n(), schema)
    assert result.fact.subject == "google"


def test_predicate_canonicalized_via_schema() -> None:
    schema = RelationSchema.default()
    result = canonicalize(_n(predicate="ceo"), schema)
    assert result.fact.predicate == "chiefExecutiveOfficer"


def test_unknown_predicate_passthrough() -> None:
    schema = RelationSchema.default()
    result = canonicalize(_n(predicate="unmapped_predicate_xyz"), schema)
    assert result.fact.predicate == "unmapped_predicate_xyz"


def test_canonical_id_changes_when_predicate_changes() -> None:
    schema = RelationSchema.default()
    original = _n(predicate="ceo")
    result = canonicalize(original, schema)
    # Predicate changed ceo -> chiefExecutiveOfficer, so id must change
    assert result.id != original.id


def test_canonical_id_stable_when_inputs_unchanged() -> None:
    schema = RelationSchema.default()
    # Already-canonical subject + predicate: ID should match `.new`'s hash.
    pre_norm = _n(subject="google", predicate="chiefExecutiveOfficer")
    result = canonicalize(pre_norm, schema)
    assert result.id == pre_norm.id


def test_nugget_fields_preserved() -> None:
    schema = RelationSchema.default()
    original = _n()
    result = canonicalize(original, schema)
    assert result.kind == original.kind
    assert result.fact.object == original.fact.object
    assert result.fact.text == original.fact.text
    assert result.validity == original.validity
    assert result.provenance == original.provenance
    assert result.epistemic.status == original.epistemic.status
