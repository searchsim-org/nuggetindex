from datetime import UTC, datetime

import pytest
from hypothesis import given
from hypothesis import strategies as st
from pydantic import ValidationError

from nuggetindex.core.enums import LifecycleStatus, NuggetKind
from nuggetindex.core.models import (
    EpistemicState,
    FactTriple,
    Nugget,
    ProvenanceRecord,
    ValidityInterval,
)


def _make_nugget(**overrides) -> Nugget:
    defaults = dict(
        kind=NuggetKind.SEMANTIC_FACT,
        fact=FactTriple(
            subject="Google",
            predicate="ceo",
            object="Pichai",
            text="Pichai is CEO of Google",
        ),
        validity=ValidityInterval(start=datetime(2015, 10, 1, tzinfo=UTC)),
        epistemic=EpistemicState(),
        provenance=(ProvenanceRecord(source_id="doc-1", evidence_span="Pichai is CEO"),),
        extraction_confidence=0.95,
        created_at=datetime(2026, 1, 1, tzinfo=UTC),
        updated_at=datetime(2026, 1, 1, tzinfo=UTC),
    )
    defaults.update(overrides)
    return Nugget.new(**defaults)


def test_nugget_id_is_16_hex_chars():
    n = _make_nugget()
    assert len(n.id) == 16


def test_nugget_id_is_deterministic():
    a = _make_nugget()
    b = _make_nugget()
    assert a.id == b.id


def test_nugget_id_changes_when_fact_changes():
    a = _make_nugget()
    b = _make_nugget(
        fact=FactTriple(subject="Google", predicate="ceo", object="Page", text="Page is CEO")
    )
    assert a.id != b.id


def test_nugget_id_changes_when_validity_start_changes():
    a = _make_nugget()
    b = _make_nugget(validity=ValidityInterval(start=datetime(2011, 4, 1, tzinfo=UTC)))
    assert a.id != b.id


def test_nugget_key_tuple():
    n = _make_nugget()
    assert n.key == ("Google", "ceo", "global")


def test_nugget_json_round_trip():
    n = _make_nugget()
    dumped = n.model_dump_json()
    restored = Nugget.model_validate_json(dumped)
    assert restored == n


def test_nugget_is_frozen():
    n = _make_nugget()
    with pytest.raises(ValidationError):
        n.kind = NuggetKind.INSTRUCTION


def test_is_retrievable_at_respects_validity_and_status():
    n = _make_nugget()
    assert n.is_retrievable_at(datetime(2020, 1, 1, tzinfo=UTC))
    # Before validity start -> not retrievable
    assert not n.is_retrievable_at(datetime(2010, 1, 1, tzinfo=UTC))


def test_deprecated_nugget_not_retrievable():
    n = _make_nugget(epistemic=EpistemicState(status=LifecycleStatus.DEPRECATED))
    assert not n.is_retrievable_at(datetime(2020, 1, 1, tzinfo=UTC))


def test_contested_nugget_is_retrievable():
    n = _make_nugget(epistemic=EpistemicState(status=LifecycleStatus.CONTESTED))
    assert n.is_retrievable_at(datetime(2020, 1, 1, tzinfo=UTC))


# Property-based invariant: JSON round-trip preserves all fields.
@given(
    subject=st.text(min_size=1, max_size=30).filter(lambda s: s.strip()),
    predicate=st.text(min_size=1, max_size=30).filter(lambda s: s.strip()),
    obj=st.text(min_size=1, max_size=30).filter(lambda s: s.strip()),
    confidence=st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
)
def test_nugget_round_trip_property(subject, predicate, obj, confidence):
    n = _make_nugget(
        fact=FactTriple(
            subject=subject,
            predicate=predicate,
            object=obj,
            text=f"{subject} {predicate} {obj}",
        ),
        extraction_confidence=confidence,
    )
    restored = Nugget.model_validate_json(n.model_dump_json())
    assert restored.id == n.id
    assert restored.fact == n.fact
    assert abs(restored.extraction_confidence - n.extraction_confidence) < 1e-9
