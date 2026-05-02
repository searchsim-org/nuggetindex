"""Tests for the optional ``subject_type`` / ``object_type`` fields on
:class:`nuggetindex.core.models.FactTriple` (fix 8).
"""

from __future__ import annotations

from nuggetindex.core.models import FactTriple


def test_fact_triple_defaults_types_to_none() -> None:
    """Legacy callers that don't set the type fields still get a valid
    FactTriple; the fields default to ``None``."""
    ft = FactTriple(
        subject="Google",
        predicate="chiefExecutiveOfficer",
        object="Sundar Pichai",
        text="Sundar Pichai is CEO of Google",
    )
    assert ft.subject_type is None
    assert ft.object_type is None


def test_fact_triple_round_trips_types_through_model_dump() -> None:
    """``subject_type`` / ``object_type`` survive Pydantic JSON round-trip."""
    ft = FactTriple(
        subject="Sundar Pichai",
        predicate="chiefExecutiveOfficer",
        object="Google",
        text="Sundar Pichai is CEO of Google",
        subject_type="PERSON",
        object_type="ORG",
    )
    # Direct dict round-trip.
    dumped = ft.model_dump()
    restored = FactTriple.model_validate(dumped)
    assert restored == ft
    assert restored.subject_type == "PERSON"
    assert restored.object_type == "ORG"
    # JSON round-trip (more stringent: keys + values both serialise).
    json_dumped = ft.model_dump_json()
    restored_json = FactTriple.model_validate_json(json_dumped)
    assert restored_json == ft
    assert restored_json.subject_type == "PERSON"
    assert restored_json.object_type == "ORG"


def test_fact_triple_accepts_case_variants() -> None:
    """The model stores whatever case the extractor returned; normalization
    happens at comparison time (see fix 9 in pipeline/entity_types.py)."""
    ft_lower = FactTriple(
        subject="Elon Musk",
        predicate="chiefExecutiveOfficer",
        object="SpaceX",
        text="Elon Musk runs SpaceX",
        subject_type="person",
        object_type="org",
    )
    assert ft_lower.subject_type == "person"
    assert ft_lower.object_type == "org"

    ft_mixed = FactTriple(
        subject="Elon Musk",
        predicate="chiefExecutiveOfficer",
        object="SpaceX",
        text="Elon Musk runs SpaceX",
        subject_type="Person",
        object_type="Org",
    )
    assert ft_mixed.subject_type == "Person"
    assert ft_mixed.object_type == "Org"


def test_fact_triple_types_preserved_when_frozen() -> None:
    """Type fields respect the frozen-model contract: no mutation allowed."""
    import pytest
    from pydantic import ValidationError

    ft = FactTriple(
        subject="A",
        predicate="p",
        object="B",
        text="t",
        subject_type="ORG",
        object_type="PERSON",
    )
    with pytest.raises(ValidationError):
        ft.subject_type = "PERSON"  # type: ignore[misc]
