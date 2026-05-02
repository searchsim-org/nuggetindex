"""Tests for the argument-type annotations on :class:`RelationSchema`."""

from __future__ import annotations

from nuggetindex.core.schema import RelationSchema


def test_default_schema_has_arg_types_for_ceo() -> None:
    schema = RelationSchema.default()
    assert schema.expected_subject_types("chiefExecutiveOfficer") == frozenset({"ORG"})
    assert schema.expected_object_types("chiefExecutiveOfficer") == frozenset({"PERSON"})


def test_unknown_predicate_returns_empty_types() -> None:
    schema = RelationSchema.default()
    assert schema.expected_subject_types("made_up_predicate") == frozenset()
    assert schema.expected_object_types("made_up_predicate") == frozenset()


def test_alias_canonicalizes_before_lookup() -> None:
    schema = RelationSchema.default()
    assert schema.expected_object_types("ceo") == schema.expected_object_types(
        "chiefExecutiveOfficer"
    )
    assert schema.expected_subject_types("ceo") == schema.expected_subject_types(
        "chiefExecutiveOfficer"
    )
