"""Tests for the ``Cardinality`` enum and ``RelationSchema.cardinality``."""

from pathlib import Path

from nuggetindex.core.enums import Cardinality
from nuggetindex.core.schema import Relation, RelationKind, RelationSchema


def test_cardinality_enum_has_three_values() -> None:
    values = {c.value for c in Cardinality}
    assert values == {"functional", "multi_valued", "event_log"}


def test_relation_default_cardinality_is_functional() -> None:
    # Back-compat: a Relation built without an explicit `cardinality` kwarg
    # defaults to FUNCTIONAL so existing schema-construction code keeps
    # behaving the same way.
    r = Relation(name="foo", kind=RelationKind.FUNCTIONAL)
    assert r.cardinality == Cardinality.FUNCTIONAL


def test_schema_cardinality_event_log_for_announced() -> None:
    schema = RelationSchema.default()
    assert schema.cardinality("announced") == Cardinality.EVENT_LOG


def test_schema_cardinality_multi_valued_for_acquired() -> None:
    # `acquired` is listed as cardinality: multi_valued in the bundled YAML.
    schema = RelationSchema.default()
    assert schema.cardinality("acquired") == Cardinality.MULTI_VALUED


def test_schema_cardinality_unknown_predicate_defaults_multi_valued() -> None:
    # The Mode-A fix: unknown predicates must NOT be treated as functional
    # (would cause false CONTESTED flags in ConflictDetector).
    schema = RelationSchema.default()
    assert schema.cardinality("made_up_predicate") == Cardinality.MULTI_VALUED


def test_schema_cardinality_alias_canonicalized() -> None:
    schema = RelationSchema.default()
    assert schema.cardinality("announces") == schema.cardinality("announced")
    assert schema.cardinality("announces") == Cardinality.EVENT_LOG


def test_schema_cardinality_derived_from_functional_when_omitted(
    tmp_path: Path,
) -> None:
    # YAML without an explicit `cardinality` key should derive it from the
    # legacy `functional` bool so existing user-supplied schemas keep working.
    yaml = tmp_path / "r.yaml"
    yaml.write_text(
        """
version: 1
predicates:
  fooFunc:
    functional: true
  fooMulti:
    functional: false
"""
    )
    schema = RelationSchema.from_yaml(yaml)
    assert schema.cardinality("fooFunc") == Cardinality.FUNCTIONAL
    assert schema.cardinality("fooMulti") == Cardinality.MULTI_VALUED


def test_schema_cardinality_yaml_explicit_overrides_functional(
    tmp_path: Path,
) -> None:
    # When both are present, `cardinality` wins.
    yaml = tmp_path / "r.yaml"
    yaml.write_text(
        """
version: 1
predicates:
  newsVerb:
    functional: false
    cardinality: event_log
    aliases: [reported]
"""
    )
    schema = RelationSchema.from_yaml(yaml)
    assert schema.cardinality("newsVerb") == Cardinality.EVENT_LOG
    assert schema.cardinality("reported") == Cardinality.EVENT_LOG
