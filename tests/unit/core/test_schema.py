import pytest

from nuggetindex.core.errors import InvalidRelationSchema
from nuggetindex.core.schema import RelationSchema


def test_default_schema_loads():
    schema = RelationSchema.default()
    assert schema.is_functional("chiefExecutiveOfficer")
    assert not schema.is_functional("boardMember")


def test_alias_resolution():
    schema = RelationSchema.default()
    # "ceo" is an alias for chiefExecutiveOfficer
    assert schema.canonicalize("ceo") == "chiefExecutiveOfficer"
    assert schema.canonicalize("CEO") == "chiefExecutiveOfficer"


def test_unknown_predicate_defaults_to_not_functional():
    schema = RelationSchema.default()
    # Changed in nuggetindex 0.3: unknown predicates are no longer treated as
    # functional.  The old conservative default caused false CONTESTED flags
    # on common news-verb predicates ("announced", "said", …) that aren't in
    # the YAML.  See Cardinality.MULTI_VALUED as the new default.
    assert not schema.is_functional("someRandomPredicate")


def test_canonicalize_unknown_returns_input():
    schema = RelationSchema.default()
    assert schema.canonicalize("weirdRelation") == "weirdRelation"


def test_invalid_yaml_raises_typed_error(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("predicates: not_a_mapping")
    with pytest.raises(InvalidRelationSchema):
        RelationSchema.from_yaml(bad)


def test_from_yaml_roundtrip(tmp_path):
    yaml_content = """
version: 1
predicates:
  foo:
    functional: true
    aliases: [f, foo_alias]
"""
    p = tmp_path / "schema.yaml"
    p.write_text(yaml_content)
    s = RelationSchema.from_yaml(p)
    assert s.is_functional("foo")
    assert s.canonicalize("f") == "foo"
