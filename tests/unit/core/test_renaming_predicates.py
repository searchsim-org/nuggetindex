"""Tests for the ``renaming`` predicate flag on :class:`RelationSchema`."""

from pathlib import Path

from nuggetindex.core.schema import RelationSchema


def test_default_schema_has_renaming_predicates() -> None:
    schema = RelationSchema.default()
    rp = schema.renaming_predicates
    assert "renamedTo" in rp
    assert "corporateName" in rp
    assert "formerlyKnownAs" in rp
    # succeededBy / precededBy are role-succession, not entity-rename.
    assert "succeededBy" not in rp
    assert "precededBy" not in rp


def test_non_renaming_predicates_not_in_set() -> None:
    schema = RelationSchema.default()
    assert "chiefExecutiveOfficer" not in schema.renaming_predicates
    assert "dateOfBirth" not in schema.renaming_predicates


def test_is_renaming_checks_canonical_and_alias() -> None:
    schema = RelationSchema.default()
    assert schema.is_renaming("renamedTo")
    assert schema.is_renaming("renamedAs")  # alias
    assert not schema.is_renaming("chiefExecutiveOfficer")
    assert not schema.is_renaming("ceo")  # alias of non-renaming pred


def test_unknown_predicate_not_renaming_by_default() -> None:
    schema = RelationSchema.default()
    assert not schema.is_renaming("totally_made_up_predicate")


def test_alias_canonicalized_in_renaming_set(tmp_path: Path) -> None:
    yaml = tmp_path / "r.yaml"
    yaml.write_text(
        """
version: 1
predicates:
  rebranded_to:
    functional: true
    renaming: true
    aliases: [renamed_as]
"""
    )
    schema = RelationSchema.from_yaml(yaml)
    assert schema.renaming_predicates == frozenset({"rebranded_to"})
    # Aliases resolve to the canonical name
    assert schema.canonicalize("renamed_as") == "rebranded_to"


def test_renaming_defaults_to_false_when_absent(tmp_path: Path) -> None:
    yaml = tmp_path / "r.yaml"
    yaml.write_text(
        """
version: 1
predicates:
  some_pred:
    functional: true
    aliases: []
"""
    )
    schema = RelationSchema.from_yaml(yaml)
    assert schema.renaming_predicates == frozenset()
    assert not schema.is_renaming("some_pred")


def test_default_entity_rename_predicates_is_narrow_whitelist() -> None:
    schema = RelationSchema.default()
    assert schema.entity_rename_predicates == frozenset(
        {"renamedTo", "formerlyKnownAs", "corporateName"}
    )


def test_entity_rename_predicates_excludes_user_supplied_succession(
    tmp_path: Path,
) -> None:
    # A user-supplied schema that (incorrectly) marks ``succeededBy`` as
    # ``renaming: true`` must still be filtered out of
    # ``entity_rename_predicates`` by the library-level whitelist.
    yaml = tmp_path / "r.yaml"
    yaml.write_text(
        """
version: 1
predicates:
  succeededBy:
    functional: true
    renaming: true
    aliases: [successor]
  renamedTo:
    functional: true
    renaming: true
    aliases: []
"""
    )
    schema = RelationSchema.from_yaml(yaml)
    # renaming_predicates reflects what the YAML said...
    assert "succeededBy" in schema.renaming_predicates
    # ...but entity_rename_predicates enforces the whitelist.
    assert "succeededBy" not in schema.entity_rename_predicates
    assert schema.entity_rename_predicates == frozenset({"renamedTo"})
