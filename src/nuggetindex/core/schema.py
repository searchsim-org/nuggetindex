"""Relation schema: predicate vocabulary + functional/multi-valued metadata.

Drives conflict detection: functional predicates trigger conflict resolution
on temporal overlap; multi-valued predicates coexist.

Predicates may also be flagged ``renaming: true`` in YAML -- meaning a
statement ``(subject, pred, object)`` encodes that the *subject becomes the
object* (rebrands, mergers, etc.). Rename chain walks in :mod:`chains` use
this flag to identify which edges are traversable.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

import yaml

from nuggetindex.core.enums import Cardinality
from nuggetindex.core.errors import InvalidRelationSchema


class RelationKind(StrEnum):
    FUNCTIONAL = "functional"
    MULTI_VALUED = "multi_valued"


@dataclass(frozen=True)
class Relation:
    name: str
    kind: RelationKind
    renaming: bool = False
    cardinality: Cardinality = Cardinality.FUNCTIONAL
    aliases: tuple[str, ...] = ()
    # Optional NER argument-type constraints (spaCy label strings, e.g.
    # ``"ORG"`` / ``"PERSON"`` / ``"GPE"``). An empty set means "no
    # constraint" -- the pipeline lets any type through. Populated from
    # ``expected_subject_types`` / ``expected_object_types`` in YAML.
    expected_subject_types: frozenset[str] = frozenset()
    expected_object_types: frozenset[str] = frozenset()


# Hard-coded narrow whitelist of predicates that are semantically
# entity-rename (the thing *itself* changed name), as opposed to
# role-succession (the *role-holder* changed but the entity did not).
# ``entity_rename_predicates`` intersects the schema's ``renaming`` flags
# with this set so user-supplied schemas cannot smuggle role-succession
# predicates (e.g. ``succeededBy``) into entity-rename walks.
_ENTITY_RENAME_WHITELIST: frozenset[str] = frozenset(
    {"renamedTo", "formerlyKnownAs", "corporateName"}
)


_DEFAULT_YAML_PATH = Path(__file__).parent / "schemas" / "default_predicates.yaml"


class RelationSchema:
    """Predicate vocabulary with alias resolution.

    Unknown predicates default to :attr:`Cardinality.MULTI_VALUED` (conservative
    default that avoids false CONTESTED flags from the conflict detector on
    news-verb / event-log predicates that aren't in the YAML). Prior to
    nuggetindex 0.3 the default was FUNCTIONAL, which caused over-flagging in
    practice. Unknown predicates are NOT considered renaming.
    """

    def __init__(self, relations: Iterable[Relation]) -> None:
        self._by_name: dict[str, Relation] = {}
        self._alias_to_canonical: dict[str, str] = {}
        materialised = list(relations)
        for r in materialised:
            self._by_name[r.name] = r
            self._alias_to_canonical[r.name.lower()] = r.name
            for a in r.aliases:
                self._alias_to_canonical[a.lower()] = r.name
        self._renaming: frozenset[str] = frozenset(r.name for r in materialised if r.renaming)

    @classmethod
    def default(cls) -> RelationSchema:
        """Load the bundled 50-predicate schema."""
        return cls.from_yaml(_DEFAULT_YAML_PATH)

    @classmethod
    def from_yaml(cls, path: Path | str) -> RelationSchema:
        path = Path(path)
        try:
            raw = yaml.safe_load(path.read_text())
        except yaml.YAMLError as e:
            raise InvalidRelationSchema(f"cannot parse {path}: {e}") from e

        if not isinstance(raw, dict) or "predicates" not in raw:
            raise InvalidRelationSchema(f"{path}: missing 'predicates' key")
        preds = raw["predicates"]
        if not isinstance(preds, dict):
            raise InvalidRelationSchema(f"{path}: 'predicates' must be a mapping")

        relations: list[Relation] = []
        for name, spec in preds.items():
            if not isinstance(spec, dict):
                raise InvalidRelationSchema(f"{path}: predicate {name} must be a mapping")
            # ``functional`` drives the pre-existing ``RelationKind``.  For
            # backward compat it still defaults to True when absent.
            functional = bool(spec.get("functional", True))
            renaming = bool(spec.get("renaming", False))
            aliases_raw = spec.get("aliases", []) or []
            if not isinstance(aliases_raw, list):
                raise InvalidRelationSchema(f"{path}: {name}.aliases must be a list")
            kind = RelationKind.FUNCTIONAL if functional else RelationKind.MULTI_VALUED

            # ``cardinality`` is the newer (orthogonal) field.  When present in
            # the YAML it takes precedence; otherwise it's derived from
            # ``kind`` so existing ``functional: true|false`` schemas keep
            # their prior behaviour.
            cardinality_raw = spec.get("cardinality")
            if cardinality_raw is not None:
                try:
                    cardinality = Cardinality(str(cardinality_raw))
                except ValueError as e:
                    raise InvalidRelationSchema(
                        f"{path}: {name}.cardinality must be one of "
                        f"{[c.value for c in Cardinality]}"
                    ) from e
            else:
                cardinality = (
                    Cardinality.FUNCTIONAL
                    if kind == RelationKind.FUNCTIONAL
                    else Cardinality.MULTI_VALUED
                )

            # Optional argument-type annotations. Both keys are lists of
            # spaCy NER labels (e.g. ``[ORG]`` / ``[PERSON, ORG]``). Missing
            # keys default to empty ``frozenset``s (no constraint).
            expected_subj_raw = spec.get("expected_subject_types", []) or []
            expected_obj_raw = spec.get("expected_object_types", []) or []
            if not isinstance(expected_subj_raw, list):
                raise InvalidRelationSchema(f"{path}: {name}.expected_subject_types must be a list")
            if not isinstance(expected_obj_raw, list):
                raise InvalidRelationSchema(f"{path}: {name}.expected_object_types must be a list")
            expected_subj = frozenset(str(t) for t in expected_subj_raw)
            expected_obj = frozenset(str(t) for t in expected_obj_raw)

            relations.append(
                Relation(
                    name=name,
                    kind=kind,
                    renaming=renaming,
                    cardinality=cardinality,
                    aliases=tuple(aliases_raw),
                    expected_subject_types=expected_subj,
                    expected_object_types=expected_obj,
                )
            )
        return cls(relations)

    def is_functional(self, predicate: str) -> bool:
        """Return True iff the (canonicalised) predicate is functional.

        For predicates present in the schema this reflects the ``functional``
        YAML flag as it always has.  For unknown predicates the behaviour was
        changed in nuggetindex 0.3: the old "conservative" default returned
        True (treating unknowns as single-valued), which caused false CONTESTED
        flags from the conflict detector on common news-verbs. The new default
        is False, consistent with :meth:`cardinality` returning
        :attr:`Cardinality.MULTI_VALUED` for unknowns.
        """
        canonical = self.canonicalize(predicate)
        r = self._by_name.get(canonical)
        if r is None:
            return False
        return r.kind == RelationKind.FUNCTIONAL

    def cardinality(self, predicate: str) -> Cardinality:
        """Return the :class:`Cardinality` of the (canonicalised) predicate.

        Lookup order:

        1. If the predicate (or one of its aliases) resolves to a schema entry
           with an explicit ``cardinality`` field, that value is returned.
        2. Otherwise it is derived from the schema entry's ``RelationKind``.
        3. Unknown predicates return :attr:`Cardinality.MULTI_VALUED` — a
           conservative default that avoids spurious CONTESTED flags on common
           news-verb / event-log predicates that aren't in the YAML.
        """
        canonical = self.canonicalize(predicate)
        r = self._by_name.get(canonical)
        if r is None:
            return Cardinality.MULTI_VALUED
        return r.cardinality

    def canonicalize(self, predicate: str) -> str:
        return self._alias_to_canonical.get(predicate.lower(), predicate)

    @property
    def renaming_predicates(self) -> frozenset[str]:
        """Canonical names of predicates flagged ``renaming: true``."""
        return self._renaming

    @property
    def entity_rename_predicates(self) -> frozenset[str]:
        """Narrow set of predicates that are semantically *entity-rename*.

        This is :pyattr:`renaming_predicates` intersected with a library-level
        hard-coded whitelist of known entity-rename predicate names
        (``renamedTo``, ``formerlyKnownAs``, ``corporateName``). The
        intersection is a guarantee: even if a user-supplied schema marks
        role-succession predicates like ``succeededBy`` / ``precededBy`` as
        ``renaming: true``, they will **not** appear in this set. Callers
        that want to walk only true entity-rename edges (e.g. strict mode
        on :meth:`NuggetStore.achain_rename`) should consume this property
        instead of :pyattr:`renaming_predicates`.
        """
        return self._renaming & _ENTITY_RENAME_WHITELIST

    def is_renaming(self, predicate: str) -> bool:
        """True iff ``predicate`` (or an alias of it) is flagged renaming."""
        return self.canonicalize(predicate) in self._renaming

    def expected_subject_types(self, predicate: str) -> frozenset[str]:
        """Return the canonical predicate's expected subject NER labels.

        Resolves ``predicate`` through the alias table first. Unknown
        predicates (and predicates with no ``expected_subject_types`` in the
        YAML) return an empty :class:`frozenset`, which callers treat as
        "no constraint".
        """
        canonical = self.canonicalize(predicate)
        r = self._by_name.get(canonical)
        if r is None:
            return frozenset()
        return r.expected_subject_types

    def expected_object_types(self, predicate: str) -> frozenset[str]:
        """Return the canonical predicate's expected object NER labels.

        Resolves ``predicate`` through the alias table first. Unknown
        predicates (and predicates with no ``expected_object_types`` in the
        YAML) return an empty :class:`frozenset`, which callers treat as
        "no constraint".
        """
        canonical = self.canonicalize(predicate)
        r = self._by_name.get(canonical)
        if r is None:
            return frozenset()
        return r.expected_object_types
