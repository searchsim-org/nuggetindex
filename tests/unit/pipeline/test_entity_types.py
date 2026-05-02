"""Tests for :mod:`nuggetindex.pipeline.entity_types`."""

from __future__ import annotations

import pytest

from nuggetindex.audit.heuristics.timex import get_nlp
from nuggetindex.core.schema import RelationSchema
from nuggetindex.pipeline.entity_types import (
    check_triple_direction,
    probe_entity_type,
)


def _require_spacy() -> None:
    if get_nlp() is None:
        pytest.skip("spaCy + en_core_web_sm not installed")


def test_probe_entity_type_person() -> None:
    _require_spacy()
    assert probe_entity_type("Tim Cook") == "PERSON"


def test_probe_entity_type_compound() -> None:
    _require_spacy()
    label = probe_entity_type("A Day In The Life of Apple's CEO")
    # Must NOT classify as a clean single-entity PERSON/ORG. Accept either
    # COMPOUND (multi-entity span) or NONE depending on spaCy's tagging; the
    # invariant we need is "not a valid single-entity mention".
    assert label in {"COMPOUND", "NONE"}


def test_probe_entity_type_empty() -> None:
    _require_spacy()
    assert probe_entity_type("") == "NONE"
    assert probe_entity_type("   ") == "NONE"


def test_probe_entity_type_unavailable_when_nlp_missing() -> None:
    # Exercise the graceful-degradation branch independent of the environment
    # by explicitly passing a None-returning stub as nlp. This is the API the
    # pipeline relies on when spaCy isn't installed.
    # Import the module via ``sys.modules`` so we get the real submodule and
    # not the shadowed ``audit.audit`` function re-exported from the package
    # ``__init__``.
    import sys

    from nuggetindex.audit.heuristics import timex as _timex_import  # noqa: F401
    from nuggetindex.pipeline import entity_types as et
    timex = sys.modules["nuggetindex.audit.heuristics.timex"]
    original = timex._NLP
    timex._NLP = None  # force the "unavailable" sentinel
    try:
        assert et.probe_entity_type("Tim Cook") == "UNAVAILABLE"
    finally:
        timex._NLP = original


def test_check_direction_ok() -> None:
    _require_spacy()
    schema = RelationSchema.default()
    # Use Apple (spaCy-recognized ORG) + Tim Cook (spaCy-recognized PERSON).
    # "Google" returns NONE from the small en_core_web_sm model, so avoid it
    # in the spaCy-fallback tests — that's exactly the failure mode fix 9 was
    # introduced to work around at the pipeline level.
    assert (
        check_triple_direction("Apple", "chiefExecutiveOfficer", "Tim Cook", schema)
        == "ok"
    )


def test_check_direction_flip() -> None:
    _require_spacy()
    schema = RelationSchema.default()
    assert (
        check_triple_direction("Tim Cook", "chiefExecutiveOfficer", "Apple", schema)
        == "flip"
    )


def test_check_direction_reject() -> None:
    _require_spacy()
    schema = RelationSchema.default()
    # Apple is ORG (subject-type OK), object is a YouTube-style title ->
    # COMPOUND / NONE, not PERSON -> reject.
    result = check_triple_direction(
        "Apple",
        "chiefExecutiveOfficer",
        "A Day In The Life of Apple's CEO",
        schema,
    )
    assert result == "reject"


def test_check_direction_passthrough_when_no_expected_types() -> None:
    # ``president`` has no expected_*_types in the default YAML, so the
    # direction check is a no-op regardless of whether spaCy is installed.
    schema = RelationSchema.default()
    assert (
        check_triple_direction("anything", "president", "something else", schema)
        == "ok"
    )


def test_check_direction_passthrough_when_spacy_unavailable() -> None:
    # Force the shared timex nlp cache to "unavailable" and confirm the
    # direction check returns "ok" for triples that would otherwise reject.
    # Import the module via ``sys.modules`` so we get the real submodule and
    # not the shadowed ``audit.audit`` function re-exported from the package
    # ``__init__``.
    import sys

    from nuggetindex.audit.heuristics import timex as _timex_import  # noqa: F401
    timex = sys.modules["nuggetindex.audit.heuristics.timex"]
    original = timex._NLP
    timex._NLP = None
    try:
        schema = RelationSchema.default()
        assert (
            check_triple_direction(
                "Apple",
                "chiefExecutiveOfficer",
                "A Day In The Life of Apple's CEO",
                schema,
            )
            == "ok"
        )
    finally:
        timex._NLP = original


# ---------------------------------------------------------------------------
# Fix 9: LLM-emitted types take priority over spaCy NER.
# ---------------------------------------------------------------------------


def test_check_direction_llm_types_flip_no_spacy_needed() -> None:
    """``(Elon Musk, chiefExecutiveOfficer, SpaceX)`` with explicit
    LLM-emitted types must flip WITHOUT consulting spaCy.

    We deliberately do not mock or touch spaCy here; the test passes even
    when spaCy / ``en_core_web_sm`` are not installed. This is the headline
    genuine fix: LLM types make the check cross-lingual / cross-domain.
    """
    schema = RelationSchema.default()
    result = check_triple_direction(
        "Elon Musk",
        "chiefExecutiveOfficer",
        "SpaceX",
        schema,
        subject_type="PERSON",
        object_type="ORG",
    )
    assert result == "flip"


def test_check_direction_llm_types_case_insensitive() -> None:
    """Lowercase / mixed-case LLM types must still match the schema's
    uppercase ``expected_*_types`` values."""
    schema = RelationSchema.default()
    # Same inverted triple as above, but with lowercase types.
    assert (
        check_triple_direction(
            "Elon Musk",
            "chiefExecutiveOfficer",
            "SpaceX",
            schema,
            subject_type="person",
            object_type="org",
        )
        == "flip"
    )
    # Mixed case.
    assert (
        check_triple_direction(
            "SpaceX",
            "chiefExecutiveOfficer",
            "Elon Musk",
            schema,
            subject_type="Org",
            object_type="Person",
        )
        == "ok"
    )


def test_check_direction_llm_types_reject() -> None:
    """A triple whose object is flagged as a WORK_OF_ART can satisfy
    neither the direct nor the flipped direction -- must be rejected."""
    schema = RelationSchema.default()
    result = check_triple_direction(
        "Apple",
        "chiefExecutiveOfficer",
        "A Day In The Life of Apple's CEO",
        schema,
        subject_type="ORG",
        object_type="WORK_OF_ART",
    )
    assert result == "reject"


def test_check_direction_falls_back_to_ner_when_types_missing() -> None:
    """When both LLM types are ``None`` the function falls back to spaCy
    NER. Skipped when spaCy isn't installed since that's the whole point
    of the LLM-types path -- there's nothing to assert on the fallback."""
    pytest.importorskip("spacy")
    if get_nlp() is None:
        pytest.skip("spaCy model en_core_web_sm not installed")
    schema = RelationSchema.default()
    # Classic inverted case: without LLM types, spaCy NER drives the flip.
    # Apple is chosen over Google because en_core_web_sm recognizes it
    # reliably as ORG; the small model returns NONE for "Google".
    result = check_triple_direction(
        "Tim Cook",
        "chiefExecutiveOfficer",
        "Apple",
        schema,
        subject_type=None,
        object_type=None,
    )
    assert result == "flip"


def test_check_direction_one_llm_type_missing_falls_back_to_ner() -> None:
    """If only one of ``subject_type`` / ``object_type`` is supplied, the
    LLM path is skipped (incomplete data) and we fall back to spaCy. When
    spaCy is unavailable the check passes through as ``ok`` -- the same
    graceful-degrade behaviour as before fix 9."""
    schema = RelationSchema.default()
    # Force spaCy missing so we isolate the "one-type-missing" branch.
    import sys

    from nuggetindex.audit.heuristics import timex as _timex_import  # noqa: F401
    timex = sys.modules["nuggetindex.audit.heuristics.timex"]
    original = timex._NLP
    timex._NLP = None
    try:
        assert (
            check_triple_direction(
                "Elon Musk",
                "chiefExecutiveOfficer",
                "SpaceX",
                schema,
                subject_type="PERSON",
                object_type=None,
            )
            == "ok"
        )
    finally:
        timex._NLP = original
