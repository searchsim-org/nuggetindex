"""Tests for ``nuggetindex.audit.heuristics.triggers`` (Task 2.3).

These tests are unconditional: the triggers module is pure stdlib + ``re``.
"""

from __future__ import annotations

from nuggetindex.audit.heuristics import scan_triggers, trigger_kinds


def test_scan_triggers_finds_role_succession() -> None:
    text = "Satya Nadella became CEO of Microsoft in 2014."
    matches = scan_triggers(text)
    role_hits = [m for m in matches if m.kind == "role_succession"]
    assert len(role_hits) >= 1
    hit = role_hits[0]
    assert hit.predicate == "succeededBy"
    assert "became CEO of Microsoft" in hit.match_text


def test_scan_triggers_finds_entity_rename() -> None:
    text = "Twitter Inc. was renamed to X Corp. in 2023."
    matches = scan_triggers(text)
    rename_hits = [m for m in matches if m.kind == "entity_rename"]
    assert len(rename_hits) >= 1
    assert rename_hits[0].predicate == "renamedTo"


def test_scan_triggers_finds_functional_relation() -> None:
    text = "Microsoft acquired LinkedIn for $26.2 billion."
    matches = scan_triggers(text)
    fn_hits = [m for m in matches if m.kind == "functional_relation"]
    assert len(fn_hits) >= 1
    assert any(m.predicate == "acquired" for m in fn_hits)


def test_scan_triggers_empty_text_returns_empty() -> None:
    assert scan_triggers("") == []


def test_scan_triggers_no_cue_returns_empty() -> None:
    assert scan_triggers("The sky is blue.") == []


def test_trigger_kinds_exposes_all_three() -> None:
    assert trigger_kinds() == {
        "role_succession",
        "entity_rename",
        "functional_relation",
    }
