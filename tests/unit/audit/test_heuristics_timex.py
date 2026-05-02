"""Tests for ``nuggetindex.audit.heuristics.timex`` (Task 2.2).

The tests that actually need spaCy + ``en_core_web_sm`` + dateparser are
guarded by ``skipif(not timex_available())`` so the suite stays green in
minimal dev envs. The unavailable-backend path is exercised unconditionally
via monkeypatching.
"""

from __future__ import annotations

from datetime import datetime

import pytest

from nuggetindex.audit.heuristics import TimeExpression, tag_timex, timex_available
from nuggetindex.audit.heuristics import timex as timex_mod


def test_timex_available_reports_truth() -> None:
    assert isinstance(timex_available(), bool)


def test_time_expression_is_frozen_dataclass() -> None:
    t = TimeExpression(span="2015", start_char=0, end_char=4, parsed=None)
    import dataclasses as _dc

    with pytest.raises(_dc.FrozenInstanceError):
        t.span = "x"  # type: ignore[misc]


@pytest.mark.skipif(
    not timex_available(), reason="spaCy + en_core_web_sm + dateparser not installed"
)
def test_tag_timex_finds_explicit_date() -> None:
    text = "Sundar Pichai became CEO on October 2, 2015."
    result = tag_timex(text)
    assert len(result) >= 1
    assert any(t.parsed is not None and t.parsed.year == 2015 for t in result)


@pytest.mark.skipif(
    not timex_available(), reason="spaCy + en_core_web_sm + dateparser not installed"
)
def test_tag_timex_empty_on_no_dates() -> None:
    assert tag_timex("The sky is blue.") == []


@pytest.mark.skipif(
    not timex_available(), reason="spaCy + en_core_web_sm + dateparser not installed"
)
def test_tag_timex_relative_uses_reference_date() -> None:
    ref = datetime(2024, 6, 1)
    result = tag_timex("last year", reference_date=ref)
    assert any(t.parsed is not None and t.parsed.year == 2023 for t in result)


def test_tag_timex_graceful_on_unavailable_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    """When the spaCy pipeline can't load, ``tag_timex`` must return ``[]``."""
    # Invalidate the cache and force ``_load_nlp`` to return the unavailable
    # sentinel regardless of what is actually installed.
    monkeypatch.setattr(timex_mod, "_NLP", timex_mod._UNAVAILABLE, raising=False)
    monkeypatch.setattr(timex_mod, "_load_nlp", lambda: None)
    assert tag_timex("Pichai became CEO on October 2, 2015.") == []


def test_timex_available_false_when_backend_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(timex_mod, "_NLP", timex_mod._UNAVAILABLE, raising=False)
    monkeypatch.setattr(timex_mod, "_load_nlp", lambda: None)
    assert timex_available() is False
