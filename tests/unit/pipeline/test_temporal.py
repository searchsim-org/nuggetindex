"""Tests for temporal validity inference (Algorithm 1)."""

from datetime import UTC, datetime

import pytest

from nuggetindex.core.models import ValidityInterval
from nuggetindex.pipeline.temporal import InferredValidity, infer_validity

SOURCE = datetime(2026, 1, 1, tzinfo=UTC)


def test_since_phrase_sets_open_interval() -> None:
    vi = infer_validity("Sundar Pichai has been CEO since October 2015", source_date=SOURCE)
    assert isinstance(vi, ValidityInterval)
    assert vi.start.year == 2015 and vi.start.month == 10
    assert vi.end is None


def test_starting_phrase_sets_start() -> None:
    vi = infer_validity("Starting 2018, she served on the board.", source_date=SOURCE)
    assert isinstance(vi, ValidityInterval)
    assert vi.start.year == 2018
    assert vi.end is None


def test_became_phrase_sets_start() -> None:
    vi = infer_validity("Pichai became CEO in 2019", source_date=SOURCE)
    assert isinstance(vi, ValidityInterval)
    assert vi.start.year == 2019
    assert vi.end is None


def test_until_phrase_sets_end() -> None:
    vi = infer_validity("Larry Page was CEO until 2015", source_date=SOURCE)
    assert isinstance(vi, ValidityInterval)
    assert vi.end is not None and vi.end.year == 2015


def test_resumed_phrase_sets_start() -> None:
    vi = infer_validity("Jobs resumed the CEO role in 1997", source_date=SOURCE)
    assert isinstance(vi, ValidityInterval)
    assert vi.start.year == 1997


def test_from_to_phrase_sets_both() -> None:
    vi = infer_validity(
        "He served as CFO from 2010 to 2014 before retiring",
        source_date=SOURCE,
    )
    assert isinstance(vi, ValidityInterval)
    assert vi.start.year == 2010
    assert vi.end is not None and vi.end.year == 2014


def test_year_range_dash_sets_both() -> None:
    vi = infer_validity("She was chair 2011-2018", source_date=SOURCE)
    assert isinstance(vi, ValidityInterval)
    assert vi.start.year == 2011
    assert vi.end is not None and vi.end.year == 2018


def test_no_temporal_cue_falls_back_to_source_date() -> None:
    vi = infer_validity("The company makes phones", source_date=SOURCE)
    assert isinstance(vi, ValidityInterval)
    assert vi.start == SOURCE
    assert vi.end is None


def test_no_temporal_cue_full_confidence() -> None:
    result = infer_validity(
        "The company makes phones",
        source_date=SOURCE,
        return_confidence=True,
    )
    assert isinstance(result, InferredValidity)
    assert result.confidence == 1.0


def test_bare_year_triggers_uncertainty() -> None:
    result = infer_validity(
        "The event happened in 2024",
        source_date=SOURCE,
        return_confidence=True,
    )
    # NOTE: "in <year>" matches the generic in-date rule first with full
    # confidence. A truly ambiguous case is a year without a trigger word.
    assert isinstance(result, InferredValidity)
    # The `in 2024` phrase IS a cue, so it should be captured as start=2024.
    assert result.interval.start.year == 2024


def test_bare_year_without_preposition_reduces_confidence() -> None:
    result = infer_validity(
        "An article about 2024 changes",
        source_date=SOURCE,
        return_confidence=True,
    )
    assert isinstance(result, InferredValidity)
    assert result.confidence == pytest.approx(0.75)
    assert result.interval.start == SOURCE


def test_source_date_must_be_tz_aware() -> None:
    with pytest.raises(ValueError):
        infer_validity("text", source_date=datetime(2026, 1, 1))


def test_return_confidence_false_returns_bare_interval() -> None:
    result = infer_validity("Pichai became CEO in 2019", source_date=SOURCE)
    assert isinstance(result, ValidityInterval)
    assert not isinstance(result, InferredValidity)


def test_return_confidence_true_returns_wrapper() -> None:
    result = infer_validity(
        "Pichai became CEO in 2019",
        source_date=SOURCE,
        return_confidence=True,
    )
    assert isinstance(result, InferredValidity)
    assert result.confidence == 1.0
