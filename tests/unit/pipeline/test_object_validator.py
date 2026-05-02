"""Tests for :func:`nuggetindex.pipeline.object_validator.is_valid_object`."""

from nuggetindex.pipeline.object_validator import is_valid_object


def test_valid_object_passes() -> None:
    ok, reason = is_valid_object("Tim Cook")
    assert ok is True
    assert reason == ""


def test_too_short_fails() -> None:
    ok, reason = is_valid_object("A")
    assert ok is False
    assert reason == "too_short"


def test_empty_after_strip_fails() -> None:
    ok, reason = is_valid_object("   ")
    assert ok is False
    assert reason == "too_short"


def test_bare_year_fails() -> None:
    ok, reason = is_valid_object("2000")
    assert ok is False
    assert reason == "bare_numeric"


def test_bare_money_fails() -> None:
    ok, reason = is_valid_object("$26.2B")
    assert ok is False
    assert reason == "bare_numeric"


def test_percent_fails() -> None:
    ok, reason = is_valid_object("15%")
    assert ok is False
    assert reason == "bare_numeric"


def test_question_fails() -> None:
    # Russian interrogative from the real-corpus Mode-B example.
    ok, reason = is_valid_object("Следующие CEO Apple?")
    assert ok is False
    assert reason == "interrogative"


def test_cjk_question_fails() -> None:
    ok, reason = is_valid_object("苹果下一个CEO？")
    assert ok is False
    assert reason == "interrogative"


def test_no_letters_fails() -> None:
    ok, reason = is_valid_object("!!!")
    assert ok is False
    assert reason == "no_letters"


def test_multiword_with_numbers_passes() -> None:
    # Contains letters; not a bare-numeric token.
    ok, reason = is_valid_object("22 years")
    assert ok is True
    assert reason == ""


def test_normal_non_english_passes() -> None:
    ok, reason = is_valid_object("Société Générale")
    assert ok is True
    assert reason == ""
