from datetime import UTC, datetime

from nuggetindex.core.models import ValidityInterval


def test_unknown_is_classmethod_returning_placeholder():
    vi = ValidityInterval.unknown()
    assert vi.source_type == "placeholder"
    assert vi.start.year == 1  # datetime.min — an unambiguous placeholder


def test_unknown_is_detectable():
    vi = ValidityInterval.unknown()
    assert vi.is_placeholder()
    regular = ValidityInterval(start=datetime(2020, 1, 1, tzinfo=UTC))
    assert not regular.is_placeholder()


def test_unknown_is_tz_aware():
    vi = ValidityInterval.unknown()
    assert vi.start.tzinfo is not None
