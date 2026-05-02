from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from nuggetindex.core.models import ValidityInterval


def test_validity_open_ended():
    vi = ValidityInterval(start=datetime(2020, 1, 1, tzinfo=UTC))
    assert vi.end is None
    assert vi.contains(datetime(2030, 1, 1, tzinfo=UTC))


def test_validity_bounded():
    vi = ValidityInterval(
        start=datetime(2020, 1, 1, tzinfo=UTC), end=datetime(2025, 1, 1, tzinfo=UTC)
    )
    assert vi.contains(datetime(2022, 6, 1, tzinfo=UTC))
    assert not vi.contains(datetime(2026, 1, 1, tzinfo=UTC))


def test_validity_end_before_start_rejected():
    with pytest.raises(ValidationError):
        ValidityInterval(
            start=datetime(2025, 1, 1, tzinfo=UTC),
            end=datetime(2020, 1, 1, tzinfo=UTC),
        )


def test_validity_requires_tz_aware_datetimes():
    with pytest.raises(ValidationError):
        ValidityInterval(start=datetime(2020, 1, 1))  # naive


def test_validity_overlaps():
    a = ValidityInterval(
        start=datetime(2020, 1, 1, tzinfo=UTC), end=datetime(2023, 1, 1, tzinfo=UTC)
    )
    b = ValidityInterval(
        start=datetime(2022, 1, 1, tzinfo=UTC), end=datetime(2024, 1, 1, tzinfo=UTC)
    )
    c = ValidityInterval(
        start=datetime(2024, 1, 1, tzinfo=UTC), end=datetime(2025, 1, 1, tzinfo=UTC)
    )
    assert a.overlaps(b)
    assert b.overlaps(a)
    assert not a.overlaps(c)


def test_validity_overlaps_open_ended():
    a = ValidityInterval(start=datetime(2020, 1, 1, tzinfo=UTC))  # open
    b = ValidityInterval(
        start=datetime(2025, 1, 1, tzinfo=UTC), end=datetime(2030, 1, 1, tzinfo=UTC)
    )
    assert a.overlaps(b)


def test_default_scope_is_global():
    vi = ValidityInterval(start=datetime(2020, 1, 1, tzinfo=UTC))
    assert vi.scope == "global"


def test_validity_known_defaults_to_true():
    vi = ValidityInterval(
        start=datetime(2020, 1, 1, tzinfo=UTC),
        end=datetime(2025, 1, 1, tzinfo=UTC),
    )
    assert vi.validity_known is True


def test_validity_known_false_round_trips():
    vi = ValidityInterval(
        start=datetime(2020, 1, 1, tzinfo=UTC),
        end=datetime(2025, 1, 1, tzinfo=UTC),
        validity_known=False,
    )
    dumped = vi.model_dump()
    assert dumped["validity_known"] is False
    restored = ValidityInterval.model_validate(dumped)
    assert restored.validity_known is False
    assert restored == vi
