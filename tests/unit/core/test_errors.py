import pytest

from nuggetindex.core.errors import (
    BackendUnavailable,
    ConflictUnresolved,
    ExtractionFailed,
    InvalidRelationSchema,
    JudgeTimeout,
    NuggetIndexError,
)


def test_all_errors_inherit_from_root():
    for exc in (
        ExtractionFailed,
        ConflictUnresolved,
        BackendUnavailable,
        InvalidRelationSchema,
        JudgeTimeout,
    ):
        assert issubclass(exc, NuggetIndexError)


def test_root_error_is_catchable_as_exception():
    with pytest.raises(Exception):  # noqa: B017 - intentionally verifying root is an Exception
        raise NuggetIndexError("boom")


def test_error_carries_message():
    try:
        raise ExtractionFailed("parse error in response")
    except ExtractionFailed as e:
        assert "parse error" in str(e)
