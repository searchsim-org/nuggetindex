"""Public-surface tests for ``nuggetindex.audit.doctor`` (Task 2.1).

The doctor module ships a stub ``scan_index`` plus two frozen dataclasses.
These tests guard the public shape only -- behaviour lands in Tasks 2.4/2.5.
"""

from __future__ import annotations

import dataclasses

import pytest

from nuggetindex import DoctorReport as TopLevelDoctorReport
from nuggetindex import scan_index as top_level_scan
from nuggetindex.audit.doctor import DoctorReport, DoctorScore, scan_index


def test_dataclasses_frozen() -> None:
    # Frozen dataclasses -- mutation should raise FrozenInstanceError
    s = DoctorScore(
        dimension="temporal_depth",
        percentage=0.0,
        ci95=(0.0, 0.0),
        n_sampled=0,
        n_total=None,
        examples=[],
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        s.percentage = 1.0  # type: ignore[misc]


def test_doctor_report_to_markdown_returns_rendered_markdown() -> None:
    r = DoctorReport(
        sample_mode="fast",
        scores=[],
        verdict="low",
        rendered_markdown="# test",
    )
    assert r.to_markdown() == "# test"


def test_top_level_reexports() -> None:
    assert TopLevelDoctorReport is DoctorReport
    assert top_level_scan is scan_index


@pytest.mark.asyncio
async def test_deep_mode_requires_extractor() -> None:
    # Task 2.5 replaced the stub with a real implementation that requires
    # an ``extractor=`` kwarg; without one, deep mode raises ValueError.
    with pytest.raises(ValueError, match="extractor"):
        await scan_index(docs=[], mode="deep")
