import pytest
from pydantic import ValidationError

from nuggetindex.core.enums import EpistemicRank, LifecycleStatus
from nuggetindex.core.models import EpistemicState


def test_epistemic_defaults():
    es = EpistemicState()
    assert es.status == LifecycleStatus.ACTIVE
    assert es.rank == EpistemicRank.NORMAL
    assert es.confidence == 1.0


def test_confidence_out_of_range_rejected():
    with pytest.raises(ValidationError):
        EpistemicState(confidence=1.5)
    with pytest.raises(ValidationError):
        EpistemicState(confidence=-0.1)
