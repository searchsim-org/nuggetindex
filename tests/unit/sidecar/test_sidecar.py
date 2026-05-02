from datetime import UTC, datetime
from pathlib import Path

import pytest

from nuggetindex import NuggetStore
from nuggetindex.core.enums import EpistemicRank, LifecycleStatus, NuggetKind
from nuggetindex.core.models import (
    EpistemicState,
    FactTriple,
    Nugget,
    ProvenanceRecord,
    ValidityInterval,
)
from nuggetindex.sidecar import Sidecar, SidecarResponse


@pytest.fixture
async def seeded_store(tmp_path: Path):
    store = NuggetStore(db_path=tmp_path / "s.db")
    now = datetime.now(tz=UTC)
    n = Nugget.new(
        kind=NuggetKind.SEMANTIC_FACT,
        fact=FactTriple(
            subject="Google",
            predicate="chiefExecutiveOfficer",
            object="Larry Page",
            text="Larry Page served as Google CEO from 2011 through 2015",
        ),
        validity=ValidityInterval(
            start=datetime(2011, 4, 4, tzinfo=UTC),
            end=datetime(2015, 10, 2, tzinfo=UTC),
        ),
        epistemic=EpistemicState(
            status=LifecycleStatus.ACTIVE,
            rank=EpistemicRank.NORMAL,
            confidence=0.9,
        ),
        provenance=(
            ProvenanceRecord(
                source_id="wiki",
                evidence_span="Larry Page served as CEO from 2011 to 2015.",
                char_start=0,
                char_end=50,
                created_at=now,
            ),
        ),
        extraction_confidence=0.9,
    )
    await store.aadd(n)
    yield store
    await store.backend.aclose()


async def test_sidecar_offline_curated_temporal_query(seeded_store):
    sidecar = Sidecar(store=seeded_store, mode="offline-curated")
    response = await sidecar.ahandle("who was Google's CEO in 2013?")
    assert isinstance(response, SidecarResponse)
    assert response.decision is not None
    assert response.decision.use_nugget is True
    assert (
        response.decision.query_time is not None
        and response.decision.query_time.year == 2013
    )
    assert response.context_block != ""
    assert "Larry Page" in response.context_block


async def test_sidecar_passthrough_on_noise(seeded_store):
    sidecar = Sidecar(store=seeded_store, mode="offline-curated")
    response = await sidecar.ahandle("the sky is blue")
    assert response.decision is not None
    assert response.decision.use_nugget is False
    assert response.context_block == ""


async def test_sidecar_just_in_time_requires_extractor(seeded_store):
    with pytest.raises(ValueError, match="extractor"):
        Sidecar(store=seeded_store, mode="just-in-time", extractor=None)
