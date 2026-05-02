from datetime import UTC, datetime

import pytest
import pytest_asyncio

from nuggetindex.core.enums import LifecycleStatus, NuggetKind
from nuggetindex.core.models import (
    EpistemicState,
    FactTriple,
    Nugget,
    ProvenanceRecord,
    ValidityInterval,
)
from nuggetindex.store.backends.sqlite import SQLiteBackend


def _n(obj, start, end=None, status=LifecycleStatus.ACTIVE) -> Nugget:
    return Nugget.new(
        kind=NuggetKind.SEMANTIC_FACT,
        fact=FactTriple(subject="Google", predicate="ceo", object=obj, text=f"{obj} is CEO"),
        validity=ValidityInterval(start=start, end=end),
        epistemic=EpistemicState(status=status),
        provenance=(ProvenanceRecord(source_id="d1", evidence_span="x"),),
    )


@pytest_asyncio.fixture
async def populated(tmp_db_path):
    b = SQLiteBackend(tmp_db_path)
    # Pichai: 2015-10 to open
    await b.aupsert(_n("Pichai", datetime(2015, 10, 1, tzinfo=UTC)))
    # Page: 2011-04 to 2015-10, deprecated
    await b.aupsert(
        _n(
            "Page",
            datetime(2011, 4, 1, tzinfo=UTC),
            end=datetime(2015, 10, 1, tzinfo=UTC),
            status=LifecycleStatus.DEPRECATED,
        )
    )
    # Schmidt: 2001 to 2011, deprecated
    await b.aupsert(
        _n(
            "Schmidt",
            datetime(2001, 1, 1, tzinfo=UTC),
            end=datetime(2011, 4, 1, tzinfo=UTC),
            status=LifecycleStatus.DEPRECATED,
        )
    )
    yield b
    await b.aclose()


@pytest.mark.asyncio
async def test_filter_active_at_query_time(populated):
    ids = await populated.afilter(query_time=datetime(2020, 1, 1, tzinfo=UTC), view="active")
    assert len(ids) == 1  # only Pichai is active at 2020


@pytest.mark.asyncio
async def test_filter_all_ignores_status(populated):
    ids = await populated.afilter(query_time=datetime(2012, 1, 1, tzinfo=UTC), view="all")
    # Page (2011-2015) is valid at 2012; Pichai (since 2015) is not; Schmidt (2001-2011) ended before
    assert len(ids) == 1


@pytest.mark.asyncio
async def test_filter_uses_null_end_open_ended(populated):
    # 2030: only Pichai (open-ended) is valid, and he's ACTIVE
    ids = await populated.afilter(query_time=datetime(2030, 1, 1, tzinfo=UTC), view="active")
    assert len(ids) == 1
