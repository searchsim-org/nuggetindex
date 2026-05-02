"""Shared fixtures for chain unit tests.

``sample_nuggets`` is the canonical Google CEO succession (Schmidt -> Page
-> Pichai) used across succession, rename, and chain-model tests.
"""

from datetime import UTC, datetime

import pytest

from nuggetindex.core.enums import NuggetKind
from nuggetindex.core.models import (
    EpistemicState,
    FactTriple,
    Nugget,
    ProvenanceRecord,
    ValidityInterval,
)


def _google_ceo(obj: str, start_year: int, end_year: int | None) -> Nugget:
    vi = ValidityInterval(
        start=datetime(start_year, 1, 1, tzinfo=UTC),
        end=datetime(end_year, 1, 1, tzinfo=UTC) if end_year else None,
    )
    return Nugget.new(
        kind=NuggetKind.SEMANTIC_FACT,
        fact=FactTriple(
            subject="Google",
            # Canonical predicate — chain lookups canonicalise queries as of
            # v0.2.1, so fixtures store under the canonical name.
            predicate="chiefExecutiveOfficer",
            object=obj,
            text=f"{obj} is CEO",
        ),
        validity=vi,
        epistemic=EpistemicState(),
        provenance=(
            ProvenanceRecord(
                source_id=f"doc-{obj}",
                evidence_span=f"{obj} is CEO",
            ),
        ),
    )


@pytest.fixture
def sample_nuggets() -> list[Nugget]:
    return [
        _google_ceo("Schmidt", 2001, 2011),
        _google_ceo("Page", 2011, 2015),
        _google_ceo("Pichai", 2015, None),
    ]
