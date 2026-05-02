"""Phase 6.5 integration: ``NuggetStore.encoder`` feeds semantic dedup.

When ``NuggetStore`` is built with ``encoder=<callable>``, the same encoder
must be passed to ``Deduplicator`` so the pipeline can collapse paraphrased
aliases ("Sundar Pichai" vs "S. Pichai") before persistence.

Two scenarios:

* **No encoder** — paraphrased objects survive as two rows (Jaccard fallback
  cannot catch aliasing when n-gram overlap is low).
* **With encoder** — a stub encoder maps both aliases to near-identical
  vectors, so cosine similarity exceeds the default 0.92 threshold and only
  one nugget is persisted.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from nuggetindex.core.enums import NuggetKind
from nuggetindex.core.models import (
    EpistemicState,
    FactTriple,
    Nugget,
    ProvenanceRecord,
    ValidityInterval,
)
from nuggetindex.extractors.base import BaseExtractor, ExtractionResult
from nuggetindex.pipeline.constructor import Document
from nuggetindex.store.base import NuggetStore


class _FixedExtractor(BaseExtractor):
    def __init__(self, results: list[ExtractionResult]) -> None:
        self._results = results

    async def aextract(self, text: str, *, context: str = "") -> list[ExtractionResult]:
        return list(self._results)


def _extraction(
    *, subject: str, predicate: str, obj: str, sentence: str, source_id: str
) -> ExtractionResult:
    n = Nugget.new(
        kind=NuggetKind.SEMANTIC_FACT,
        fact=FactTriple(subject=subject, predicate=predicate, object=obj, text=sentence),
        validity=ValidityInterval(start=datetime(2019, 1, 1, tzinfo=UTC)),
        epistemic=EpistemicState(),
        provenance=(ProvenanceRecord(source_id=source_id, evidence_span=sentence),),
    )
    return ExtractionResult(nugget=n, confidence=0.95, rationale=None)


def _paraphrase_encoder() -> Any:
    """Encoder that maps 'Pichai' aliases to near-identical vectors."""
    mapping = {
        "Sundar Pichai": np.array([1.0, 0.01, 0.0, 0.0], dtype="float32"),
        "S. Pichai": np.array([0.99, 0.01, 0.0, 0.0], dtype="float32"),
        # A genuinely distinct object:
        "Larry Page": np.array([0.0, 0.0, 1.0, 0.0], dtype="float32"),
    }

    def encode(texts: list[str]) -> np.ndarray:
        return np.stack(
            [mapping.get(t, np.array([0.0, 0.0, 0.0, 1.0], dtype="float32")) for t in texts],
            axis=0,
        )

    return encode


@pytest.mark.asyncio
async def test_without_encoder_paraphrases_survive_as_two_rows(
    tmp_db_path: Path,
) -> None:
    """Baseline: Jaccard fallback misses paraphrase aliasing."""
    # Two extractions with aliased CEO names on the same key.
    store = NuggetStore(
        tmp_db_path,
        extractor=_FixedExtractor(
            [
                _extraction(
                    subject="Google",
                    predicate="ceo",
                    obj="Sundar Pichai",
                    sentence="Sundar Pichai is CEO of Google",
                    source_id="d1",
                ),
                _extraction(
                    subject="Google",
                    predicate="ceo",
                    obj="S. Pichai",
                    sentence="S. Pichai is CEO of Google",
                    source_id="d1",
                ),
            ]
        ),
        encoder=None,  # explicit: no encoder
    )
    doc = Document(
        source_id="d1",
        text="Pichai stuff",
        source_date=datetime(2019, 6, 1, tzinfo=UTC),
    )
    await store.aingest(doc)
    # Jaccard on "Sundar Pichai" vs "S. Pichai" is below the 0.85 threshold
    # (only ~0.25 character-trigram overlap once spaces differ), so two rows
    # survive. This is the baseline we want semantic dedup to improve on.
    assert await store.acount() >= 2
    await store.aclose()


@pytest.mark.asyncio
async def test_with_encoder_paraphrases_collapse_to_one_row(
    tmp_db_path: Path,
) -> None:
    """Task 6.5: encoder attached on NuggetStore feeds semantic dedup."""
    store = NuggetStore(
        tmp_db_path,
        extractor=_FixedExtractor(
            [
                _extraction(
                    subject="Google",
                    predicate="ceo",
                    obj="Sundar Pichai",
                    sentence="Sundar Pichai is CEO of Google",
                    source_id="d1",
                ),
                _extraction(
                    subject="Google",
                    predicate="ceo",
                    obj="S. Pichai",
                    sentence="S. Pichai is CEO of Google",
                    source_id="d1",
                ),
            ]
        ),
        encoder=_paraphrase_encoder(),
    )
    doc = Document(
        source_id="d1",
        text="Pichai stuff",
        source_date=datetime(2019, 6, 1, tzinfo=UTC),
    )
    await store.aingest(doc)
    # Cosine("Sundar Pichai", "S. Pichai") ~ 1.0 via the stub encoder —
    # above the 0.92 threshold -> dedup catches it.
    assert await store.acount() == 1
    await store.aclose()


@pytest.mark.asyncio
async def test_encoder_does_not_collapse_genuinely_different_objects(
    tmp_db_path: Path,
) -> None:
    """Guardrail: different object values survive even with an encoder."""
    store = NuggetStore(
        tmp_db_path,
        extractor=_FixedExtractor(
            [
                _extraction(
                    subject="Google",
                    predicate="ceo",
                    obj="Sundar Pichai",
                    sentence="Pichai CEO Google",
                    source_id="d1",
                ),
                _extraction(
                    subject="Google",
                    predicate="ceo",
                    obj="Larry Page",
                    sentence="Page CEO Google",
                    source_id="d1",
                ),
            ]
        ),
        encoder=_paraphrase_encoder(),
    )
    # Conflict detection may flip these to CONTESTED, but dedup should not
    # collapse them (cosine ~ 0, well below the 0.92 threshold).
    await store.aingest(
        Document(
            source_id="d1",
            text="Pichai and Page",
            source_date=datetime(2019, 6, 1, tzinfo=UTC),
        )
    )
    assert await store.acount() == 2
    await store.aclose()
