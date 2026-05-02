"""QualityGate: partition extraction results by confidence.

Wraps any `BaseExtractor` and splits its output into three buckets:

* ``accepted`` -- confidence at or above ``accept_threshold`` (default 0.85)
* ``deferred`` -- between ``review_threshold`` and ``accept_threshold``; also
  appended to a JSONL review queue for the annotation tool (Improvement D)
* ``rejected`` -- below ``review_threshold``

The review queue is one JSON object per line, with stable field names so
downstream tooling can evolve independently.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from nuggetindex.extractors.base import (
    BaseExtractor,
    ExtractionResult,
    accepts_source_id,
)


@dataclass
class QualityGateResult:
    accepted: list[ExtractionResult] = field(default_factory=list)
    deferred: list[ExtractionResult] = field(default_factory=list)
    rejected: list[ExtractionResult] = field(default_factory=list)


class QualityGate:
    """Confidence-based gate around an extractor.

    Deferred results are appended to ``review_queue_path`` as JSONL so a
    human annotator (or a follow-up batch job) can triage them without
    blocking the main pipeline.
    """

    def __init__(
        self,
        extractor: BaseExtractor,
        *,
        accept_threshold: float = 0.85,
        review_threshold: float = 0.6,
        review_queue_path: Path | str = Path("review_queue.jsonl"),
    ) -> None:
        if review_threshold > accept_threshold:
            raise ValueError(
                "review_threshold must be <= accept_threshold"
            )
        self.extractor = extractor
        self.accept_threshold = accept_threshold
        self.review_threshold = review_threshold
        self.review_queue_path = Path(review_queue_path)

    async def aextract(
        self,
        text: str,
        *,
        context: str = "",
        source_id: str | None = None,
    ) -> QualityGateResult:
        # Forward source_id only if the wrapped extractor accepts it, so the
        # gate stays transparent to legacy extractors that predate the 0.2
        # convention. ``BaseExtractor.aextract`` abstractly declares only
        # ``text`` and ``context``; we pass ``source_id`` via **kwargs so
        # mypy doesn't complain about the optional 0.2 kwarg at call sites.
        kwargs: dict[str, str] = {"context": context}
        if source_id is not None and accepts_source_id(self.extractor):
            kwargs["source_id"] = source_id
        raw = await self.extractor.aextract(text, **kwargs)
        out = QualityGateResult()
        for r in raw:
            if r.confidence >= self.accept_threshold:
                out.accepted.append(r)
            elif r.confidence >= self.review_threshold:
                out.deferred.append(r)
                self._log_deferred(r, source_text=text, context=context)
            else:
                out.rejected.append(r)
        return out

    def _log_deferred(
        self,
        result: ExtractionResult,
        *,
        source_text: str,
        context: str,
    ) -> None:
        row = {
            "nugget": json.loads(result.nugget.model_dump_json()),
            "confidence": result.confidence,
            "rationale": result.rationale,
            "source_text": source_text,
            "context": context,
            "extractor": type(self.extractor).__name__,
            "timestamp": datetime.now(UTC).isoformat(),
        }
        self.review_queue_path.parent.mkdir(parents=True, exist_ok=True)
        with self.review_queue_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")
