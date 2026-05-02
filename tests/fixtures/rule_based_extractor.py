"""Regex-based zero-dependency extractor.

Not for production accuracy -- it is a fallback useful for tests, air-gapped
demos, and smoke checks. It splits text into sentences and applies a small
set of hand-written patterns that cover common forms:

* ``X is Y`` / ``X are Y`` / ``X was Y`` / ``X were Y``
* ``X founded in Y``
* ``X CEO of Y`` (role phrase)

Confidence for each match is in ``[0.3, 0.6]`` -- comfortably below the
default QualityGate acceptance threshold. Temporal inference happens in the
pipeline's temporal stage; this extractor emits
``ValidityInterval.unknown()`` as a placeholder so re-ingesting the same
document is idempotent (the old behaviour stamped ``datetime.now(UTC)``
into the content hash).
"""

from __future__ import annotations

import re

from nuggetindex.core.enums import NuggetKind
from nuggetindex.core.models import (
    EpistemicState,
    FactTriple,
    Nugget,
    ProvenanceRecord,
    ValidityInterval,
)
from nuggetindex.extractors.base import BaseExtractor, ExtractionResult

# Sentence splitter: split on ". " or a newline, then drop empties.
_SENT_SPLIT = re.compile(r"(?:\.\s+|\n+)")

# Role phrase like "CEO of" must run before the generic "is" rule because
# an "is CEO of" sentence would otherwise also match the "is" pattern.
_CEO_OF = re.compile(
    r"^(?P<person>.+?)\s+(?:is|was|became)\s+CEO\s+of\s+(?P<company>.+?)\s*\.?$",
    re.IGNORECASE,
)

_FOUNDED_IN = re.compile(
    r"^(?P<subj>.+?)\s+(?:was|were)\s+founded\s+in\s+(?P<obj>.+?)\s*\.?$",
    re.IGNORECASE,
)

_IS_PATTERN = re.compile(
    r"^(?P<subj>.+?)\s+(?P<pred>is|are|was|were)\s+(?P<obj>.+?)\s*\.?$",
    re.IGNORECASE,
)

_PRED_CONFIDENCE: dict[str, float] = {
    "ceo": 0.55,
    "founded": 0.55,
    "is": 0.40,
    "are": 0.40,
    "was": 0.40,
    "were": 0.40,
}

# Interrogative pronouns that show up as spurious "subjects" when the rule
# patterns match question-shaped passages (e.g. "Who is the CEO of Google?").
# See findings A1.
_INTERROGATIVE_SUBJECTS: frozenset[str] = frozenset(
    {
        "who",
        "what",
        "where",
        "when",
        "why",
        "which",
        "whose",
        "how",
        "whom",
        "whether",
    }
)


def _clean(s: str) -> str:
    return s.strip().rstrip(".").strip()


def _is_valid_subject(subject: str) -> bool:
    """Reject empty, too-short, or interrogative-pronoun subjects."""
    s = subject.strip().lower()
    if len(s) < 2:
        return False
    return s not in _INTERROGATIVE_SUBJECTS


class RuleBasedExtractor(BaseExtractor):
    """Offline, regex-driven extractor. No network, no models."""

    # The pipeline consults this attribute to decide whether a missing
    # ``Document.source_date`` should raise the "placeholder validity"
    # warning. The rule-based extractor emits ``ValidityInterval.unknown()``
    # for every match, so it opts in.
    emits_placeholder_validity: bool = True

    def __init__(self, *, source_id: str = "rule-based") -> None:
        self._source_id = source_id

    async def aextract(
        self,
        text: str,
        *,
        context: str = "",
        source_id: str | None = None,
    ) -> list[ExtractionResult]:
        if not text or not text.strip():
            return []

        sentences = [s.strip() for s in _SENT_SPLIT.split(text) if s.strip()]
        results: list[ExtractionResult] = []
        effective_source_id = source_id or self._source_id
        for sentence in sentences:
            triple = self._match(sentence)
            if triple is None:
                continue
            subject, predicate, obj = triple
            if not _is_valid_subject(subject):
                continue
            confidence = _PRED_CONFIDENCE.get(predicate, 0.4)
            nugget = Nugget.new(
                kind=NuggetKind.SEMANTIC_FACT,
                fact=FactTriple(
                    subject=subject,
                    predicate=predicate,
                    object=obj,
                    text=sentence,
                ),
                validity=ValidityInterval.unknown(),
                epistemic=EpistemicState(confidence=confidence),
                provenance=(
                    ProvenanceRecord(
                        source_id=effective_source_id,
                        evidence_span=sentence,
                    ),
                ),
                extraction_confidence=confidence,
            )
            results.append(
                ExtractionResult(
                    nugget=nugget,
                    confidence=confidence,
                    rationale=f"rule:{predicate}",
                )
            )
        return results

    @staticmethod
    def _match(sentence: str) -> tuple[str, str, str] | None:
        """Try each pattern in priority order. Return (subj, pred, obj) or None."""
        m = _CEO_OF.match(sentence)
        if m:
            # "X is CEO of Y" -> subject=Y company, predicate=ceo, object=X person
            return _clean(m.group("company")), "ceo", _clean(m.group("person"))

        m = _FOUNDED_IN.match(sentence)
        if m:
            return _clean(m.group("subj")), "founded", _clean(m.group("obj"))

        m = _IS_PATTERN.match(sentence)
        if m:
            return (
                _clean(m.group("subj")),
                m.group("pred").lower(),
                _clean(m.group("obj")),
            )
        return None
