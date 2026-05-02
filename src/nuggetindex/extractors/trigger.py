"""LLM-free :class:`BaseExtractor` driven by trigger-verb patterns + spaCy NER.

Elevates :mod:`nuggetindex.audit.heuristics.triggers` (fast-doctor's hint
layer) into a first-class extractor: the "60% of the value at 0% LLM cost"
path for users who can't afford or don't want LLM extraction.

What it does:

* Runs :func:`scan_triggers` over the input text.
* For each :class:`TriggerMatch`:

  - slices ``subject`` / ``object`` out of the raw text at the match spans;
  - maps role-succession matches ("X became CEO of Y") to the more specific
    canonical predicate (``chiefExecutiveOfficer`` etc.) and swaps
    subject/object so the resulting triple matches the schema's
    ``(ORG, role, PERSON)`` direction;
  - probes each mention through :func:`probe_entity_type` when spaCy is
    available (``subject_type`` / ``object_type`` otherwise ``None``);
  - rejects objects that fail :func:`is_valid_object`;
  - emits one :class:`ExtractionResult` with a flat ``0.5`` confidence —
    strictly lower than a typical LLM call, so downstream quality gates /
    callers can filter confidently.

Gracefully degrades without spaCy (types default to ``None``; the pipeline's
direction-flip step falls through to ``"ok"``) and without the default
schema (canonicalization is a best-effort pass).

The extractor emits :meth:`ValidityInterval.unknown` so the pipeline's
temporal-inference stage can overwrite validity with the document's
``source_date`` -- mirroring how :class:`RuleBasedExtractor` works. Without
real calendar extraction we do not attempt to parse temporal expressions
from the match text.
"""

from __future__ import annotations

import re

from nuggetindex.audit.heuristics.triggers import TriggerMatch, scan_triggers
from nuggetindex.core.enums import NuggetKind
from nuggetindex.core.models import (
    EpistemicState,
    FactTriple,
    Nugget,
    ProvenanceRecord,
    ValidityInterval,
)
from nuggetindex.extractors.base import BaseExtractor, ExtractionResult
from nuggetindex.pipeline.object_validator import is_valid_object

# Role-phrase -> canonical schema predicate. Lookup is lower-cased.
# These keys are the role strings embedded in the role_succession patterns
# of ``audit/heuristics/triggers.py`` -- keeping the mapping here (rather
# than in ``triggers.py``) preserves backward-compat for the triggers
# module's own tests, which lock the coarse ``succeededBy`` predicate.
_ROLE_PHRASES: tuple[tuple[str, str], ...] = (
    ("chief executive officer", "chiefExecutiveOfficer"),
    ("chief operating officer", "chiefOperatingOfficer"),
    ("chief executive", "chiefExecutiveOfficer"),
    ("ceo", "chiefExecutiveOfficer"),
    ("coo", "chiefOperatingOfficer"),
    ("cfo", "chiefFinancialOfficer"),
    ("cto", "chiefTechnologyOfficer"),
    ("chairman", "chairperson"),
    ("chairwoman", "chairperson"),
    ("chairperson", "chairperson"),
    ("chair", "chairperson"),
    ("president", "president"),
    ("director", "director"),
)


def _match_role(match_text: str) -> str | None:
    """Return the canonical role predicate found in ``match_text``, if any.

    Uses simple case-insensitive substring checks. The tuple above is in
    length-descending precedence order so "chief executive officer" wins
    over "chief executive".
    """
    low = match_text.lower()
    for phrase, canonical in _ROLE_PHRASES:
        if re.search(r"\b" + re.escape(phrase) + r"\b", low):
            return canonical
    return None


class TriggerExtractor(BaseExtractor):
    """LLM-free extractor driven by trigger-verb patterns + spaCy NER.

    Uses the closed-set trigger patterns in
    :mod:`nuggetindex.audit.heuristics.triggers` (``role_succession`` /
    ``entity_rename`` / ``functional_relation``). For each trigger match in
    the document, emits a :class:`Nugget` with:

    * subject + object taken from the trigger's capture groups (with
      role-direction swap for ``role_succession`` hits);
    * predicate = the canonical predicate the pattern maps to, refined to a
      role-specific predicate when a role phrase is present;
    * ``subject_type`` / ``object_type`` probed via spaCy NER when
      available, else ``None``;
    * ``validity`` = :meth:`ValidityInterval.unknown`, so the pipeline's
      temporal-inference stage takes over (``source_date`` fallback);
    * ``extraction_confidence`` = ``0.5`` (lower than LLM; callers can
      filter).

    Gracefully degrades without spaCy -- types default to ``None``;
    direction-flip downstream falls through to ``"ok"``.
    """

    # The pipeline inspects this attribute to decide whether a missing
    # ``Document.source_date`` should raise the "placeholder validity"
    # warning. We emit ``ValidityInterval.unknown()`` for every match, so
    # we opt in.
    emits_placeholder_validity: bool = True

    def __init__(
        self,
        *,
        min_trigger_confidence: float = 0.5,
        source_id: str = "trigger-extract",
    ) -> None:
        if not 0.0 <= min_trigger_confidence <= 1.0:
            raise ValueError(
                "min_trigger_confidence must be within [0.0, 1.0], "
                f"got {min_trigger_confidence!r}"
            )
        self._min_confidence = float(min_trigger_confidence)
        self._default_source_id = source_id

    async def aextract(
        self,
        text: str,
        *,
        context: str = "",  # noqa: ARG002 -- surfaced for API symmetry.
        source_id: str | None = None,
    ) -> list[ExtractionResult]:
        if not text or not text.strip():
            return []

        confidence = self._min_confidence
        effective_source_id = source_id or self._default_source_id

        # Lazy probe import: keeps the core import path free of spaCy's
        # transitive deps when the user never calls ``aextract``.
        from nuggetindex.pipeline.entity_types import probe_entity_type

        results: list[ExtractionResult] = []
        for match in scan_triggers(text):
            triple = self._build_triple(text, match)
            if triple is None:
                continue
            subject, predicate, obj = triple
            subject_type = _probe_type_safe(probe_entity_type, subject)
            object_type = _probe_type_safe(probe_entity_type, obj)

            nugget = Nugget.new(
                kind=NuggetKind.SEMANTIC_FACT,
                fact=FactTriple(
                    subject=subject,
                    predicate=predicate,
                    object=obj,
                    text=match.match_text,
                    subject_type=subject_type,
                    object_type=object_type,
                ),
                validity=ValidityInterval.unknown(),
                epistemic=EpistemicState(confidence=confidence),
                provenance=(
                    ProvenanceRecord(
                        source_id=effective_source_id,
                        evidence_span=match.match_text,
                    ),
                ),
                extraction_confidence=confidence,
            )
            results.append(
                ExtractionResult(
                    nugget=nugget,
                    confidence=confidence,
                    rationale=f"trigger:{match.kind}:{predicate}",
                )
            )
        return results

    @staticmethod
    def _build_triple(
        text: str,
        match: TriggerMatch,
    ) -> tuple[str, str, str] | None:
        """Return ``(subject, predicate, object)`` or ``None`` to skip.

        Handles the role-succession direction swap + predicate refinement
        (``succeededBy`` -> ``chiefExecutiveOfficer`` etc.). Applies
        :func:`is_valid_object` to the emitted object so downstream stages
        don't see obvious malformed values.
        """
        subject = text[match.subject_span[0]:match.subject_span[1]].strip()
        if match.object_span == (0, 0):
            obj = ""
        else:
            obj = text[match.object_span[0]:match.object_span[1]].strip()
        if not subject or not obj:
            return None

        predicate = match.predicate
        if match.kind == "role_succession":
            role_pred = _match_role(match.match_text)
            if role_pred is not None:
                # Schema direction for role predicates is (ORG, role, PERSON):
                # "Satya Nadella became CEO of Microsoft" pattern captures
                # subj="Satya Nadella", obj="Microsoft"; swap to
                # (Microsoft, chiefExecutiveOfficer, Satya Nadella).
                subject, obj = obj, subject
                predicate = role_pred

        ok, _reason = is_valid_object(obj)
        if not ok:
            return None
        return subject, predicate, obj


def _probe_type_safe(probe_fn, mention: str) -> str | None:
    """Call ``probe_fn`` defensively.

    Returns ``None`` when spaCy is unavailable or the probe yields an
    empty / sentinel result, so downstream comparisons never see the
    ``"UNAVAILABLE"`` / ``"NONE"`` strings that aren't valid NER labels.
    """
    if not mention:
        return None
    try:
        label = probe_fn(mention)
    except Exception:  # pragma: no cover -- defensive; probe is pure-Python
        return None
    if not label or label in {"UNAVAILABLE", "NONE"}:
        return None
    return label
