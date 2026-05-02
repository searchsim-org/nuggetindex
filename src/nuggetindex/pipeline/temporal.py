"""Stage 3: temporal validity inference (Algorithm 1, paper §3.3, spec §5.4).

Given evidence text and a document source date, produce a ``ValidityInterval``
plus an optional confidence multiplier for ambiguous cues.

The patterns handled (in precedence order):

* ``from <date> to <date>`` / ``<date>-<date>``     -> start + end
* ``since <date>`` / ``starting <date>`` / ``from <date>``  -> start only
* ``until <date>`` / ``was X until <date>``         -> end only
* ``became X in <date>`` / ``X in <date>``          -> start only
* ``resumed <role> in <date>`` / ``retook ... in <date>``   -> start only
* bare year (e.g. ``"event happened in 2024"``)     -> start=source_date,
  confidence multiplier 0.75 (spec §5.4)
* no temporal cue                                   -> start=source_date, end=None
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Literal, overload

import dateparser

from nuggetindex.core.models import ValidityInterval


@dataclass
class InferredValidity:
    """Result of temporal inference.

    ``confidence`` is a multiplier to apply to ``EpistemicState.confidence``:
    1.0 when a clear cue was matched, 0.75 for ambiguous (year-only) cases,
    per spec §5.4.
    """

    interval: ValidityInterval
    confidence: float = 1.0


_DATEPARSER_SETTINGS = {"RETURN_AS_TIMEZONE_AWARE": True}


def _parse(s: str) -> datetime | None:
    """Parse a date string, returning a tz-aware UTC datetime or None."""
    parsed: datetime | None = dateparser.parse(s, settings=_DATEPARSER_SETTINGS)
    if parsed is None:
        return None
    if parsed.tzinfo is None:
        # dateparser sometimes returns naive despite the setting
        parsed = parsed.replace(tzinfo=UTC)
    return parsed


# --- Pattern order matters: range forms before single forms, specific before general. ---

# "from <date> to <date>"
_FROM_TO = re.compile(
    r"\bfrom\s+([A-Za-z0-9 ,.\-/]+?)\s+to\s+([A-Za-z0-9 ,.\-/]+?)"
    r"(?=[.,;)]|\s+(?:and|or|but|while|when|where|which|who|then|that|before|after|until|since)\b|$)",
    re.IGNORECASE,
)

# "<YYYY>-<YYYY>" explicit year range (avoid matching inside ISO dates like 2019-03)
_YEAR_RANGE = re.compile(r"\b((?:19|20)\d{2})\s*[-–]\s*((?:19|20)\d{2})\b")

# "since <date>" / "starting <date>" / "from <date>" (no "to")
_SINCE = re.compile(
    r"\b(?:since|starting|from)\s+([A-Za-z0-9 ,.\-/]+?)"
    r"(?=[.,;)]|\s+(?:and|or|but|while|when|where|which|who|then|that|has|have|was|were|is|are|until)\b|$)",
    re.IGNORECASE,
)

# "until <date>"
_UNTIL = re.compile(
    r"\buntil\s+([A-Za-z0-9 ,.\-/]+?)"
    r"(?=[.,;)]|\s+(?:and|or|but|while|when|where|which|who|then|that)\b|$)",
    re.IGNORECASE,
)

# "became ... in <date>" / "resumed ... in <date>" / "retook ... in <date>"
_BECAME_IN = re.compile(
    r"\b(?:became|resumed|retook|took\s+over)\b.*?\bin\s+([A-Za-z0-9 ,.\-/]+?)"
    r"(?=[.,;)]|\s+(?:and|or|but|while|when|where|which|who|then|that)\b|$)",
    re.IGNORECASE,
)

# generic "in <date>" (e.g. "Pichai in 2019")
_IN_DATE = re.compile(
    r"\bin\s+((?:19|20)\d{2}(?:[-/\s][A-Za-z0-9 ,.\-/]*)?)"
    r"(?=[.,;)]|\s+(?:and|or|but|while|when|where|which|who|then|that)\b|$)",
    re.IGNORECASE,
)

# bare year (fallback — ambiguity flag)
_BARE_YEAR = re.compile(r"\b(19|20)\d{2}\b")


@overload
def infer_validity(
    text: str,
    *,
    source_date: datetime,
    prior: ValidityInterval | None = ...,
) -> ValidityInterval: ...


@overload
def infer_validity(
    text: str,
    *,
    source_date: datetime,
    return_confidence: Literal[False],
    prior: ValidityInterval | None = ...,
) -> ValidityInterval: ...


@overload
def infer_validity(
    text: str,
    *,
    source_date: datetime,
    return_confidence: Literal[True],
    prior: ValidityInterval | None = ...,
) -> InferredValidity: ...


def infer_validity(
    text: str,
    *,
    source_date: datetime,
    return_confidence: bool = False,
    prior: ValidityInterval | None = None,
) -> ValidityInterval | InferredValidity:
    """Infer ``ValidityInterval`` from evidence text and document source date.

    Pass ``return_confidence=True`` to receive the full ``InferredValidity``
    wrapper (including the confidence multiplier); otherwise returns the bare
    ``ValidityInterval``.

    ``prior`` is an optional baseline interval (e.g. one already supplied by
    an LLM extractor from a structured-output temporal-expression field).
    When the text contains no temporal cue, ``prior`` is returned unchanged
    instead of the ``source_date`` fallback, so LLM-emitted concrete
    intervals aren't clobbered by weaker rule-based inference.
    """
    if source_date.tzinfo is None:
        raise ValueError("source_date must be timezone-aware")

    result = _infer(text, source_date=source_date, prior=prior)
    return result if return_confidence else result.interval


def _infer(
    text: str,
    *,
    source_date: datetime,
    prior: ValidityInterval | None = None,
) -> InferredValidity:
    # 1. "from A to B" — explicit range
    m = _FROM_TO.search(text)
    if m:
        start = _parse(m.group(1))
        end = _parse(m.group(2))
        if start is not None and end is not None and end > start:
            return InferredValidity(ValidityInterval(start=start, end=end))

    # 2. "YYYY-YYYY" — year range
    m = _YEAR_RANGE.search(text)
    if m:
        start = _parse(m.group(1))
        end = _parse(m.group(2))
        if start is not None and end is not None and end > start:
            return InferredValidity(ValidityInterval(start=start, end=end))

    # 3. "until <date>" — end only; start falls back to source_date
    m_until = _UNTIL.search(text)
    if m_until:
        end = _parse(m_until.group(1))
        if end is not None:
            start_candidate = source_date
            if end <= start_candidate:
                # End is in the past relative to source_date; use a conservative
                # open start so the interval is valid (start < end).
                start_candidate = end.replace(year=end.year - 1)
            return InferredValidity(
                ValidityInterval(start=start_candidate, end=end)
            )

    # 4. "since <date>" / "starting <date>" / "from <date>" — start only
    m = _SINCE.search(text)
    if m:
        start = _parse(m.group(1))
        if start is not None:
            return InferredValidity(ValidityInterval(start=start))

    # 5. "became|resumed|retook ... in <date>" — start only
    m = _BECAME_IN.search(text)
    if m:
        start = _parse(m.group(1))
        if start is not None:
            return InferredValidity(ValidityInterval(start=start))

    # 6. generic "in <YEAR>" — start only, full confidence (explicit cue)
    m = _IN_DATE.search(text)
    if m:
        start = _parse(m.group(1))
        if start is not None:
            return InferredValidity(ValidityInterval(start=start))

    # 7. Bare year without a trigger phrase — ambiguous; use source_date but
    # reduce confidence (spec §5.4). A concrete ``prior`` supplied by the
    # caller (e.g. an LLM-emitted interval) is considered stronger than a
    # bare-year signal, so we keep it.
    if _BARE_YEAR.search(text):
        if prior is not None and not prior.is_placeholder():
            return InferredValidity(prior, confidence=0.75)
        return InferredValidity(
            ValidityInterval(start=source_date, validity_known=False),
            confidence=0.75,
        )

    # 8. No temporal cue at all. Prefer the caller-supplied ``prior`` if
    # present (don't clobber concrete intervals with a source_date fallback).
    if prior is not None and not prior.is_placeholder():
        return InferredValidity(prior)
    return InferredValidity(ValidityInterval(start=source_date, validity_known=False))
