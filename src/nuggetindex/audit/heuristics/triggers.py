"""Closed-set trigger-verb pattern matching for the doctor scan (fast mode).

Hand-curated regex patterns that flag likely sites of three relation kinds:

* ``role_succession`` -- "X became CEO of Y", "X succeeded Y as ...",
  "X replaced Y as ..." (canonical predicate ``succeededBy``).
* ``entity_rename`` -- "X renamed to Y", "X formerly known as Y",
  "X merged into Y" (canonical predicates ``renamedTo`` /
  ``formerlyKnownAs`` / ``corporateName``).
* ``functional_relation`` -- "X acquired Y", "X headquartered in Y",
  "X priced at ..." (canonical predicates ``acquired`` /
  ``headquarteredIn`` / ``priceOf``).

This is a **hint layer**, not a relation extractor. Patterns are kept short
and capture best-effort subject/object boundaries. Downstream code in the
orchestrator (Phase 2.4) is responsible for normalising rename direction and
deciding which hits to promote.

The module is stdlib-only (``re`` + ``dataclasses``) so it works even when
the ``[doctor]`` extra is not installed.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

# Cache-key component used by :class:`nuggetindex.extractors.cache.CachedExtractor`.
# Bump when ``_PATTERNS`` or the emitted predicate mapping changes in a way
# that would produce different ``TriggerMatch`` output for the same input.
TRIGGER_VERSION = "v2026-04"

TriggerKind = Literal[
    "role_succession",
    "entity_rename",
    "functional_relation",
]


@dataclass(frozen=True)
class TriggerMatch:
    """A single trigger-pattern hit.

    ``subject_span`` / ``object_span`` are character offsets into the original
    text for the first / second capture groups. Single-object patterns that
    don't have a natural second entity record ``object_span = (0, 0)`` as a
    convention.
    """

    kind: TriggerKind
    predicate: str
    subject_span: tuple[int, int]
    object_span: tuple[int, int]
    match_text: str


# (pattern, kind, canonical-predicate). Order matters: earlier patterns win
# ties when two patterns match the same span. Each pattern captures the
# subject in group 1 and, where applicable, the object / price in group 2.
_PATTERNS: list[tuple[re.Pattern[str], TriggerKind, str]] = [
    # --- role_succession --------------------------------------------------
    (
        re.compile(
            r"\b([A-Z][\w .']+?)\s+(?:became|was named|was appointed)\s+"
            r"(?:the\s+new\s+)?"
            r"(?:CEO|chairman|president|director|chair|chief executive|"
            r"chief operating officer|COO|CFO|CTO)\s+of\s+"
            r"([A-Z][\w .'&]+)",
            re.IGNORECASE,
        ),
        "role_succession",
        "succeededBy",
    ),
    (
        re.compile(
            r"\b([A-Z][\w .']+?)\s+succeeded\s+([A-Z][\w .']+)\s+as\s+"
            r"(?:CEO|chairman|president|chief executive)",
            re.IGNORECASE,
        ),
        "role_succession",
        "succeededBy",
    ),
    (
        re.compile(
            r"\b([A-Z][\w .']+?)\s+(?:replaced|took over from)\s+"
            r"([A-Z][\w .']+)\s+as\s+",
            re.IGNORECASE,
        ),
        "role_succession",
        "succeededBy",
    ),
    # --- entity_rename ----------------------------------------------------
    (
        re.compile(
            r"\b([A-Z][\w .'&]+?)\s+(?:was\s+)?(?:renamed|rebranded)\s+"
            r"(?:to|as)\s+([A-Z][\w .'&]+)",
            re.IGNORECASE,
        ),
        "entity_rename",
        "renamedTo",
    ),
    (
        re.compile(
            r"\b([A-Z][\w .'&]+?)\s+formerly known as\s+([A-Z][\w .'&]+)",
            re.IGNORECASE,
        ),
        "entity_rename",
        "formerlyKnownAs",
    ),
    (
        re.compile(
            r"\b([A-Z][\w .'&]+?)\s+merged into\s+([A-Z][\w .'&]+)",
            re.IGNORECASE,
        ),
        "entity_rename",
        "corporateName",
    ),
    # --- functional_relation ---------------------------------------------
    (
        re.compile(
            r"\b([A-Z][\w .'&]+?)\s+acquired\s+([A-Z][\w .'&]+?)"
            r"(?:\s+for\s+\$?[\d,.]+\s*(?:billion|million))?",
            re.IGNORECASE,
        ),
        "functional_relation",
        "acquired",
    ),
    (
        re.compile(
            r"\b([A-Z][\w .'&]+?)\s+(?:is\s+)?headquartered in\s+"
            r"([A-Z][\w .'&]+)",
            re.IGNORECASE,
        ),
        "functional_relation",
        "headquarteredIn",
    ),
    (
        re.compile(
            r"\b([A-Z][\w .'&]+?)\s+priced\s+at\s+"
            r"(\$?[\d,.]+\s*(?:billion|million))",
            re.IGNORECASE,
        ),
        "functional_relation",
        "priceOf",
    ),
]


_KINDS: set[TriggerKind] = {
    "role_succession",
    "entity_rename",
    "functional_relation",
}


def trigger_kinds() -> set[TriggerKind]:
    """Return the closed set of kinds recognised by :func:`scan_triggers`."""
    return set(_KINDS)


def _overlaps(a: tuple[int, int], b: tuple[int, int]) -> bool:
    """Half-open interval overlap; ``(0, 0)`` is treated as an empty sentinel."""
    if a == (0, 0) or b == (0, 0):
        return False
    return a[0] < b[1] and b[0] < a[1]


def scan_triggers(text: str) -> list[TriggerMatch]:
    """Run the hand-curated trigger regex set against ``text``.

    Returns all non-overlapping matches. When two patterns hit overlapping
    spans, the earlier pattern in ``_PATTERNS`` wins; that's the only overlap
    resolution done -- this is a hint layer, not a full extractor.
    """
    if not text:
        return []

    out: list[TriggerMatch] = []
    used_spans: list[tuple[int, int]] = []

    for pattern, kind, predicate in _PATTERNS:
        for m in pattern.finditer(text):
            full_span = m.span(0)
            if any(_overlaps(full_span, used) for used in used_spans):
                continue

            subject_span = m.span(1)
            # All current patterns have a group 2 (object or price). If a
            # future single-capture pattern is added, missing-group yields
            # (-1, -1) from re; normalise that to the (0, 0) sentinel.
            try:
                object_span = m.span(2)
            except IndexError:  # pragma: no cover -- defensive
                object_span = (0, 0)
            if object_span == (-1, -1):  # pragma: no cover -- defensive
                object_span = (0, 0)

            out.append(
                TriggerMatch(
                    kind=kind,
                    predicate=predicate,
                    subject_span=subject_span,
                    object_span=object_span,
                    match_text=m.group(0),
                )
            )
            used_spans.append(full_span)

    return out
