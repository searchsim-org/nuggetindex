"""Cheap named-entity extraction for the doctor scan (fast mode).

Piggy-backs on spaCy's ``en_core_web_sm`` pipeline -- the same one already
loaded by :mod:`nuggetindex.audit.heuristics.timex` -- to surface entities
that can play the subject/object role in functional and rename relations.

The label set is deliberately narrow::

    ORG, PERSON, GPE, PRODUCT, EVENT, WORK_OF_ART, LAW

DATE/TIME fall under :mod:`timex`; CARDINAL / ORDINAL / QUANTITY / MONEY /
PERCENT / NORP / FAC / LOC / LANGUAGE are filtered out because they rarely
act as subjects or objects in the relations the doctor surfaces.

Backend availability mirrors :mod:`timex`: returns ``[]`` /
``ner_available() == False`` when spaCy or the English model is missing.
"""

from __future__ import annotations

from dataclasses import dataclass

from nuggetindex.audit.heuristics.timex import get_nlp

# Labels worth surfacing. See module docstring for rationale.
_WANTED_LABELS: frozenset[str] = frozenset(
    {"ORG", "PERSON", "GPE", "PRODUCT", "EVENT", "WORK_OF_ART", "LAW"}
)


@dataclass(frozen=True)
class Entity:
    """A named entity tagged in a piece of text."""

    text: str
    label: str
    start_char: int
    end_char: int


def ner_available() -> bool:
    """Return ``True`` iff spaCy + ``en_core_web_sm`` are importable/loadable.

    Cheap after the first call (results are cached via the shared ``timex``
    module-level pipeline).
    """
    return get_nlp() is not None


def extract_entities(text: str) -> list[Entity]:
    """Extract named entities from ``text``.

    Best-effort: returns an empty list if spaCy / the English model is
    unavailable, or if ``text`` is empty. Only labels in the narrow set
    documented in the module docstring are returned.
    """
    if not text:
        return []

    nlp = get_nlp()
    if nlp is None:
        return []

    doc = nlp(text)
    out: list[Entity] = []
    for ent in doc.ents:
        if ent.label_ not in _WANTED_LABELS:
            continue
        out.append(
            Entity(
                text=ent.text,
                label=ent.label_,
                start_char=ent.start_char,
                end_char=ent.end_char,
            )
        )
    return out
