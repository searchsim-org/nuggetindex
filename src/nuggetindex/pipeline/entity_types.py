"""Entity-type probe + triple-direction check for the pipeline.

Uses the shared spaCy pipeline loaded by
:func:`nuggetindex.audit.heuristics.timex.get_nlp` to classify a short
entity mention into one of spaCy's NER label strings. Gracefully degrades
when spaCy isn't installed: :func:`probe_entity_type` returns the sentinel
``"UNAVAILABLE"`` and :func:`check_triple_direction` passes every triple
through as ``"ok"``.

This module is deliberately additive: it does not rewrite nuggets itself,
it only classifies a mention or a triple's direction. The pipeline stage in
``constructor.py`` consumes the result to emit a flipped / rejected triple
before the existing conflict detector runs.
"""

from __future__ import annotations

from typing import Any


def probe_entity_type(mention: str, nlp: Any | None = None) -> str:
    """Classify ``mention`` into a single spaCy NER label when possible.

    Returns
    -------
    str
        * spaCy label (e.g. ``"PERSON"`` / ``"ORG"`` / ``"GPE"`` /
          ``"PRODUCT"`` / ``"DATE"`` / ...) when ``mention`` is cleanly
          recognized as exactly one entity that spans the whole string.
        * ``"COMPOUND"`` when spaCy recognizes multiple entities in the
          mention, or one entity that covers only part of the string (e.g.
          ``"A Day In The Life of Apple's CEO"`` contains a ``PERSON``
          substring but is not itself a single entity).
        * ``"NONE"`` when the mention is empty or spaCy finds no entities.
        * ``"UNAVAILABLE"`` when spaCy is not installed. Callers should
          treat this as "skip the type check" to preserve graceful
          degradation.

    Parameters
    ----------
    mention:
        The raw surface form to classify. Whitespace is stripped.
    nlp:
        Optional preloaded spaCy ``Language``. When ``None`` (the common
        case), the shared pipeline from
        :func:`nuggetindex.audit.heuristics.timex.get_nlp` is used.
    """
    if nlp is None:
        # Lazy import so the core package still imports when spaCy isn't
        # installed -- ``get_nlp()`` itself is safe to call either way.
        from nuggetindex.audit.heuristics.timex import get_nlp

        nlp = get_nlp()
    if nlp is None:
        return "UNAVAILABLE"
    text = mention.strip()
    if not text:
        return "NONE"
    doc = nlp(text)
    if not doc.ents:
        return "NONE"
    # Single entity covering the entire cleaned input -- a clean type hit.
    if len(doc.ents) == 1 and doc.ents[0].text.strip() == text:
        return doc.ents[0].label_
    # Multi-entity span OR partial-coverage span: not a clean single-entity
    # mention. This is the signal we use to reject YouTube-style titles that
    # contain a real entity as a substring (e.g. "A Day In The Life of
    # Apple's CEO" -> COMPOUND, because spaCy tags "Apple" inside it).
    return "COMPOUND"


def check_triple_direction(
    subject: str,
    predicate: str,
    object: str,  # noqa: A002  -- mirrors FactTriple's field name.
    schema: Any,
    nlp: Any | None = None,
    *,
    subject_type: str | None = None,
    object_type: str | None = None,
) -> str:
    """Check that ``(subject, predicate, object)`` matches the schema's types.

    Decision order:

    1. If BOTH ``subject_type`` and ``object_type`` are provided (the
       LLM-emitted path, fix 9), use them directly -- no spaCy call.
    2. Otherwise fall back to the spaCy NER probe on the raw mentions.
    3. If neither source yields a type, return ``"ok"`` (graceful degrade).

    Returns one of:

    * ``"ok"`` -- direct arg-type check passes, OR the predicate has no
      expected arg-types, OR no type source is available. The caller keeps
      the triple unchanged.
    * ``"flip"`` -- the direct check fails BUT reversing subject and
      object would satisfy both constraints. The caller should swap
      subject/object (and their types) before persisting.
    * ``"reject"`` -- neither direction's types match; the caller must
      drop the triple.

    LLM-emitted types are uppercased before comparison so lowercase /
    mixed-case values (``"person"``, ``"Org"``) behave the same as the
    spaCy-compatible uppercase labels used in the schema YAML.
    """
    expected_subj = schema.expected_subject_types(predicate)
    expected_obj = schema.expected_object_types(predicate)
    if not expected_subj and not expected_obj:
        return "ok"

    # Path 1: prefer LLM-emitted types when both are present. The LLM sees
    # full sentence context so these are strictly more reliable than spaCy
    # NER on an isolated mention (and they work cross-lingually).
    if subject_type is not None and object_type is not None:
        subj_type = subject_type.strip().upper()
        obj_type = object_type.strip().upper()
    else:
        # Path 2: spaCy NER probe fallback. probe_entity_type() returns the
        # sentinel ``"UNAVAILABLE"`` when spaCy isn't installed.
        subj_type = probe_entity_type(subject, nlp=nlp)
        obj_type = probe_entity_type(object, nlp=nlp)
        # Path 3: spaCy missing on either probe -> graceful degrade.
        if subj_type == "UNAVAILABLE" or obj_type == "UNAVAILABLE":
            return "ok"

    # Direct direction.
    direct_ok = (not expected_subj or subj_type in expected_subj) and (
        not expected_obj or obj_type in expected_obj
    )
    if direct_ok:
        return "ok"
    # Flip direction (subject <-> object).
    flip_ok = (not expected_subj or obj_type in expected_subj) and (
        not expected_obj or subj_type in expected_obj
    )
    if flip_ok:
        return "flip"
    return "reject"
