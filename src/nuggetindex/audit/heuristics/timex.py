"""Temporal-expression tagging for the doctor scan (fast mode).

Uses spaCy's ``en_core_web_sm`` DATE/TIME entities, resolved to concrete
:class:`datetime` values via ``dateparser`` when possible. Both dependencies
are **optional** and live behind the ``doctor`` extra; this module must import
cleanly even when they are absent. When the backend is unavailable,
:func:`tag_timex` returns an empty list and :func:`timex_available` returns
``False``.

Installation
------------

The ``doctor`` extra pulls the Python packages::

    pip install "nuggetindex[doctor]"

Note that spaCy's English model is **not** installable from PyPI via an
extras marker -- after installing the extra, download it explicitly::

    python -m spacy download en_core_web_sm

Without the model, :func:`timex_available` returns ``False`` even if the
``spacy`` package is present.

Thread-safety
-------------

The loaded spaCy pipeline is cached in a module-level global
(``_NLP``). The cache is populated on the first call to :func:`tag_timex`
from an "unloaded" state; subsequent calls read the cache. This is a
read-after-first-write pattern, safe enough for the single-process,
primarily-sequential audit use case. No lock is held. Callers that truly
need concurrent first-call initialisation should warm the cache by calling
:func:`timex_available` once at startup.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any

# Sentinel values for the lazy spaCy cache.
_UNLOADED: Any = object()
_UNAVAILABLE: Any = None  # distinct from "not yet attempted"

# Module-level cache: one of {_UNLOADED, _UNAVAILABLE, <spacy Language>}.
_NLP: Any = _UNLOADED


@dataclass(frozen=True)
class TimeExpression:
    """A tagged temporal expression in a piece of text.

    ``parsed`` is the best-effort concrete datetime resolved by
    ``dateparser``. It may be ``None`` when the expression is present but
    ambiguous (e.g. bare "summer", "recently").
    """

    span: str
    start_char: int
    end_char: int
    parsed: datetime | None


def _load_nlp() -> Any:
    """Return the cached spaCy pipeline, or ``None`` if unavailable.

    Side-effect: populates ``_NLP`` on first call.
    """
    global _NLP
    if _NLP is _UNLOADED:
        try:
            import spacy  # type: ignore[import-not-found]
        except ImportError:
            _NLP = _UNAVAILABLE
            return _NLP
        try:
            _NLP = spacy.load("en_core_web_sm")
        except (OSError, ImportError):
            # OSError: model not downloaded. ImportError: rare, but some
            # spaCy builds raise it when a component can't be constructed.
            _NLP = _UNAVAILABLE
    return _NLP


def get_nlp() -> Any | None:
    """Return the shared spaCy pipeline, or ``None`` if unavailable.

    Public accessor around the module-level cache so other heuristic modules
    (e.g. :mod:`nuggetindex.audit.heuristics.ner`) can reuse the same loaded
    ``en_core_web_sm`` pipeline rather than loading a second copy. Populates
    the cache on first call; cheap on subsequent calls.
    """
    return _load_nlp()


def timex_available() -> bool:
    """Return ``True`` iff spaCy + ``en_core_web_sm`` + dateparser all import.

    Cheap after the first call (results are cached). The first call may load
    the spaCy model and a copy of ``dateparser``, which can take ~1s.
    """
    nlp = _load_nlp()
    if nlp is None:
        return False
    try:
        import dateparser  # type: ignore[import-not-found]  # noqa: F401
    except ImportError:
        return False
    return True


def tag_timex(
    text: str,
    *,
    reference_date: datetime | None = None,
) -> list[TimeExpression]:
    """Tag temporal expressions in ``text``.

    Best-effort: returns an empty list if spaCy / the English model /
    dateparser are unavailable. Relative expressions ("last year", "two
    weeks ago") resolve against ``reference_date`` when provided.
    """
    if not text:
        return []

    nlp = _load_nlp()
    if nlp is None:
        return []

    try:
        import dateparser  # type: ignore[import-not-found]
    except ImportError:
        return []

    settings: dict[str, Any] = {}
    if reference_date is not None:
        settings["RELATIVE_BASE"] = reference_date

    doc = nlp(text)
    out: list[TimeExpression] = []
    for ent in doc.ents:
        if ent.label_ not in {"DATE", "TIME"}:
            continue
        parsed: datetime | None
        try:
            parsed = dateparser.parse(ent.text, settings=settings) if settings else dateparser.parse(ent.text)
        except Exception:
            # dateparser is noisy on garbage input; treat parse failure as
            # "ambiguous but detected".
            parsed = None
        out.append(
            TimeExpression(
                span=ent.text,
                start_char=ent.start_char,
                end_char=ent.end_char,
                parsed=parsed,
            )
        )
    return out
