"""Cheap language detection for doctor-scan stratification.

This module offers a single public helper, :func:`_detect_language`, used by
:mod:`nuggetindex.audit.heuristics.sample` to bucket documents by language
before drawing a stratified sample.

Design notes
------------
* The optional ``langdetect`` dependency (``[doctor]`` extra) is tried first.
  It is pure Python and tiny, so it is safe to import at call time.
* If ``langdetect`` is not installed, we fall back to a first-letter
  Unicode-block heuristic that returns one of ``en``, ``ru``, ``zh``, ``ar``,
  or ``unk``. This keeps the heuristic deterministic and dependency-free in
  the worst case.
* Text is capped at 500 characters before detection — language identification
  is dominated by the first few tokens, and the cap keeps the doctor scan's
  per-doc latency bounded.
"""

from __future__ import annotations

import contextlib

_TEXT_SNIPPET_CAP = 500


def _detect_language(text: str) -> str:
    """Return an ISO-639-1-ish language code for ``text``.

    Parameters
    ----------
    text:
        Raw document text. Only the first ``_TEXT_SNIPPET_CAP`` characters
        are inspected.

    Returns
    -------
    str
        A short language code (``"en"``, ``"ru"``, ``"zh"``, ``"ar"``, ...)
        or ``"unk"`` when the input is empty or detection fails.
    """
    if not text:
        return "unk"
    snippet = text[:_TEXT_SNIPPET_CAP]
    try:
        import langdetect  # type: ignore[import-not-found]

        # langdetect.detect is non-deterministic by default; seeding keeps
        # the doctor scan reproducible across runs.
        with contextlib.suppress(Exception):  # pragma: no cover -- defensive
            langdetect.DetectorFactory.seed = 0  # type: ignore[attr-defined]
        try:
            return str(langdetect.detect(snippet))
        except Exception:
            # LangDetectException or similar -- fall through to heuristic.
            pass
    except ImportError:
        pass

    return _fallback_language(snippet)


def _fallback_language(snippet: str) -> str:
    """Classify ``snippet`` by the first non-space character's Unicode block.

    This is a conservative heuristic: it only tries to separate common scripts.
    Everything else collapses to ``"unk"``. We avoid per-character histograms
    so the cost stays O(1) in ``len(snippet)``.
    """
    for ch in snippet:
        if ch.isspace():
            continue
        cp = ord(ch)
        # CJK Unified Ideographs + common supplementary ranges.
        if 0x4E00 <= cp <= 0x9FFF or 0x3400 <= cp <= 0x4DBF:
            return "zh"
        # Cyrillic.
        if 0x0400 <= cp <= 0x04FF or 0x0500 <= cp <= 0x052F:
            return "ru"
        # Arabic.
        if 0x0600 <= cp <= 0x06FF or 0x0750 <= cp <= 0x077F:
            return "ar"
        # Basic Latin + Latin-1 Supplement -- default to English.
        if 0x0020 <= cp <= 0x024F:
            return "en"
        return "unk"
    return "unk"


__all__ = ["_detect_language"]
