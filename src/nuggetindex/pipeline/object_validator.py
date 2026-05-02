"""Reject nuggets whose extracted object is obviously malformed.

Language-agnostic, cheap, deterministic. Runs AFTER the LLM extractor's raw
output and BEFORE the pipeline's conflict-detection stage. Dropping malformed
objects at this point prevents bogus CONTESTED flags driven by LLM noise
(bare years, interrogative titles, empty-after-strip objects).
"""

from __future__ import annotations

import re

# Pure-numeric / money / percentage: e.g. "2000", "42.5", "$26.2B", "15%"
_PURE_NUMERIC_RE = re.compile(
    r"^[\s]*[\d.,$€£¥%\-+]+\s*(billion|million|thousand|B|M|K)?\s*$",
    re.IGNORECASE,
)
# No alphabetic characters at all (letters in any script — ASCII or Unicode).
_NO_LETTERS_RE = re.compile(r"^[^a-zA-Z\u00C0-\uFFFF]+$")
# Question-ending (any script)
_ENDS_WITH_QUESTION_RE = re.compile(r"[?？]\s*$")


def is_valid_object(obj: str) -> tuple[bool, str]:
    """Return ``(ok, reason)``. ``ok=True`` if the object passes basic sanity.

    Rules:
      * Must be ≥ 2 chars after strip.
      * Must contain at least one alphabetic character.
      * Must not be a pure numeric / currency / percent token.
      * Must not end with an interrogative mark (Western ``?`` or CJK ``？``).
    """
    s = obj.strip()
    if len(s) < 2:
        return False, "too_short"
    # Check bare-numeric BEFORE the generic "no letters" rule so e.g. "2000"
    # and "$26.2B" are reported with the more specific reason.
    if _PURE_NUMERIC_RE.match(s):
        return False, "bare_numeric"
    if _NO_LETTERS_RE.match(s):
        return False, "no_letters"
    if _ENDS_WITH_QUESTION_RE.search(s):
        return False, "interrogative"
    return True, ""
