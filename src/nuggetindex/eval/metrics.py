"""QA metric primitives used by the eval harness.

``exact_match`` and ``f1_score`` are the SQuAD-style classics with the
usual string-normalisation rules: lowercase, strip punctuation, collapse
whitespace, drop a leading ``a``/``an``/``the``. The normalisers are kept
simple on purpose — the harness is ballpark-accuracy, not leaderboard
submission.
"""

from __future__ import annotations

import re
import string
from collections import Counter

_ARTICLE_RE = re.compile(r"\b(a|an|the)\b", re.IGNORECASE)
_WHITESPACE_RE = re.compile(r"\s+")
_PUNCT_TABLE = str.maketrans("", "", string.punctuation)


def _normalize(text: str) -> str:
    """Lowercase, strip punctuation + articles, collapse whitespace."""
    if not text:
        return ""
    lowered = text.lower()
    no_punct = lowered.translate(_PUNCT_TABLE)
    no_articles = _ARTICLE_RE.sub(" ", no_punct)
    return _WHITESPACE_RE.sub(" ", no_articles).strip()


def exact_match(prediction: str, expected: str) -> bool:
    """Return True iff ``_normalize(prediction) == _normalize(expected)``.

    Treats missing predictions as wrong. A blank expected answer matches
    only a blank prediction so tests with "no answer" behave sensibly.
    """
    return _normalize(prediction) == _normalize(expected)


def f1_score(prediction: str, expected: str) -> float:
    """Token-overlap F1 between ``prediction`` and ``expected`` (both normalised).

    Returns ``0.0`` when either side is empty and they disagree on emptiness;
    returns ``1.0`` when both are empty (the "no answer" degenerate case).
    """
    pred_tokens = _normalize(prediction).split()
    exp_tokens = _normalize(expected).split()
    if not pred_tokens and not exp_tokens:
        return 1.0
    if not pred_tokens or not exp_tokens:
        return 0.0
    common = Counter(pred_tokens) & Counter(exp_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(exp_tokens)
    return 2 * precision * recall / (precision + recall)


def contains_expected(context: str, expected: str) -> bool:
    """Cheap oracle: does ``context`` contain all expected tokens?

    Used as the default fallback ``answerer`` when the caller doesn't
    provide an LLM. Not a leaderboard metric — this exists so the
    harness can still produce a green signal offline.
    """
    if not expected:
        return False
    norm_ctx = _normalize(context)
    norm_exp = _normalize(expected)
    if not norm_exp:
        return False
    # Substring match at the normalised level handles multi-token
    # expected answers (``"Larry Page"``) and short single-token ones.
    return norm_exp in norm_ctx


__all__ = ["contains_expected", "exact_match", "f1_score"]
