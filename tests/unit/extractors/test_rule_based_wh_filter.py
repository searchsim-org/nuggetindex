"""Regression tests: RuleBasedExtractor must not emit interrogative-pronoun subjects.

Covers findings A1 — rule-based patterns like ``X is Y`` would greedily match
question-shaped passages (``Who is the CEO of Google?``) and emit
``(who, is, ...)`` triples. The extractor now drops any subject whose
stripped lowercase form is an interrogative pronoun.
"""

from __future__ import annotations

import pytest

from tests.fixtures import RuleBasedExtractor

_INTERROGATIVES = [
    "Who",
    "What",
    "Where",
    "When",
    "Why",
    "Which",
    "Whose",
    "How",
    "Whom",
    "Whether",
]


@pytest.mark.parametrize("pronoun", _INTERROGATIVES)
@pytest.mark.asyncio
async def test_rule_based_drops_interrogative_subjects(pronoun: str) -> None:
    ext = RuleBasedExtractor()
    text = f"{pronoun} is the CEO of Google?"
    results = await ext.aextract(text)
    subjects = {r.nugget.fact.subject.lower() for r in results}
    assert pronoun.lower() not in subjects, (
        f"rule-based extractor should drop {pronoun!r} as a subject; got {subjects}"
    )


@pytest.mark.asyncio
async def test_rule_based_still_emits_legitimate_subjects() -> None:
    ext = RuleBasedExtractor()
    results = await ext.aextract("Sundar Pichai is the CEO of Google.")
    subjects = {r.nugget.fact.subject.lower() for r in results}
    assert any("pichai" in s or "google" in s or "sundar" in s for s in subjects), (
        f"extractor should still find real subjects in a normal sentence; got {subjects}"
    )
