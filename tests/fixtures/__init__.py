"""Test-only fixtures.

``RuleBasedExtractor`` used to live in ``nuggetindex.extractors.rule_based``
but was removed from the public library (noisy output hurt first-time users
more than it helped). It now ships here as a cheap offline extractor that
existing tests can still import without pulling an LLM or the ``trigger``
extractor.
"""

from tests.fixtures.rule_based_extractor import RuleBasedExtractor

__all__ = ["RuleBasedExtractor"]
