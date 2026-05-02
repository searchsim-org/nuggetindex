"""Prompt templates bundled with the extractor package.

Exposes a version string that callers (chiefly :class:`CachedExtractor`)
include in cache keys so a prompt change invalidates stored results. The
string is purely a cache-key component — no prompt parsing depends on it.

Bump the suffix whenever an included prompt file changes in a way that
would produce different extractions (wording, structured-output schema,
instructions). The module-level ``Path`` attributes below are convenient
pointers for callers that want to read a specific prompt.
"""

from __future__ import annotations

from pathlib import Path

PROMPT_VERSION = "v2026-04"

_PROMPTS_DIR = Path(__file__).parent
EXTRACTION_PROMPT_PATH = _PROMPTS_DIR / "extraction.md"
CHAIN_RESOLVER_PROMPT_PATH = _PROMPTS_DIR / "chain_resolver.md"
JUDGE_PROMPT_PATH = _PROMPTS_DIR / "judge.md"

__all__ = [
    "CHAIN_RESOLVER_PROMPT_PATH",
    "EXTRACTION_PROMPT_PATH",
    "JUDGE_PROMPT_PATH",
    "PROMPT_VERSION",
]
