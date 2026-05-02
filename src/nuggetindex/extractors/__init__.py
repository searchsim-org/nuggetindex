"""Extractors: turn natural-language text into governed Nuggets.

Public re-exports for ergonomic imports:

    from nuggetindex.extractors import (
        BaseExtractor, LLMExtractor, QualityGate,
        LLMConfig, build_client,
    )

Provider SDK imports inside the client modules are lazy, so this import
block is safe even when optional extras (openai, anthropic, etc.) are not
installed.
"""

from nuggetindex.extractors.base import BaseExtractor, ExtractionResult
from nuggetindex.extractors.cache import CachedExtractor
from nuggetindex.extractors.clients.base import LLMClient, LLMConfig, build_client
from nuggetindex.extractors.llm import ExtractionPayload, LLMExtractor
from nuggetindex.extractors.quality import QualityGate, QualityGateResult
from nuggetindex.extractors.trigger import TriggerExtractor

__all__ = [
    "BaseExtractor",
    "CachedExtractor",
    "ExtractionPayload",
    "ExtractionResult",
    "LLMClient",
    "LLMConfig",
    "LLMExtractor",
    "QualityGate",
    "QualityGateResult",
    "TriggerExtractor",
    "build_client",
]
