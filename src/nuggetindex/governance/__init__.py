"""Governance: framework-agnostic postprocessor and session-cache helpers.

The ``GovernancePostProcessor`` is the Tier-1 adoption wedge that wraps any
retriever in a small, async-safe layer that:

* maintains a content-addressed session cache of nuggets extracted from
  retrieved passages,
* cross-references new retrievals against that cache so DEPRECATED passages
  get filtered and CONTESTED ones get a ``[DISPUTED]`` prefix,
* is itself framework-agnostic — LangChain / LlamaIndex / Haystack adapters
  will translate their native types to and from ``RetrievedPassage``.
"""
from nuggetindex.governance.postprocessor import (
    GovernancePostProcessor,
    RetrievedPassage,
)
from nuggetindex.governance.session_cache import default_cache_path, passage_hash

__all__ = [
    "GovernancePostProcessor",
    "RetrievedPassage",
    "default_cache_path",
    "passage_hash",
]
