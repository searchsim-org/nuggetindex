"""Sidecar runtime: drop-in layer on top of any retriever.

Two modes:

- ``offline-curated``: queries a pre-built :class:`~nuggetindex.NuggetStore`
  alongside the original retriever. Good for frozen corpora.
- ``just-in-time``: runs the extractor live on the top-K retrieved passages.
  Good for live corpora where re-indexing is impractical.

Public API::

    from nuggetindex.sidecar import Sidecar, Router, ContextFormatter, SidecarResponse

    sidecar = Sidecar(store=my_nugget_store, mode="offline-curated")
    response = await sidecar.ahandle("who was Google's CEO in 2013?")
    prompt = response.context_block + "\\n\\nUser question: " + query
"""

from nuggetindex.sidecar.context import ContextFormatter
from nuggetindex.sidecar.freshness import FreshnessChecker, FreshnessResult
from nuggetindex.sidecar.jit_cache import JITPassageCache
from nuggetindex.sidecar.modes import JustInTime, OfflineCurated, SidecarMode
from nuggetindex.sidecar.router import Router, RouterDecision
from nuggetindex.sidecar.sidecar import Sidecar, SidecarResponse

__all__ = [
    "ContextFormatter",
    "FreshnessChecker",
    "FreshnessResult",
    "JITPassageCache",
    "JustInTime",
    "OfflineCurated",
    "Router",
    "RouterDecision",
    "Sidecar",
    "SidecarMode",
    "SidecarResponse",
]
