"""Temporal provenance chains over :class:`NuggetStore`.

Three chain kinds are exposed via :class:`nuggetindex.NuggetStore`:

* :meth:`NuggetStore.achain_succession` -- ordered history for one key.
* :meth:`NuggetStore.achain_rename` -- walk renaming-predicate edges.
* :meth:`NuggetStore.achain_join` -- bounded 1--3 hop functional joins.

All three return a :class:`NuggetChain`. Ambiguous steps raise
:class:`ChainAmbiguousError` on the default (no-LLM) path; pass an opt-in
:class:`ChainResolver` to delegate disambiguation to an LLM.
"""

from nuggetindex.chains.models import ChainEdge, ChainEdgeType, NuggetChain
from nuggetindex.chains.resolver import ChainResolution, ChainResolver

__all__ = [
    "ChainEdge",
    "ChainEdgeType",
    "ChainResolution",
    "ChainResolver",
    "NuggetChain",
]
