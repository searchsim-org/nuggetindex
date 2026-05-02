"""Haystack 2.x integration package.

Importing this module transitively triggers each submodule's module-level
``_require_haystack()`` call. Keeping the guard at the leaf submodule level
(and NOT in ``nuggetindex.integrations.__init__``) preserves the invariant
that merely importing ``nuggetindex.integrations`` never raises when
optional framework extras are missing.

v0.2 adds :class:`NuggetDocumentStore` — a full Haystack 2.x
``DocumentStore`` implementation backed by a :class:`NuggetStore`. v0.1
intentionally deferred this component; the filter DSL and base-protocol
surface area made sense once the SQL allowlist was established during v0.1
review.
"""
from nuggetindex.integrations.haystack.chain_retriever import NuggetChainRetriever
from nuggetindex.integrations.haystack.constructor import NuggetConstructor
from nuggetindex.integrations.haystack.doctor import doctor
from nuggetindex.integrations.haystack.document_store import NuggetDocumentStore
from nuggetindex.integrations.haystack.retriever import NuggetIndexRetriever
from nuggetindex.integrations.haystack.sidecar import NuggetSidecarComponent

__all__ = [
    "NuggetChainRetriever",
    "NuggetConstructor",
    "NuggetDocumentStore",
    "NuggetIndexRetriever",
    "NuggetSidecarComponent",
    "doctor",
]
