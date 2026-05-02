"""LlamaIndex integration package.

Importing this module triggers the ``[llamaindex]`` extras check in each of
the submodules' module-level ``_require_llamaindex()`` call. Keeping the
guard at the leaf submodule level (and NOT in
``nuggetindex.integrations.__init__``) preserves the invariant that merely
importing ``nuggetindex.integrations`` never raises when optional framework
extras are missing.
"""
from nuggetindex.integrations.llamaindex.chain_retriever import NuggetChainRetriever
from nuggetindex.integrations.llamaindex.doctor import doctor
from nuggetindex.integrations.llamaindex.ingestion import NuggetTransformation
from nuggetindex.integrations.llamaindex.postprocessor import GovernancePostProcessor
from nuggetindex.integrations.llamaindex.retriever import NuggetIndexRetriever
from nuggetindex.integrations.llamaindex.sidecar import NuggetSidecarNodePostprocessor

__all__ = [
    "GovernancePostProcessor",
    "NuggetChainRetriever",
    "NuggetIndexRetriever",
    "NuggetSidecarNodePostprocessor",
    "NuggetTransformation",
    "doctor",
]
