"""LangChain integration package.

Importing this module triggers the ``[langchain]`` extras check in each of
the submodules' module-level ``_require_langchain()`` call. Keeping the
guard at the leaf submodule level (and NOT in
``nuggetindex.integrations.__init__``) preserves the invariant that merely
importing ``nuggetindex.integrations`` never raises when optional framework
extras are missing.
"""
from nuggetindex.integrations.langchain.chain_retriever import NuggetChainRetriever
from nuggetindex.integrations.langchain.doctor import doctor
from nuggetindex.integrations.langchain.loader import NuggetConstructionLoader
from nuggetindex.integrations.langchain.postprocessor import GovernanceFilter
from nuggetindex.integrations.langchain.retriever import NuggetIndexRetriever
from nuggetindex.integrations.langchain.sidecar import NuggetSidecarRunnable

__all__ = [
    "GovernanceFilter",
    "NuggetChainRetriever",
    "NuggetConstructionLoader",
    "NuggetIndexRetriever",
    "NuggetSidecarRunnable",
    "doctor",
]
