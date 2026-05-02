"""Corpus adapters for :func:`nuggetindex.auto.auto`'s unbiased bootstrap.

Exports:

* :class:`CorpusSource`      -- the protocol ``auto()`` consumes.
* :class:`JsonlCorpus`       -- flat-file source.
* :class:`VespaCorpus`       -- Vespa-style BM25 REST API adapter.
* :class:`HaystackCorpus`    -- Haystack 2.x DocumentStore adapter.
* :class:`LlamaIndexCorpus`  -- LlamaIndex VectorStoreIndex / node-iterable
  adapter.
* :class:`WebSearchCorpus`   -- SearXNG (+ optional Camoufox) web-search
  corpus, with :class:`ProxyPool` rotation and :class:`CaptchaDetector`
  escalation.

All adapters satisfy :class:`CorpusSource` (async ``sample`` / ``search``)
and lazy-import their heavy dependencies so the core package stays importable
without any of the framework extras installed.
"""

from __future__ import annotations

from nuggetindex.adapters.base import CorpusSource
from nuggetindex.adapters.elasticsearch import ElasticsearchCorpus
from nuggetindex.adapters.haystack_corpus import HaystackCorpus
from nuggetindex.adapters.jsonl import JsonlCorpus
from nuggetindex.adapters.llamaindex_corpus import LlamaIndexCorpus
from nuggetindex.adapters.opensearch import OpenSearchCorpus
from nuggetindex.adapters.pinecone import PineconeCorpus
from nuggetindex.adapters.qdrant import QdrantCorpus
from nuggetindex.adapters.searxng import (
    CaptchaDetector,
    ProxyEntry,
    ProxyPool,
)
from nuggetindex.adapters.vespa import VespaCorpus
from nuggetindex.adapters.web_search import WebSearchCorpus

__all__ = [
    "CaptchaDetector",
    "CorpusSource",
    "ElasticsearchCorpus",
    "HaystackCorpus",
    "JsonlCorpus",
    "LlamaIndexCorpus",
    "OpenSearchCorpus",
    "PineconeCorpus",
    "ProxyEntry",
    "ProxyPool",
    "QdrantCorpus",
    "VespaCorpus",
    "WebSearchCorpus",
]
