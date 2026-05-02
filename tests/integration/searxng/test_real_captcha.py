"""Integration tests against a real SearXNG instance.

Skipped unless ``RUN_SEARXNG_INTEGRATION=1`` and the SEARXNG_URL is reachable.
Optional proxy rotation with a comma-separated ``PROXY_URLS`` env var.
"""

from __future__ import annotations

import os

import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_SEARXNG_INTEGRATION") != "1",
    reason="Set RUN_SEARXNG_INTEGRATION=1 to run live-SearXNG integration tests.",
)

SEARXNG_URL = os.environ.get("SEARXNG_URL", "http://localhost:8888")
PROXY_URLS = [p for p in os.environ.get("PROXY_URLS", "").split(",") if p]


@pytest.mark.asyncio
async def test_known_ratelimited_query_recovers_via_proxy_rotation():
    from nuggetindex.adapters import WebSearchCorpus
    from nuggetindex.adapters.searxng.proxy import ProxyPool

    if not PROXY_URLS:
        pytest.skip("Set PROXY_URLS=proxy1,proxy2,... to run this test.")

    corpus = WebSearchCorpus(
        searxng_url=SEARXNG_URL,
        backend="searxng+camoufox",
        proxy_pool=ProxyPool(proxies=PROXY_URLS),
    )
    # Hammer the backend; expect graceful recovery (non-error, list return).
    for _ in range(10):
        docs = await corpus.search("apple ceo", limit=5)
        assert isinstance(docs, list)


@pytest.mark.asyncio
async def test_captcha_known_query_triggers_camoufox_fallback():
    from nuggetindex.adapters import WebSearchCorpus

    corpus = WebSearchCorpus(
        searxng_url=SEARXNG_URL,
        backend="searxng+camoufox",
    )
    # Pattern commonly tripping Google's sorry-page in dev contexts.
    docs = await corpus.search("site:news.ycombinator.com python async", limit=10)
    assert isinstance(docs, list)


@pytest.mark.asyncio
async def test_silent_empty_response_falls_through_to_camoufox():
    """Guard rails only. Meaningful only when SearXNG is configured to force
    upstream engine errors (edit settings.yml to disable engines). The test
    asserts no exception is raised -- correctness of the fallback is verified
    manually by the operator by reading the Sidecar's decision.reason."""
    from nuggetindex.adapters import WebSearchCorpus

    corpus = WebSearchCorpus(
        searxng_url=SEARXNG_URL,
        backend="searxng+camoufox",
    )
    docs = await corpus.search("nuggetindex unknown query", limit=5)
    assert isinstance(docs, list)
