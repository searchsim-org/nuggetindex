import httpx
import pytest

from nuggetindex.adapters import CorpusSource, ProxyPool, WebSearchCorpus


def test_web_search_corpus_is_corpus_source():
    corpus = WebSearchCorpus(searxng_url="http://localhost:8888")
    assert isinstance(corpus, CorpusSource)


@pytest.mark.asyncio
async def test_random_ids_raises():
    corpus = WebSearchCorpus(searxng_url="http://localhost:8888")
    with pytest.raises(NotImplementedError):
        await corpus.sample(mode="random_ids", n=5)


@pytest.mark.asyncio
async def test_searxng_happy_path_returns_documents():
    def handler(request):
        return httpx.Response(
            200,
            json={
                "results": [
                    {"url": "https://ex/a", "title": "t", "content": "c"},
                    {"url": "https://ex/b", "title": "t2", "content": "c2"},
                ],
            },
        )

    corpus = WebSearchCorpus(
        searxng_url="http://localhost:8888",
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )
    docs = await corpus.search("Apple CEO", limit=10)
    assert len(docs) == 2
    assert docs[0].source_id == "https://ex/a"


@pytest.mark.asyncio
async def test_captcha_engages_camoufox_fallback(monkeypatch):
    def handler(request):
        return httpx.Response(429)

    from nuggetindex.adapters.searxng import camoufox_backend as cb

    class _FakePage:
        async def goto(self, *a, **kw):
            return None

        async def content(self):
            return (
                '<html><body><div class="g">'
                '<a href="https://cf/a"><h3>t</h3></a>'
                '<div class="VwiC3b">c</div></div></body></html>'
            )

        async def close(self):
            return None

    class _FakeBrowser:
        async def new_page(self):
            return _FakePage()

        async def close(self):
            return None

    class _FakeCtx:
        async def __aenter__(self):
            return _FakeBrowser()

        async def __aexit__(self, *exc):
            return False

    monkeypatch.setattr(cb, "AsyncCamoufox", lambda **kw: _FakeCtx())

    corpus = WebSearchCorpus(
        searxng_url="http://localhost:8888",
        backend="searxng+camoufox",
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )
    docs = await corpus.search("Apple CEO", limit=5)
    assert len(docs) == 1
    assert docs[0].source_id == "https://cf/a"


@pytest.mark.asyncio
async def test_searxng_only_mode_does_not_engage_camoufox_on_captcha():
    def handler(request):
        return httpx.Response(429)

    corpus = WebSearchCorpus(
        searxng_url="http://localhost:8888",
        backend="searxng",
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )
    docs = await corpus.search("Apple CEO", limit=5)
    assert docs == []


@pytest.mark.asyncio
async def test_proxy_pool_is_threaded_through():
    pool = ProxyPool(proxies=["http://p1:1"])
    corpus = WebSearchCorpus(
        searxng_url="http://localhost:8888",
        proxy_pool=pool,
    )
    assert corpus._client.proxy_pool is pool
