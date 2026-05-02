import pytest

from nuggetindex.adapters.searxng.camoufox_backend import CamoufoxBackend
from nuggetindex.adapters.searxng.proxy import ProxyPool


class _FakePage:
    def __init__(self, html: str):
        self._html = html

    async def goto(self, url, **kw):
        return None

    async def content(self):
        return self._html

    async def close(self):
        return None


class _FakeBrowser:
    def __init__(self, html: str):
        self._html = html

    async def new_page(self):
        return _FakePage(self._html)

    async def close(self):
        return None


class _FakeCamoufoxCtx:
    def __init__(self, html: str, captured: dict | None = None):
        self._html = html
        self._captured = captured

    async def __aenter__(self):
        return _FakeBrowser(self._html)

    async def __aexit__(self, *exc):
        return False


@pytest.mark.asyncio
async def test_camoufox_backend_scrapes_google_results(monkeypatch):
    html = """
    <html><body>
      <div class="g">
        <a href="https://example.com/a"><h3>Apple names new CEO</h3></a>
        <div class="VwiC3b">Apple announced today...</div>
      </div>
      <div class="g">
        <a href="https://example.com/b"><h3>Tim Cook era ends</h3></a>
        <div class="VwiC3b">After 15 years...</div>
      </div>
    </body></html>
    """
    captured: dict = {}

    def fake_camoufox(**kwargs):
        captured.update(kwargs)
        return _FakeCamoufoxCtx(html)

    monkeypatch.setattr(
        "nuggetindex.adapters.searxng.camoufox_backend.AsyncCamoufox",
        fake_camoufox,
    )
    backend = CamoufoxBackend(engine="google")
    results = await backend.search("Apple CEO", limit=5)
    assert len(results) == 2
    assert results[0]["url"] == "https://example.com/a"
    assert "new CEO" in results[0]["title"]


@pytest.mark.asyncio
async def test_camoufox_backend_uses_proxy_from_pool(monkeypatch):
    captured: dict = {}

    def fake_camoufox(**kwargs):
        captured.update(kwargs)
        return _FakeCamoufoxCtx("<html></html>")

    monkeypatch.setattr(
        "nuggetindex.adapters.searxng.camoufox_backend.AsyncCamoufox",
        fake_camoufox,
    )
    pool = ProxyPool(proxies=["http://user:pass@1.2.3.4:8080"])
    backend = CamoufoxBackend(engine="google", proxy_pool=pool)
    await backend.search("q", limit=1)
    assert captured["proxy"]["server"] == "http://user:pass@1.2.3.4:8080"


@pytest.mark.asyncio
async def test_camoufox_backend_rotates_proxy_on_captcha(monkeypatch):
    call_count = {"n": 0}
    clean_html = (
        '<html><body><div class="g">'
        '<a href="https://ex/x"><h3>t</h3></a>'
        '<div class="VwiC3b">c</div></div></body></html>'
    )
    sorry_html = (
        "<html><body>Our systems have detected unusual traffic from your network."
        "</body></html>"
    )

    def fake_camoufox(**kwargs):
        call_count["n"] += 1
        return _FakeCamoufoxCtx(sorry_html if call_count["n"] == 1 else clean_html)

    monkeypatch.setattr(
        "nuggetindex.adapters.searxng.camoufox_backend.AsyncCamoufox",
        fake_camoufox,
    )
    pool = ProxyPool(proxies=["http://p1:1", "http://p2:2"])
    backend = CamoufoxBackend(engine="google", proxy_pool=pool)
    results = await backend.search("q", limit=5)
    assert len(results) == 1
    assert call_count["n"] == 2
    assert pool.active_count() == 1  # first proxy quarantined


def test_camoufox_backend_unsupported_engine_raises():
    with pytest.raises(ValueError, match="unsupported"):
        CamoufoxBackend(engine="altavista")
