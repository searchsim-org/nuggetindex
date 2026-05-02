from datetime import timedelta

import httpx
import pytest

from nuggetindex.adapters.searxng.client import SearxngClient, SearxngResponse
from nuggetindex.adapters.searxng.proxy import ProxyPool


@pytest.mark.asyncio
async def test_clean_response_yields_results():
    def handler(request):
        params = dict(request.url.params)
        assert params.get("q") == "Apple CEO"
        return httpx.Response(200, json={
            "results": [{"url": "https://ex", "title": "t", "content": "c"}],
        })
    client = SearxngClient(
        base_url="http://searxng.local",
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )
    response = await client.search("Apple CEO", limit=5)
    assert isinstance(response, SearxngResponse)
    assert len(response.results) == 1
    assert response.was_captcha is False


@pytest.mark.asyncio
async def test_rate_limit_triggers_proxy_rotation():
    call_log = []

    def handler(request):
        call_log.append(dict(request.url.params))
        if len(call_log) < 3:
            return httpx.Response(429, text="Too Many Requests")
        return httpx.Response(200, json={
            "results": [{"url": "https://ex", "title": "t", "content": "c"}],
        })

    pool = ProxyPool(
        proxies=["http://p1:1", "http://p2:2", "http://p3:3"],
        quarantine_duration=timedelta(minutes=5),
    )
    client = SearxngClient(
        base_url="http://searxng.local",
        proxy_pool=pool,
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )
    response = await client.search("Apple", limit=3)
    assert len(call_log) == 3
    assert response.was_captcha is False
    # 2 proxies burned, 1 still active
    assert pool.active_count() == 1


@pytest.mark.asyncio
async def test_captcha_after_all_proxies_returns_was_captcha_true():
    def handler(request):
        return httpx.Response(429, text="Too Many Requests")
    pool = ProxyPool(proxies=["http://p1:1"])
    client = SearxngClient(
        base_url="http://searxng.local",
        proxy_pool=pool,
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )
    response = await client.search("Apple", limit=3)
    assert response.was_captcha is True
    assert response.captcha_category == "rate_limit"
    assert response.results == []


@pytest.mark.asyncio
async def test_cloudflare_challenge_classified():
    def handler(request):
        return httpx.Response(503, text='<html>cf-browser-verification</html>')
    client = SearxngClient(
        base_url="http://searxng.local",
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )
    response = await client.search("Apple", limit=3)
    assert response.was_captcha is True
    assert response.captcha_category == "cloudflare"
