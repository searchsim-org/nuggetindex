from datetime import timedelta

import pytest

from nuggetindex.adapters.searxng.proxy import ProxyPool


def test_empty_pool_yields_none():
    pool = ProxyPool(proxies=[])
    assert pool.next() is None


def test_round_robin_rotation():
    pool = ProxyPool(proxies=["http://a:1", "http://b:2", "http://c:3"])
    assert pool.next().url == "http://a:1"
    assert pool.next().url == "http://b:2"
    assert pool.next().url == "http://c:3"
    assert pool.next().url == "http://a:1"


def test_quarantined_proxies_are_skipped():
    pool = ProxyPool(proxies=["http://a:1", "http://b:2"], quarantine_duration=timedelta(minutes=5))
    a = pool.next()
    pool.mark_failed(a)
    b = pool.next()
    assert b.url == "http://b:2"
    assert pool.next().url == "http://b:2"


def test_all_proxies_quarantined_raises():
    pool = ProxyPool(proxies=["http://a:1"])
    a = pool.next()
    pool.mark_failed(a)
    with pytest.raises(RuntimeError, match="exhausted"):
        pool.next()


def test_health_recovers_after_quarantine_expiry():
    import time

    pool = ProxyPool(proxies=["http://a:1"], quarantine_duration=timedelta(microseconds=1))
    a = pool.next()
    pool.mark_failed(a)
    time.sleep(0.01)
    assert pool.next() is not None


def test_mark_success_clears_consecutive_failures():
    pool = ProxyPool(proxies=["http://a:1"])
    a = pool.next()
    pool.mark_failed(a)
    pool.mark_success(a)
    assert pool.next().url == "http://a:1"
