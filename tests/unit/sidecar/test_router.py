from datetime import UTC, datetime

import pytest

from nuggetindex.sidecar.router import Router, RouterDecision


@pytest.fixture
def router() -> Router:
    return Router()


@pytest.fixture
def now() -> datetime:
    return datetime(2026, 4, 19, tzinfo=UTC)


def test_router_passthrough_on_noise(router, now):
    d = router.classify("hello world", now=now)
    assert d.use_nugget is False
    assert d.query_time is None


def test_router_temporal_year(router, now):
    d = router.classify("who was Google's CEO in 2013?", now=now)
    assert d.use_nugget is True
    assert d.query_time is not None
    assert d.query_time.year == 2013


def test_router_temporal_as_of(router, now):
    d = router.classify("list Google's CEO as of today", now=now)
    assert d.use_nugget is True
    assert d.query_time is not None


def test_router_functional_keyword_triggers_even_without_time(router, now):
    d = router.classify("what is the CEO of Microsoft?", now=now)
    assert d.use_nugget is True


def test_router_reason_string_nonempty_when_triggered(router, now):
    d = router.classify("who was Apple's CEO in 2011?", now=now)
    assert d.reason != ""


def test_router_llm_fallback_defers_when_cheap_path_fails(now):
    def classifier(q, n):
        return RouterDecision(use_nugget=True, reason="llm_said_yes")

    router = Router(llm_classifier=classifier)
    d = router.classify("the sky is blue", now=now)
    assert d.use_nugget is True
    assert "llm" in d.reason.lower()


def test_router_before_extracts_prior_year(router, now):
    d = router.classify("who led Google before 2015?", now=now)
    assert d.use_nugget is True
    assert d.query_time is not None
    assert d.query_time.year == 2014


def test_router_llm_fallback_not_consulted_when_cheap_path_wins(now):
    calls: list[tuple[str, datetime]] = []

    def classifier(q, n):
        calls.append((q, n))
        return RouterDecision(use_nugget=False, reason="should_not_run")

    router = Router(llm_classifier=classifier)
    d = router.classify("who was Apple's CEO in 2011?", now=now)
    assert d.use_nugget is True
    assert calls == []  # cheap path already decided


def test_router_llm_fallback_passthrough_when_classifier_returns_garbage(now):
    def classifier(q, n):
        return "not a decision"

    router = Router(llm_classifier=classifier)
    d = router.classify("the sky is blue", now=now)
    assert d.use_nugget is False
    assert "unknown" in d.reason.lower()
