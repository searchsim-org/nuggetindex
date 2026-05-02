"""Tests for LLMConfig, the LLMClient protocol, and build_client dispatch."""

from __future__ import annotations

import pytest

from nuggetindex.extractors.clients.base import LLMClient, LLMConfig, build_client


def test_llm_config_requires_provider_and_model() -> None:
    cfg = LLMConfig(provider="openai", model="gpt-4o-mini")
    assert cfg.provider == "openai"
    assert cfg.model == "gpt-4o-mini"


def test_llm_config_defaults() -> None:
    cfg = LLMConfig(provider="openai", model="gpt-4o-mini")
    assert cfg.temperature == 0.0
    assert cfg.max_tokens == 2048
    assert cfg.timeout_seconds == 30.0
    assert cfg.extra == {}
    assert cfg.api_key is None
    assert cfg.base_url is None


def test_llm_config_extra_is_per_instance() -> None:
    a = LLMConfig(provider="openai", model="m1")
    b = LLMConfig(provider="openai", model="m2")
    a.extra["k"] = "v"
    assert b.extra == {}


def test_build_client_unknown_provider_raises() -> None:
    with pytest.raises(ValueError, match="unknown provider"):
        build_client(LLMConfig(provider="nope", model="x"))


def test_llm_client_is_a_runtime_protocol() -> None:
    # A minimal structural implementation should satisfy the protocol.
    class _Dummy:
        async def achat_structured(self, messages, response_model):  # type: ignore[no-untyped-def]
            return response_model()

    assert isinstance(_Dummy(), LLMClient)


def test_build_client_openai_dispatches(monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("openai")
    cfg = LLMConfig(provider="openai", model="gpt-4o-mini", api_key="sk-test")
    client = build_client(cfg)
    # Class name and module should match the openai client.
    assert type(client).__name__ == "OpenAIClient"
