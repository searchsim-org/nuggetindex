"""Tests for ``LLMConfig.api_key`` SecretStr coercion + redacting repr."""

from __future__ import annotations

from pydantic import SecretStr

from nuggetindex.extractors.clients.base import LLMConfig


def test_string_api_key_stored_as_secret() -> None:
    cfg = LLMConfig(provider="openai", model="gpt-4o-mini", api_key="sk-test123")
    assert isinstance(cfg.api_key, SecretStr)
    assert cfg.api_key.get_secret_value() == "sk-test123"


def test_repr_redacts_api_key() -> None:
    cfg = LLMConfig(
        provider="openai", model="gpt-4o-mini", api_key="sk-secret-deadbeef"
    )
    r = repr(cfg)
    assert "sk-secret-deadbeef" not in r
    assert "redacted" in r.lower() or "****" in r


def test_none_api_key_stays_none() -> None:
    cfg = LLMConfig(provider="openai", model="gpt-4o-mini", api_key=None)
    assert cfg.api_key is None
    assert "redacted" not in repr(cfg).lower()


def test_secret_str_input_passthrough() -> None:
    ss = SecretStr("pre-wrapped")
    cfg = LLMConfig(provider="openai", model="gpt-4o-mini", api_key=ss)
    assert cfg.api_key is ss or cfg.api_key.get_secret_value() == "pre-wrapped"
