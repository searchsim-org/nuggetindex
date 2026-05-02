"""``asdict_redacted(cfg)`` — safe alternative to ``dataclasses.asdict`` for
``LLMConfig`` logging. Unlike ``asdict``, the returned dict carries the
string ``"<redacted>"`` in place of the ``SecretStr`` api_key so calling
code can't accidentally ``get_secret_value()`` it later. (findings-A6)
"""
from __future__ import annotations

from nuggetindex.extractors.clients.base import LLMConfig, asdict_redacted


def test_asdict_redacted_hides_api_key():
    cfg = LLMConfig(provider="openai", model="gpt-4o-mini", api_key="sk-secret")
    d = asdict_redacted(cfg)
    assert d["api_key"] == "<redacted>"
    assert d["model"] == "gpt-4o-mini"
    assert d["provider"] == "openai"


def test_asdict_redacted_with_none_api_key():
    cfg = LLMConfig(provider="openai", model="gpt-4o-mini", api_key=None)
    d = asdict_redacted(cfg)
    assert d["api_key"] is None
