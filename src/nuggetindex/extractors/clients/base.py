"""LLMConfig dataclass + LLMClient protocol + build_client factory.

Each concrete provider client implements the `LLMClient` protocol: one
method, `achat_structured`, which takes a list of chat messages and a
Pydantic response model and returns an instance of that model.

`build_client` dispatches to the correct concrete client based on
`cfg.provider`. Provider modules are imported lazily inside each branch
so installing nuggetindex without (e.g.) the ``[anthropic]`` extra does
not break ``from nuggetindex.extractors.clients import build_client``.
"""

from __future__ import annotations

from dataclasses import asdict as _asdict
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel, SecretStr

__all__ = [
    "LLMConfig",
    "LLMClient",
    "asdict_redacted",
    "build_client",
    "unwrap_api_key",
]


@dataclass
class LLMConfig:
    """Provider-agnostic config for a structured-output LLM call.

    ``api_key`` accepts either a raw ``str`` or a Pydantic ``SecretStr``.
    Raw strings are wrapped in ``SecretStr`` during ``__post_init__`` so
    accidental logging via ``repr()`` / ``print()`` never leaks the value.
    Concrete client adapters unwrap the secret via ``get_secret_value()``
    immediately before handing it to the underlying SDK.
    """

    provider: str  # one of: openai | anthropic | google | ollama | openrouter | openai_compat
    model: str
    api_key: SecretStr | str | None = None
    base_url: str | None = None
    temperature: float = 0.0
    max_tokens: int = 2048
    timeout_seconds: float = 30.0
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # Wrap raw strings in SecretStr so repr()/str() never leaks them.
        # SecretStr inputs pass through untouched; None stays None.
        if isinstance(self.api_key, str):
            self.api_key = SecretStr(self.api_key)

    def __repr__(self) -> str:
        redacted: str | None = "<redacted>" if self.api_key is not None else None
        return (
            f"LLMConfig(provider={self.provider!r}, model={self.model!r}, "
            f"api_key={redacted}, base_url={self.base_url!r}, "
            f"temperature={self.temperature}, max_tokens={self.max_tokens}, "
            f"timeout_seconds={self.timeout_seconds}, extra={self.extra!r})"
        )


@runtime_checkable
class LLMClient(Protocol):
    """Provider-agnostic structured-output LLM client."""

    async def achat_structured(
        self,
        messages: list[dict[str, Any]],
        response_model: type[BaseModel],
    ) -> BaseModel: ...


def asdict_redacted(cfg: LLMConfig) -> dict[str, Any]:
    """Like :func:`dataclasses.asdict` but redacts ``api_key``.

    ``dataclasses.asdict(LLMConfig(api_key="sk-..."))`` returns the
    underlying :class:`~pydantic.SecretStr` instance unchanged, so anything
    that subsequently calls ``get_secret_value()`` on the returned dict
    leaks the raw key. Use this helper when logging or serialising an
    :class:`LLMConfig`: the ``api_key`` field is replaced with the literal
    string ``"<redacted>"`` when set, and left as ``None`` when unset.
    (findings-A6)
    """
    d = _asdict(cfg)
    if d.get("api_key") is not None:
        d["api_key"] = "<redacted>"
    return d


def unwrap_api_key(api_key: SecretStr | str | None) -> str | None:
    """Return the raw API-key string, unwrapping ``SecretStr`` if needed.

    Concrete client adapters call this immediately before passing the value
    to an underlying SDK. Keeping the unwrap at the SDK boundary minimises
    the number of places a raw key lives in memory.
    """
    if api_key is None:
        return None
    if isinstance(api_key, SecretStr):
        return api_key.get_secret_value()
    return api_key


def build_client(cfg: LLMConfig) -> LLMClient:
    """Return the concrete client matching ``cfg.provider``.

    Provider SDKs are imported lazily inside each branch so this factory
    itself has no hard dependencies on any provider package.
    """
    if cfg.provider == "openai":
        from nuggetindex.extractors.clients.openai import OpenAIClient

        return OpenAIClient(cfg)
    if cfg.provider == "anthropic":
        from nuggetindex.extractors.clients.anthropic import AnthropicClient

        return AnthropicClient(cfg)
    if cfg.provider == "google":
        from nuggetindex.extractors.clients.google import GoogleClient

        return GoogleClient(cfg)
    if cfg.provider == "ollama":
        from nuggetindex.extractors.clients.ollama import OllamaClient

        return OllamaClient(cfg)
    if cfg.provider == "openrouter":
        from nuggetindex.extractors.clients.openrouter import OpenRouterClient

        return OpenRouterClient(cfg)
    if cfg.provider == "openai_compat":
        from nuggetindex.extractors.clients.openai_compat import OpenAICompatClient

        return OpenAICompatClient(cfg)
    raise ValueError(f"unknown provider: {cfg.provider}")
