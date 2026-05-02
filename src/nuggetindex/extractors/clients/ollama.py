"""Ollama structured-output client.

Ollama speaks the OpenAI chat-completions protocol, so we reuse the OpenAI
SDK pointed at the local Ollama HTTP endpoint. The ``[ollama]`` extra
guards against accidentally requiring Ollama at runtime when it is not
wanted. The ``openai`` SDK (in the ``[openai]`` extra) is also needed.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from nuggetindex.extractors.clients.base import LLMConfig, unwrap_api_key

_DEFAULT_OLLAMA_URL = "http://localhost:11434/v1"


def _require_ollama_sdk() -> tuple[Any, Any]:
    """Import ``instructor`` and ``openai.AsyncOpenAI`` (Ollama-compatible)."""
    try:
        import instructor
        import ollama  # noqa: F401  -- presence check only
        from openai import AsyncOpenAI
    except ImportError as e:  # pragma: no cover - exercised via stub in tests
        raise ImportError(
            "nuggetindex[ollama] and nuggetindex[openai] are required. "
            "Run: pip install nuggetindex[ollama,openai]"
        ) from e
    return instructor, AsyncOpenAI


class OllamaClient:
    """Structured-output chat client talking to a local Ollama server."""

    def __init__(self, cfg: LLMConfig) -> None:
        instructor, AsyncOpenAI = _require_ollama_sdk()
        self.cfg = cfg
        base_url = cfg.base_url or _DEFAULT_OLLAMA_URL
        self._raw_client = instructor.from_openai(
            AsyncOpenAI(
                api_key=unwrap_api_key(cfg.api_key) or "ollama",
                base_url=base_url,
                timeout=cfg.timeout_seconds,
            ),
            mode=instructor.Mode.JSON,
        )

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type((ConnectionError, TimeoutError)),
    )
    async def achat_structured(
        self,
        messages: list[dict[str, Any]],
        response_model: type[BaseModel],
    ) -> BaseModel:
        result: BaseModel = await self._raw_client.chat.completions.create(
            model=self.cfg.model,
            messages=messages,
            response_model=response_model,
            temperature=self.cfg.temperature,
            max_tokens=self.cfg.max_tokens,
        )
        return result
