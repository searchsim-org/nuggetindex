"""OpenRouter structured-output client.

OpenRouter exposes an OpenAI-compatible API, so this is a thin wrapper over
the OpenAI SDK with a fixed default ``base_url``. The ``openai`` SDK from
the ``[openai]`` extra is required.
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

_DEFAULT_OPENROUTER_URL = "https://openrouter.ai/api/v1"


def _require_openrouter_sdk() -> tuple[Any, Any]:
    """Import ``instructor`` and ``openai.AsyncOpenAI`` or raise."""
    try:
        import instructor
        from openai import AsyncOpenAI
    except ImportError as e:  # pragma: no cover - exercised via stub in tests
        raise ImportError(
            "nuggetindex[openai] not installed. Run: pip install nuggetindex[openai]"
        ) from e
    return instructor, AsyncOpenAI


class OpenRouterClient:
    """Structured-output chat client for any model hosted on OpenRouter."""

    def __init__(self, cfg: LLMConfig) -> None:
        instructor, AsyncOpenAI = _require_openrouter_sdk()
        self.cfg = cfg
        base_url = cfg.base_url or _DEFAULT_OPENROUTER_URL
        self._raw_client = instructor.from_openai(
            AsyncOpenAI(
                api_key=unwrap_api_key(cfg.api_key),
                base_url=base_url,
                timeout=cfg.timeout_seconds,
            )
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
