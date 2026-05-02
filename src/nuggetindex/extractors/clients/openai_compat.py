"""Generic OpenAI-compatible structured-output client.

For self-hosted gateways, vLLM, LM Studio, or any other service that
exposes the OpenAI chat-completions protocol. ``base_url`` is required.
The ``openai`` SDK from the ``[openai]`` extra is required.
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


def _require_openai_compat_sdk() -> tuple[Any, Any]:
    """Import ``instructor`` and ``openai.AsyncOpenAI`` or raise."""
    try:
        import instructor
        from openai import AsyncOpenAI
    except ImportError as e:  # pragma: no cover - exercised via stub in tests
        raise ImportError(
            "nuggetindex[openai] not installed. Run: pip install nuggetindex[openai]"
        ) from e
    return instructor, AsyncOpenAI


class OpenAICompatClient:
    """Structured-output chat client pointed at any OpenAI-compatible URL."""

    def __init__(self, cfg: LLMConfig) -> None:
        if not cfg.base_url:
            raise ValueError("openai_compat provider requires an explicit base_url")
        instructor, AsyncOpenAI = _require_openai_compat_sdk()
        self.cfg = cfg
        self._raw_client = instructor.from_openai(
            AsyncOpenAI(
                api_key=unwrap_api_key(cfg.api_key) or "not-needed",
                base_url=cfg.base_url,
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
