"""OpenAI structured-output client.

Uses ``instructor`` to coerce chat completions into a given Pydantic model.
Retries network-level failures up to three times with exponential backoff.

The ``openai`` and ``instructor`` SDKs live in the ``[openai]`` extra, so
they are imported lazily inside ``_require_openai_sdk`` -- users who install
``nuggetindex`` without the extra can still ``import
nuggetindex.extractors.clients`` without ``ImportError``; they only hit the
missing-dep error when they try to build the client.
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


def _require_openai_sdk() -> tuple[Any, Any]:
    """Import ``instructor`` and ``openai.AsyncOpenAI`` or raise a friendly error."""
    try:
        import instructor
        from openai import AsyncOpenAI
    except ImportError as e:  # pragma: no cover - exercised via stub in tests
        raise ImportError(
            "nuggetindex[openai] not installed. Run: pip install nuggetindex[openai]"
        ) from e
    return instructor, AsyncOpenAI


class OpenAIClient:
    """Structured-output chat client backed by OpenAI + instructor."""

    def __init__(self, cfg: LLMConfig) -> None:
        instructor, AsyncOpenAI = _require_openai_sdk()
        self.cfg = cfg
        self._raw_client = instructor.from_openai(
            AsyncOpenAI(
                api_key=unwrap_api_key(cfg.api_key),
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
