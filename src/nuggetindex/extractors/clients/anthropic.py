"""Anthropic structured-output client.

Uses ``instructor.from_anthropic`` on top of ``anthropic.AsyncAnthropic``.
The SDK is part of the optional ``[anthropic]`` extra, so the import
happens lazily inside ``_require_anthropic_sdk``.
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


def _require_anthropic_sdk() -> tuple[Any, Any]:
    """Import ``instructor`` and ``anthropic.AsyncAnthropic`` or raise."""
    try:
        import instructor
        from anthropic import AsyncAnthropic
    except ImportError as e:  # pragma: no cover - exercised via stub in tests
        raise ImportError(
            "nuggetindex[anthropic] not installed. Run: pip install nuggetindex[anthropic]"
        ) from e
    return instructor, AsyncAnthropic


class AnthropicClient:
    """Structured-output chat client backed by Anthropic + instructor."""

    def __init__(self, cfg: LLMConfig) -> None:
        instructor, AsyncAnthropic = _require_anthropic_sdk()
        self.cfg = cfg
        self._raw_client = instructor.from_anthropic(
            AsyncAnthropic(
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
        # Anthropic's API splits system prompts from the message list.
        system_prompt = ""
        user_messages: list[dict[str, Any]] = []
        for m in messages:
            if m.get("role") == "system":
                system_prompt += m.get("content", "")
            else:
                user_messages.append(m)

        result: BaseModel = await self._raw_client.messages.create(
            model=self.cfg.model,
            messages=user_messages,
            system=system_prompt or None,
            response_model=response_model,
            temperature=self.cfg.temperature,
            max_tokens=self.cfg.max_tokens,
        )
        return result
