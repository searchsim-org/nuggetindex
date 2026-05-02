"""Google Gemini structured-output client.

Uses ``instructor.from_gemini`` (falls back to ``from_genai`` on older
instructor releases) over ``google.generativeai``. The SDK is part of the
optional ``[google]`` extra.
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


def _require_google_sdk() -> tuple[Any, Any]:
    """Import ``instructor`` and ``google.generativeai`` or raise."""
    try:
        import google.generativeai as genai
        import instructor
    except ImportError as e:  # pragma: no cover - exercised via stub in tests
        raise ImportError(
            "nuggetindex[google] not installed. "
            "Run: pip install nuggetindex[google]"
        ) from e
    return instructor, genai


class GoogleClient:
    """Structured-output chat client backed by Gemini + instructor."""

    def __init__(self, cfg: LLMConfig) -> None:
        instructor, genai = _require_google_sdk()
        self.cfg = cfg
        api_key_str = unwrap_api_key(cfg.api_key)
        if api_key_str:
            genai.configure(api_key=api_key_str)
        native_model = genai.GenerativeModel(model_name=cfg.model)
        # Use whichever structured-output helper is available on the installed
        # instructor version.
        factory = getattr(
            instructor, "from_gemini", getattr(instructor, "from_genai", None)
        )
        if factory is None:  # pragma: no cover - guarded by import above
            raise ImportError(
                "Installed instructor version has no Gemini/GenAI adapter. "
                "Upgrade instructor."
            )
        self._raw_client = factory(client=native_model, use_async=True)

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
        result: BaseModel = await self._raw_client.messages.create(
            messages=messages,
            response_model=response_model,
            generation_config={
                "temperature": self.cfg.temperature,
                "max_output_tokens": self.cfg.max_tokens,
            },
        )
        return result
