"""Stub-backed tests for the OpenRouter structured-output client."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
from pydantic import BaseModel

pytest.importorskip("openai")
pytest.importorskip("instructor")

import nuggetindex.extractors.clients.openrouter as mod  # noqa: E402
from nuggetindex.extractors.clients.base import LLMConfig  # noqa: E402

_TRANSCRIPT = (
    Path(__file__).resolve().parents[3]
    / "fixtures"
    / "llm_transcripts"
    / "openrouter_extraction_01.json"
)


class Triple(BaseModel):
    subject: str
    predicate: str
    object: str


@pytest.mark.asyncio
async def test_openrouter_structured_output_via_instructor() -> None:
    cfg = LLMConfig(
        provider="openrouter",
        model="anthropic/claude-3-haiku",
        api_key="sk-or-test",
    )
    client = mod.OpenRouterClient(cfg)

    fake_raw = MagicMock()
    fake_raw.chat.completions.create = AsyncMock(
        return_value=Triple(
            subject="Microsoft", predicate="ceo", object="Satya Nadella"
        )
    )
    client._raw_client = fake_raw

    result = await client.achat_structured(
        messages=[{"role": "user", "content": "Satya Nadella is CEO of Microsoft."}],
        response_model=Triple,
    )
    assert isinstance(result, Triple)
    assert result.subject == "Microsoft"

    kwargs = fake_raw.chat.completions.create.await_args.kwargs
    assert kwargs["model"] == "anthropic/claude-3-haiku"


def test_openrouter_uses_default_base_url() -> None:
    cfg = LLMConfig(
        provider="openrouter",
        model="anthropic/claude-3-haiku",
        api_key="sk-or-test",
    )
    client = mod.OpenRouterClient(cfg)
    # AsyncOpenAI instance is nested; check base_url on the underlying client
    # produced by instructor.from_openai (instructor keeps a reference).
    raw_client = getattr(client._raw_client, "client", None) or getattr(
        client._raw_client, "_client", None
    )
    if raw_client is not None:
        assert "openrouter.ai" in str(raw_client.base_url)


@pytest.mark.asyncio
async def test_openrouter_uses_recorded_transcript() -> None:
    transcript = json.loads(_TRANSCRIPT.read_text())
    cfg = LLMConfig(
        provider="openrouter", model=transcript["model"], api_key="sk-or-test"
    )
    client = mod.OpenRouterClient(cfg)
    fake_raw = MagicMock()
    fake_raw.chat.completions.create = AsyncMock(
        return_value=Triple(**transcript["response"])
    )
    client._raw_client = fake_raw
    result = await client.achat_structured(
        messages=transcript["request"]["messages"],
        response_model=Triple,
    )
    assert result.subject == "Microsoft"


def test_openrouter_client_raises_when_sdk_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _boom() -> tuple[object, object]:
        raise ImportError(
            "nuggetindex[openai] not installed. Run: pip install nuggetindex[openai]"
        )

    monkeypatch.setattr(mod, "_require_openrouter_sdk", _boom)
    with pytest.raises(ImportError, match=r"nuggetindex\[openai\] not installed"):
        mod.OpenRouterClient(
            LLMConfig(provider="openrouter", model="anthropic/claude-3-haiku")
        )
