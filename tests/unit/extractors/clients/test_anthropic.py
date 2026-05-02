"""Stub-backed tests for the Anthropic structured-output client."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
from pydantic import BaseModel

import nuggetindex.extractors.clients.anthropic as mod
from nuggetindex.extractors.clients.base import LLMConfig

_TRANSCRIPT = (
    Path(__file__).resolve().parents[3]
    / "fixtures"
    / "llm_transcripts"
    / "anthropic_extraction_01.json"
)


class Triple(BaseModel):
    subject: str
    predicate: str
    object: str


def _build_client_with_fake_raw(raw: object) -> mod.AnthropicClient:
    """Instantiate without running the real __init__ (SDK may be missing)."""
    client = object.__new__(mod.AnthropicClient)
    client.cfg = LLMConfig(provider="anthropic", model="claude-3-haiku-20240307")
    client._raw_client = raw
    return client


@pytest.mark.asyncio
async def test_anthropic_structured_output_via_instructor() -> None:
    fake_raw = MagicMock()
    fake_raw.messages.create = AsyncMock(
        return_value=Triple(subject="Apple", predicate="ceo", object="Tim Cook")
    )
    client = _build_client_with_fake_raw(fake_raw)

    result = await client.achat_structured(
        messages=[
            {"role": "system", "content": "You extract atomic facts from text."},
            {"role": "user", "content": "Tim Cook is CEO of Apple."},
        ],
        response_model=Triple,
    )
    assert isinstance(result, Triple)
    assert result.subject == "Apple"

    kwargs = fake_raw.messages.create.await_args.kwargs
    assert kwargs["model"] == "claude-3-haiku-20240307"
    assert kwargs["response_model"] is Triple
    assert kwargs["system"] == "You extract atomic facts from text."
    assert kwargs["messages"] == [
        {"role": "user", "content": "Tim Cook is CEO of Apple."},
    ]


@pytest.mark.asyncio
async def test_anthropic_uses_recorded_transcript() -> None:
    transcript = json.loads(_TRANSCRIPT.read_text())
    fake_raw = MagicMock()
    fake_raw.messages.create = AsyncMock(return_value=Triple(**transcript["response"]))
    client = _build_client_with_fake_raw(fake_raw)
    result = await client.achat_structured(
        messages=transcript["request"]["messages"],
        response_model=Triple,
    )
    assert result.subject == "Apple"


def test_anthropic_client_raises_when_sdk_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _boom() -> tuple[object, object]:
        raise ImportError(
            "nuggetindex[anthropic] not installed. Run: pip install nuggetindex[anthropic]"
        )

    monkeypatch.setattr(mod, "_require_anthropic_sdk", _boom)
    with pytest.raises(ImportError, match=r"nuggetindex\[anthropic\] not installed"):
        mod.AnthropicClient(LLMConfig(provider="anthropic", model="claude-3-haiku-20240307"))
