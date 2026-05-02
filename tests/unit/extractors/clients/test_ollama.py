"""Stub-backed tests for the Ollama structured-output client."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
from pydantic import BaseModel

import nuggetindex.extractors.clients.ollama as mod
from nuggetindex.extractors.clients.base import LLMConfig

_TRANSCRIPT = (
    Path(__file__).resolve().parents[3]
    / "fixtures"
    / "llm_transcripts"
    / "ollama_extraction_01.json"
)


class Triple(BaseModel):
    subject: str
    predicate: str
    object: str


def _build_client_with_fake_raw(raw: object) -> mod.OllamaClient:
    client = object.__new__(mod.OllamaClient)
    client.cfg = LLMConfig(provider="ollama", model="llama3.2")
    client._raw_client = raw
    return client


@pytest.mark.asyncio
async def test_ollama_structured_output_via_instructor() -> None:
    fake_raw = MagicMock()
    fake_raw.chat.completions.create = AsyncMock(
        return_value=Triple(
            subject="Elon Musk", predicate="founded", object="SpaceX"
        )
    )
    client = _build_client_with_fake_raw(fake_raw)

    result = await client.achat_structured(
        messages=[{"role": "user", "content": "Elon Musk founded SpaceX."}],
        response_model=Triple,
    )
    assert isinstance(result, Triple)
    assert result.object == "SpaceX"

    kwargs = fake_raw.chat.completions.create.await_args.kwargs
    assert kwargs["model"] == "llama3.2"
    assert kwargs["response_model"] is Triple


@pytest.mark.asyncio
async def test_ollama_uses_recorded_transcript() -> None:
    transcript = json.loads(_TRANSCRIPT.read_text())
    fake_raw = MagicMock()
    fake_raw.chat.completions.create = AsyncMock(
        return_value=Triple(**transcript["response"])
    )
    client = _build_client_with_fake_raw(fake_raw)
    result = await client.achat_structured(
        messages=transcript["request"]["messages"],
        response_model=Triple,
    )
    assert result.subject == "Elon Musk"


def test_ollama_client_raises_when_sdk_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _boom() -> tuple[object, object]:
        raise ImportError(
            "nuggetindex[ollama] and nuggetindex[openai] are required. "
            "Run: pip install nuggetindex[ollama,openai]"
        )

    monkeypatch.setattr(mod, "_require_ollama_sdk", _boom)
    with pytest.raises(ImportError, match=r"nuggetindex\[ollama\]"):
        mod.OllamaClient(LLMConfig(provider="ollama", model="llama3.2"))
