"""Stub-backed tests for the OpenAI structured-output client."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
from pydantic import BaseModel

pytest.importorskip("openai")
pytest.importorskip("instructor")

from nuggetindex.extractors.clients.base import LLMConfig  # noqa: E402
from nuggetindex.extractors.clients.openai import OpenAIClient  # noqa: E402

_TRANSCRIPT = (
    Path(__file__).resolve().parents[3]
    / "fixtures"
    / "llm_transcripts"
    / "openai_extraction_01.json"
)


class Triple(BaseModel):
    subject: str
    predicate: str
    object: str


@pytest.mark.asyncio
async def test_openai_structured_output_via_instructor() -> None:
    cfg = LLMConfig(provider="openai", model="gpt-4o-mini", api_key="sk-test")
    client = OpenAIClient(cfg)

    fake_raw = MagicMock()
    fake_raw.chat.completions.create = AsyncMock(
        return_value=Triple(subject="Google", predicate="ceo", object="Sundar Pichai")
    )
    client._raw_client = fake_raw

    result = await client.achat_structured(
        messages=[{"role": "user", "content": "Sundar Pichai is CEO of Google."}],
        response_model=Triple,
    )
    assert isinstance(result, Triple)
    assert result.subject == "Google"
    assert result.predicate == "ceo"
    assert result.object == "Sundar Pichai"

    # Parameters forwarded correctly.
    fake_raw.chat.completions.create.assert_awaited_once()
    kwargs = fake_raw.chat.completions.create.await_args.kwargs
    assert kwargs["model"] == "gpt-4o-mini"
    assert kwargs["response_model"] is Triple
    assert kwargs["temperature"] == 0.0
    assert kwargs["max_tokens"] == 2048


@pytest.mark.asyncio
async def test_openai_client_uses_recorded_transcript() -> None:
    transcript = json.loads(_TRANSCRIPT.read_text())
    cfg = LLMConfig(provider="openai", model=transcript["model"], api_key="sk-test")
    client = OpenAIClient(cfg)

    fake_raw = MagicMock()
    fake_raw.chat.completions.create = AsyncMock(return_value=Triple(**transcript["response"]))
    client._raw_client = fake_raw

    result = await client.achat_structured(
        messages=transcript["request"]["messages"],
        response_model=Triple,
    )
    assert result.subject == "Google"


def test_openai_client_raises_when_sdk_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    """If the SDK import fails, _require_openai_sdk() must raise a helpful error."""
    import nuggetindex.extractors.clients.openai as mod

    def _boom() -> tuple[object, object]:
        raise ImportError("nuggetindex[openai] not installed. Run: pip install nuggetindex[openai]")

    monkeypatch.setattr(mod, "_require_openai_sdk", _boom)
    with pytest.raises(ImportError, match=r"nuggetindex\[openai\] not installed"):
        mod.OpenAIClient(LLMConfig(provider="openai", model="gpt-4o-mini"))
