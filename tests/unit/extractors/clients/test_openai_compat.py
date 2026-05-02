"""Stub-backed tests for the generic OpenAI-compatible client."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
from pydantic import BaseModel

pytest.importorskip("openai")
pytest.importorskip("instructor")

import nuggetindex.extractors.clients.openai_compat as mod  # noqa: E402
from nuggetindex.extractors.clients.base import LLMConfig  # noqa: E402

_TRANSCRIPT = (
    Path(__file__).resolve().parents[3]
    / "fixtures"
    / "llm_transcripts"
    / "openai_compat_extraction_01.json"
)


class Triple(BaseModel):
    subject: str
    predicate: str
    object: str


@pytest.mark.asyncio
async def test_openai_compat_requires_base_url() -> None:
    with pytest.raises(ValueError, match="base_url"):
        mod.OpenAICompatClient(LLMConfig(provider="openai_compat", model="mistral-7b"))


@pytest.mark.asyncio
async def test_openai_compat_structured_output() -> None:
    cfg = LLMConfig(
        provider="openai_compat",
        model="mistral-7b",
        base_url="http://localhost:8000/v1",
    )
    client = mod.OpenAICompatClient(cfg)
    fake_raw = MagicMock()
    fake_raw.chat.completions.create = AsyncMock(
        return_value=Triple(
            subject="Ada Lovelace", predicate="wrote", object="first algorithm"
        )
    )
    client._raw_client = fake_raw

    result = await client.achat_structured(
        messages=[{"role": "user", "content": "Ada Lovelace wrote the first algorithm."}],
        response_model=Triple,
    )
    assert result.subject == "Ada Lovelace"
    kwargs = fake_raw.chat.completions.create.await_args.kwargs
    assert kwargs["model"] == "mistral-7b"


@pytest.mark.asyncio
async def test_openai_compat_uses_recorded_transcript() -> None:
    transcript = json.loads(_TRANSCRIPT.read_text())
    cfg = LLMConfig(
        provider="openai_compat",
        model=transcript["model"],
        base_url=transcript["base_url"],
    )
    client = mod.OpenAICompatClient(cfg)
    fake_raw = MagicMock()
    fake_raw.chat.completions.create = AsyncMock(
        return_value=Triple(**transcript["response"])
    )
    client._raw_client = fake_raw
    result = await client.achat_structured(
        messages=transcript["request"]["messages"],
        response_model=Triple,
    )
    assert result.subject == "Ada Lovelace"


def test_openai_compat_client_raises_when_sdk_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _boom() -> tuple[object, object]:
        raise ImportError(
            "nuggetindex[openai] not installed. Run: pip install nuggetindex[openai]"
        )

    monkeypatch.setattr(mod, "_require_openai_compat_sdk", _boom)
    with pytest.raises(ImportError, match=r"nuggetindex\[openai\] not installed"):
        mod.OpenAICompatClient(
            LLMConfig(
                provider="openai_compat",
                model="mistral-7b",
                base_url="http://localhost:8000/v1",
            )
        )
