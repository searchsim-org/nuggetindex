"""Stub-backed tests for the Google/Gemini structured-output client."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
from pydantic import BaseModel

import nuggetindex.extractors.clients.google as mod
from nuggetindex.extractors.clients.base import LLMConfig

_TRANSCRIPT = (
    Path(__file__).resolve().parents[3]
    / "fixtures"
    / "llm_transcripts"
    / "google_extraction_01.json"
)


class Triple(BaseModel):
    subject: str
    predicate: str
    object: str


def _build_client_with_fake_raw(raw: object) -> mod.GoogleClient:
    client = object.__new__(mod.GoogleClient)
    client.cfg = LLMConfig(provider="google", model="gemini-1.5-flash")
    client._raw_client = raw
    return client


@pytest.mark.asyncio
async def test_google_structured_output_via_instructor() -> None:
    fake_raw = MagicMock()
    fake_raw.messages.create = AsyncMock(
        return_value=Triple(
            subject="Larry Page", predicate="co-founded", object="Google"
        )
    )
    client = _build_client_with_fake_raw(fake_raw)

    result = await client.achat_structured(
        messages=[{"role": "user", "content": "Larry Page co-founded Google."}],
        response_model=Triple,
    )
    assert isinstance(result, Triple)
    assert result.subject == "Larry Page"

    kwargs = fake_raw.messages.create.await_args.kwargs
    assert kwargs["response_model"] is Triple
    gen_cfg = kwargs["generation_config"]
    assert gen_cfg["temperature"] == 0.0
    assert gen_cfg["max_output_tokens"] == 2048


@pytest.mark.asyncio
async def test_google_uses_recorded_transcript() -> None:
    transcript = json.loads(_TRANSCRIPT.read_text())
    fake_raw = MagicMock()
    fake_raw.messages.create = AsyncMock(
        return_value=Triple(**transcript["response"])
    )
    client = _build_client_with_fake_raw(fake_raw)
    result = await client.achat_structured(
        messages=transcript["request"]["messages"],
        response_model=Triple,
    )
    assert result.subject == "Larry Page"


def test_google_client_raises_when_sdk_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _boom() -> tuple[object, object]:
        raise ImportError(
            "nuggetindex[google] not installed. Run: pip install nuggetindex[google]"
        )

    monkeypatch.setattr(mod, "_require_google_sdk", _boom)
    with pytest.raises(ImportError, match=r"nuggetindex\[google\] not installed"):
        mod.GoogleClient(LLMConfig(provider="google", model="gemini-1.5-flash"))
