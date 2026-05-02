"""Tests for the `prompt_path=` constructor kwarg on `LLMExtractor`.

Users who depend on the pre-0.2.1 literal-predicate prompt can opt out of the
canonicalisation rewrite by pointing the extractor at their own prompt file.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock

import pytest
from pydantic import BaseModel

from nuggetindex.extractors.clients.base import LLMConfig
from nuggetindex.extractors.llm import ExtractionPayload, LLMExtractor


class _StubClient:
    def __init__(self, payload: ExtractionPayload) -> None:
        self._payload = payload
        self.achat_structured = AsyncMock(side_effect=self._return)

    async def _return(
        self, messages: list[dict[str, Any]], response_model: type[BaseModel]
    ) -> BaseModel:
        assert response_model is ExtractionPayload
        return self._payload


def _cfg() -> LLMConfig:
    return LLMConfig(provider="openai", model="gpt-4o-mini", api_key="sk-test")


@pytest.mark.asyncio
async def test_prompt_path_override_uses_custom_prompt(tmp_path: Path) -> None:
    """`prompt_path=` redirects the extractor to a user-supplied prompt file.

    The custom prompt must contain both `# System` and `# User` sections; its
    `# System` body becomes the system message sent to the client and both
    placeholders (`{text}`, `{context_hint}`) still interpolate.
    """
    custom = tmp_path / "custom_prompt.md"
    custom.write_text(
        "# System\n"
        "Custom system instructions: literal predicates only.\n"
        "\n"
        "# User\n"
        "Extract from:\n{text}\n{context_hint}\n"
    )

    stub = _StubClient(ExtractionPayload(facts=[]))
    ex = LLMExtractor(_cfg(), client=stub, prompt_path=custom)  # type: ignore[arg-type]

    # The raw prompt text is kept on the instance for inspection.
    assert "Custom system instructions" in ex._prompt
    # And the parsed sections both point at the override.
    assert "literal predicates only" in ex._system_template
    assert "{text}" in ex._user_template

    await ex.aextract("some text", context="doc-2024")
    messages = stub.achat_structured.await_args.args[0]
    assert messages[0]["role"] == "system"
    assert "literal predicates only" in messages[0]["content"]
    assert "some text" in messages[-1]["content"]
    assert "Context: doc-2024" in messages[-1]["content"]
