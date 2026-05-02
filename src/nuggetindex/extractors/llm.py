"""LLM-backed extractor: load prompt, call structured-output client, build Nuggets.

The prompt template lives in ``prompts/extraction.md`` and is parsed once at
module import into a system block and a user block. The user block contains
``{text}`` and ``{context_hint}`` placeholders that are filled per call
via ``str.format`` -- simpler and less fragile than repeatedly slicing the
Markdown on every extraction.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from nuggetindex.core.enums import NuggetKind
from nuggetindex.core.models import (
    EpistemicState,
    FactTriple,
    Nugget,
    ProvenanceRecord,
    ValidityInterval,
)
from nuggetindex.extractors.base import BaseExtractor, ExtractionResult
from nuggetindex.extractors.clients.base import LLMClient, LLMConfig, build_client

_PROMPT_PATH = Path(__file__).parent / "prompts" / "extraction.md"
_DEFAULT_PROMPT_PATH = _PROMPT_PATH


def _load_prompt_sections(path: Path) -> tuple[str, str]:
    """Split the prompt file into (system_template, user_template)."""
    text = path.read_text()
    # Split into named sections by '# <Name>' headers.
    parts = re.split(r"(?m)^#\s+(\w+)\s*\n", text)
    # parts[0] is the preamble (often empty); afterward pairs of (name, body).
    sections: dict[str, str] = {}
    for i in range(1, len(parts), 2):
        name = parts[i].strip().lower()
        body = parts[i + 1] if i + 1 < len(parts) else ""
        sections[name] = body.strip()
    try:
        return sections["system"], sections["user"]
    except KeyError as e:  # pragma: no cover - only triggered by prompt edits
        raise ValueError("extraction.md must contain both '# System' and '# User' sections") from e


_SYSTEM_TEMPLATE, _USER_TEMPLATE = _load_prompt_sections(_PROMPT_PATH)


class _TripleOut(BaseModel):
    """One structured triple as returned by the LLM.

    ``subject_type`` and ``object_type`` carry the NER labels the LLM was
    asked to emit alongside each triple (see ``prompts/extraction.md``).
    Both default to ``None`` so older prompts / cooperating-but-forgetful
    models still parse: downstream code treats ``None`` as "no LLM type"
    and falls back to spaCy NER where available.
    """

    subject: str
    predicate: str
    object: str
    evidence_span: str
    temporal_expression: str | None = None
    confidence: float = Field(default=0.8, ge=0.0, le=1.0)
    subject_type: str | None = None
    object_type: str | None = None


class ExtractionPayload(BaseModel):
    """Top-level structured response the LLM is asked to produce."""

    facts: list[_TripleOut]


class LLMExtractor(BaseExtractor):
    """Extractor that delegates triple extraction to a structured-output LLM."""

    def __init__(
        self,
        cfg: LLMConfig,
        *,
        source_id_fn: Callable[[], str] | None = None,
        client: LLMClient | None = None,
        prompt_path: Path | None = None,
    ) -> None:
        self.cfg = cfg
        self.client: LLMClient = client if client is not None else build_client(cfg)
        self.source_id_fn: Callable[[], str] = source_id_fn or (lambda: "llm-extract")
        self._prompt_path: Path = prompt_path or _DEFAULT_PROMPT_PATH
        if prompt_path is None:
            # Fast path: use the module-level pre-parsed default templates.
            self._system_template = _SYSTEM_TEMPLATE
            self._user_template = _USER_TEMPLATE
        else:
            self._system_template, self._user_template = _load_prompt_sections(self._prompt_path)
        self._prompt: str = self._prompt_path.read_text()

    def _build_messages(self, text: str, context: str) -> list[dict[str, Any]]:
        user = self._user_template.format(
            text=text,
            context_hint=(f"Context: {context}" if context else ""),
        )
        return [
            {"role": "system", "content": self._system_template},
            {"role": "user", "content": user},
        ]

    async def aextract(
        self,
        text: str,
        *,
        context: str = "",
        source_id: str | None = None,
    ) -> list[ExtractionResult]:
        if not text or not text.strip():
            return []
        messages = self._build_messages(text, context)
        payload = await self.client.achat_structured(messages, ExtractionPayload)
        assert isinstance(payload, ExtractionPayload)

        results: list[ExtractionResult] = []
        now = datetime.now(UTC)
        effective_source_id = source_id or self.source_id_fn()
        for triple in payload.facts:
            nugget = Nugget.new(
                kind=NuggetKind.SEMANTIC_FACT,
                fact=FactTriple(
                    subject=triple.subject,
                    predicate=triple.predicate,
                    object=triple.object,
                    text=triple.evidence_span,
                    # LLM-emitted entity types (fix 8). When the LLM omits
                    # them (older prompt, non-cooperating model) the fields
                    # stay ``None`` and the pipeline falls back to spaCy NER.
                    subject_type=triple.subject_type,
                    object_type=triple.object_type,
                ),
                validity=ValidityInterval(start=now),  # temporal inference: Phase 4
                epistemic=EpistemicState(confidence=triple.confidence),
                provenance=(
                    ProvenanceRecord(
                        source_id=effective_source_id,
                        evidence_span=triple.evidence_span,
                    ),
                ),
                extraction_confidence=triple.confidence,
            )
            results.append(
                ExtractionResult(
                    nugget=nugget,
                    confidence=triple.confidence,
                )
            )
        return results
