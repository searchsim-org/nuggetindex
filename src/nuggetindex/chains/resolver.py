"""Opt-in LLM disambiguator for ambiguous chain steps.

:class:`ChainResolver` wraps an :class:`LLMClient` the same way
:class:`nuggetindex.pipeline.judge.LLMJudge` does. It is **never constructed
by default** -- if a chain method receives ``resolver=None``, ambiguous steps
raise :class:`ChainAmbiguousError`.

Every adjudication is logged with the full candidate list and chosen index
to ``~/.nuggetindex/chain_resolver_log.jsonl`` (override the home dir via
the ``NUGGETINDEX_HOME`` env var). The log makes chain walks reproducible
for audits.
"""

from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from nuggetindex.core.models import Nugget
from nuggetindex.extractors.clients.base import LLMClient, LLMConfig, build_client

_PROMPT_PATH = (
    Path(__file__).parent.parent
    / "extractors"
    / "prompts"
    / "chain_resolver.md"
)


def _load_prompt() -> str:
    return _PROMPT_PATH.read_text(encoding="utf-8")


def _default_log_path() -> Path:
    home_override = os.environ.get("NUGGETINDEX_HOME")
    home = Path(home_override) if home_override else (Path.home() / ".nuggetindex")
    return home / "chain_resolver_log.jsonl"


class ChainResolution(BaseModel):
    """The resolver's verdict for an ambiguous step."""

    picked: Nugget
    rationale: str


class _ResolverResponse(BaseModel):
    """Structured LLM response: index into the candidate list + rationale."""

    picked_index: int = Field(..., ge=0)
    rationale: str = Field(default="", max_length=400)


class ChainResolver:
    """Opt-in LLM disambiguation for ambiguous chain walks.

    Usage::

        resolver = ChainResolver.default()
        chain = await store.achain_rename(
            subject="Twitter", resolver=resolver,
        )

    The resolver lazily builds the underlying :class:`LLMClient` only on
    the first :meth:`adisambiguate` call, so constructing the resolver
    itself never requires a working API key.
    """

    def __init__(
        self,
        cfg: LLMConfig,
        *,
        client: LLMClient | None = None,
        log_path: Path | str | None = None,
    ) -> None:
        self.cfg = cfg
        self._prompt = _load_prompt()
        # Lazy client construction: only build on the first adisambiguate
        # call so ``ChainResolver.default()`` doesn't require an API key at
        # construction time.
        self._client: LLMClient | None = client
        self._log_path: Path = (
            Path(log_path) if log_path is not None else _default_log_path()
        )

    @classmethod
    def default(cls) -> ChainResolver:
        """Default config: OpenAI ``gpt-4o-mini``. Does not require an API
        key at construction -- the key is only needed on the first
        :meth:`adisambiguate` call.
        """
        return cls(LLMConfig(provider="openai", model="gpt-4o-mini"))

    def _get_client(self) -> LLMClient:
        if self._client is None:
            self._client = build_client(self.cfg)
        return self._client

    async def adisambiguate(
        self, *, candidates: list[Nugget], context: str,
    ) -> ChainResolution:
        if not candidates:
            raise ValueError("adisambiguate requires at least one candidate")
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": self._prompt},
            {"role": "user", "content": _render_user(candidates, context)},
        ]
        client = self._get_client()
        resp = await client.achat_structured(messages, _ResolverResponse)
        assert isinstance(resp, _ResolverResponse)
        idx = resp.picked_index
        # Defensive: clamp out-of-range indices rather than crashing the walker.
        if idx < 0 or idx >= len(candidates):
            idx = 0
        resolution = ChainResolution(
            picked=candidates[idx], rationale=resp.rationale
        )
        self._log(candidates, context, idx, resp.rationale)
        return resolution

    def _log(
        self,
        candidates: list[Nugget],
        context: str,
        picked_idx: int,
        rationale: str,
    ) -> None:
        try:
            self._log_path.parent.mkdir(parents=True, exist_ok=True)
        except OSError:
            return
        row = {
            "context": context,
            "picked_idx": picked_idx,
            "picked_id": candidates[picked_idx].id,
            "candidate_ids": [c.id for c in candidates],
            "rationale": rationale,
            "timestamp": datetime.now(UTC).isoformat(),
        }
        try:
            with self._log_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(row) + "\n")
        except OSError:
            pass


def _render_user(candidates: list[Nugget], context: str) -> str:
    lines = [f"Context: {context}", "", "Candidates:"]
    for i, n in enumerate(candidates):
        end = n.validity.end.date().isoformat() if n.validity.end else "open"
        lines.append(
            f"  [{i}] {n.fact.subject} -- {n.fact.predicate} -- {n.fact.object} "
            f"(valid {n.validity.start.date().isoformat()} to {end}; "
            f"evidence sources: {len(n.provenance)}; "
            f"status: {n.epistemic.status})"
        )
    lines.append("")
    lines.append("Pick the candidate index that best continues the chain.")
    return "\n".join(lines)
