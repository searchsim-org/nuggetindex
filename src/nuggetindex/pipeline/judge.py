"""LLM-as-judge conflict adjudicator (Improvement C, spec §5.5).

Invoked by the conflict detector only on ``(1,1)``-evidence ambiguous cases:
both sides functional, both temporally overlapping, and no evidence-count
asymmetry to pick a winner via rules.

The judge returns one of four ``JudgeDecision`` enum members. Decisions are
logged to ``~/.nuggetindex/judge_log.jsonl`` for audit.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from nuggetindex.core.models import Nugget
from nuggetindex.extractors.clients.base import LLMClient, LLMConfig, build_client

_PROMPT_PATH = Path(__file__).parent.parent / "extractors" / "prompts" / "judge.md"


def _load_prompt() -> str:
    return _PROMPT_PATH.read_text(encoding="utf-8")


class JudgeDecision(StrEnum):
    A_WINS = "A_WINS"
    B_WINS = "B_WINS"
    GENUINELY_CONTESTED = "GENUINELY_CONTESTED"
    NEED_MORE_EVIDENCE = "NEED_MORE_EVIDENCE"


class _JudgeResponse(BaseModel):
    decision: JudgeDecision
    rationale: str = Field(default="", max_length=500)


class LLMJudge:
    """Adjudicate ``(1,1)``-evidence contested pairs via a structured-output LLM call."""

    def __init__(
        self,
        cfg: LLMConfig,
        *,
        client: LLMClient | None = None,
        log_path: Path | str | None = None,
    ) -> None:
        self.cfg = cfg
        self.client: LLMClient = client if client is not None else build_client(cfg)
        self._prompt = _load_prompt()
        self._log_path: Path = (
            Path(log_path)
            if log_path is not None
            else Path.home() / ".nuggetindex" / "judge_log.jsonl"
        )

    @classmethod
    def default(cls) -> LLMJudge:
        """Default config: OpenAI gpt-4o-mini. Not auto-created; opt-in only."""
        return cls(LLMConfig(provider="openai", model="gpt-4o-mini"))

    async def aadjudicate(self, incoming: Nugget, existing: Nugget) -> JudgeDecision:
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": self._prompt},
            {"role": "user", "content": _render_user(incoming, existing)},
        ]
        resp = await self.client.achat_structured(messages, _JudgeResponse)
        assert isinstance(resp, _JudgeResponse)
        self._log(incoming, existing, resp)
        return resp.decision

    def _log(self, a: Nugget, b: Nugget, resp: _JudgeResponse) -> None:
        try:
            self._log_path.parent.mkdir(parents=True, exist_ok=True)
        except OSError:
            # Logging is best-effort; failures should not break adjudication.
            return
        row = {
            "a_id": a.id,
            "b_id": b.id,
            "a_object": a.fact.object,
            "b_object": b.fact.object,
            "decision": str(resp.decision),
            "rationale": resp.rationale,
            "timestamp": datetime.now(UTC).isoformat(),
        }
        try:
            with self._log_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(row) + "\n")
        except OSError:
            pass


def _render_user(a: Nugget, b: Nugget) -> str:
    a_end = a.validity.end.date().isoformat() if a.validity.end else "open"
    b_end = b.validity.end.date().isoformat() if b.validity.end else "open"
    return (
        f"Fact A: {a.fact.subject} -- {a.fact.predicate} -- {a.fact.object} "
        f"(valid {a.validity.start.date().isoformat()} to {a_end}; "
        f"evidence sources: {len(a.provenance)})\n"
        f"Fact B: {b.fact.subject} -- {b.fact.predicate} -- {b.fact.object} "
        f"(valid {b.validity.start.date().isoformat()} to {b_end}; "
        f"evidence sources: {len(b.provenance)})\n"
        f"Decide which (if either) is correct at the overlap, or mark as "
        f"genuinely contested / insufficient evidence."
    )
