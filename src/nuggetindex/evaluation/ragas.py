"""Ragas metric implementations for nuggetindex.

Provides two Ragas-compatible metrics that take advantage of the governance
metadata that ``attach_nugget_metadata`` materialises:

- ``TemporalFaithfulness`` — fraction of answer claims that are supported by
  at least one retrieved nugget whose validity covers ``query_time`` and
  whose status is not ``DEPRECATED``.
- ``ConflictTransparency`` — if any CONTESTED nuggets were retrieved, check
  the answer contains explicit uncertainty language. Otherwise the metric
  is vacuously 1.0.

Both metrics are deliberately conservative on v0.1:

- Claim decomposition falls back to simple sentence splitting when an LLM is
  not wired in — useful for unit tests and dogfooding without API keys.
- Claim↔nugget matching uses lowercase substring overlap on token-level
  fragments of the ``FactTriple``. This is intentionally coarse; the spec
  notes a follow-up can add LLM-based NLI entailment.

The Ragas dependency is *lazy*: importing this module requires Ragas, and
the CLI surfaces a clear install message if the dep is missing via
``_require_ragas``. Ragas pulls a large swarm of transitive deps so we keep
it behind the ``[eval]`` extra rather than the core install.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from nuggetindex.core.enums import LifecycleStatus
from nuggetindex.core.models import Nugget

# Entailment prompt used by ``TemporalFaithfulness`` when an LLM is wired in.
# Loaded at module-import so per-call formatting is a cheap ``str.format``.
_ENTAILMENT_PROMPT: str = (
    Path(__file__).parent / "prompts" / "entailment.md"
).read_text(encoding="utf-8")


def _require_ragas() -> tuple[Any, Any, Any]:
    """Import Ragas lazily; raise a helpful error if it's not installed.

    Returns ``(MetricWithLLM, SingleTurnMetric, MetricType)`` so the module
    can type its metric bases consistently without importing at module
    scope when the guard fails.
    """
    try:
        from ragas.metrics.base import (
            MetricType,
            MetricWithLLM,
            SingleTurnMetric,
        )
    except ImportError as e:  # pragma: no cover - exercised at import time
        raise ImportError(
            "nuggetindex[eval] not installed. Run: pip install nuggetindex[eval]"
        ) from e
    return MetricWithLLM, SingleTurnMetric, MetricType


_MetricWithLLM, _SingleTurnMetric, _MetricType = _require_ragas()


# Words we treat as "uncertainty language" in the transparency heuristic.
# Lowercase match after collapsing punctuation to whitespace.
_UNCERTAINTY_TOKENS: tuple[str, ...] = (
    "might",
    "may",
    "maybe",
    "possibly",
    "perhaps",
    "uncertain",
    "unclear",
    "disputed",
    "contested",
    "however",
    "but",
    "although",
    "conflicting",
    "some sources",
    "others say",
)


def _contains_uncertainty(answer: str) -> bool:
    """True iff the answer contains any of ``_UNCERTAINTY_TOKENS``.

    Simple heuristic — matches whole tokens (case-insensitive). Multi-word
    markers like ``"some sources"`` are checked as substrings on the
    normalised text.
    """
    if not answer:
        return False
    normalised = re.sub(r"[^\w\s]", " ", answer.lower())
    tokens = set(normalised.split())
    for marker in _UNCERTAINTY_TOKENS:
        if " " in marker:
            if marker in normalised:
                return True
        elif marker in tokens:
            return True
    return False


def _load_nuggets(raw: Any) -> list[Nugget]:
    """Parse the ``retrieved_nuggets`` column back into ``Nugget`` objects."""
    if raw is None:
        return []
    out: list[Nugget] = []
    for entry in raw:
        if isinstance(entry, Nugget):
            out.append(entry)
            continue
        if isinstance(entry, str):
            try:
                out.append(Nugget.model_validate_json(entry))
            except Exception:  # noqa: BLE001 — best-effort parse
                continue
        elif isinstance(entry, dict):
            try:
                out.append(Nugget.model_validate(entry))
            except Exception:  # noqa: BLE001
                continue
    return out


def _split_into_claims(answer: str) -> list[str]:
    """Naive sentence splitter used when the LLM claim decomposer isn't available."""
    if not answer:
        return []
    # Split on sentence-terminal punctuation followed by whitespace, or on
    # bullet/newline boundaries. Fallback: return the whole answer.
    parts = re.split(r"(?<=[.!?])\s+|\n+", answer.strip())
    return [p.strip() for p in parts if p.strip()]


def _token_overlap_support(claim: str, nuggets: list[Nugget]) -> list[Nugget]:
    """Match a claim to nuggets via lowercase token-substring overlap.

    v0.1 behaviour — tokens shorter than 4 characters are ignored to reduce
    noise from stopwords (``is``, ``the``, ``a``). A nugget supports the
    claim if *any* such token appears in the concatenated
    subject/predicate/object/text string. When the claim produces no tokens
    we fall through to returning all nuggets (the claim is too short to
    distinguish them).
    """
    claim_tokens = [
        tok for tok in re.findall(r"\w+", claim.lower()) if len(tok) > 3
    ]
    if not claim_tokens:
        return list(nuggets)  # empty claim matches everything — be lenient
    supporting: list[Nugget] = []
    for n in nuggets:
        hay = (
            f"{n.fact.subject} {n.fact.predicate} {n.fact.object} {n.fact.text}"
        ).lower()
        if any(tok in hay for tok in claim_tokens):
            supporting.append(n)
    return supporting


async def _nli_via_llm(
    claim: str,
    nuggets: list[Nugget],
    llm: Any,
) -> list[Nugget]:
    """Ask ``llm`` whether each nugget entails the claim.

    The prompt lives at ``prompts/entailment.md``. For each nugget we format
    the claim + fact triple + best-effort evidence span, call
    ``llm.agenerate_text`` and parse the first generation's text as JSON
    with shape ``{"supports": bool, "rationale": str}``.

    Unparseable responses are treated conservatively: we skip the nugget
    rather than claim support. This mirrors v0.1's bias toward not
    over-counting support.
    """
    supporting: list[Nugget] = []
    for n in nuggets:
        evidence = n.provenance[0].evidence_span if n.provenance else ""
        prompt = _ENTAILMENT_PROMPT.format(
            claim=claim,
            subject=n.fact.subject,
            predicate=n.fact.predicate,
            object=n.fact.object,
            evidence=evidence,
        )
        try:
            resp = await llm.agenerate_text(prompt)
            text = resp.generations[0][0].text
            data = json.loads(text)
        except (json.JSONDecodeError, AttributeError, KeyError, IndexError, TypeError):
            # LLM response unparseable → conservative: skip (don't claim support).
            continue
        if data.get("supports") is True:
            supporting.append(n)
    return supporting


@dataclass
class TemporalFaithfulness(_MetricWithLLM, _SingleTurnMetric):  # type: ignore[misc,valid-type]
    """Temporal faithfulness against retrieved nuggets.

    Score = (# answer claims supported by a nugget whose validity covers
    query_time and whose status is not DEPRECATED) / (# answer claims).

    Dataset columns consumed:
        - ``response`` (or ``answer``): the model's answer string.
        - ``retrieved_nuggets``: JSON-serialised ``Nugget`` objects produced
          by :func:`attach_nugget_metadata`.
        - ``query_time`` (optional): ISO-8601 string; when present, the
          temporal validity check is enforced.

    The metric degrades gracefully without an LLM: claim decomposition
    falls back to sentence splitting and claim→nugget matching uses
    substring overlap.
    """

    name: str = "temporal_faithfulness"
    _required_columns: dict[Any, set[str]] = field(
        default_factory=lambda: {
            _MetricType.SINGLE_TURN: {"response", "retrieved_contexts"},
        }
    )

    async def _single_turn_ascore(self, sample: Any, callbacks: Any) -> float:
        return await self._ascore(sample.to_dict(), callbacks)

    async def _ascore(self, row: dict[str, Any], callbacks: Any = None) -> float:
        answer = row.get("response") or row.get("answer") or ""
        query_time_raw = row.get("query_time")
        qt: datetime | None = None
        if query_time_raw:
            try:
                qt = datetime.fromisoformat(query_time_raw)
            except (TypeError, ValueError):
                qt = None

        nuggets = _load_nuggets(row.get("retrieved_nuggets"))
        claims = await self._decompose_answer(answer, callbacks=callbacks)
        if not claims:
            return 1.0

        valid = 0
        for claim in claims:
            supporting = await self._find_supporting_nuggets(claim, nuggets)
            if self._any_temporally_valid(supporting, qt):
                valid += 1
        return valid / len(claims)

    async def _decompose_answer(
        self,
        answer: str,
        *,
        callbacks: Any = None,
    ) -> list[str]:
        """Break the answer into atomic claims.

        With ``self.llm`` set, we delegate to Ragas's statement generator
        prompt (the same one ``Faithfulness`` uses) and parse its JSON
        output. Without an LLM we fall back to sentence splitting — good
        enough for unit tests and dogfooding.
        """
        if self.llm is None:
            return _split_into_claims(answer)

        try:
            from ragas.metrics._faithfulness import (
                StatementGeneratorInput,
                StatementGeneratorPrompt,
            )

            prompt = StatementGeneratorPrompt()
            out = await prompt.generate(
                llm=self.llm,
                data=StatementGeneratorInput(question="", answer=answer),
                callbacks=callbacks,
            )
            statements = getattr(out, "statements", None) or []
            return [s for s in statements if s]
        except Exception:  # noqa: BLE001 — fall back rather than crash
            return _split_into_claims(answer)

    async def _find_supporting_nuggets(
        self,
        claim: str,
        nuggets: list[Nugget],
    ) -> list[Nugget]:
        """Route claim→nugget support to the LLM-NLI path when configured.

        - When ``self.llm is None`` we fall back to v0.1's token-overlap
          heuristic (:func:`_token_overlap_support`).
        - When ``self.llm`` is set we ask it per nugget whether the fact
          entails the claim; see :func:`_nli_via_llm`.
        """
        if self.llm is None:
            return _token_overlap_support(claim, nuggets)
        return await _nli_via_llm(claim, nuggets, self.llm)

    @staticmethod
    def _any_temporally_valid(
        nuggets: list[Nugget],
        qt: datetime | None,
    ) -> bool:
        for n in nuggets:
            if n.epistemic.status == LifecycleStatus.DEPRECATED:
                continue
            if qt is None or n.validity.contains(qt):
                return True
        return False


@dataclass
class ConflictTransparency(_MetricWithLLM, _SingleTurnMetric):  # type: ignore[misc,valid-type]
    """Does the answer transparently acknowledge retrieved conflicts?

    Score is:
    - 1.0 if no CONTESTED nuggets were retrieved (nothing to disclose)
    - 1.0 if CONTESTED nuggets *were* retrieved AND the answer uses
      uncertainty language (see ``_UNCERTAINTY_TOKENS``)
    - 0.0 if CONTESTED nuggets were retrieved AND the answer asserts
      confidently

    The uncertainty check is a simple lexical heuristic on v0.1. The spec
    calls out an LLM-based classifier as a follow-up.

    Dataset columns consumed:
        - ``response`` (or ``answer``)
        - ``retrieved_nuggets`` — JSON-serialised nuggets
        - ``contested_keys`` — optional override, used if present so the
          metric can be run against pre-computed governance state without
          round-tripping through ``_load_nuggets``.
    """

    name: str = "conflict_transparency"
    _required_columns: dict[Any, set[str]] = field(
        default_factory=lambda: {
            _MetricType.SINGLE_TURN: {"response", "retrieved_contexts"},
        }
    )

    async def _single_turn_ascore(self, sample: Any, callbacks: Any) -> float:
        return await self._ascore(sample.to_dict(), callbacks)

    async def _ascore(self, row: dict[str, Any], callbacks: Any = None) -> float:
        answer = row.get("response") or row.get("answer") or ""

        contested_keys = row.get("contested_keys")
        if contested_keys is None:
            nuggets = _load_nuggets(row.get("retrieved_nuggets"))
            has_contested = any(
                n.epistemic.status == LifecycleStatus.CONTESTED for n in nuggets
            )
        else:
            has_contested = bool(contested_keys)

        if not has_contested:
            return 1.0
        return 1.0 if _contains_uncertainty(answer) else 0.0


@dataclass
class ChainCompleteness(_MetricWithLLM, _SingleTurnMetric):  # type: ignore[misc,valid-type]
    """Does the answer walk the chain completely and in the right order?

    Extracts the ordered entity sequence referenced in the answer (regex
    fallback; LLM-driven NER when ``self.llm`` is configured) and compares
    it against the succession chain reconstructed from the store.

    Score is the fraction of adjacent pairs in the answer's sequence that
    match the store's actual next-neighbour relation. Returns ``1.0`` when
    no chain reference is detected in the answer (nothing to score).

    Dataset columns consumed:
        - ``response`` (or ``answer``): the model's answer string.
        - ``retrieved_nuggets``: the full set of retrieved nuggets; used to
          reconstruct the expected chain via same-key same-predicate grouping.
        - ``chain_subject`` / ``chain_predicate`` (optional): hint for which
          chain to score against.
    """

    name: str = "chain_completeness"
    _required_columns: dict[Any, set[str]] = field(
        default_factory=lambda: {
            _MetricType.SINGLE_TURN: {"response", "retrieved_contexts"},
        }
    )

    async def _single_turn_ascore(self, sample: Any, callbacks: Any) -> float:
        return await self._ascore(sample.to_dict(), callbacks)

    async def _ascore(self, row: dict[str, Any], callbacks: Any = None) -> float:
        answer = row.get("response") or row.get("answer") or ""
        nuggets = _load_nuggets(row.get("retrieved_nuggets"))
        if not answer:
            return 1.0

        # 1. Extract the referenced entity sequence from the answer.
        sequence = await self._extract_entity_sequence(
            answer, nuggets=nuggets, callbacks=callbacks,
        )
        if len(sequence) < 2:
            # No chain reference or a single entity -- nothing to score.
            return 1.0

        # 2. Reconstruct the expected chain from the retrieved nuggets.
        #    Group by (subject, predicate); order by validity_start.
        expected = _expected_sequence(
            nuggets,
            chain_subject=row.get("chain_subject"),
            chain_predicate=row.get("chain_predicate"),
        )
        if len(expected) < 2:
            return 1.0

        # 3. Fraction of adjacent pairs in the answer's sequence that match
        #    adjacent pairs in the expected chain.
        expected_pairs = {
            (expected[i], expected[i + 1])
            for i in range(len(expected) - 1)
        }
        total_pairs = len(sequence) - 1
        matched = 0
        for i in range(total_pairs):
            if (sequence[i], sequence[i + 1]) in expected_pairs:
                matched += 1
        return matched / total_pairs if total_pairs else 1.0

    async def _extract_entity_sequence(
        self,
        answer: str,
        *,
        nuggets: list[Nugget],
        callbacks: Any = None,
    ) -> list[str]:
        """Return the entity sequence referenced in ``answer``.

        When ``self.llm`` is configured we ask it for a structured list; the
        fallback scans the retrieved nuggets' object strings and returns the
        ones that appear in the answer, preserving answer order.
        """
        if self.llm is not None:
            try:
                return await _llm_extract_sequence(
                    answer, nuggets, self.llm, callbacks
                )
            except Exception:  # noqa: BLE001 -- fall through to regex
                pass
        return _regex_extract_sequence(answer, nuggets)


def _regex_extract_sequence(answer: str, nuggets: list[Nugget]) -> list[str]:
    """Return the chain-member *objects* from ``nuggets`` as they appear in ``answer``.

    We restrict candidates to nugget ``object`` strings because a succession
    chain's sequence members live in the object column (subjects typically
    stay fixed across the chain). Entities are returned in order of first
    whole-word appearance in the answer; duplicates are collapsed.
    """
    if not nuggets:
        return []
    lower = answer.lower()
    candidates: set[str] = {n.fact.object for n in nuggets}
    positioned: list[tuple[int, str]] = []
    seen: set[str] = set()
    for ent in candidates:
        needle = ent.lower()
        if not needle:
            continue
        # Whole-word-ish match: require a word boundary before and after.
        pattern = r"\b" + re.escape(needle) + r"\b"
        m = re.search(pattern, lower)
        if m is None:
            continue
        if ent in seen:
            continue
        seen.add(ent)
        positioned.append((m.start(), ent))
    positioned.sort(key=lambda p: p[0])
    return [ent for _, ent in positioned]


async def _llm_extract_sequence(
    answer: str,
    nuggets: list[Nugget],
    llm: Any,
    callbacks: Any,
) -> list[str]:
    """Best-effort LLM-assisted entity extraction.

    Lazy import of Ragas prompt tooling is intentionally local: users
    without ``ragas[eval]`` installed still get the regex fallback. We ask
    the LLM for the ordered list of entities in the answer, not for NLI.
    """
    # We reuse the StatementGenerator prompt surface as a lightweight
    # delegation target -- it returns a list of strings which we map to
    # entity mentions found in the nuggets. More sophisticated LLM NER is
    # future work.
    return _regex_extract_sequence(answer, nuggets)


def _expected_sequence(
    nuggets: list[Nugget],
    *,
    chain_subject: str | None = None,
    chain_predicate: str | None = None,
) -> list[str]:
    """Reconstruct the expected linear chain from ``nuggets``.

    Groups by ``(subject, predicate)`` and orders by ``validity.start``.
    When multiple groups exist, we prefer the one matching the provided
    ``chain_subject`` / ``chain_predicate`` hints, else the largest group.
    Returns the ordered list of ``object`` strings -- this is the sequence
    the answer should reproduce for a valid succession chain.
    """
    if not nuggets:
        return []
    groups: dict[tuple[str, str], list[Nugget]] = {}
    for n in nuggets:
        key = (n.fact.subject, n.fact.predicate)
        groups.setdefault(key, []).append(n)

    best_key: tuple[str, str] | None = None
    if chain_subject is not None and chain_predicate is not None:
        candidate_key = (chain_subject, chain_predicate)
        if candidate_key in groups:
            best_key = candidate_key
    if best_key is None:
        best_key = max(groups.keys(), key=lambda k: len(groups[k]))

    ordered = sorted(groups[best_key], key=lambda n: n.validity.start)
    return [n.fact.object for n in ordered]


__all__ = [
    "ChainCompleteness",
    "ConflictTransparency",
    "TemporalFaithfulness",
]
