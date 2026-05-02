"""``nuggetindex doctor`` -- scan any RAG index for temporal / conflict / rename damage.

Thin CLI wrapper around :func:`nuggetindex.audit.doctor.scan_index`. Accepts a
document source (``.jsonl`` today; a SQLite ``nuggetindex.db`` is attempted by
reading its ``passages`` table) and renders a :class:`DoctorReport` either to
stdout (Markdown, default) or to a file (Markdown / JSON, inferred from the
``--report`` extension).

Each JSONL line must be an object with at least the required fields::

    {"source_id": "doc-1", "text": "...", "uri": null, "source_date": "2023-04-11T00:00:00Z"}

``uri`` and ``source_date`` are optional; ``source_id`` and ``text`` are
required. ``source_date`` accepts anything :meth:`datetime.fromisoformat`
groks; trailing ``Z`` is rewritten to ``+00:00`` first.

For ``--mode deep`` the ``--extractor`` flag selects the extractor:

* Model ids prefixed ``claude-`` route to Anthropic, ``gemini-`` to Google,
  and everything else is treated as an OpenAI-compatible model id. The
  underlying clients read ``OPENAI_API_KEY`` / ``UP_API_KEY`` / the relevant
  provider key from the environment. The legacy ``rule_based`` sentinel was
  removed in favour of the LLM extractor and the upcoming
  ``TriggerExtractor`` (LLM-free, pattern + NER).

Fast mode relies on the ``[doctor]`` extra (spaCy + dateparser) for the
temporal-depth and temporal-drift scores; trigger / rename scoring works
without it. Install with::

    pip install 'nuggetindex[doctor]'
    python -m spacy download en_core_web_sm

``en_core_web_sm`` is a spaCy model, not a PyPI package, so it is not part of
the ``[doctor]`` extra; the command above fetches it on demand.

Exit codes
----------
* ``0`` -- scan completed and either printed to stdout or written to ``--report``.
* ``1`` -- user error (missing / unreadable input; bad ``--mode``; unknown
  extension on ``--report``; deep-mode extractor construction failed).
* ``2`` -- deep-mode ingestion failure rate exceeded 30 %.
"""

from __future__ import annotations

import asyncio
import dataclasses
import json
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import typer

from nuggetindex.audit.doctor import DoctorReport, scan_index
from nuggetindex.pipeline.constructor import Document

_DEEP_FAILURE_EXIT_THRESHOLD = 0.30


def _parse_source_date(raw: Any) -> datetime | None:
    """Best-effort ISO-8601 parser for the JSONL ``source_date`` field.

    Returns ``None`` when ``raw`` is missing / empty / unparseable. Strips a
    trailing ``Z`` before handing the string to :meth:`datetime.fromisoformat`,
    which does not accept the zulu suffix before Python 3.11. Naive datetimes
    are coerced to UTC so the sampler's date bucketing stays consistent.
    """
    if raw is None:
        return None
    if isinstance(raw, datetime):
        return raw if raw.tzinfo else raw.replace(tzinfo=UTC)
    s = str(raw).strip()
    if not s:
        return None
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(s)
    except ValueError:
        return None
    return dt if dt.tzinfo else dt.replace(tzinfo=UTC)


def _load_jsonl_docs(path: Path, *, verbose: bool) -> list[Document]:
    """Parse each non-blank line of ``path`` into a :class:`Document`.

    Required keys: ``source_id``, ``text``. Lines missing either, or whose
    JSON fails to parse, are skipped with a stderr warning (but do not abort
    the scan -- honest partial results beat a fail-closed doctor).
    """
    docs: list[Document] = []
    total = 0
    with path.open("r", encoding="utf-8") as fh:
        for lineno, raw in enumerate(fh, 1):
            line = raw.strip()
            if not line:
                continue
            total += 1
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                typer.echo(
                    f"doctor: skipping {path.name}:{lineno} -- invalid JSON: {exc}",
                    err=True,
                )
                continue
            source_id = row.get("source_id")
            text = row.get("text")
            if not source_id or not text:
                typer.echo(
                    f"doctor: skipping {path.name}:{lineno} -- missing required "
                    "'source_id' or 'text'",
                    err=True,
                )
                continue
            docs.append(
                Document(
                    source_id=str(source_id),
                    text=str(text),
                    uri=row.get("uri"),
                    source_date=_parse_source_date(row.get("source_date")),
                )
            )
    if verbose:
        typer.echo(
            f"doctor: loaded {len(docs)} document(s) from {path} (skipped {total - len(docs)}).",
            err=True,
        )
    return docs


async def _load_sqlite_docs(path: Path, *, verbose: bool) -> list[Document]:
    """Pull every passage row out of a ``nuggetindex.db`` SQLite store.

    The store's ``passages`` table is the canonical place for raw document
    text, so one :class:`Document` per row is the honest mapping.
    ``source_date`` is left ``None`` here -- the passages table does not
    carry a per-doc date (nugget ``validity.start`` is inferred, not
    authoritative for the source document itself). Callers who need date
    stratification should provide JSONL. If the store was built without raw
    passages, ``alist_source_ids`` returns an empty list; the caller handles
    that as an empty-scan case.
    """
    from nuggetindex.store import NuggetStore

    store = NuggetStore(db_path=path)
    try:
        source_ids = await store.backend.alist_source_ids()
        records = await store.backend.aget_passage_records(source_ids)
    finally:
        await store.aclose()

    docs: list[Document] = [
        Document(source_id=sid, text=text, uri=None, source_date=None)
        for sid, (text, _meta) in records.items()
        if text
    ]
    if verbose:
        typer.echo(f"doctor: loaded {len(docs)} passage(s) from {path}.", err=True)
    return docs


def _build_extractor(name: str) -> Any:
    """Instantiate the deep-mode extractor matching ``name``.

    ``"trigger"`` builds a LLM-free :class:`TriggerExtractor` (the new CLI
    default; no API key required). Any other string is treated as a model
    id and routed to a provider by prefix (``claude-`` -> anthropic,
    ``gemini-`` -> google, default -> openai). Construction is lazy so the
    ``[openai]`` / ``[anthropic]`` extras are only imported when actually
    needed. The legacy ``rule_based`` sentinel emits a
    :class:`DeprecationWarning` and delegates to ``trigger``.
    """
    import warnings

    if name == "rule_based":
        warnings.warn(
            "--extractor rule_based is deprecated: delegating to "
            "TriggerExtractor ('trigger'). Pass --extractor trigger to "
            "silence this warning.",
            DeprecationWarning,
            stacklevel=2,
        )
        name = "trigger"
    if name == "trigger":
        from nuggetindex.extractors.trigger import TriggerExtractor

        return TriggerExtractor()

    from nuggetindex.extractors.clients.base import LLMConfig
    from nuggetindex.extractors.llm import LLMExtractor

    if name.startswith("claude-"):
        provider = "anthropic"
    elif name.startswith(("gemini-", "models/gemini-")):
        provider = "google"
    else:
        provider = "openai"
    return LLMExtractor(LLMConfig(provider=provider, model=name))


class _LoggingDocList(list[Document]):
    """List subclass that logs each iteration step to stderr.

    Used in ``--verbose`` deep mode so users see per-doc progress. The
    ``scan_index`` sampler + ``_deep_scan`` each iterate the document list
    at most once, so logging on ``__iter__`` is the right place to emit the
    ``Ingested doc N / M`` line.
    """

    def __init__(self, docs: list[Document]) -> None:
        super().__init__(docs)

    def __iter__(self) -> Iterator[Document]:
        total = len(self)
        for i, d in enumerate(super().__iter__(), 1):
            typer.echo(f"doctor: Ingested doc {i} / {total}", err=True)
            yield d


def _write_report(report: DoctorReport, out: Path) -> None:
    """Dispatch on ``out``'s extension to pick markdown vs JSON serialization.

    ``dataclasses.asdict`` walks ``DoctorScore.ci95`` (a tuple) into a JSON
    list, which is what callers expect. ``default=str`` is belt-and-braces
    for any stray datetime / Path values that might sneak into future
    dimensions; today the report contains only plain primitives + tuples.
    """
    ext = out.suffix.lower()
    if ext in {".md", ".markdown"}:
        out.write_text(report.rendered_markdown, encoding="utf-8")
        return
    if ext == ".json":
        out.write_text(
            json.dumps(dataclasses.asdict(report), indent=2, default=str),
            encoding="utf-8",
        )
        return
    raise typer.BadParameter(
        f"--report must end in .md, .markdown, or .json (got {out.suffix!r})",
        param_hint="--report",
    )


def doctor_command(
    index_path: Path = typer.Option(
        ...,
        "--index-path",
        help=(
            "Path to a .jsonl file (one document per line with "
            "'source_id'/'text') or a nuggetindex.db SQLite store."
        ),
    ),
    mode: str = typer.Option(
        "fast",
        "--mode",
        help="Scan mode: 'fast' (heuristics only) or 'deep' (uses --extractor).",
    ),
    sample_size: int = typer.Option(
        500,
        "--sample-size",
        help="Target number of documents to draw for the estimate.",
    ),
    stratify_by: str = typer.Option(
        "composite",
        "--stratify-by",
        help=(
            "Sampler stratification: 'composite' (default; language x domain),"
            " 'source_date', 'domain', 'language', or 'none'."
        ),
    ),
    dedup_near_duplicates: bool = typer.Option(
        False,
        "--dedup-near-duplicates/--no-dedup-near-duplicates",
        help=(
            "Apply a SimHash-based near-duplicate filter to the sample "
            "(64-bit hash, 3-bit Hamming threshold). Default off."
        ),
    ),
    report: Path | None = typer.Option(
        None,
        "--report",
        help=(
            "Write the report to this file (.md or .json). If omitted the "
            "Markdown report is printed to stdout."
        ),
    ),
    extractor: str = typer.Option(
        "trigger",
        "--extractor",
        help=(
            "Deep-mode extractor: 'trigger' (default; LLM-free) or a model "
            "id routed by prefix (claude-* -> Anthropic, gemini-* -> "
            "Google, otherwise OpenAI-compatible). The legacy 'rule_based' "
            "alias is accepted with a deprecation warning."
        ),
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        help="Emit per-doc progress + load counts on stderr.",
    ),
) -> None:
    """Scan any index for temporal/conflict/rename damage.

    Fast mode uses heuristics only (spaCy TIMEX + trigger-verb regex). Deep
    mode additionally routes a stratified sample through a configured
    extractor so conflict / drift signals reflect the structured output of
    the actual extraction pipeline. See the module docstring for install
    notes (spaCy model download, API-key envvars).
    """
    # --- arg validation --------------------------------------------------
    if mode not in {"fast", "deep"}:
        typer.echo(
            f"doctor: --mode must be 'fast' or 'deep' (got {mode!r}).",
            err=True,
        )
        raise typer.Exit(code=1)
    if stratify_by not in {"source_date", "none", "domain", "language", "composite"}:
        typer.echo(
            f"doctor: --stratify-by must be one of 'source_date', 'none', "
            f"'domain', 'language', 'composite' (got {stratify_by!r}).",
            err=True,
        )
        raise typer.Exit(code=1)
    if not index_path.exists():
        typer.echo(f"doctor: --index-path not found: {index_path}", err=True)
        raise typer.Exit(code=1)

    # --- load documents --------------------------------------------------
    suffix = index_path.suffix.lower()
    if suffix in {".jsonl", ".ndjson"}:
        docs = _load_jsonl_docs(index_path, verbose=verbose)
    elif suffix in {".db", ".sqlite", ".sqlite3"}:
        try:
            docs = asyncio.run(_load_sqlite_docs(index_path, verbose=verbose))
        except Exception as exc:  # noqa: BLE001 -- surface to the user
            typer.echo(
                f"doctor: failed to read SQLite store {index_path}: "
                f"{type(exc).__name__}: {exc}\n"
                "SQLite input for `doctor` reads the passages table; if your "
                "store was built without raw passages, pass a .jsonl file "
                "instead. See TODO in src/nuggetindex/cli/doctor.py",
                err=True,
            )
            raise typer.Exit(code=1) from exc
    else:
        typer.echo(
            f"doctor: unsupported --index-path extension {index_path.suffix!r}; "
            "expected .jsonl, .ndjson, .db, .sqlite, or .sqlite3.",
            err=True,
        )
        raise typer.Exit(code=1)

    # --- build extractor (deep only) ------------------------------------
    extractor_obj: Any | None = None
    if mode == "deep":
        try:
            extractor_obj = _build_extractor(extractor)
        except Exception as exc:  # noqa: BLE001 -- config / import failures
            typer.echo(
                f"doctor: failed to build extractor {extractor!r}: {type(exc).__name__}: {exc}",
                err=True,
            )
            raise typer.Exit(code=1) from exc
        if verbose:
            docs = _LoggingDocList(docs)

    # --- scan ------------------------------------------------------------
    try:
        doctor_report: DoctorReport = asyncio.run(
            scan_index(
                docs=docs,
                mode=mode,  # type: ignore[arg-type]
                sample_size=sample_size,
                stratify_by=stratify_by,  # type: ignore[arg-type]
                extractor=extractor_obj,
                dedup_near_duplicates=dedup_near_duplicates,
            )
        )
    except ValueError as exc:
        # scan_index raises ValueError for bad mode / missing extractor.
        typer.echo(f"doctor: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    # --- emit ------------------------------------------------------------
    if report is not None:
        try:
            _write_report(doctor_report, report)
        except typer.BadParameter as exc:
            typer.echo(f"doctor: {exc.message}", err=True)
            raise typer.Exit(code=1) from exc
        if verbose:
            typer.echo(f"doctor: wrote report to {report}", err=True)
    else:
        typer.echo(doctor_report.rendered_markdown)

    # --- deep-mode failure-rate exit code -------------------------------
    # The rendered markdown carries a ``> N of M ingestions failed; ...``
    # preamble when the deep-scan warning triggers (see
    # ``_render_markdown_scorecard``). Parse it so we can exit 2 per spec
    # without re-running the scan.
    if mode == "deep":
        md = doctor_report.rendered_markdown
        first = md.splitlines()[0] if md else ""
        if first.startswith("> ") and "ingestions failed" in first:
            try:
                _, rest = first.split("> ", 1)
                n_failed_str, _, rest2 = rest.partition(" of ")
                m_total_str, _, _ = rest2.partition(" ingestions")
                n_failed = int(n_failed_str)
                m_total = int(m_total_str)
            except (ValueError, IndexError):
                return
            if m_total > 0 and n_failed / m_total > _DEEP_FAILURE_EXIT_THRESHOLD:
                raise typer.Exit(code=2)


__all__ = ["doctor_command"]
