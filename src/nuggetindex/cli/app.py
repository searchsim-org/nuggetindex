"""Top-level Typer app for ``nuggetindex``.

Registers every subcommand in :mod:`nuggetindex.cli` and exposes three
global flags that every command inherits via the app callback:

* ``--log-level`` -- forwarded to stdlib logging + structlog.
* ``--no-color`` -- disables Rich color output for pipes / CI logs.
* ``--config <yaml>`` -- optional YAML config file; values are stashed on
  the ``typer.Context`` object so individual subcommands can read them.

Each subcommand is a plain function in its own module (``audit.py``,
``build.py``, ...); we register them via ``app.command(name=...)``
rather than ``add_typer`` so positional arguments work naturally.
"""

from __future__ import annotations

import logging
from pathlib import Path

import typer
from rich.console import Console

from nuggetindex.cli.audit import audit_command
from nuggetindex.cli.auto import auto_command
from nuggetindex.cli.build import build_command, ingest_command
from nuggetindex.cli.chain import chain_command
from nuggetindex.cli.diff import diff_command
from nuggetindex.cli.doctor import doctor_command
from nuggetindex.cli.estimate_cost import estimate_cost_command
from nuggetindex.cli.eval import eval_command
from nuggetindex.cli.inspect import inspect_command
from nuggetindex.cli.judge_replay import judge_replay_command
from nuggetindex.cli.query import query_command
from nuggetindex.cli.resolve import resolve_command
from nuggetindex.cli.review import review_command
from nuggetindex.cli.schema import schema_app
from nuggetindex.cli.seeds import seeds_app
from nuggetindex.cli.serve import serve_command

app = typer.Typer(
    help="nuggetindex - governed atomic-fact index for RAG",
    no_args_is_help=True,
    add_completion=False,
)

console = Console()


def _configure_logging(log_level: str) -> None:
    """Configure stdlib logging + structlog at ``log_level``."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper(), logging.INFO),
        format="%(levelname)s %(name)s: %(message)s",
    )
    try:
        import structlog

        structlog.configure(
            wrapper_class=structlog.make_filtering_bound_logger(
                getattr(logging, log_level.upper(), logging.INFO)
            ),
        )
    except ImportError:
        pass


@app.callback()
def main(
    ctx: typer.Context,
    log_level: str = typer.Option(
        "INFO",
        "--log-level",
        help="Python log level (DEBUG / INFO / WARNING / ERROR).",
    ),
    no_color: bool = typer.Option(
        False,
        "--no-color",
        help="Disable Rich color output.",
    ),
    config: Path | None = typer.Option(
        None,
        "--config",
        help="Optional YAML config file (values exposed via ctx.obj).",
    ),
) -> None:
    """Global flags callback. Runs before every subcommand."""
    _configure_logging(log_level)
    if no_color:
        console.no_color = True

    cfg: dict[str, object] = {}
    if config is not None and config.exists():
        try:
            import yaml

            cfg = yaml.safe_load(config.read_text(encoding="utf-8")) or {}
        except Exception:
            cfg = {}
    ctx.obj = {"config": cfg, "no_color": no_color, "log_level": log_level}


app.command(name="audit", help="Zero-index audit of retrieved passages.")(audit_command)
app.command(name="build", help="Build a NuggetStore from a folder of text files.")(
    build_command
)
app.command(name="ingest", help="Append-only ingest into an existing NuggetStore.")(
    ingest_command
)
app.command(name="query", help="Query a NuggetStore (hybrid retrieval).")(query_command)
app.command(
    name="chain",
    help="Walk a temporal provenance chain (succession | rename | join).",
)(chain_command)
app.command(name="inspect", help="Dump NuggetStore statistics.")(inspect_command)
app.command(name="diff", help="Diff two NuggetStore databases.")(diff_command)
app.command(
    name="judge-replay", help="Summarize an LLM-judge log file."
)(judge_replay_command)
app.command(
    name="review", help="Summarize the review queue of deferred extractions."
)(review_command)
app.command(
    name="resolve",
    help="Adjudicate Contested facts: pin a winner, suppress the losers.",
)(resolve_command)
app.command(
    name="doctor",
    help="Scan any index for temporal/conflict/rename damage.",
)(doctor_command)
app.command(
    name="estimate-cost",
    help="Preview the LLM cost / wall-time for a JSONL corpus without ingesting.",
)(estimate_cost_command)
app.command(
    name="auto",
    help="One-call facade: discovery -> seeds -> ingest -> sidecar.",
)(auto_command)
app.command(
    name="eval",
    help="Score a baseline retriever vs a sidecar-augmented one on a benchmark.",
)(eval_command)
app.command(
    name="serve",
    help="Launch the nuggetindex HTTP API (FastAPI + uvicorn).",
)(serve_command)
app.add_typer(
    seeds_app,
    name="seeds",
    help="Propose seed queries for offline nuggetindex construction.",
)
app.add_typer(
    schema_app,
    name="schema",
    help="Corpus-driven predicate discovery.",
)


if __name__ == "__main__":  # pragma: no cover - manual entry point
    app()
