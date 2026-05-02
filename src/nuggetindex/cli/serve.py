"""`nuggetindex serve` CLI entry point.

Thin Typer command that launches uvicorn against ``nuggetindex.serve.create_app``.
"""
from __future__ import annotations

from pathlib import Path

import typer


def serve_command(
    db_path: Path = typer.Option(..., "--db", help="Path to the NuggetStore SQLite file."),
    host: str = typer.Option("127.0.0.1", "--host"),
    port: int = typer.Option(8080, "--port"),
    mode: str = typer.Option(
        "offline-curated", "--mode",
        help="Sidecar mode: 'offline-curated' or 'just-in-time'.",
    ),
    reload: bool = typer.Option(False, "--reload", help="Auto-reload on code changes (dev only)."),
) -> None:
    """Launch the nuggetindex HTTP API."""
    try:
        import uvicorn
    except ImportError as exc:  # pragma: no cover
        raise typer.BadParameter(
            "uvicorn is not installed. Install the [serve] extra: "
            "pip install 'nuggetindex[serve]'",
        ) from exc
    from nuggetindex.serve import create_app

    app = create_app(db_path=db_path, mode=mode)
    uvicorn.run(app, host=host, port=port, reload=reload)
