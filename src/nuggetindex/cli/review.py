"""``nuggetindex review`` -- summarize the deferred-extractions queue."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

console = Console()


def _bucket(confidence: float) -> str:
    """Classify a confidence into coarse bands for the histogram."""
    if confidence < 0.5:
        return "<0.50"
    if confidence < 0.7:
        return "0.50-0.70"
    if confidence < 0.85:
        return "0.70-0.85"
    return ">=0.85"


def review_command(
    queue: Path = typer.Option(
        Path("review_queue.jsonl"),
        "--queue",
        help="Path to the review-queue JSONL file.",
    ),
) -> None:
    """Summarize the queue by confidence bucket + extractor."""
    if not queue.exists():
        typer.echo(f"Error: review queue not found: {queue}", err=True)
        raise typer.Exit(code=2)

    rows: list[dict[str, object]] = []
    with queue.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    if not rows:
        console.print("[yellow]Review queue is empty.[/yellow]")
        return

    by_bucket: Counter[str] = Counter()
    by_extractor: Counter[str] = Counter()
    for r in rows:
        try:
            conf = float(r.get("confidence", 0.0))  # type: ignore[arg-type]
        except (TypeError, ValueError):
            conf = 0.0
        by_bucket[_bucket(conf)] += 1
        by_extractor[str(r.get("extractor", "unknown"))] += 1

    console.print(f"[bold]Review queue:[/bold] {len(rows)} deferred entries in {queue}")

    bucket_table = Table(title="By confidence")
    bucket_table.add_column("bucket", style="cyan")
    bucket_table.add_column("count", justify="right")
    for bucket in ["<0.50", "0.50-0.70", "0.70-0.85", ">=0.85"]:
        if by_bucket[bucket]:
            bucket_table.add_row(bucket, str(by_bucket[bucket]))
    console.print(bucket_table)

    extractor_table = Table(title="By extractor")
    extractor_table.add_column("extractor", style="cyan")
    extractor_table.add_column("count", justify="right")
    for name, count in by_extractor.most_common():
        extractor_table.add_row(name, str(count))
    console.print(extractor_table)
