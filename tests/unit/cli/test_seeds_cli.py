"""``nuggetindex seeds propose`` CLI tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from nuggetindex.cli.app import app

runner = CliRunner()


@pytest.fixture
def jsonl_corpus(tmp_path: Path) -> Path:
    path = tmp_path / "docs.jsonl"
    lines = [
        json.dumps({"source_id": f"d{i}", "text": text})
        for i, text in enumerate(
            [
                "Larry Page became CEO of Google in 2011.",
                "Sundar Pichai became CEO of Google in 2015.",
                "Microsoft acquired LinkedIn for 26 billion.",
                "Microsoft acquired GitHub.",
                "Twitter was renamed to X in 2023.",
                "Facebook was renamed to Meta.",
            ]
        )
    ] * 3
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def test_cli_prints_seed_table(jsonl_corpus: Path) -> None:
    result = runner.invoke(
        app,
        [
            "seeds",
            "propose",
            "--index-path",
            str(jsonl_corpus),
            "--budget",
            "5",
            "--sample-size",
            "50",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "seed" in result.stdout.lower()


def test_cli_writes_json(jsonl_corpus: Path, tmp_path: Path) -> None:
    out = tmp_path / "seeds.json"
    result = runner.invoke(
        app,
        [
            "seeds",
            "propose",
            "--index-path",
            str(jsonl_corpus),
            "--budget",
            "5",
            "--sample-size",
            "50",
            "--out",
            str(out),
        ],
    )
    assert result.exit_code == 0, result.output
    data = json.loads(out.read_text())
    assert "seeds" in data and isinstance(data["seeds"], list)
