"""``nuggetindex schema discover`` CLI tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml
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
                "Satya Nadella became CEO of Microsoft in 2014.",
                "Microsoft acquired LinkedIn for 26 billion.",
                "Microsoft acquired GitHub.",
                "Twitter was renamed to X in 2023.",
                "Facebook was renamed to Meta.",
            ]
        )
    ] * 3
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def test_cli_prints_yaml_to_stdout(jsonl_corpus: Path) -> None:
    result = runner.invoke(
        app,
        [
            "schema",
            "discover",
            "--index",
            str(jsonl_corpus),
            "--sample-size",
            "50",
            "--min-frequency",
            "2",
        ],
    )
    assert result.exit_code == 0, result.output
    # YAML is emitted on stdout; markdown report on stderr.
    assert "predicates" in result.stdout


def test_cli_writes_yaml_out(jsonl_corpus: Path, tmp_path: Path) -> None:
    out = tmp_path / "proposal.yaml"
    result = runner.invoke(
        app,
        [
            "schema",
            "discover",
            "--index",
            str(jsonl_corpus),
            "--sample-size",
            "50",
            "--min-frequency",
            "2",
            "--out",
            str(out),
        ],
    )
    assert result.exit_code == 0, result.output
    assert out.exists()
    parsed = yaml.safe_load(out.read_text())
    assert "predicates" in parsed
    assert isinstance(parsed["predicates"], dict)


def test_cli_writes_markdown_report(jsonl_corpus: Path, tmp_path: Path) -> None:
    out = tmp_path / "proposal.yaml"
    report = tmp_path / "report.md"
    result = runner.invoke(
        app,
        [
            "schema",
            "discover",
            "--index",
            str(jsonl_corpus),
            "--sample-size",
            "50",
            "--min-frequency",
            "2",
            "--out",
            str(out),
            "--report",
            str(report),
        ],
    )
    assert result.exit_code == 0, result.output
    assert report.exists()
    md = report.read_text()
    assert md.startswith("# Schema proposal")


def test_cli_errors_on_missing_index(tmp_path: Path) -> None:
    missing = tmp_path / "nope.jsonl"
    result = runner.invoke(
        app,
        ["schema", "discover", "--index", str(missing)],
    )
    assert result.exit_code == 1
    assert "not found" in (result.stdout + result.stderr).lower()


def test_cli_errors_on_bad_extension(tmp_path: Path) -> None:
    bad = tmp_path / "docs.txt"
    bad.write_text("irrelevant\n", encoding="utf-8")
    result = runner.invoke(
        app,
        ["schema", "discover", "--index", str(bad)],
    )
    assert result.exit_code == 1
    assert "unsupported" in (result.stdout + result.stderr).lower()
