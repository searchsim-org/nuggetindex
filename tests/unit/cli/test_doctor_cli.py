"""``nuggetindex doctor`` CLI tests."""

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
        json.dumps({"source_id": "d1", "text": "Microsoft acquired LinkedIn for $26.2 billion."}),
        json.dumps({"source_id": "d2", "text": "Microsoft acquired LinkedIn for $26.4 billion."}),
        json.dumps({"source_id": "d3", "text": "Twitter Inc. was renamed to X Corp. in 2023."}),
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def test_doctor_cli_fast_mode_prints_report(jsonl_corpus: Path) -> None:
    result = runner.invoke(app, ["doctor", "--index-path", str(jsonl_corpus), "--mode", "fast"])
    assert result.exit_code == 0, result.output
    assert "doctor scan" in result.stdout.lower()
    assert "verdict" in result.stdout.lower()


def test_doctor_cli_fast_mode_writes_markdown(jsonl_corpus: Path, tmp_path: Path) -> None:
    out = tmp_path / "report.md"
    result = runner.invoke(
        app,
        ["doctor", "--index-path", str(jsonl_corpus), "--mode", "fast", "--report", str(out)],
    )
    assert result.exit_code == 0, result.output
    assert out.exists()
    assert "doctor scan" in out.read_text().lower()


def test_doctor_cli_fast_mode_writes_json(jsonl_corpus: Path, tmp_path: Path) -> None:
    out = tmp_path / "report.json"
    result = runner.invoke(
        app,
        ["doctor", "--index-path", str(jsonl_corpus), "--mode", "fast", "--report", str(out)],
    )
    assert result.exit_code == 0, result.output
    data = json.loads(out.read_text())
    assert data["sample_mode"] == "fast"
    assert "scores" in data and len(data["scores"]) == 4
    assert data["verdict"] in {"low", "medium", "high"}


def test_doctor_cli_deep_mode_default_is_trigger(jsonl_corpus: Path) -> None:
    """The new deep-mode default (``--extractor trigger``) runs offline."""
    result = runner.invoke(
        app, ["doctor", "--index-path", str(jsonl_corpus), "--mode", "deep"]
    )
    assert result.exit_code == 0, result.output


def test_doctor_cli_deep_mode_rule_based_delegates_to_trigger(
    jsonl_corpus: Path,
) -> None:
    """Legacy ``--extractor rule_based`` now delegates to TriggerExtractor
    with a deprecation warning (softened from the old hard rejection)."""
    with pytest.warns(DeprecationWarning, match="rule_based"):
        result = runner.invoke(
            app,
            [
                "doctor",
                "--index-path",
                str(jsonl_corpus),
                "--mode",
                "deep",
                "--extractor",
                "rule_based",
            ],
        )
    assert result.exit_code == 0, result.output


def test_doctor_cli_missing_index_path_errors() -> None:
    result = runner.invoke(app, ["doctor", "--mode", "fast"])
    assert result.exit_code != 0
