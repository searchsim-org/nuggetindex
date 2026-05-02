"""``nuggetindex review`` CLI tests."""

from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from nuggetindex.cli.app import app

runner = CliRunner()


def _write_queue(tmp_path: Path) -> Path:
    queue = tmp_path / "review.jsonl"
    rows = [
        {
            "nugget": {},
            "confidence": 0.65,
            "rationale": "",
            "source_text": "t",
            "context": "",
            "extractor": "RuleBasedExtractor",
            "timestamp": "2026-01-01T00:00:00+00:00",
        },
        {
            "nugget": {},
            "confidence": 0.80,
            "rationale": "",
            "source_text": "t",
            "context": "",
            "extractor": "LLMExtractor",
            "timestamp": "2026-01-02T00:00:00+00:00",
        },
    ]
    with queue.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    return queue


def test_review_summarizes_queue(tmp_path: Path) -> None:
    queue = _write_queue(tmp_path)
    result = runner.invoke(app, ["review", "--queue", str(queue)])
    assert result.exit_code == 0, result.output
    assert "Review queue" in result.stdout
    assert "By confidence" in result.stdout
    assert "By extractor" in result.stdout


def test_review_missing_queue_errors(tmp_path: Path) -> None:
    result = runner.invoke(app, ["review", "--queue", str(tmp_path / "absent.jsonl")])
    assert result.exit_code != 0


def test_review_empty_queue(tmp_path: Path) -> None:
    queue = tmp_path / "empty.jsonl"
    queue.write_text("")
    result = runner.invoke(app, ["review", "--queue", str(queue)])
    assert result.exit_code == 0
    assert "empty" in result.stdout.lower()
