"""``nuggetindex judge-replay`` CLI tests."""

from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from nuggetindex.cli.app import app

runner = CliRunner()


def _write_log(tmp_path: Path) -> Path:
    log = tmp_path / "judge.jsonl"
    rows = [
        {
            "a_id": "x",
            "b_id": "y",
            "a_object": "Pichai",
            "b_object": "Page",
            "decision": "A_WINS",
            "rationale": "evidence supports A",
            "timestamp": "2026-01-01T00:00:00+00:00",
        },
        {
            "a_id": "z",
            "b_id": "w",
            "a_object": "X",
            "b_object": "Y",
            "decision": "GENUINELY_CONTESTED",
            "rationale": "both plausible",
            "timestamp": "2026-01-02T00:00:00+00:00",
        },
    ]
    with log.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    # blank line + malformed line should be skipped cleanly
    with log.open("a") as f:
        f.write("\n")
        f.write("{not json\n")
    return log


def test_judge_replay_prints_histogram(tmp_path: Path) -> None:
    log = _write_log(tmp_path)
    result = runner.invoke(app, ["judge-replay", str(log)])
    assert result.exit_code == 0, result.output
    assert "A_WINS" in result.stdout
    assert "GENUINELY_CONTESTED" in result.stdout
    assert "2 entries" in result.stdout


def test_judge_replay_missing_log_errors(tmp_path: Path) -> None:
    result = runner.invoke(app, ["judge-replay", str(tmp_path / "absent.jsonl")])
    assert result.exit_code != 0
