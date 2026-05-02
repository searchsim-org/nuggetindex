"""``nuggetindex build`` / ``nuggetindex ingest`` CLI tests."""

from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from nuggetindex.cli.app import app

runner = CliRunner()


def _write_corpus(tmp_path: Path) -> Path:
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "a.txt").write_text("Sundar Pichai became CEO of Google in 2015.\n")
    (corpus / "b.md").write_text("Satya Nadella became CEO of Microsoft in 2014.\n")
    return corpus


def test_build_creates_db_with_trigger_default(tmp_path: Path) -> None:
    """The new default ``--model trigger`` builds a DB with no API key / no LLM."""
    corpus = _write_corpus(tmp_path)
    db = tmp_path / "ni.db"
    result = runner.invoke(
        app,
        ["build", str(corpus), "--db", str(db)],
    )
    assert result.exit_code == 0, result.output
    assert db.exists()
    assert "build complete" in result.stdout


def test_build_explicit_trigger_model(tmp_path: Path) -> None:
    """``--model trigger`` is the documented opt-in spelling."""
    corpus = _write_corpus(tmp_path)
    db = tmp_path / "ni.db"
    result = runner.invoke(app, ["build", str(corpus), "--db", str(db), "--model", "trigger"])
    assert result.exit_code == 0, result.output
    assert db.exists()


def test_build_missing_folder_errors(tmp_path: Path) -> None:
    missing = tmp_path / "nope"
    result = runner.invoke(app, ["build", str(missing), "--db", str(tmp_path / "x.db")])
    assert result.exit_code != 0


def test_ingest_appends_with_trigger(tmp_path: Path) -> None:
    corpus = _write_corpus(tmp_path)
    db = tmp_path / "ni.db"
    first = runner.invoke(app, ["build", str(corpus), "--db", str(db), "--model", "trigger"])
    assert first.exit_code == 0, first.output
    # Add another file + re-run via ingest
    (corpus / "c.txt").write_text("Tim Cook became CEO of Apple in 2011.\n")
    second = runner.invoke(app, ["ingest", str(corpus), "--db", str(db), "--model", "trigger"])
    assert second.exit_code == 0, second.output
    assert "ingest complete" in second.stdout


def test_build_rule_based_delegates_to_trigger(tmp_path: Path) -> None:
    """Legacy ``--model rule_based`` now delegates to the trigger extractor
    with a deprecation warning rather than hard-rejecting."""
    corpus = _write_corpus(tmp_path)
    db = tmp_path / "ni.db"
    with pytest.warns(DeprecationWarning, match="rule_based"):
        result = runner.invoke(
            app,
            ["build", str(corpus), "--db", str(db), "--model", "rule_based"],
        )
    assert result.exit_code == 0, result.output
    assert db.exists()


def test_build_dry_run_prints_estimate(tmp_path: Path) -> None:
    """``--dry-run`` skips the real ingest and prints the cost estimate."""
    corpus = _write_corpus(tmp_path)
    db = tmp_path / "ni.db"
    result = runner.invoke(
        app,
        [
            "build",
            str(corpus),
            "--db",
            str(db),
            "--model",
            "gpt-4o-mini",
            "--dry-run",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "Ingest cost estimate" in result.stdout
    # DB must not be created on a dry run.
    assert not db.exists()


def test_build_cache_flag_wraps_extractor(tmp_path: Path) -> None:
    """``--cache`` creates a SQLite cache file next to the DB."""
    corpus = _write_corpus(tmp_path)
    db = tmp_path / "ni.db"
    cache = tmp_path / "ni-cache.db"
    result = runner.invoke(
        app,
        [
            "build",
            str(corpus),
            "--db",
            str(db),
            "--model",
            "trigger",
            "--cache",
            str(cache),
        ],
    )
    assert result.exit_code == 0, result.output
    assert db.exists()
    assert cache.exists(), "cache file should be created by CachedExtractor"
