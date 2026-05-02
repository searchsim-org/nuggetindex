"""Global CLI wiring: root --help, subcommand --help, global flags."""

from __future__ import annotations

from typer.testing import CliRunner

from nuggetindex.cli.app import app

runner = CliRunner()


def test_root_help_exits_zero_and_lists_all_subcommands() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    for name in (
        "audit",
        "build",
        "ingest",
        "query",
        "inspect",
        "diff",
        "judge-replay",
        "review",
    ):
        assert name in result.stdout


def test_root_no_args_shows_help() -> None:
    """``no_args_is_help=True`` on the root app means bare invocation prints help."""
    result = runner.invoke(app, [])
    # Typer returns a non-zero exit when printing help via ``no_args_is_help``.
    assert "Usage" in result.stdout or "Usage" in (result.stderr or "")


def test_global_flags_accepted() -> None:
    """Global flags must parse cleanly even when followed by --help."""
    result = runner.invoke(app, ["--log-level", "DEBUG", "--no-color", "--help"])
    assert result.exit_code == 0


def test_each_subcommand_help() -> None:
    for name in (
        "audit",
        "build",
        "ingest",
        "query",
        "inspect",
        "diff",
        "judge-replay",
        "review",
    ):
        result = runner.invoke(app, [name, "--help"])
        assert result.exit_code == 0, f"{name} --help returned {result.exit_code}"
