from typer.testing import CliRunner

from nuggetindex.cli.app import app

runner = CliRunner()


def test_serve_help_shows_flags():
    result = runner.invoke(app, ["serve", "--help"])
    assert result.exit_code == 0
    assert "--db" in result.stdout
    assert "--host" in result.stdout
    assert "--port" in result.stdout
    assert "--mode" in result.stdout


def test_serve_requires_db():
    result = runner.invoke(app, ["serve"])
    assert result.exit_code != 0
