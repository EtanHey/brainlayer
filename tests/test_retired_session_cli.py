"""Session history remains available without a session-enrichment command."""

from typer.testing import CliRunner

from brainlayer import cli

runner = CliRunner()


def test_cli_does_not_advertise_retired_session_enrichment():
    result = runner.invoke(cli.app, ["--help"])
    assert result.exit_code == 0
    assert "enrich-sessions" not in result.stdout
    assert "git-overlay" in result.stdout


def test_retired_session_command_fails_before_database_or_backend(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("retired command opened a database")

    monkeypatch.setattr(cli, "get_db_path", forbidden)
    monkeypatch.setattr("brainlayer.paths.get_db_path", forbidden)
    monkeypatch.setattr("brainlayer.vector_store.VectorStore", forbidden)
    result = runner.invoke(cli.app, ["enrich-sessions"])
    assert result.exit_code == 2, result.output
    assert "No such command" in result.output
