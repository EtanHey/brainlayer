"""Tests for brainlayer enrich CLI routing."""

from types import SimpleNamespace
from unittest.mock import MagicMock

from typer.testing import CliRunner

from brainlayer.cli import app

runner = CliRunner()


def test_cli_enrich_mode_realtime_routes_to_controller(monkeypatch):
    monkeypatch.setattr("brainlayer.cli.get_db_path", lambda: "/tmp/test.db")
    monkeypatch.setattr("brainlayer.vector_store.VectorStore", lambda path: MagicMock())
    called = {}

    def fake_realtime(store, limit=25, since_hours=24, **kwargs):
        called.update({"store": store, "limit": limit, "since_hours": since_hours, **kwargs})
        return SimpleNamespace(mode="realtime", attempted=1, enriched=1, skipped=0, failed=0, errors=[])

    monkeypatch.setattr("brainlayer.enrichment_controller.enrich_realtime", fake_realtime)

    result = runner.invoke(app, ["enrich", "--mode", "realtime", "--limit", "9", "--since-hours", "12"])

    assert result.exit_code == 0
    assert called["limit"] == 9
    assert called["since_hours"] == 12


def test_cli_enrich_mode_batch_routes_to_controller(monkeypatch):
    monkeypatch.setattr("brainlayer.cli.get_db_path", lambda: "/tmp/test.db")
    monkeypatch.setattr("brainlayer.vector_store.VectorStore", lambda path: MagicMock())
    called = {}

    def fake_batch(store, phase="run", limit=5000, **kwargs):
        called.update({"phase": phase, "limit": limit, **kwargs})
        return SimpleNamespace(mode="batch", attempted=0, enriched=0, skipped=0, failed=0, errors=[])

    monkeypatch.setattr("brainlayer.enrichment_controller.enrich_batch", fake_batch)

    result = runner.invoke(app, ["enrich", "--mode", "batch", "--phase", "poll", "--limit", "50"])

    assert result.exit_code == 0
    assert called["phase"] == "poll"
    assert called["limit"] == 50


def test_cli_enrich_mode_local_routes_to_controller(monkeypatch):
    monkeypatch.setattr("brainlayer.cli.get_db_path", lambda: "/tmp/test.db")
    monkeypatch.setattr("brainlayer.vector_store.VectorStore", lambda path: MagicMock())
    called = {}

    def fake_local(store, limit=100, parallel=2, backend="mlx"):
        called.update({"limit": limit, "parallel": parallel, "backend": backend})
        return SimpleNamespace(mode="local", attempted=1, enriched=1, skipped=0, failed=0, errors=[])

    monkeypatch.setattr("brainlayer.enrichment_controller.enrich_local", fake_local)

    result = runner.invoke(
        app,
        ["enrich", "--mode", "local", "--limit", "8", "--parallel", "3", "--backend", "mlx"],
    )

    assert result.exit_code == 0
    assert called == {"limit": 8, "parallel": 3, "backend": "mlx"}


def test_cli_enrich_stats_prints_progress(monkeypatch):
    store = MagicMock()
    store.get_enrichment_stats.return_value = {
        "total_chunks": 10,
        "enriched": 4,
        "percent": 40.0,
        "remaining": 6,
        "by_intent": {},
    }
    monkeypatch.setattr("brainlayer.cli.get_db_path", lambda: "/tmp/test.db")
    monkeypatch.setattr("brainlayer.vector_store.VectorStore", lambda path: store)

    result = runner.invoke(app, ["enrich", "--stats"])

    assert result.exit_code == 0
    assert "Total:" in result.stdout
    assert "Enriched:" in result.stdout
    assert "Remaining:" in result.stdout


def test_cli_enrich_invalid_mode_rejected():
    result = runner.invoke(app, ["enrich", "--mode", "wrong"])

    assert result.exit_code != 0
