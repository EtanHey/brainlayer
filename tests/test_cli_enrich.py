"""Tests for brainlayer enrichment and maintenance CLI routing."""

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


def test_cli_enrich_mode_batch_submit_routes_to_cloud_backfill(monkeypatch):
    monkeypatch.setattr("brainlayer.cli.get_db_path", lambda: "/tmp/test.db")
    called = {}

    def fake_backfill(db_path, model, dry_run=False, sample=0, no_sanitize=False, submit_only=False):
        called.update(
            {
                "db_path": str(db_path),
                "model": model,
                "dry_run": dry_run,
                "sample": sample,
                "no_sanitize": no_sanitize,
                "submit_only": submit_only,
            }
        )

    monkeypatch.setattr("brainlayer.cloud_backfill.run_full_backfill", fake_backfill)

    result = runner.invoke(app, ["enrich", "--mode", "batch", "--phase", "submit", "--limit", "50"])

    assert result.exit_code == 0
    assert called["db_path"] == "/tmp/test.db"
    assert called["sample"] == 50
    assert called["submit_only"] is True


def test_cli_enrich_mode_batch_submit_defaults_to_full_batch(monkeypatch):
    monkeypatch.setattr("brainlayer.cli.get_db_path", lambda: "/tmp/test.db")
    called = {}

    def fake_backfill(db_path, model, dry_run=False, sample=0, no_sanitize=False, submit_only=False):
        called.update(
            {
                "db_path": str(db_path),
                "model": model,
                "dry_run": dry_run,
                "sample": sample,
                "no_sanitize": no_sanitize,
                "submit_only": submit_only,
            }
        )

    monkeypatch.setattr("brainlayer.cloud_backfill.run_full_backfill", fake_backfill)

    result = runner.invoke(app, ["enrich", "--mode", "batch", "--phase", "submit"])

    assert result.exit_code == 0
    assert called["db_path"] == "/tmp/test.db"
    assert called["sample"] == 0
    assert called["submit_only"] is True


def test_cli_enrich_mode_local_is_rejected():
    result = runner.invoke(app, ["enrich", "--mode", "local"])

    assert result.exit_code != 0
    assert "Invalid mode: local" in result.stdout


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


def test_cli_decay_routes_to_decay_job(monkeypatch):
    called = {}

    def fake_decay_job(db_path, dry_run=False, batch_size=10_000):
        called.update({"db_path": str(db_path), "dry_run": dry_run, "batch_size": batch_size})
        return {
            "rows_processed": 10,
            "archived_rows": 3,
            "pinned_rows": 1,
            "average_decay": 0.25,
            "duration_seconds": 2.5,
            "dry_run": dry_run,
        }

    monkeypatch.setattr("brainlayer.cli.get_db_path", lambda: "/tmp/test.db")
    monkeypatch.setattr("brainlayer.decay_job.run_decay_job", fake_decay_job)

    result = runner.invoke(app, ["decay", "--dry-run", "--batch-size", "50"])

    assert result.exit_code == 0
    assert called == {"db_path": "/tmp/test.db", "dry_run": True, "batch_size": 50}
    assert "Decay job complete" in result.stdout


def test_cli_wal_checkpoint_routes_to_helper(monkeypatch):
    called = {}

    def fake_run(mode):
        called["mode"] = mode
        return {
            "db": "/tmp/test.db",
            "mode": mode,
            "wal_before": "1.0MB",
            "wal_after": "0.0B",
            "wal_before_bytes": 1024 * 1024,
            "wal_after_bytes": 0,
            "busy": 0,
            "log_pages": 10,
            "checkpointed_pages": 10,
        }

    monkeypatch.setattr("brainlayer.wal_checkpoint.run_wal_checkpoint", fake_run)

    result = runner.invoke(app, ["wal-checkpoint", "--mode", "truncate"])

    assert result.exit_code == 0
    assert called == {"mode": "TRUNCATE"}
    assert "Checkpoint (TRUNCATE): 10/10 pages" in result.stdout
