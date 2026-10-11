"""Tests for brainlayer enrichment and maintenance CLI routing."""

import json
from types import SimpleNamespace
from unittest.mock import MagicMock

from typer.testing import CliRunner

from brainlayer.cli import app
from tests.retirement_helpers import forbid_enrichment_controller

runner = CliRunner()


def _assert_enrichment_retired(result):
    assert result.exit_code == 1
    assert "Enrichment has been retired" in result.stdout


def test_cli_enrich_mode_realtime_does_not_call_controller(monkeypatch):
    monkeypatch.setattr("brainlayer.cli.get_db_path", lambda: "/tmp/test.db")
    monkeypatch.setattr("brainlayer.vector_store.VectorStore", lambda path: MagicMock())
    called = {}

    forbid_enrichment_controller(monkeypatch)

    result = runner.invoke(app, ["enrich", "--mode", "realtime", "--limit", "9", "--since-hours", "12"])

    _assert_enrichment_retired(result)
    assert called == {}


def test_cli_provenance_sweep_routes_to_answer_leg(monkeypatch):
    monkeypatch.setattr("brainlayer.cli.get_db_path", lambda: "/tmp/test.db")
    monkeypatch.setattr("brainlayer.vector_store.VectorStore", lambda path: MagicMock())
    called = {}

    def fake_sweep(store, limit=100, enable_operational_evidence=False):
        called.update(
            {
                "store": store,
                "limit": limit,
                "enable_operational_evidence": enable_operational_evidence,
            }
        )
        return SimpleNamespace(swept=2, superseded_count=1, pending_confirm_count=1, skipped_personal_count=0)

    monkeypatch.setattr("brainlayer.provenance_integration.sweep_provenance_queue", fake_sweep)

    result = runner.invoke(app, ["provenance", "sweep", "--limit", "2", "--operational-evidence"])

    assert result.exit_code == 0
    assert called["limit"] == 2
    assert called["enable_operational_evidence"] is True
    assert "swept=2" in result.stdout
    assert "superseded=1" in result.stdout


def test_cli_provenance_pending_lists_confirm_and_reject_actions(monkeypatch):
    monkeypatch.setattr("brainlayer.cli.get_db_path", lambda: "/tmp/test.db")
    monkeypatch.setattr("brainlayer.vector_store.VectorStore", lambda path: MagicMock())
    monkeypatch.setattr(
        "brainlayer.provenance_integration.list_pending_confirm",
        lambda store: [
            {
                "id": "pending-control",
                "entity": "controlLayer",
                "attribute": "ARBITRATION",
                "value": "CONTROLLAYER_DECIDES",
                "chunk_id": "c-infer",
                "provenance_class": "AGENT-INFERENCE",
                "reason": "test reason",
                "created_at": "2026-06-01T00:00:00Z",
            }
        ],
    )

    result = runner.invoke(app, ["provenance", "pending"])

    assert result.exit_code == 0
    assert (
        "pending-control · controlLayer · ARBITRATION · CONTROLLAYER_DECIDES · chunk=c-infer · confirm|reject"
        in result.stdout
    )


def test_cli_enrich_supervisor_does_not_call_controller(monkeypatch):
    monkeypatch.setattr("brainlayer.cli.get_db_path", lambda: "/tmp/test.db")
    called = {}

    forbid_enrichment_controller(monkeypatch)

    result = runner.invoke(app, ["enrich", "--mode", "realtime", "--supervisor"])

    _assert_enrichment_retired(result)
    assert called == {}


def test_cli_enrich_supervisor_rejects_explicit_since_hours(monkeypatch):
    monkeypatch.setattr("brainlayer.cli.get_db_path", lambda: "/tmp/test.db")
    called = {}

    forbid_enrichment_controller(monkeypatch)

    result = runner.invoke(app, ["enrich", "--mode", "realtime", "--supervisor", "--since-hours", "8760"])

    _assert_enrichment_retired(result)
    assert called == {}


def test_cli_enrich_supervisor_does_not_install_signal_handlers(monkeypatch):
    monkeypatch.setattr("brainlayer.cli.get_db_path", lambda: "/tmp/test.db")
    called = {}

    forbid_enrichment_controller(monkeypatch)

    result = runner.invoke(app, ["enrich", "--mode", "realtime", "--supervisor"])

    _assert_enrichment_retired(result)
    assert called == {}


def test_cli_enrich_mode_batch_submit_does_not_call_cloud_backfill(monkeypatch):
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

    monkeypatch.setattr("brainlayer.cloud_backfill.run_full_backfill", fake_backfill, raising=False)

    result = runner.invoke(app, ["enrich", "--mode", "batch", "--phase", "submit", "--limit", "50"])

    _assert_enrichment_retired(result)
    assert called == {}


def test_cli_enrich_mode_batch_submit_does_not_submit_full_batch(monkeypatch):
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

    monkeypatch.setattr("brainlayer.cloud_backfill.run_full_backfill", fake_backfill, raising=False)

    result = runner.invoke(app, ["enrich", "--mode", "batch", "--phase", "submit"])

    _assert_enrichment_retired(result)
    assert called == {}


def test_cli_enrich_mode_batch_drain_submit_does_not_call_cloud_backfill(monkeypatch):
    monkeypatch.setattr("brainlayer.cli.get_db_path", lambda: "/tmp/test.db")
    called = {}

    def fake_backfill(
        db_path,
        model,
        dry_run=False,
        sample=0,
        no_sanitize=False,
        submit_only=False,
        drain_backlog=False,
    ):
        called.update(
            {
                "db_path": str(db_path),
                "model": model,
                "dry_run": dry_run,
                "sample": sample,
                "no_sanitize": no_sanitize,
                "submit_only": submit_only,
                "drain_backlog": drain_backlog,
            }
        )

    monkeypatch.setattr("brainlayer.cloud_backfill.run_full_backfill", fake_backfill, raising=False)

    result = runner.invoke(app, ["enrich", "--mode", "batch", "--phase", "drain-submit", "--limit", "50"])

    _assert_enrichment_retired(result)
    assert called == {}


def test_cli_enrich_mode_local_is_rejected():
    result = runner.invoke(app, ["enrich", "--mode", "local"])

    _assert_enrichment_retired(result)


def test_cli_enrich_stats_does_not_read_progress(monkeypatch):
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

    _assert_enrichment_retired(result)
    store.get_enrichment_stats.assert_not_called()


def test_cli_enrich_invalid_mode_rejected():
    result = runner.invoke(app, ["enrich", "--mode", "wrong"])

    _assert_enrichment_retired(result)


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


def test_cli_wal_checkpoint_reports_guard_timeout_as_json_error(monkeypatch):
    from brainlayer.wal_checkpoint import CheckpointGuardTimeout

    def fake_run(_mode):
        raise CheckpointGuardTimeout("checkpoint guard acquisition timed out after 10.0s")

    monkeypatch.setattr("brainlayer.wal_checkpoint.run_wal_checkpoint", fake_run)

    result = runner.invoke(app, ["wal-checkpoint", "--json"])

    assert result.exit_code == 1
    assert json.loads(result.stdout) == {"error": "checkpoint guard acquisition timed out after 10.0s"}


def test_retired_enrich_is_hidden_and_does_not_resolve_database(monkeypatch):
    from typer.main import get_command

    def forbidden_db():
        raise AssertionError("retired command resolved a DB")

    monkeypatch.setattr("brainlayer.cli.get_db_path", forbidden_db)
    assert get_command(app).commands["enrich"].hidden is True
    _assert_enrichment_retired(runner.invoke(app, ["enrich"]))
