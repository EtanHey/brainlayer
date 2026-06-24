from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

from typer.testing import CliRunner

from brainlayer.vector_store import VectorStore

NOW = datetime(2026, 6, 22, 12, 0, tzinfo=UTC)


def _loaded_launchctl(args: list[str]) -> SimpleNamespace:
    assert args[:2] == ["launchctl", "print"]
    return SimpleNamespace(returncode=0, stdout="", stderr="")


def _hotlane_ps() -> str:
    return "123 /usr/bin/python scripts/hotlane_brainbar_daemon.py --interval 1 --backlog-batch 128\n"


def _insert_chunk(
    store: VectorStore,
    chunk_id: str,
    *,
    content: str | None = None,
    created_at: str = "2026-06-22T10:00:00+00:00",
) -> None:
    text = content or f"doctor fixture content {chunk_id}"
    cursor = store.conn.cursor()
    cursor.execute(
        """
        INSERT INTO chunks (
            id, content, metadata, source_file, project, content_type,
            value_type, char_count, source, created_at, enriched_at, enrich_status,
            summary, tags, importance, chunk_origin, seen_count, last_seen_at,
            content_class
        ) VALUES (?, ?, '{}', 'doctor-fixture.jsonl', 'doctor-fixture', 'note',
            'HIGH', ?, 'test', ?, NULL, NULL,
            NULL, NULL, NULL, 'raw', 1, ?,
            'human-authored')
        """,
        (chunk_id, text, len(text), created_at, created_at),
    )


def _insert_vector(store: VectorStore, chunk_id: str, first: float) -> None:
    store._upsert_chunk_vector(store.conn.cursor(), chunk_id, [first] + [0.0] * 1023)


def _build_db(path: Path, *, missing_recent_vector: bool = False, fts_desync: bool = False) -> None:
    store = VectorStore(path)
    try:
        _insert_chunk(store, "doctor-healthy-1", content="doctor alpha healthy searchable content")
        _insert_chunk(store, "doctor-healthy-2", content="doctor beta healthy searchable content")
        _insert_vector(store, "doctor-healthy-1", 1.0)
        _insert_vector(store, "doctor-healthy-2", 2.0)
        if missing_recent_vector:
            _insert_chunk(
                store,
                "doctor-recent-missing-vector",
                content="doctor recent chunk with no vector should fail the D2 gate",
            )
        if fts_desync:
            store.conn.cursor().execute("DELETE FROM chunks_fts WHERE chunk_id = 'doctor-healthy-1'")
        store.conn.cursor().execute("PRAGMA wal_checkpoint(TRUNCATE)")
    finally:
        store.close()


def _add_enrichment_backlog(path: Path) -> None:
    store = VectorStore(path)
    try:
        _insert_chunk(
            store,
            "doctor-enrichment-backlog",
            content="doctor enrichment backlog content long enough to require realtime Gemini enrichment",
        )
        _insert_vector(store, "doctor-enrichment-backlog", 3.0)
        store.conn.cursor().execute("PRAGMA wal_checkpoint(TRUNCATE)")
    finally:
        store.close()


def _write_drain_health(path: Path, *, updated_at: datetime, drained_total: int = 0) -> None:
    path.write_text(
        json.dumps(
            {
                "drain_cycles": 3,
                "drained_total": drained_total,
                "updated_at": updated_at.isoformat(),
            }
        ),
        encoding="utf-8",
    )


def _write_daily_cap_reached(cost_dir: Path, *, now: datetime, spent_usd: float = 5.0) -> None:
    cost_dir.mkdir()
    (cost_dir / "enrich-daily-cost.json").write_text(
        json.dumps({"date": now.astimezone().date().isoformat(), "spent_usd": spent_usd}),
        encoding="utf-8",
    )


def _doctor_config(tmp_path: Path, db_path: Path):
    from brainlayer.doctor import DoctorConfig

    queue_dir = tmp_path / "queue"
    queue_dir.mkdir()
    watcher_health_path = tmp_path / "watcher-health.json"
    drain_health_path = tmp_path / "drain-health.json"
    watcher_health_path.write_text(json.dumps({"poll_count": 12}), encoding="utf-8")
    drain_health_path.write_text(json.dumps({"drained_total": 34}), encoding="utf-8")
    return DoctorConfig(
        db_path=db_path,
        queue_dir=queue_dir,
        watcher_health_path=watcher_health_path,
        drain_health_path=drain_health_path,
        queue_movement_sample_seconds=0,
    )


def test_run_doctor_exits_zero_on_healthy_fixture(tmp_path):
    from brainlayer.doctor import run_doctor

    db_path = tmp_path / "healthy.db"
    _build_db(db_path)

    result = run_doctor(
        _doctor_config(tmp_path, db_path),
        ps_output_fn=_hotlane_ps,
        command_runner=_loaded_launchctl,
        now_fn=lambda: NOW,
    )

    assert result.exit_code == 0
    assert result.ok is True
    assert not [issue for issue in result.issues if issue.severity == "fatal"]


def test_run_doctor_exits_nonzero_for_recent_unvectored_chunk(tmp_path):
    from brainlayer.doctor import run_doctor

    db_path = tmp_path / "missing-vector.db"
    _build_db(db_path, missing_recent_vector=True)

    result = run_doctor(
        _doctor_config(tmp_path, db_path),
        ps_output_fn=_hotlane_ps,
        command_runner=_loaded_launchctl,
        now_fn=lambda: NOW,
    )

    assert result.exit_code == 1
    assert result.ok is False
    assert any(issue.code == "recent_unvectored_chunks" for issue in result.issues)


def test_run_doctor_reports_fts_desync_without_rebuilding(tmp_path):
    from brainlayer.doctor import run_doctor

    db_path = tmp_path / "fts-desync.db"
    _build_db(db_path, fts_desync=True)

    result = run_doctor(
        _doctor_config(tmp_path, db_path),
        ps_output_fn=_hotlane_ps,
        command_runner=_loaded_launchctl,
        now_fn=lambda: NOW,
    )

    assert result.exit_code == 1
    assert any(issue.code == "fts5_desync" for issue in result.issues)

    store = VectorStore(db_path)
    try:
        assert store.conn.cursor().execute("SELECT COUNT(*) FROM chunks_fts").fetchone()[0] == 1
    finally:
        store.close()


def test_run_doctor_exits_nonzero_when_queue_backlog_is_not_moving(tmp_path):
    from brainlayer.doctor import run_doctor

    db_path = tmp_path / "queue-stalled.db"
    _build_db(db_path)
    config = _doctor_config(tmp_path, db_path)
    (config.queue_dir / "pending.jsonl").write_text('{"kind":"watcher_chunk"}\n', encoding="utf-8")

    result = run_doctor(
        config,
        ps_output_fn=_hotlane_ps,
        command_runner=_loaded_launchctl,
        now_fn=lambda: NOW,
    )

    assert result.exit_code == 1
    assert any(issue.code == "queue_not_moving_with_backlog" for issue in result.issues)


def test_run_doctor_fails_loudly_when_loaded_drain_heartbeat_stale_with_backlog_without_quota(tmp_path, monkeypatch):
    from brainlayer.doctor import run_doctor

    monkeypatch.setenv("BRAINLAYER_ENRICH_COST_DIR", str(tmp_path / "cost"))
    monkeypatch.setenv("BRAINLAYER_ENRICH_DAILY_USD_CAP", "5.0")
    db_path = tmp_path / "drain-liveness-stalled.db"
    _build_db(db_path)
    _add_enrichment_backlog(db_path)
    config = _doctor_config(tmp_path, db_path)
    _write_drain_health(config.drain_health_path, updated_at=NOW - timedelta(minutes=10))

    result = run_doctor(
        config,
        ps_output_fn=_hotlane_ps,
        command_runner=_loaded_launchctl,
        now_fn=lambda: NOW,
    )

    issue = next(issue for issue in result.issues if issue.code == "drain_liveness_stalled")
    assert result.exit_code == 1
    assert issue.severity == "fatal"
    assert "DRAIN_LIVENESS_STALLED" in issue.message
    assert issue.details["backlog_count"] == 1
    assert issue.details["drain_label"] == "com.brainlayer.drain"


def test_run_doctor_does_not_fail_drain_liveness_when_heartbeat_is_advancing(tmp_path, monkeypatch):
    from brainlayer.doctor import run_doctor

    monkeypatch.setenv("BRAINLAYER_ENRICH_COST_DIR", str(tmp_path / "cost"))
    monkeypatch.setenv("BRAINLAYER_ENRICH_DAILY_USD_CAP", "5.0")
    db_path = tmp_path / "drain-liveness-moving.db"
    _build_db(db_path)
    _add_enrichment_backlog(db_path)
    config = _doctor_config(tmp_path, db_path)
    _write_drain_health(config.drain_health_path, updated_at=NOW - timedelta(seconds=30))

    result = run_doctor(
        config,
        ps_output_fn=_hotlane_ps,
        command_runner=_loaded_launchctl,
        now_fn=lambda: NOW,
    )

    assert result.exit_code == 0
    assert result.ok is True
    assert not [issue for issue in result.issues if issue.code == "drain_liveness_stalled"]
    assert any(issue.code == "enrichment_idle_with_backlog" for issue in result.issues)


def test_run_doctor_keeps_loaded_but_idle_warning_only_when_daily_cap_blocks_enrichment(tmp_path, monkeypatch):
    from brainlayer.doctor import run_doctor

    cost_dir = tmp_path / "cost"
    monkeypatch.setenv("BRAINLAYER_ENRICH_COST_DIR", str(cost_dir))
    monkeypatch.setenv("BRAINLAYER_ENRICH_DAILY_USD_CAP", "5.0")
    _write_daily_cap_reached(cost_dir, now=NOW)
    db_path = tmp_path / "drain-liveness-quota.db"
    _build_db(db_path)
    _add_enrichment_backlog(db_path)
    config = _doctor_config(tmp_path, db_path)
    _write_drain_health(config.drain_health_path, updated_at=NOW - timedelta(minutes=10))

    result = run_doctor(
        config,
        ps_output_fn=_hotlane_ps,
        command_runner=_loaded_launchctl,
        now_fn=lambda: NOW,
    )

    assert result.exit_code == 0
    assert result.ok is True
    assert not [issue for issue in result.issues if issue.severity == "fatal"]
    assert any(issue.code == "enrichment_idle_with_backlog" for issue in result.issues)
    assert any(issue.code == "drain_liveness_quota_blocked" for issue in result.issues)


def test_doctor_cli_is_registered_with_json_and_db_options():
    from typer.main import get_command

    from brainlayer.cli import app

    result = CliRunner().invoke(app, ["doctor", "--help"])

    assert result.exit_code == 0
    doctor_command = get_command(app).commands["doctor"]
    option_decls = {opt for param in doctor_command.params for opt in getattr(param, "opts", [])}
    assert "--json" in option_decls
    assert "--db" in option_decls


def test_doctor_cli_json_uses_injected_runner_and_db_option(tmp_path, monkeypatch):
    import brainlayer.cli as cli
    from brainlayer.cli import app

    db_path = tmp_path / "cli-fixture.db"
    seen: dict[str, Path] = {}

    class FakeResult:
        ok = True
        exit_code = 0
        chunk_count = 2
        recent_unvectored_chunks = 0
        queue_count = 0
        issues = []

        def to_dict(self) -> dict:
            return {"ok": self.ok, "exit_code": self.exit_code, "db_path": str(seen["db_path"])}

    def fake_run(config):
        seen["db_path"] = config.db_path
        return FakeResult()

    monkeypatch.setattr(cli, "_run_doctor_cli", fake_run)

    result = CliRunner().invoke(app, ["doctor", "--json", "--db", str(db_path)])

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["ok"] is True
    assert payload["db_path"] == str(db_path)
