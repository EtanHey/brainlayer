from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest
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
    cost_dir.mkdir(parents=True, exist_ok=True)
    (cost_dir / "enrich-daily-cost.json").write_text(
        json.dumps({"date": now.astimezone().date().isoformat(), "spent_usd": spent_usd}),
        encoding="utf-8",
    )


def _write_corrupt_daily_cap_counter(cost_dir: Path) -> None:
    cost_dir.mkdir(parents=True, exist_ok=True)
    (cost_dir / "enrich-daily-cost.json").write_text('{"date": ', encoding="utf-8")


def _write_invalid_spent_daily_cap_counter(cost_dir: Path, *, now: datetime, spent_usd: object = "oops") -> None:
    cost_dir.mkdir(parents=True, exist_ok=True)
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

    monkeypatch.delenv("BRAINLAYER_ENRICH_COST_DIR", raising=False)
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
    assert issue.details["queue_count"] == 0
    assert issue.details["enrichment_backlog"] == 1
    assert issue.details["drain_label"] == "com.brainlayer.drain"


def test_run_doctor_does_not_fail_drain_liveness_when_heartbeat_is_advancing(tmp_path, monkeypatch):
    from brainlayer.doctor import run_doctor

    monkeypatch.delenv("BRAINLAYER_ENRICH_COST_DIR", raising=False)
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


def test_run_doctor_suppresses_enrichment_only_stale_liveness_when_drain_cycles_advance(tmp_path, monkeypatch):
    import brainlayer.doctor as doctor

    monkeypatch.delenv("BRAINLAYER_ENRICH_COST_DIR", raising=False)
    monkeypatch.setenv("BRAINLAYER_ENRICH_DAILY_USD_CAP", "5.0")
    db_path = tmp_path / "drain-liveness-enrichment-cycles-advance.db"
    _build_db(db_path)
    _add_enrichment_backlog(db_path)
    config = _doctor_config(tmp_path, db_path)
    stale_updated_at = (NOW - timedelta(minutes=10)).isoformat()
    drain_reads = 0
    original_load_json = doctor._load_json

    def fake_load_json(path: Path) -> dict:
        nonlocal drain_reads
        if path == config.drain_health_path:
            drain_reads += 1
            return {
                "drain_cycles": 20 + drain_reads,
                "drained_total": 40,
                "updated_at": stale_updated_at,
            }
        return original_load_json(path)

    monkeypatch.setattr(doctor, "_load_json", fake_load_json)

    result = doctor.run_doctor(
        config,
        ps_output_fn=_hotlane_ps,
        command_runner=_loaded_launchctl,
        now_fn=lambda: NOW,
    )

    assert result.exit_code == 0
    assert result.ok is True
    assert drain_reads == 2
    assert any(issue.code == "enrichment_idle_with_backlog" for issue in result.issues)
    assert not [issue for issue in result.issues if issue.code == "drain_liveness_stalled"]


def test_run_doctor_keeps_loaded_but_idle_warning_only_when_daily_cap_blocks_enrichment(tmp_path, monkeypatch):
    from brainlayer.doctor import run_doctor

    monkeypatch.delenv("BRAINLAYER_ENRICH_COST_DIR", raising=False)
    canonical_dir = tmp_path / "canonical"
    canonical_dir.mkdir()
    monkeypatch.setenv("BRAINLAYER_DB", str(canonical_dir / "brainlayer.db"))
    monkeypatch.setenv("BRAINLAYER_ENRICH_DAILY_USD_CAP", "5.0")
    db_path = tmp_path / "drain-liveness-quota.db"
    _write_daily_cap_reached(db_path.parent, now=NOW)
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


def test_run_doctor_honors_enrich_cost_dir_for_drain_liveness_quota(tmp_path, monkeypatch):
    from brainlayer.doctor import run_doctor

    cost_dir = tmp_path / "configured-cost-dir"
    monkeypatch.setenv("BRAINLAYER_ENRICH_COST_DIR", str(cost_dir))
    monkeypatch.setenv("BRAINLAYER_ENRICH_DAILY_USD_CAP", "5.0")
    db_path = tmp_path / "drain-liveness-env-quota.db"
    _write_daily_cap_reached(cost_dir, now=NOW)
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
    assert not [issue for issue in result.issues if issue.code == "drain_liveness_stalled"]


def test_run_doctor_fails_loudly_when_drain_liveness_quota_counter_is_corrupt(tmp_path, monkeypatch):
    from brainlayer.doctor import run_doctor

    monkeypatch.delenv("BRAINLAYER_ENRICH_COST_DIR", raising=False)
    monkeypatch.setenv("BRAINLAYER_ENRICH_DAILY_USD_CAP", "5.0")
    db_path = tmp_path / "drain-liveness-corrupt-quota.db"
    _write_corrupt_daily_cap_counter(db_path.parent)
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
    assert result.ok is False
    assert issue.severity == "fatal"
    assert "DRAIN_LIVENESS_STALLED" in issue.message
    assert not [issue for issue in result.issues if issue.code == "drain_liveness_quota_blocked"]
    assert not [issue for issue in result.issues if issue.code == "drain_liveness_blocker_unknown"]


@pytest.mark.parametrize("invalid_spend", ["oops", "nan", "inf", "-1"])
def test_run_doctor_fails_loudly_when_drain_liveness_quota_counter_has_invalid_spend(
    tmp_path, monkeypatch, invalid_spend
):
    from brainlayer.doctor import run_doctor

    monkeypatch.delenv("BRAINLAYER_ENRICH_COST_DIR", raising=False)
    monkeypatch.setenv("BRAINLAYER_ENRICH_DAILY_USD_CAP", "5.0")
    db_path = tmp_path / "drain-liveness-invalid-counter.db"
    _write_invalid_spent_daily_cap_counter(db_path.parent, now=NOW, spent_usd=invalid_spend)
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
    assert result.ok is False
    assert issue.severity == "fatal"
    assert "DRAIN_LIVENESS_STALLED" in issue.message
    assert not [issue for issue in result.issues if issue.code == "drain_liveness_quota_blocked"]
    assert not [issue for issue in result.issues if issue.code == "drain_liveness_blocker_unknown"]


def test_run_doctor_does_not_let_enrichment_quota_mask_durable_queue_liveness(tmp_path, monkeypatch):
    from brainlayer.doctor import run_doctor

    monkeypatch.delenv("BRAINLAYER_ENRICH_COST_DIR", raising=False)
    monkeypatch.setenv("BRAINLAYER_ENRICH_DAILY_USD_CAP", "5.0")
    db_path = tmp_path / "drain-liveness-queue-quota.db"
    _write_daily_cap_reached(db_path.parent, now=NOW)
    _build_db(db_path)
    config = _doctor_config(tmp_path, db_path)
    (config.queue_dir / "pending.jsonl").write_text('{"kind":"watcher_chunk"}\n', encoding="utf-8")
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
    assert issue.details["backlog_count"] == 1
    assert issue.details["queue_count"] == 1
    assert issue.details["enrichment_backlog"] == 0
    assert not [issue for issue in result.issues if issue.code == "drain_liveness_quota_blocked"]


def test_run_doctor_skips_drain_liveness_when_drain_label_is_disabled(tmp_path, monkeypatch):
    from brainlayer.doctor import run_doctor

    monkeypatch.delenv("BRAINLAYER_ENRICH_COST_DIR", raising=False)
    monkeypatch.setenv("BRAINLAYER_ENRICH_DAILY_USD_CAP", "5.0")
    db_path = tmp_path / "drain-liveness-disabled-label.db"
    _build_db(db_path)
    _add_enrichment_backlog(db_path)
    config = _doctor_config(tmp_path, db_path)
    config.drain_label = ""
    _write_drain_health(config.drain_health_path, updated_at=NOW - timedelta(minutes=10))

    result = run_doctor(
        config,
        ps_output_fn=_hotlane_ps,
        command_runner=_loaded_launchctl,
        now_fn=lambda: NOW,
    )

    assert result.exit_code == 0
    assert result.ok is True
    assert any(issue.code == "enrichment_idle_with_backlog" for issue in result.issues)
    assert not [issue for issue in result.issues if issue.code.startswith("drain_liveness")]


def test_run_doctor_suppresses_stale_liveness_when_drain_counter_advances(tmp_path, monkeypatch):
    import brainlayer.doctor as doctor

    monkeypatch.delenv("BRAINLAYER_ENRICH_COST_DIR", raising=False)
    monkeypatch.setenv("BRAINLAYER_ENRICH_DAILY_USD_CAP", "5.0")
    db_path = tmp_path / "drain-liveness-counter-advances.db"
    _build_db(db_path)
    config = _doctor_config(tmp_path, db_path)
    (config.queue_dir / "pending.jsonl").write_text('{"kind":"watcher_chunk"}\n', encoding="utf-8")
    stale_updated_at = (NOW - timedelta(minutes=10)).isoformat()
    drain_reads = 0
    original_load_json = doctor._load_json

    def fake_load_json(path: Path) -> dict:
        nonlocal drain_reads
        if path == config.drain_health_path:
            drain_reads += 1
            return {
                "drain_cycles": 3 + drain_reads,
                "drained_total": 40 + drain_reads,
                "updated_at": stale_updated_at,
            }
        return original_load_json(path)

    monkeypatch.setattr(doctor, "_load_json", fake_load_json)

    result = doctor.run_doctor(
        config,
        ps_output_fn=_hotlane_ps,
        command_runner=_loaded_launchctl,
        now_fn=lambda: NOW,
    )

    assert result.exit_code == 0
    assert result.ok is True
    assert drain_reads == 2
    assert not [issue for issue in result.issues if issue.code == "drain_liveness_stalled"]
    assert not [issue for issue in result.issues if issue.code == "queue_not_moving_with_backlog"]


def test_run_doctor_suppresses_stale_liveness_but_fails_queue_when_only_drain_cycles_advance(
    tmp_path,
    monkeypatch,
):
    import brainlayer.doctor as doctor

    monkeypatch.delenv("BRAINLAYER_ENRICH_COST_DIR", raising=False)
    monkeypatch.setenv("BRAINLAYER_ENRICH_DAILY_USD_CAP", "5.0")
    db_path = tmp_path / "drain-liveness-cycles-advance.db"
    _build_db(db_path)
    config = _doctor_config(tmp_path, db_path)
    (config.queue_dir / "pending.jsonl").write_text('{"kind":"watcher_chunk"}\n', encoding="utf-8")
    stale_updated_at = (NOW - timedelta(minutes=10)).isoformat()
    drain_reads = 0
    original_load_json = doctor._load_json

    def fake_load_json(path: Path) -> dict:
        nonlocal drain_reads
        if path == config.drain_health_path:
            drain_reads += 1
            return {
                "drain_cycles": 20 + drain_reads,
                "drained_total": 40,
                "updated_at": stale_updated_at,
            }
        return original_load_json(path)

    monkeypatch.setattr(doctor, "_load_json", fake_load_json)

    result = doctor.run_doctor(
        config,
        ps_output_fn=_hotlane_ps,
        command_runner=_loaded_launchctl,
        now_fn=lambda: NOW,
    )

    assert result.exit_code == 1
    assert result.ok is False
    assert drain_reads == 2
    assert not [issue for issue in result.issues if issue.code == "drain_liveness_stalled"]
    assert any(issue.code == "queue_not_moving_with_backlog" for issue in result.issues)


def test_run_doctor_uses_sample_time_for_post_sample_drain_liveness(tmp_path, monkeypatch):
    import brainlayer.doctor as doctor

    monkeypatch.delenv("BRAINLAYER_ENRICH_COST_DIR", raising=False)
    monkeypatch.setenv("BRAINLAYER_ENRICH_DAILY_USD_CAP", "5.0")
    db_path = tmp_path / "drain-liveness-sample-time.db"
    _build_db(db_path)
    config = _doctor_config(tmp_path, db_path)
    config.drain_liveness_stale_seconds = 300
    (config.queue_dir / "pending.jsonl").write_text('{"kind":"watcher_chunk"}\n', encoding="utf-8")
    drain_reads = 0
    now_reads = 0
    original_load_json = doctor._load_json

    def fake_load_json(path: Path) -> dict:
        nonlocal drain_reads
        if path == config.drain_health_path:
            drain_reads += 1
            return {
                "drain_cycles": 20,
                "drained_total": 40,
                "updated_at": (NOW - timedelta(seconds=299)).isoformat(),
            }
        return original_load_json(path)

    def fake_now() -> datetime:
        nonlocal now_reads
        now_reads += 1
        return NOW if now_reads == 1 else NOW + timedelta(seconds=2)

    monkeypatch.setattr(doctor, "_load_json", fake_load_json)

    result = doctor.run_doctor(
        config,
        ps_output_fn=_hotlane_ps,
        command_runner=_loaded_launchctl,
        now_fn=fake_now,
    )

    issue = next(issue for issue in result.issues if issue.code == "drain_liveness_stalled")
    assert result.exit_code == 1
    assert drain_reads == 2
    assert now_reads == 2
    assert issue.details["heartbeat_age_seconds"] == 301.0


def test_run_doctor_suppresses_stale_liveness_when_sampled_heartbeat_is_fresh(tmp_path, monkeypatch):
    import brainlayer.doctor as doctor

    monkeypatch.delenv("BRAINLAYER_ENRICH_COST_DIR", raising=False)
    monkeypatch.setenv("BRAINLAYER_ENRICH_DAILY_USD_CAP", "5.0")
    db_path = tmp_path / "drain-liveness-sampled-fresh.db"
    _build_db(db_path)
    config = _doctor_config(tmp_path, db_path)
    (config.queue_dir / "pending.jsonl").write_text('{"kind":"watcher_chunk"}\n', encoding="utf-8")
    drain_reads = 0
    watcher_reads = 0
    original_load_json = doctor._load_json

    def fake_load_json(path: Path) -> dict:
        nonlocal drain_reads, watcher_reads
        if path == config.drain_health_path:
            drain_reads += 1
            return {
                "drain_cycles": 20,
                "drained_total": 40,
                "updated_at": (NOW - timedelta(minutes=10)).isoformat() if drain_reads == 1 else NOW.isoformat(),
            }
        if path == config.watcher_health_path:
            watcher_reads += 1
            return {"poll_count": 12 + watcher_reads}
        return original_load_json(path)

    monkeypatch.setattr(doctor, "_load_json", fake_load_json)

    result = doctor.run_doctor(
        config,
        ps_output_fn=_hotlane_ps,
        command_runner=_loaded_launchctl,
        now_fn=lambda: NOW,
    )

    assert result.exit_code == 0
    assert result.ok is True
    assert drain_reads == 2
    assert watcher_reads == 2
    assert not [issue for issue in result.issues if issue.code == "drain_liveness_stalled"]
    assert not [issue for issue in result.issues if issue.code == "queue_not_moving_with_backlog"]


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
