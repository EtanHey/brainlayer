from __future__ import annotations

import datetime as dt
import json
import plistlib
import subprocess
from pathlib import Path

import apsw
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


def _write_jsonl(path: Path, events: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(event) + "\n" for event in events), encoding="utf-8")


def _create_enrichment_db(path: Path) -> None:
    conn = apsw.Connection(str(path))
    try:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute(
            """
            CREATE TABLE chunks (
                id TEXT PRIMARY KEY,
                content TEXT,
                summary TEXT,
                enriched_at TEXT,
                enrich_status TEXT,
                content_hash TEXT
            )
            """
        )
        conn.execute(
            """
            INSERT INTO chunks (id, content, summary, enriched_at, enrich_status, content_hash)
            VALUES
                ('already-done', 'same content', 'old summary', '2026-05-30T00:00:00Z', 'success', 'hash-ok'),
                ('needs-update', 'same content', NULL, NULL, NULL, 'hash-ok'),
                ('hash-moved', 'new content', 'old summary', '2026-05-30T00:00:00Z', 'success', 'hash-new')
            """
        )
    finally:
        conn.close()


def _config(tmp_path: Path, *, now: dt.datetime):
    from brainlayer.maintenance import MaintenanceConfig

    return MaintenanceConfig(
        db_path=tmp_path / "brainlayer.db",
        queue_dir=tmp_path / "queue",
        quarantine_root=tmp_path / "quarantine",
        log_path=tmp_path / "maintenance.log",
        repo_root=REPO_ROOT,
        now_fn=lambda: now,
        quiet_window_start_hour=4,
        quiet_window_duration_minutes=120,
        idle_sample_seconds=0,
        recent_write_grace_seconds=0,
    )


@pytest.fixture(autouse=True)
def fake_watchdog_hold(monkeypatch, tmp_path):
    from brainlayer import maintenance

    monkeypatch.setattr(maintenance, "run_command", lambda args, **kwargs: subprocess.CompletedProcess(args, 0, "", ""))
    monkeypatch.setattr(maintenance, "PAUSE_SENTINEL_PATH", tmp_path / "pause.sentinel")


def test_off_window_gate_aborts_before_touching_queue(tmp_path, monkeypatch):
    from brainlayer import maintenance

    config = _config(tmp_path, now=dt.datetime(2026, 5, 30, 2, 30, tzinfo=dt.timezone.utc))
    config.queue_dir.mkdir(parents=True)
    queued = config.queue_dir / "enrichment-stale.jsonl"
    queued.write_text("{}\n", encoding="utf-8")
    commands: list[list[str]] = []
    monkeypatch.setattr(maintenance, "collect_lsof_entries", lambda _paths: [])
    monkeypatch.setattr(maintenance, "run_command", lambda args, **_kwargs: commands.append(args))

    with pytest.raises(maintenance.MaintenanceAbort, match="outside quiet window"):
        maintenance.run_maintenance("light", config=config, dry_run=True)

    assert queued.exists()
    assert commands == []


def test_lsof_gate_aborts_on_unexpected_writer(tmp_path, monkeypatch):
    from brainlayer import maintenance

    config = _config(tmp_path, now=dt.datetime(2026, 5, 30, 4, 5, tzinfo=dt.timezone.utc))
    config.db_path.write_bytes(b"db")
    config.queue_dir.mkdir(parents=True)
    monkeypatch.setattr(
        maintenance,
        "collect_lsof_entries",
        lambda _paths: [
            maintenance.LsofEntry(pid=4242, command="python3", fd="9u", path=str(config.db_path)),
        ],
    )

    with pytest.raises(maintenance.MaintenanceAbort, match="unexpected writer"):
        maintenance.run_maintenance("light", config=config, dry_run=True)


def test_lsof_gate_accepts_packaged_drain_module_writer(tmp_path, monkeypatch):
    from brainlayer import maintenance

    config = _config(tmp_path, now=dt.datetime(2026, 5, 30, 4, 5, tzinfo=dt.timezone.utc))
    config.db_path.write_bytes(b"db")
    monkeypatch.setattr(
        maintenance,
        "collect_lsof_entries",
        lambda _paths: [
            maintenance.LsofEntry(
                pid=4242,
                command="python -m brainlayer.drain",
                fd="9u",
                path=str(config.db_path),
            ),
        ],
    )

    maintenance._check_lsof_clean(config)


def test_recent_queue_activity_abort_message_reports_count(tmp_path, monkeypatch):
    from brainlayer import maintenance

    config = _config(tmp_path, now=dt.datetime(2026, 5, 30, 4, 5, tzinfo=dt.timezone.utc))
    config.recent_write_grace_seconds = 300
    config.queue_dir.mkdir(parents=True)
    (config.queue_dir / "a.jsonl").write_text("{}\n", encoding="utf-8")
    (config.queue_dir / "z.jsonl").write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(maintenance, "collect_lsof_entries", lambda _paths: [])

    with pytest.raises(maintenance.MaintenanceAbort, match=r"2 file\(s\) modified recently"):
        maintenance.run_maintenance("light", config=config, dry_run=True)


def test_dry_run_runs_gates_and_reports_without_moving_queue_files(tmp_path, monkeypatch):
    from brainlayer import maintenance

    config = _config(tmp_path, now=dt.datetime(2026, 5, 30, 4, 5, tzinfo=dt.timezone.utc))
    _create_enrichment_db(config.db_path)
    stale = config.queue_dir / "enrichment-stale.jsonl"
    _write_jsonl(
        stale,
        [
            {
                "kind": "enrichment_update",
                "chunk_id": "already-done",
                "content_hash": "hash-ok",
                "enrichment": {"summary": "new summary"},
            }
        ],
    )
    monkeypatch.setattr(maintenance, "collect_lsof_entries", lambda _paths: [])

    result = maintenance.run_maintenance("light", config=config, dry_run=True)

    assert result.dry_run is True
    assert result.stale_queue.quarantined_files == 0
    assert result.stale_queue.candidate_files == 1
    assert stale.exists()
    assert not config.quarantine_root.exists()


def test_weekly_backup_waits_for_daily_then_reuses_verified_receipt(tmp_path, monkeypatch):
    from brainlayer import maintenance

    config = _config(tmp_path, now=dt.datetime(2026, 9, 27, 4, 0, tzinfo=dt.UTC))
    config.backup_staging_dir = tmp_path / "staging"
    config.backup_log_path = tmp_path / "backup.log"
    config.backup_wait_poll_seconds = 1
    clock = [0.0]
    calls = [0]

    def held(_path):
        calls[0] += 1
        return calls[0] == 1

    def finish(_seconds):
        clock[0] += 1
        _write_jsonl(
            config.backup_log_path,
            [
                {
                    "attempted_at": "2026-09-27T03:17:00+00:00",
                    "db": str(config.db_path),
                    "uploaded": True,
                    "verified": True,
                    "drive_file": {"id": "verified-daily"},
                }
            ],
        )

    monkeypatch.setattr(maintenance, "_backup_lock_is_held", held)
    monkeypatch.setattr(maintenance.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(maintenance.time, "sleep", finish)
    monkeypatch.setattr(maintenance, "_run_bounded_weekly_backup", lambda *_args: pytest.fail("duplicate backup"))

    assert maintenance._weekly_backup(config)["drive_file"]["id"] == "verified-daily"
    assert calls[0] == 2


def test_weekly_backup_timeout_aborts_before_destructive_work(tmp_path, monkeypatch):
    from brainlayer import maintenance

    config = _config(tmp_path, now=dt.datetime(2026, 9, 27, 4, 0, tzinfo=dt.UTC))
    config.backup_wait_timeout_seconds = 2
    config.backup_wait_poll_seconds = 1
    clock = [0.0]
    monkeypatch.setattr(maintenance, "_backup_lock_is_held", lambda _path: True)
    monkeypatch.setattr(maintenance.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(maintenance.time, "sleep", lambda seconds: clock.__setitem__(0, clock[0] + seconds))
    monkeypatch.setattr(maintenance, "_run_bounded_weekly_backup", lambda *_args: pytest.fail("backup started"))

    with pytest.raises(maintenance.MaintenanceAbort, match="backup wait timed out") as exc:
        maintenance._weekly_backup(config)
    assert exc.value.code == 76

    _create_enrichment_db(config.db_path)
    monkeypatch.setattr(maintenance, "collect_lsof_entries", lambda _paths: [])
    monkeypatch.setattr(maintenance, "_service_is_loaded", lambda _service, **kwargs: False)
    monkeypatch.setattr(maintenance, "_bootout_service", lambda _service: False)
    checkpoints = []
    monkeypatch.setattr(maintenance, "_checkpoint_full", lambda _db_path: checkpoints.append(1) or (0, 0, 0))
    monkeypatch.setattr(maintenance, "_verify_search_latency", lambda _db_path: 1.0)
    monkeypatch.setattr(maintenance, "_vacuum", lambda _db_path: pytest.fail("VACUUM started"))
    with pytest.raises(maintenance.MaintenanceAbort) as full_exc:
        maintenance.run_maintenance("full", config=config)
    assert full_exc.value.code == 76
    assert checkpoints == [1]
    assert json.loads(config.log_path.read_text().splitlines()[-1])["backup_status"] == "unavailable"


def test_weekly_backup_without_running_backup_runs_normally(tmp_path, monkeypatch):
    from brainlayer import maintenance

    config = _config(tmp_path, now=dt.datetime(2026, 9, 27, 4, 0, tzinfo=dt.UTC))
    config.backup_staging_dir = tmp_path / "staging"
    config.backup_log_path = tmp_path / "backup.log"
    monkeypatch.setattr(maintenance, "_backup_lock_is_held", lambda _path: False)
    deadlines = []

    def backup(_config, timeout_seconds):
        deadlines.append(timeout_seconds)
        return {"verified": True, "uploaded": True, "drive_file": {"id": "verified-weekly"}}

    monkeypatch.setattr(maintenance, "_run_bounded_weekly_backup", backup)

    assert maintenance._weekly_backup(config)["drive_file"]["id"] == "verified-weekly"
    assert deadlines == [7200]


def test_fresh_weekly_backup_uses_supervised_process_and_kills_stuck_group(tmp_path, monkeypatch):
    from brainlayer import maintenance

    config = _config(tmp_path, now=dt.datetime(2026, 9, 27, 4, 0, tzinfo=dt.UTC))
    config.backup_staging_dir = tmp_path / "staging"
    config.backup_log_path = tmp_path / "backup.log"
    observed = {}

    class StuckProcess:
        pid = 45210

        def communicate(self, timeout=None):
            if timeout is not None:
                raise maintenance.subprocess.TimeoutExpired("backup", timeout)
            return "", ""

    def popen(args, **kwargs):
        observed["args"] = args
        observed["kwargs"] = kwargs
        return StuckProcess()

    monkeypatch.setattr(maintenance.subprocess, "Popen", popen)
    monkeypatch.setattr(maintenance.os, "killpg", lambda pid, sig: observed.update(pid=pid, signal=sig))

    with pytest.raises(maintenance.MaintenanceAbort, match="timed out") as exc:
        maintenance._run_bounded_weekly_backup(config, 45)
    assert exc.value.code == 76
    assert observed["args"][-2:] == ["-m", "brainlayer.backup_daily"]
    assert observed["kwargs"]["env"]["BRAINLAYER_BACKUP_TIMEOUT_SECONDS"] == "45"
    assert observed["kwargs"]["env"]["BRAINLAYER_BACKUP_REUSE_VERIFIED_MAX_AGE_HOURS"] == "6"
    assert observed["kwargs"]["start_new_session"] is True
    assert observed["pid"] == 45210
    assert observed["signal"] == maintenance.signal.SIGKILL


@pytest.mark.parametrize("broken_reader", ["_backup_lock_is_held", "_recent_verified_backup"])
def test_weekly_backup_read_errors_use_vacuum_skip_code(tmp_path, monkeypatch, broken_reader):
    from brainlayer import maintenance

    config = _config(tmp_path, now=dt.datetime(2026, 9, 27, 4, 5, tzinfo=dt.UTC))
    monkeypatch.setattr(maintenance, "_backup_lock_is_held", lambda _path: False)
    monkeypatch.setattr(maintenance, "_recent_verified_backup", lambda _config: None)
    monkeypatch.setattr(
        maintenance,
        broken_reader,
        lambda _arg: (_ for _ in ()).throw(PermissionError("synthetic denial")),
    )
    monkeypatch.setattr(maintenance, "_run_bounded_weekly_backup", lambda *_args: pytest.fail("backup started"))

    with pytest.raises(maintenance.MaintenanceAbort, match="VACUUM skipped") as exc:
        maintenance._weekly_backup(config)
    assert exc.value.code == 76


def test_full_rechecks_window_after_backup_wait_and_skips_vacuum(tmp_path, monkeypatch):
    from brainlayer import maintenance

    clock = [dt.datetime(2026, 9, 27, 4, 5, tzinfo=dt.UTC)]
    config = _config(tmp_path, now=clock[0])
    config.now_fn = lambda: clock[0]
    _create_enrichment_db(config.db_path)
    monkeypatch.setattr(maintenance, "collect_lsof_entries", lambda _paths: [])

    def finished_backup(_config):
        clock[0] = dt.datetime(2026, 9, 27, 6, 1, tzinfo=dt.UTC)
        return {"verified": True, "uploaded": True, "drive_file": {"id": "verified"}}

    monkeypatch.setattr(maintenance, "_weekly_backup", finished_backup)
    monkeypatch.setattr(
        maintenance, "_service_is_loaded", lambda _service: pytest.fail("service probed after gate failure")
    )
    monkeypatch.setattr(
        maintenance, "_bootout_service", lambda _service: pytest.fail("service booted out after gate failure")
    )
    monkeypatch.setattr(maintenance, "_checkpoint_full", lambda _path: pytest.fail("checkpoint after gate failure"))
    monkeypatch.setattr(maintenance, "_verify_search_latency", lambda _path: 1.0)
    monkeypatch.setattr(maintenance, "_vacuum", lambda _path: pytest.fail("VACUUM started after quiet window"))

    with pytest.raises(maintenance.MaintenanceAbort, match="outside quiet window") as exc:
        maintenance.run_maintenance("full", config=config)
    assert exc.value.code == 76
    assert json.loads(config.log_path.read_text().splitlines()[-1])["backup_status"] == "gates_failed"


def test_weekly_backup_rechecks_after_lock_race(tmp_path, monkeypatch):
    from brainlayer import maintenance

    config = _config(tmp_path, now=dt.datetime(2026, 9, 27, 4, 0, tzinfo=dt.UTC))
    config.backup_log_path = tmp_path / "backup.log"
    config.backup_wait_poll_seconds = 0
    calls = [0]

    def backup(*_args):
        calls[0] += 1
        if calls[0] == 1:
            _write_jsonl(
                config.backup_log_path,
                [
                    {
                        "attempted_at": "2026-09-27T03:17:00+00:00",
                        "db": str(config.db_path),
                        "uploaded": True,
                        "verified": True,
                        "drive_file": {"id": "verified-daily"},
                    }
                ],
            )
            raise maintenance.BackupAlreadyRunningError("backup already running")
        pytest.fail("duplicate backup after daily completion")

    monkeypatch.setattr(maintenance, "_backup_lock_is_held", lambda _path: False)
    monkeypatch.setattr(maintenance, "_run_bounded_weekly_backup", backup)

    assert maintenance._weekly_backup(config)["drive_file"]["id"] == "verified-daily"
    assert calls[0] == 1


def test_weekly_child_rechecks_verified_receipt_after_acquiring_backup_lock(tmp_path, monkeypatch):
    from brainlayer import backup_daily

    db_path = tmp_path / "brainlayer.db"
    log_path = tmp_path / "backup.log"
    staging_dir = tmp_path / "staging"
    monkeypatch.setenv("BRAINLAYER_BACKUP_REUSE_VERIFIED_MAX_AGE_HOURS", "6")

    @backup_daily._serialized_backup_run
    def backup(db_path, staging_dir, log_path):
        pytest.fail("duplicate backup after daily completion")

    # The weekly parent observed no receipt. The daily finishes before the child
    # acquires .backup.lock; the child must recheck while holding that lock.
    _write_jsonl(
        log_path,
        [
            {
                "attempted_at": dt.datetime.now(dt.UTC).isoformat(),
                "db": str(db_path),
                "uploaded": True,
                "verified": True,
                "drive_file": {"id": "verified-daily"},
            }
        ],
    )
    assert backup(db_path=db_path, staging_dir=staging_dir, log_path=log_path)["drive_file"]["id"] == "verified-daily"


def test_weekly_child_does_not_reuse_unverified_receipt(tmp_path, monkeypatch):
    from brainlayer import backup_daily

    db_path = tmp_path / "brainlayer.db"
    log_path = tmp_path / "backup.log"
    monkeypatch.setenv("BRAINLAYER_BACKUP_REUSE_VERIFIED_MAX_AGE_HOURS", "6")
    _write_jsonl(
        log_path,
        [
            {
                "attempted_at": dt.datetime.now(dt.UTC).isoformat(),
                "db": str(db_path),
                "uploaded": True,
                "verified": False,
                "drive_file": {"id": "unverified"},
            }
        ],
    )
    runs = []

    @backup_daily._serialized_backup_run
    def backup(db_path, staging_dir, log_path):
        runs.append(1)
        return {"uploaded": True, "verified": True, "drive_file": {"id": "fresh"}}

    assert backup(db_path=db_path, staging_dir=tmp_path / "staging", log_path=log_path)["drive_file"]["id"] == "fresh"
    assert runs == [1]


def test_full_dry_run_reports_backup_contention_without_waiting(tmp_path, monkeypatch):
    from brainlayer import maintenance

    config = _config(tmp_path, now=dt.datetime(2026, 9, 27, 4, 0, tzinfo=dt.UTC))
    _create_enrichment_db(config.db_path)
    monkeypatch.setattr(maintenance, "collect_lsof_entries", lambda _paths: [])
    monkeypatch.setattr(maintenance, "_backup_lock_is_held", lambda _path: True)
    monkeypatch.setattr(maintenance, "_run_bounded_weekly_backup", lambda *_args: pytest.fail("backup started"))

    result = maintenance.run_maintenance("full", config=config, dry_run=True)

    assert "would wait for running backup" in result.actions


def test_coordinated_fts_repair_resumes_writers_after_repair_error(tmp_path, monkeypatch):
    from brainlayer import maintenance, runtime_store

    events = []
    monkeypatch.setattr(maintenance, "_service_is_loaded", lambda _service, **kwargs: False)
    monkeypatch.setattr(maintenance, "_bootout_service", lambda service: events.append(("stop", service)) or True)
    monkeypatch.setattr(maintenance, "_resume_service", lambda _root, service: events.append(("start", service)))

    def fail_open(_db_path):
        events.append(("open", "writer"))
        raise RuntimeError("writer unavailable")

    monkeypatch.setattr(runtime_store, "WriterRuntimeStore", fail_open)
    with pytest.raises(RuntimeError, match="writer unavailable"):
        maintenance.run_coordinated_fts_repair(tmp_path / "fixture.db", repo_root=tmp_path)
    assert events == [
        ("stop", "fleet-watchdog"),
        *(("stop", service) for service in maintenance.DEFAULT_SERVICES),
        ("open", "writer"),
        *(("start", service) for service in maintenance.DEFAULT_SERVICES),
        ("start", "fleet-watchdog"),
    ]


def test_scheduled_fts_repair_waits_for_weekly_maintenance_lock(tmp_path, monkeypatch):
    from brainlayer import maintenance

    db_path = tmp_path / "fixture.db"
    held = (tmp_path / ".maintenance.lock").open("a+b")
    maintenance.fcntl.flock(held, maintenance.fcntl.LOCK_EX)
    clock = [0.0]
    monkeypatch.setattr(maintenance, "MAINTENANCE_LOCK_TIMEOUT_SECONDS", 2)
    monkeypatch.setattr(maintenance.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(maintenance.time, "sleep", lambda seconds: clock.__setitem__(0, clock[0] + seconds))
    monkeypatch.setattr(maintenance, "_service_is_loaded", lambda _service: pytest.fail("repair started"))
    try:
        with pytest.raises(maintenance.MaintenanceAbort, match="maintenance lock timed out") as exc:
            maintenance.run_coordinated_fts_repair(db_path, repo_root=tmp_path)
        assert exc.value.code == 77
    finally:
        maintenance.fcntl.flock(held, maintenance.fcntl.LOCK_UN)
        held.close()


def test_coordinated_fts_repair_resumes_every_successful_bootout(tmp_path, monkeypatch):
    from brainlayer import maintenance, runtime_store

    resumed = []
    monkeypatch.setattr(maintenance, "_service_is_loaded", lambda service, **kwargs: False)
    monkeypatch.setattr(maintenance, "_bootout_service", lambda _service: True)
    monkeypatch.setattr(maintenance, "_resume_service", lambda _root, service: resumed.append(service))

    class FakeStore:
        def __init__(self, _db_path):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            pass

        def repair_fts(self, **_kwargs):
            return {"chunks_fts": 0}

    monkeypatch.setattr(runtime_store, "WriterRuntimeStore", FakeStore)
    maintenance.run_coordinated_fts_repair(tmp_path / "fixture.db", repo_root=tmp_path)
    assert resumed == [*maintenance.DEFAULT_SERVICES, "fleet-watchdog"]


def test_stale_queue_quarantine_moves_only_already_enriched_matching_hash(tmp_path):
    from brainlayer.maintenance import quarantine_stale_queue_files

    db_path = tmp_path / "brainlayer.db"
    queue_dir = tmp_path / "queue"
    quarantine_root = tmp_path / "quarantine"
    _create_enrichment_db(db_path)
    stale = queue_dir / "enrichment-stale.jsonl"
    needs_update = queue_dir / "enrichment-needs-update.jsonl"
    mismatch = queue_dir / "enrichment-mismatch.jsonl"
    missing = queue_dir / "enrichment-missing.jsonl"
    _write_jsonl(
        stale,
        [
            {
                "kind": "enrichment_update",
                "chunk_id": "already-done",
                "content_hash": "hash-ok",
                "enrichment": {"summary": "new summary"},
            }
        ],
    )
    _write_jsonl(
        needs_update,
        [
            {
                "kind": "enrichment_update",
                "chunk_id": "needs-update",
                "content_hash": "hash-ok",
                "enrichment": {"summary": "real update"},
            }
        ],
    )
    _write_jsonl(
        mismatch,
        [
            {
                "kind": "enrichment_update",
                "chunk_id": "hash-moved",
                "content_hash": "hash-old",
                "enrichment": {"summary": "stale hash mismatch"},
            }
        ],
    )
    _write_jsonl(
        missing,
        [
            {
                "kind": "enrichment_update",
                "chunk_id": "missing",
                "content_hash": "hash-ok",
                "enrichment": {"summary": "missing chunk"},
            }
        ],
    )

    result = quarantine_stale_queue_files(
        db_path=db_path,
        queue_dir=queue_dir,
        quarantine_root=quarantine_root,
        dry_run=False,
        now=dt.datetime(2026, 5, 30, 4, 10, tzinfo=dt.timezone.utc),
    )

    assert result.scanned_files == 4
    assert result.candidate_files == 1
    assert result.quarantined_files == 1
    assert not stale.exists()
    assert needs_update.exists()
    assert mismatch.exists()
    assert missing.exists()
    quarantined = list(quarantine_root.rglob("enrichment-stale.jsonl"))
    assert len(quarantined) == 1


def test_stale_queue_quarantine_preserves_entity_bearing_redundant_enrichment(tmp_path):
    from brainlayer.maintenance import quarantine_stale_queue_files

    db_path = tmp_path / "brainlayer.db"
    queue_dir = tmp_path / "queue"
    quarantine_root = tmp_path / "quarantine"
    _create_enrichment_db(db_path)
    provenance_only = queue_dir / "enrichment-provenance-only.jsonl"
    _write_jsonl(
        provenance_only,
        [
            {
                "kind": "enrichment_update",
                "chunk_id": "already-done",
                "content_hash": "hash-ok",
                "enrichment": {"summary": "redundant summary"},
                "entities": [{"name": "controlLayer"}],
            }
        ],
    )

    result = quarantine_stale_queue_files(
        db_path=db_path,
        queue_dir=queue_dir,
        quarantine_root=quarantine_root,
        dry_run=False,
        now=dt.datetime(2026, 5, 30, 4, 10, tzinfo=dt.timezone.utc),
    )

    assert result.scanned_files == 1
    assert result.candidate_files == 0
    assert result.quarantined_files == 0
    assert provenance_only.exists()
    assert not quarantine_root.exists()


def test_stale_queue_quarantine_preserves_empty_entity_redundant_enrichment(tmp_path):
    from brainlayer.maintenance import quarantine_stale_queue_files

    db_path = tmp_path / "brainlayer.db"
    queue_dir = tmp_path / "queue"
    quarantine_root = tmp_path / "quarantine"
    _create_enrichment_db(db_path)
    provenance_empty = queue_dir / "enrichment-empty-entities.jsonl"
    _write_jsonl(
        provenance_empty,
        [
            {
                "kind": "enrichment_update",
                "chunk_id": "already-done",
                "content_hash": "hash-ok",
                "enrichment": {"summary": "redundant summary"},
                "entities": [],
            }
        ],
    )

    result = quarantine_stale_queue_files(
        db_path=db_path,
        queue_dir=queue_dir,
        quarantine_root=quarantine_root,
        dry_run=False,
        now=dt.datetime(2026, 5, 30, 4, 10, tzinfo=dt.timezone.utc),
    )

    assert result.scanned_files == 1
    assert result.candidate_files == 0
    assert result.quarantined_files == 0
    assert provenance_empty.exists()
    assert not quarantine_root.exists()


def test_stale_queue_quarantine_preserves_provenance_class_redundant_enrichment(tmp_path):
    from brainlayer.maintenance import quarantine_stale_queue_files

    db_path = tmp_path / "brainlayer.db"
    queue_dir = tmp_path / "queue"
    quarantine_root = tmp_path / "quarantine"
    _create_enrichment_db(db_path)
    provenance_only = queue_dir / "enrichment-provenance-class.jsonl"
    _write_jsonl(
        provenance_only,
        [
            {
                "kind": "enrichment_update",
                "chunk_id": "already-done",
                "content_hash": "hash-ok",
                "enrichment": {"summary": "redundant summary"},
                "provenance_class": "RAW-ETAN-DIRECT",
            }
        ],
    )

    result = quarantine_stale_queue_files(
        db_path=db_path,
        queue_dir=queue_dir,
        quarantine_root=quarantine_root,
        dry_run=False,
        now=dt.datetime(2026, 5, 30, 4, 10, tzinfo=dt.timezone.utc),
    )

    assert result.scanned_files == 1
    assert result.candidate_files == 0
    assert result.quarantined_files == 0
    assert provenance_only.exists()
    assert not quarantine_root.exists()


def test_maintenance_launchd_plists_and_installer_wiring():
    nightly_path = REPO_ROOT / "scripts/launchd/com.brainlayer.maintenance-nightly.plist"
    weekly_path = REPO_ROOT / "scripts/launchd/com.brainlayer.maintenance-weekly.plist"

    with nightly_path.open("rb") as handle:
        nightly = plistlib.load(handle)
    with weekly_path.open("rb") as handle:
        weekly = plistlib.load(handle)

    assert nightly["Label"] == "com.brainlayer.maintenance-nightly"
    assert weekly["Label"] == "com.brainlayer.maintenance-weekly"
    assert "--light" in nightly["ProgramArguments"]
    assert "--full" in weekly["ProgramArguments"]
    assert nightly["StartCalendarInterval"] == {"Hour": 4, "Minute": 0}
    assert weekly["StartCalendarInterval"] == {"Weekday": 0, "Hour": 4, "Minute": 0}
    for plist in (nightly, weekly):
        assert plist["Nice"] == 10
        assert plist["ProcessType"] == "Background"
        assert plist["EnvironmentVariables"]["BRAINLAYER_REPO_ROOT"] == "__BRAINLAYER_DIR__"
        assert plist["EnvironmentVariables"]["BRAINLAYER_LAUNCHD_DIR"] == "__BRAINLAYER_LAUNCHD_DIR__"

    install = (REPO_ROOT / "scripts/launchd/install.sh").read_text(encoding="utf-8")
    assert "maintenance-nightly" in install
    assert "maintenance-weekly" in install


def test_resume_service_uses_configured_launchd_dir_for_packaged_installs(tmp_path, monkeypatch):
    from brainlayer import maintenance

    launchd_dir = tmp_path / "site-packages" / "brainlayer" / "launchd"
    launchd_dir.mkdir(parents=True)
    (launchd_dir / "install.sh").write_text("#!/usr/bin/env bash\n", encoding="utf-8")
    (launchd_dir / "brainlayer.env.example").write_text(
        "BRAINLAYER_GEMINI_SERVICE_TIER=flex\n",
        encoding="utf-8",
    )
    commands: list[list[str]] = []
    monkeypatch.setenv("BRAINLAYER_LAUNCHD_DIR", str(launchd_dir))
    monkeypatch.setattr(maintenance, "run_command", lambda args, **_kwargs: commands.append(list(args)))

    maintenance._resume_service(tmp_path / "site-packages", "watch")

    assert commands == [[str(launchd_dir / "install.sh"), "watch"]]


def test_maintenance_resume_attempts_all_services_after_mid_resume_failure(tmp_path, monkeypatch):
    from brainlayer import maintenance

    config = _config(tmp_path, now=dt.datetime(2026, 5, 30, 4, 5, tzinfo=dt.timezone.utc))
    config.queue_dir.mkdir(parents=True)
    config.db_path.write_bytes(b"db")
    quiesced: list[str] = []
    resumed: list[str] = []
    monkeypatch.setattr(maintenance, "collect_lsof_entries", lambda _paths: [])
    monkeypatch.setattr(maintenance, "_service_is_loaded", lambda _service, **kwargs: False)
    monkeypatch.setattr(maintenance, "_bootout_service", lambda service: quiesced.append(service) or True)
    monkeypatch.setattr(maintenance, "_checkpoint_full", lambda _db_path: (0, 0, 0))
    monkeypatch.setattr(maintenance, "_verify_search_latency", lambda _db_path: 1.0)

    def fail_mid_resume(_repo_root: Path, service: str) -> None:
        resumed.append(service)
        if service == "index":
            raise RuntimeError("bootstrap I/O error")

    monkeypatch.setattr(maintenance, "_resume_service", fail_mid_resume)

    with pytest.raises(maintenance.MaintenanceAbort, match="failed to resume 1 launchd service"):
        maintenance.run_maintenance("light", config=config)

    assert quiesced == ["fleet-watchdog", *maintenance.DEFAULT_SERVICES]
    assert resumed == [*maintenance.DEFAULT_SERVICES, "fleet-watchdog"]


def test_maintenance_resumes_successful_bootout_even_after_stale_loaded_probe(tmp_path, monkeypatch, capsys):
    from brainlayer import maintenance

    config = _config(tmp_path, now=dt.datetime(2026, 5, 30, 4, 5, tzinfo=dt.timezone.utc))
    config.queue_dir.mkdir(parents=True)
    config.db_path.write_bytes(b"db")
    events: list[tuple[str, str]] = []
    resumed: list[str] = []
    monkeypatch.setattr(maintenance, "collect_lsof_entries", lambda _paths: [])

    def fake_launchctl(args, **_kwargs):
        if args[1] in {"disable", "enable", "print-disabled"}:
            return subprocess.CompletedProcess(args, 0, "", "")
        assert args[:2] == ["launchctl", "print"]
        service = "fleet-watchdog" if args[-1].endswith("fleet-watchdog") else args[-1].rsplit(".", 1)[-1]
        if ("probe", service) not in events:
            events.append(("probe", service))
        returncode = 113 if service in {"watch", "enrichment"} or ("bootout", service) in events else 0
        stderr = "Could not find service" if returncode else ""
        return maintenance.subprocess.CompletedProcess(args, returncode, "", stderr)

    def fake_bootout(service: str) -> bool:
        events.append(("bootout", service))
        return service != "watch"

    monkeypatch.setattr(maintenance, "run_command", fake_launchctl)
    monkeypatch.setattr(maintenance, "_bootout_service", fake_bootout)
    monkeypatch.setattr(maintenance, "_checkpoint_full", lambda _db_path: (0, 0, 0))
    monkeypatch.setattr(maintenance, "_verify_search_latency", lambda _db_path: 1.0)
    monkeypatch.setattr(maintenance, "_service_is_deliberately_paused", lambda _service, **kwargs: False)
    monkeypatch.setattr(maintenance, "_resume_service", lambda _root, service: resumed.append(service))

    maintenance.run_maintenance("light", config=config)

    assert "watch" not in resumed, "maintenance resurrected a service that was down before quiesce"
    expected_services = list(maintenance.DEFAULT_SERVICES)
    expected_events = [("probe", service) for service in expected_services]
    expected_events.extend([("bootout", "fleet-watchdog"), ("probe", "fleet-watchdog")])
    expected_events.extend(("bootout", service) for service in expected_services)
    assert events == expected_events
    assert resumed == ["index", "drain", "fleet-watchdog"]
    assert "maintenance did not boot out service watch; leaving it down" in capsys.readouterr().err


def test_maintenance_aborts_before_db_work_when_launchd_state_is_unsafe(tmp_path, monkeypatch):
    from brainlayer import maintenance

    config = _config(tmp_path, now=dt.datetime(2026, 5, 30, 4, 5, tzinfo=dt.timezone.utc))
    config.queue_dir.mkdir(parents=True)
    config.db_path.write_bytes(b"db")
    commands: list[list[str]] = []
    monkeypatch.setattr(maintenance, "collect_lsof_entries", lambda _paths: [])

    def fake_launchctl(args, **_kwargs):
        commands.append(args)
        return maintenance.subprocess.CompletedProcess(args, 1, "", "launchctl temporarily unavailable")

    monkeypatch.setattr(maintenance, "run_command", fake_launchctl)
    monkeypatch.setattr(maintenance, "_checkpoint_full", lambda _db_path: (0, 0, 0))
    monkeypatch.setattr(maintenance, "_verify_search_latency", lambda _db_path: 1.0)

    with pytest.raises(maintenance.MaintenanceAbort, match="cannot determine whether launchd service watch is loaded"):
        maintenance.run_maintenance("light", config=config)

    assert commands == [["launchctl", "print", f"gui/{maintenance.os.getuid()}/com.brainlayer.watch"]]
    monkeypatch.setattr(maintenance, "_bootout_service", lambda _service: False)
    monkeypatch.setattr(maintenance, "_service_is_loaded", lambda _service, **kwargs: True)
    monkeypatch.setattr(maintenance, "run_command", lambda args, **kwargs: subprocess.CompletedProcess(args, 0, "", ""))
    with pytest.raises(maintenance.MaintenanceAbort, match="failed to quiesce launchd service fleet-watchdog"):
        maintenance._quiesce_services(("watch",), {})


def test_maintenance_body_error_reports_resume_failures_as_exception_note(tmp_path, monkeypatch):
    from brainlayer import maintenance

    config = _config(tmp_path, now=dt.datetime(2026, 5, 30, 4, 5, tzinfo=dt.timezone.utc))
    config.queue_dir.mkdir(parents=True)
    config.db_path.write_bytes(b"db")
    resumed: list[str] = []
    monkeypatch.setattr(maintenance, "collect_lsof_entries", lambda _paths: [])
    monkeypatch.setattr(maintenance, "_service_is_loaded", lambda _service, **kwargs: False)
    monkeypatch.setattr(maintenance, "_bootout_service", lambda _service: True)
    monkeypatch.setattr(
        maintenance,
        "_checkpoint_full",
        lambda _db_path: (_ for _ in ()).throw(maintenance.MaintenanceAbort("checkpoint failed")),
    )

    def fail_mid_resume(_repo_root: Path, service: str) -> None:
        resumed.append(service)
        if service == "index":
            raise RuntimeError("bootstrap I/O error")

    monkeypatch.setattr(maintenance, "_resume_service", fail_mid_resume)

    with pytest.raises(maintenance.MaintenanceAbort, match="checkpoint failed") as exc_info:
        maintenance.run_maintenance("light", config=config)

    assert resumed == [*maintenance.DEFAULT_SERVICES, "fleet-watchdog"]
    assert "failed to resume 1 launchd service: index: bootstrap I/O error" in exc_info.value.reason
    assert any(
        "failed to resume 1 launchd service: index: bootstrap I/O error" in note
        for note in getattr(exc_info.value, "__notes__", [])
    )


def _latency_clock(ticks):
    def clock():
        try:
            return next(ticks)
        except StopIteration:
            pytest.fail("search latency probe exceeded its clock-sample budget")

    return clock


@pytest.mark.parametrize("fts", [False, True])
def test_search_latency_uses_five_warm_samples_and_ignores_outlier(tmp_path, monkeypatch, fts):
    import sqlite3

    from brainlayer import maintenance

    path = tmp_path / "latency.db"
    with sqlite3.connect(path) as conn:
        conn.execute("CREATE TABLE chunks(id TEXT)")
        if fts:
            conn.execute("CREATE VIRTUAL TABLE chunks_fts USING fts5(content)")
            conn.execute("INSERT INTO chunks_fts VALUES('brainlayer')")
    queries = []
    connect = maintenance.sqlite3.connect

    class TracedConnection:
        def __init__(self, *args, **kwargs):
            self.conn = connect(*args, **kwargs)

        def execute(self, sql, *args):
            queries.append(sql)
            return self.conn.execute(sql, *args)

        def close(self):
            self.conn.close()

    monkeypatch.setattr(maintenance.sqlite3, "connect", TracedConnection)
    ticks = iter([0, 0.04, 1, 1.041, 2, 2.9, 3, 3.039, 4, 4.042])
    monkeypatch.setattr(maintenance.time, "perf_counter", _latency_clock(ticks))
    assert maintenance._verify_search_latency(path) == pytest.approx(41)
    search_queries = [q for q in queries if "sqlite_master" not in q]
    assert len(search_queries) == 6  # one warm-up, then five timed samples
    assert len(set(search_queries)) == 1


def test_pathological_warm_search_latency_is_failure_not_deferral(tmp_path, monkeypatch):
    _create_enrichment_db(tmp_path / "latency.db")
    from brainlayer import maintenance

    ticks = iter([0, 0.6, 1, 1.6, 2, 2.6, 3, 3.6, 4, 4.6])
    monkeypatch.setattr(maintenance.time, "perf_counter", _latency_clock(ticks))
    with pytest.raises(maintenance.MaintenanceAbort, match="search latency") as caught:
        maintenance._verify_search_latency(tmp_path / "latency.db")
    assert caught.value.code == 1


def test_small_latency_overshoot_completes_with_logged_warning(tmp_path, monkeypatch):
    from brainlayer import maintenance

    config = _config(tmp_path, now=dt.datetime(2026, 5, 30, 4, 5, tzinfo=dt.UTC))
    _create_enrichment_db(config.db_path)
    monkeypatch.setattr(maintenance, "_run_gates", lambda _config: None)
    monkeypatch.setattr(maintenance, "_quiesce_services", lambda *_args: None)
    monkeypatch.setattr(maintenance, "_service_is_loaded", lambda _service, **kwargs: False)
    monkeypatch.setattr(maintenance, "_resume_services", lambda *_args: [])
    monkeypatch.setattr(maintenance, "_checkpoint_full", lambda _path: (0, 0, 0))
    monkeypatch.setattr(maintenance, "_verify_search_latency", lambda _path: 52.5)
    monkeypatch.setattr(maintenance, "_emit_telemetry", lambda _event: None)
    result = maintenance.run_maintenance("light", config=config)
    payload = maintenance._result_to_dict(result)
    assert payload["warnings"] == ["post-maintenance search latency above target: 52.5ms > 50.0ms"]
    event = json.loads(config.log_path.read_text().splitlines()[-1])
    assert event["warnings"] == payload["warnings"]
    monkeypatch.setattr(maintenance, "run_maintenance", lambda *args, **kwargs: result)
    assert maintenance.main(["--light", "--dry-run"]) == 0


@pytest.mark.parametrize("latency_ms", [50, 52.5, 500])
def test_nonpathological_warm_latency_does_not_abort(tmp_path, monkeypatch, latency_ms):
    from brainlayer import maintenance

    path = tmp_path / "latency.db"
    _create_enrichment_db(path)
    ticks = iter(t for i in range(5) for t in (i, i + latency_ms / 1000))
    monkeypatch.setattr(maintenance.time, "perf_counter", _latency_clock(ticks))
    assert maintenance._verify_search_latency(path) == pytest.approx(latency_ms)


@pytest.mark.parametrize("service", ["enrich", "enrichment"])
def test_maintenance_defaults_and_explicit_resume_cannot_activate_retired_services(tmp_path, monkeypatch, service):
    from brainlayer import maintenance

    assert service not in maintenance.DEFAULT_SERVICES
    assert service not in maintenance.REFEED_SERVICES
    commands = []
    monkeypatch.setenv("BRAINLAYER_LAUNCHD_DIR", str(tmp_path))
    monkeypatch.setattr(maintenance, "run_command", lambda args, **kw: commands.append(args))
    with pytest.raises(maintenance.MaintenanceAbort, match="retired"):
        maintenance._resume_service(tmp_path, service)
    assert commands == []
