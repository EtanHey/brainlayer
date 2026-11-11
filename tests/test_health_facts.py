"""Verdict collectors: a source that is missing, malformed, stale or slow is UNOBSERVED, never a raise."""

from __future__ import annotations

import json
import os
import sqlite3
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

import brainlayer.health_facts as hf
from brainlayer.health_check import HealthCheckConfig, HealthCheckResult, HealthIssue
from brainlayer.health_verdict import UNOBSERVED, evaluate
from tests.verdict_evidence import backup_receipt, write_evidence

NOW = datetime(2026, 11, 2, 6, tzinfo=UTC)


def _config(tmp_path: Path, **overrides) -> HealthCheckConfig:
    paths = write_evidence(tmp_path, NOW)
    db = tmp_path / "brainlayer.db"
    with sqlite3.connect(db) as conn:
        conn.executescript(
            """
            CREATE TABLE chunks (id TEXT PRIMARY KEY, content TEXT, archived_at TEXT, superseded_by TEXT,
                                 aggregated_into TEXT, created_at TEXT);
            CREATE TABLE chunk_vectors_rowids (id TEXT PRIMARY KEY);
            """
        )
    return HealthCheckConfig(
        db_path=db,
        state_path=tmp_path / "health-state.json",
        queue_dir=tmp_path / "queue",
        pending_stores_path=tmp_path / "pending-stores.jsonl",
        pause_sentinel_path=tmp_path / "pause.sentinel",
        drain_health_path=tmp_path / "drain-health.json",
        **{**paths, **overrides},
    )


def _collect(config, prior=None, result=None):
    result = result or HealthCheckResult(checked_at=NOW.isoformat(), ok=True, canary_ok=True, canary_status="results")
    return hf.collect_facts(config, result, prior, NOW)


def test_fresh_evidence_fills_every_fact(tmp_path: Path) -> None:
    facts, state = _collect(_config(tmp_path))
    assert facts.backup_local_verified_at == facts.backup_offsite_verified_at == NOW - timedelta(hours=1)
    assert facts.transcripts_verified_at == NOW - timedelta(hours=1)
    assert (facts.drive_auth, facts.integrity, facts.store_canary_ok, facts.search_canary_ok) == (
        "ok",
        "ok",
        True,
        True,
    )
    assert facts.watcher_updated_at == NOW and facts.oldest_pending_store_at is None
    assert facts.embedding_stalled_since is None
    assert facts.disk_free_bytes > 0 and facts.disk_peak_bytes > 0 and facts.disk_floor_bytes == 2 * 1024**3
    assert state["store_canary_seen"] == (NOW - timedelta(hours=1)).isoformat()


@pytest.mark.parametrize(
    ("field", "source", "content", "expected"),
    [
        ("backup_local_verified_at", "backup_log_path", None, None),  # missing: none on record
        ("backup_local_verified_at", "backup_log_path", "not json\n", UNOBSERVED),
        ("transcripts_verified_at", "jsonl_backup_log_path", None, None),
        ("transcripts_verified_at", "jsonl_backup_log_path", "{torn\n", UNOBSERVED),
        ("watcher_updated_at", "watcher_health_path", None, None),
        ("watcher_updated_at", "watcher_health_path", '{"poll_count": 3}', UNOBSERVED),
        ("drive_auth", "drive_token_path", None, "consent_needed"),
        ("drive_auth", "drive_token_path", "[]", UNOBSERVED),
        ("drive_auth", "drive_token_path", '{"consented_at": "2026-11-01T00:00:00+00:00"}', "consent_needed"),
        (
            "drive_auth",
            "drive_token_path",
            '{"refresh_token": "r", "refresh_expires_at": "2026-10-20T00:00:00+00:00"}',
            "consent_needed",
        ),
        ("store_canary_ok", "store_roundtrip_path", None, UNOBSERVED),
        ("store_canary_ok", "store_roundtrip_path", '{"ok": "yes"}', UNOBSERVED),
        (
            "store_canary_ok",
            "store_roundtrip_path",
            json.dumps({"checked_at": (NOW - timedelta(hours=37)).isoformat(), "ok": True}),
            UNOBSERVED,
        ),
        ("store_canary_ok", "store_roundtrip_path", json.dumps({"checked_at": NOW.isoformat(), "ok": False}), False),
    ],
)
def test_source_missing_malformed_or_stale(tmp_path: Path, field, source, content, expected) -> None:
    config = _config(tmp_path)
    path = getattr(config, source)
    path.unlink()
    if content is not None:
        path.write_text(content)
    facts, _ = _collect(config)
    assert getattr(facts, field) == expected if expected is not UNOBSERVED else getattr(facts, field) is UNOBSERVED


def test_drive_receipt_after_consent_decides_and_reconsent_clears_it(tmp_path: Path) -> None:
    config = _config(tmp_path)
    lost = {
        "attempted_at": (NOW - timedelta(minutes=30)).isoformat(),
        "backup_log_provenance": "real",
        "error_type": "DriveCredentialError",
        "error": "Drive access expired: Reconnect in BrainBar",
    }
    with config.backup_log_path.open("a") as log:
        log.write(json.dumps(lost) + "\n")
    assert _collect(config)[0].drive_auth == "consent_needed"
    config.drive_token_path.write_text(
        json.dumps({"refresh_token": "r", "consented_at": (NOW - timedelta(minutes=10)).isoformat()})
    )
    assert _collect(config)[0].drive_auth == "ok"


def test_integrity_comes_from_confirmed_receipts_only(tmp_path: Path) -> None:
    config = _config(tmp_path)
    config.backup_log_path.write_text(
        json.dumps(
            backup_receipt(
                NOW - timedelta(hours=1), local_integrity="row 7 missing from index", verified=False, uploaded=False
            )
        )
        + "\n"
    )
    assert _collect(config)[0].integrity == "corrupt"
    config.backup_log_path.write_text(json.dumps({**backup_receipt(NOW), "local_integrity": "skipped"}) + "\n")
    assert _collect(config)[0].integrity is UNOBSERVED


def _verdict_ticks(config, ticks):
    """Run collectors + policy tick by tick, carrying both persisted states opaquely."""
    collectors, verdict_state, out = None, None, []
    for at, before in ticks:
        if before:
            before()
        result = HealthCheckResult(checked_at=at.isoformat(), ok=True, canary_ok=True, canary_status="results")
        facts, collectors = hf.collect_facts(config, result, collectors, at)
        verdict = evaluate(facts, prior=verdict_state)
        verdict_state = verdict.state
        out.append((facts, verdict))
    return out


def test_nightly_store_round_trip_fails_once_per_receipt_latches_and_recovers(tmp_path: Path) -> None:
    config = _config(tmp_path)

    def receipt(at, ok):
        return lambda: config.store_roundtrip_path.write_text(json.dumps({"checked_at": at.isoformat(), "ok": ok}))

    def ticks_after(night, ok, count=3):
        first = [(night + timedelta(minutes=5), receipt(night, ok))]
        return first + [(night + timedelta(minutes=5 * (i + 2)), None) for i in range(count - 1)]

    nights = [NOW + timedelta(days=d) for d in range(5)]
    for path in (config.watcher_health_path,):
        path.write_text(json.dumps({"updated_at": (NOW + timedelta(days=9)).isoformat()}))  # keep ingest out of it
    plan = ticks_after(nights[0], False) + ticks_after(nights[1], False) + ticks_after(nights[2], False)
    plan += ticks_after(nights[3], True)
    runs = _verdict_ticks(config, plan)
    canary = [facts.store_canary_ok for facts, _ in runs]
    stores = [verdict.check("stores").state for _, verdict in runs]
    # A failed receipt counts once, then the latch holds it; the third failed night is red.
    assert canary[:3] == [False, UNOBSERVED, UNOBSERVED] and stores[:3] == ["amber"] * 3
    assert stores[3:6] == ["amber"] * 3 and stores[6:9] == ["red"] * 3
    # A fresh success is positive evidence every tick: recovering, then green.
    assert canary[9:] == [True, True, True] and stores[9:] == ["red", "green", "green"]


def test_revoked_snapshot_stops_being_the_clock_end_to_end(tmp_path: Path) -> None:
    config = _config(tmp_path)
    good = backup_receipt(NOW - timedelta(hours=1))
    revocation = {**good, "attempted_at": NOW.isoformat(), "drive_md5_match": False, "verified": False}

    def revoke():
        with config.backup_log_path.open("a") as log:
            log.write(json.dumps(revocation) + "\n")

    (first, before), (after_facts, after) = _verdict_ticks(config, [(NOW, None), (NOW + timedelta(minutes=5), revoke)])
    assert before.check("backup_local").state == "green"
    assert after_facts.backup_local_verified_at is None
    assert after_facts.backup_offsite_verified_at is None
    assert after.check("backup_local").state == after.check("backup_offsite").state == "red"
    assert after.check("backup_local").age_s is None


def test_oldest_pending_store_spans_replay_file_and_drain_queue(tmp_path: Path) -> None:
    config = _config(tmp_path)
    config.pending_stores_path.write_text(
        json.dumps({"queued_at": (NOW - timedelta(minutes=20)).isoformat()}) + "\n{}\n"
    )
    config.queue_dir.mkdir()
    queued = config.queue_dir / "watcher-1.jsonl"
    queued.write_text("{}\n")
    old = (NOW - timedelta(hours=2)).timestamp()
    os.utime(queued, (old, old))
    assert _collect(config)[0].oldest_pending_store_at == NOW - timedelta(hours=2)
    # A drain that progressed after the file was queued is moving, not stuck behind it.
    config.drain_health_path.write_text(json.dumps({"last_progress_at": (NOW - timedelta(minutes=5)).isoformat()}))
    assert _collect(config)[0].oldest_pending_store_at == NOW - timedelta(minutes=20)
    config.pending_stores_path.write_text("not json\n")
    assert _collect(config)[0].oldest_pending_store_at is UNOBSERVED


def test_paused_drain_holds_its_queue_and_the_clock_restarts_at_resume(tmp_path: Path) -> None:
    config = _config(tmp_path)
    config.queue_dir.mkdir()
    queued = config.queue_dir / "watcher-1.jsonl"
    queued.write_text("{}\n")
    old = (NOW - timedelta(hours=2)).timestamp()
    os.utime(queued, (old, old))
    config.pause_sentinel_path.write_text(
        json.dumps({"labels": [config.drain_label], "expires_at": (NOW + timedelta(hours=1)).isoformat()})
    )
    paused, state = _collect(config)
    assert paused.oldest_pending_store_at is None and "drain paused" in paused.check_notes["stores"]
    config.pause_sentinel_path.unlink()
    resumed, _ = _collect(config, prior=state)
    assert resumed.oldest_pending_store_at == NOW


def test_embedding_stall_starts_when_the_backlog_stops_shrinking() -> None:
    window = NOW - timedelta(hours=24)
    assert hf.embedding_fact(0, {}, NOW, window) == (None, {"count": 0, "since": None, "window_start": None})
    since, state = hf.embedding_fact(5, {}, NOW, window)
    assert since == NOW and state["window_start"] == window.isoformat()
    later = NOW + timedelta(hours=2)
    assert hf.embedding_window_start(state, later) == window  # anchored while the backlog is outstanding
    assert hf.embedding_fact(7, state, later, window)[0] == NOW  # growing: still not shrinking since NOW
    assert hf.embedding_fact(4, state, later, window)[0] == later  # shrank: progress
    assert hf.embedding_fact(UNOBSERVED, state, later, window) == (UNOBSERVED, state)


def test_embedding_counts_recent_unvectored_chunks(tmp_path: Path) -> None:
    config = _config(tmp_path)
    with sqlite3.connect(config.db_path) as conn:
        conn.executemany(
            "INSERT INTO chunks (id, content, created_at) VALUES (?, 'x', ?)",
            [("a", (NOW - timedelta(minutes=5)).isoformat()), ("b", (NOW - timedelta(days=3)).isoformat())],
        )
    facts, state = _collect(config)
    assert facts.embedding_stalled_since == NOW and state["embedding"]["count"] == 1


def test_slow_or_failing_collector_is_unobserved_within_the_timeout(tmp_path: Path, monkeypatch) -> None:
    config = _config(tmp_path)
    monkeypatch.setattr(hf, "disk_fact", lambda *_: time.sleep(2))
    monkeypatch.setattr(hf, "watcher_fact", lambda *_: 1 / 0)
    started = time.monotonic()
    facts, _ = hf.collect_facts(config, HealthCheckResult(checked_at="", ok=True), None, NOW, timeout=0.3)
    assert time.monotonic() - started < 1.5
    assert facts.disk_free_bytes is UNOBSERVED and facts.watcher_updated_at is UNOBSERVED
    assert facts.backup_local_verified_at == NOW - timedelta(hours=1)


def test_issues_split_into_blind_codes_check_context_and_notes() -> None:
    result = HealthCheckResult(
        checked_at=NOW.isoformat(),
        ok=False,
        issues=[
            HealthIssue("process_snapshot_failed", "critical", "ps failed"),
            HealthIssue("hotlane_dead", "critical", "hotlane not running"),
            HealthIssue("job_alert_maintenance-full", "warning", "VACUUM skipped"),
            HealthIssue("retired_service_config", "critical", "retired label"),
        ],
    )
    blind, notes, context = hf.classify_issues(result)
    assert blind == ("process_snapshot_failed",)
    assert context == {"embedding": "hotlane not running"}
    assert [note.code for note in notes] == ["job_alert_maintenance-full", "retired_service_config"]


def test_search_canary_excused_by_a_valid_episode_is_unobserved() -> None:
    episode = {
        "kind": "snapshot",
        "owner": "backup",
        "episode_id": "e1",
        "expected_codes": ["brain_search_canary_failed"],
        "started_at": (NOW - timedelta(minutes=1)).isoformat(),
        "observed_at": NOW.isoformat(),
        "deadline": (NOW + timedelta(minutes=10)).isoformat(),
    }
    failed = HealthCheckResult(checked_at="", ok=False, canary_status="retrieval_failed")
    assert hf.search_canary_fact(failed, NOW) is False
    failed.issues.append(HealthIssue("brain_search_canary_failed", "critical", "refused", expected_episode=episode))
    assert hf.search_canary_fact(failed, NOW) is UNOBSERVED
    assert hf.search_canary_fact(HealthCheckResult(checked_at="", ok=True), NOW) is UNOBSERVED  # never ran


def test_maintenance_records_the_nightly_store_round_trip_receipt(tmp_path: Path) -> None:
    from brainlayer import maintenance
    from brainlayer.health_facts import store_canary_fact

    receipt_path = tmp_path / "store-roundtrip.json"
    config = maintenance.MaintenanceConfig(
        db_path=tmp_path / "brainlayer.db",
        store_roundtrip_fn=lambda _db: (True, 0.25, "vector_retrieved"),
        store_roundtrip_path=receipt_path,
    )
    receipt = maintenance._store_roundtrip(config)
    assert receipt == json.loads(receipt_path.read_text()) and receipt["ok"] is True
    assert store_canary_fact(receipt_path, datetime.now(UTC), None)[0] is True

    def broken(_db):
        raise RuntimeError("vec0 unavailable")

    failed = maintenance._store_roundtrip(
        maintenance.MaintenanceConfig(
            db_path=tmp_path / "x.db", store_roundtrip_fn=broken, store_roundtrip_path=tmp_path / "r.json"
        )
    )
    assert failed["ok"] is False and "vec0 unavailable" in failed["reason"]  # never fails maintenance
    assert maintenance.MaintenanceConfig().store_roundtrip_fn is None  # library callers opt in


# --- Review R1 regressions (each fails on deab4c7c) -------------------------------------------


@pytest.mark.parametrize("mode", ["light", "full"])
def test_scheduled_no_config_run_records_the_store_round_trip_once_after_resume(tmp_path, monkeypatch, mode):
    """B1: the decorated, config-less entry point main() calls must enable doctor's round trip."""
    import subprocess

    from brainlayer import maintenance

    real_config = maintenance.MaintenanceConfig
    receipt = tmp_path / "store-roundtrip.json"
    temp = {
        "db_path": tmp_path / "brainlayer.db",
        "queue_dir": tmp_path / "queue",
        "quarantine_root": tmp_path / "quarantine",
        "log_path": tmp_path / "maintenance.log",
        "backup_staging_dir": tmp_path / "staging",
        "store_roundtrip_path": receipt,
    }
    monkeypatch.setattr(maintenance, "MaintenanceConfig", lambda **kw: real_config(**{**temp, **kw}))
    events: list[str] = []
    monkeypatch.setattr(maintenance, "run_command", lambda args, **_kw: subprocess.CompletedProcess(args, 0, "", ""))
    monkeypatch.setattr(maintenance, "PAUSE_SENTINEL_PATH", tmp_path / "pause.sentinel")
    monkeypatch.setattr(maintenance, "_assert_running_merged_code", lambda _root: None)
    monkeypatch.setattr(maintenance, "_run_gates", lambda _config: None)
    monkeypatch.setattr(maintenance, "_weekly_backup", lambda _config: None)
    # #1582: full probes the freelist before the hold; a low ratio takes its no-VACUUM completion path.
    monkeypatch.setattr(maintenance, "_freelist_ratio", lambda _path: 0.0)
    monkeypatch.setattr(maintenance, "_quiesce_services", lambda *_args: events.append("quiesce"))
    monkeypatch.setattr(maintenance, "_service_is_loaded", lambda _service, **_kw: False)
    monkeypatch.setattr(maintenance, "_resume_services", lambda *_args: events.append("resume") or [])
    monkeypatch.setattr(maintenance, "_checkpoint_full", lambda _path: (0, 0, 0))
    monkeypatch.setattr(maintenance, "_verify_search_latency", lambda _path: 1.0)
    monkeypatch.setattr(maintenance, "_emit_telemetry", lambda _event: None)
    monkeypatch.setattr(
        maintenance, "_doctor_roundtrip", lambda db: events.append(f"roundtrip:{db}") or (True, 0.1, "vector_retrieved")
    )

    maintenance.run_maintenance(mode)

    assert events == ["quiesce", "resume", f"roundtrip:{temp['db_path']}"]  # once, after services resume
    assert json.loads(receipt.read_text())["ok"] is True
    logged = json.loads(temp["log_path"].read_text().splitlines()[-1])["store_roundtrip"]
    assert logged["ok"] is True and logged["reason"] == "vector_retrieved"

    # Explicit library opt-out still holds, and burn never runs it.
    events.clear()
    maintenance.run_maintenance(mode, config=real_config(**{**temp, "store_roundtrip_fn": None}))
    maintenance.run_maintenance("burn", config=real_config(**{**temp, "store_roundtrip_fn": lambda _db: 1 / 0}))
    assert not any(event.startswith("roundtrip") for event in events)


def _unblock_fifo(path: Path) -> None:
    try:
        os.close(os.open(path, os.O_WRONLY | os.O_NONBLOCK))
    except OSError:
        pass


def _collect_in_thread(config, timeout: float, wait_s: float = 3.0):
    import threading

    out: dict = {}
    worker = threading.Thread(
        target=lambda: out.update(
            facts=hf.collect_facts(config, HealthCheckResult(checked_at="", ok=True), None, NOW, timeout=timeout)[0]
        ),
        daemon=True,
    )
    started = time.monotonic()
    worker.start()
    worker.join(wait_s)
    return out.get("facts"), time.monotonic() - started


@pytest.mark.parametrize(
    ("source", "field"),
    [("drive_token_path", "drive_auth"), ("pending_stores_path", "oldest_pending_store_at")],
)
def test_a_read_that_never_returns_is_bounded_by_the_shared_deadline(tmp_path, source, field) -> None:
    """B2: a FIFO with no writer blocks open() forever, like a stalled mount; it must not hold the tick."""
    config = _config(tmp_path)
    fifo = getattr(config, source)
    fifo.unlink(missing_ok=True)
    os.mkfifo(fifo)
    try:
        facts, elapsed = _collect_in_thread(config, timeout=0.3)
        assert facts is not None, f"collect_facts still blocked on {source} after {elapsed:.1f}s"
        assert elapsed < 1.5
        assert getattr(facts, field) is UNOBSERVED
        assert facts.backup_local_verified_at == NOW - timedelta(hours=1)  # the rest still report
    finally:
        _unblock_fifo(fifo)


def test_a_stuck_collector_worker_cannot_keep_the_producer_process_alive(tmp_path) -> None:
    """B2: an abandoned worker must not be joined at interpreter exit."""
    import subprocess
    import sys

    script = (
        "import threading, time\n"
        "import brainlayer.health_facts as hf\n"
        "got = hf._run_bounded({'stuck': threading.Event().wait, 'ok': lambda: 7}, time.monotonic() + 0.2)\n"
        "print(got['stuck'] is hf.UNOBSERVED, got['ok'])\n"
    )
    src = str(Path(hf.__file__).resolve().parents[1])  # the source tree under test
    env = {**os.environ, "PYTHONPATH": src + os.pathsep + os.environ.get("PYTHONPATH", "")}
    started = time.monotonic()
    try:
        done = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, timeout=20, env=env)
    except subprocess.TimeoutExpired:
        pytest.fail("the producer process stayed alive behind a stuck collector worker")
    assert done.returncode == 0, done.stderr
    assert done.stdout.split() == ["True", "7"]
    assert time.monotonic() - started < 20


def _gib(n: float) -> int:
    return int(n * 1024**3)


@pytest.mark.parametrize(
    ("evidence", "dot"),
    [
        (None, "red"),  # no capability evidence: budget the larger (legacy) requirement
        ("legacy", "red"),  # the staging filesystem fell back to legacy last time
        ("reclaim", "green"),  # the last backup here reclaimed its raw: the smaller budget applies
    ],
)
def test_disk_budgets_the_mode_the_next_backup_will_use(tmp_path, monkeypatch, evidence, dot) -> None:
    """B3: 30 GiB Available sits between the reclaim (~22 GiB) and legacy (~42 GiB) budgets."""
    from types import SimpleNamespace

    from brainlayer import backup_daily as bd
    from brainlayer import disk_space

    config = _config(tmp_path)
    monkeypatch.setattr(bd, "_database_logical_size_bytes", lambda _db: _gib(20))
    monkeypatch.setattr(bd, "_estimated_gzip_bytes", lambda _dir, _size: _gib(10))
    monkeypatch.setattr(bd, "_should_run_full_verify", lambda _date: True)
    monkeypatch.setattr(disk_space, "measure_disk_space", lambda _path: SimpleNamespace(available_bytes=_gib(30)))
    probes: list = []
    monkeypatch.setattr(bd, "_probe_raw_reclamation", lambda *a: probes.append(a) or (True, None))
    reclaim = bd._peak_backup_required_bytes(_gib(20), _gib(10), full=True, reclaim_raw=True)
    legacy = bd._peak_backup_required_bytes(_gib(20), _gib(10), full=True, reclaim_raw=False)
    assert reclaim + 2 * bd.MIN_RAW_FREE_BYTES < _gib(30) < legacy
    if evidence is not None:
        snapshot = str(config.backup_staging_dir / "2026-11-02.db.zst")
        basis = "volume_delta" if evidence == "reclaim" else None
        config.backup_log_path.write_text(
            json.dumps(backup_receipt(NOW - timedelta(hours=1), snapshot=snapshot, reclaim_basis=basis)) + "\n"
        )

    facts, _ = _collect(config)

    assert facts.disk_peak_bytes == (reclaim if evidence == "reclaim" else legacy)
    assert evaluate(facts).check("disk").state == dot
    assert probes == []  # never the write-based hole-punch probe on a health tick


def test_reclaim_capability_needs_an_artifact_receipt_for_this_staging_dir(tmp_path) -> None:
    staging = tmp_path / "staging"
    staging.mkdir()
    here = {"snapshot": str(staging / "a.db.zst")}
    assert hf.reclaim_capability([], staging) is None
    assert hf.reclaim_capability([{**here}], staging) is None  # pre-reclaim receipt: no evidence
    assert hf.reclaim_capability([{**here, "reclaim_basis": "block_drop_no_snapshot"}], staging) is True
    assert hf.reclaim_capability([{**here, "reclaim_basis": None}], staging) is False
    elsewhere = {"snapshot": "/Volumes/other/a.db.zst", "reclaim_basis": "volume_delta"}
    assert hf.reclaim_capability([{**here, "reclaim_basis": None}, elsewhere], staging) is None
    assert hf.disk_budget(40, 20, None) == 40 and hf.disk_budget(40, 20, True) == 20


def test_embedding_stall_does_not_recover_when_its_rows_age_out(tmp_path) -> None:
    """B4: time advances, nothing in the DB changes, the stuck row must stay counted."""
    config = _config(tmp_path)
    with sqlite3.connect(config.db_path) as conn:
        conn.execute(
            "INSERT INTO chunks (id, content, created_at) VALUES ('stuck', 'x', ?)",
            ((NOW - timedelta(hours=23, minutes=55)).isoformat(),),
        )
    state, verdict_state, seen = None, None, []
    ticks = [NOW + timedelta(minutes=m) for m in range(0, 12, 2)]  # the row ages out at minute 5
    ticks += [NOW + timedelta(minutes=m) for m in range(20, 7 * 60 + 1, 10)]  # continuous ticks for 7 h
    for at in ticks:
        result = HealthCheckResult(checked_at=at.isoformat(), ok=True, canary_ok=True, canary_status="results")
        facts, state = hf.collect_facts(config, result, state, at)
        verdict = evaluate(facts, prior=verdict_state)
        verdict_state = verdict.state
        seen.append((state["embedding"]["count"], facts.embedding_stalled_since, verdict.check("embedding").state))
    assert [count for count, _, _ in seen] == [1] * len(ticks)
    assert all(since == NOW for _, since, _ in seen)
    assert seen[3][2] == "green" and seen[-1][2] == "red"  # under an hour, then a real 7 h stall

    # Only completion (or a lifecycle change) repays it.
    later = NOW + timedelta(hours=7, minutes=5)
    with sqlite3.connect(config.db_path) as conn:
        conn.execute("INSERT INTO chunk_vectors_rowids (id) VALUES ('stuck')")
    facts, done = hf.collect_facts(config, HealthCheckResult(checked_at="", ok=True), state, later)
    assert facts.embedding_stalled_since is None and done["embedding"]["count"] == 0


def test_embedding_lifecycle_change_repays_the_stall(tmp_path) -> None:
    config = _config(tmp_path)
    with sqlite3.connect(config.db_path) as conn:
        conn.execute("INSERT INTO chunks (id, content, created_at) VALUES ('s', 'x', ?)", (NOW.isoformat(),))
    _, state = _collect(config)
    with sqlite3.connect(config.db_path) as conn:
        conn.execute("UPDATE chunks SET archived_at = ? WHERE id = 's'", (NOW.isoformat(),))
    facts, _ = hf.collect_facts(config, HealthCheckResult(checked_at="", ok=True), state, NOW + timedelta(hours=30))
    assert facts.embedding_stalled_since is None


def test_a_newer_raising_probe_replaces_an_older_success(tmp_path) -> None:
    """B5: the newest attempt is the evidence; a raise is a timestamped failed attempt."""
    from brainlayer import maintenance

    receipt_path = tmp_path / "store-roundtrip.json"
    old = {"checked_at": (datetime.now(UTC) - timedelta(hours=1)).isoformat(), "ok": True, "reason": "vector_retrieved"}
    receipt_path.write_text(json.dumps(old))

    def raising(_db):
        raise RuntimeError("new probe failed")

    config = maintenance.MaintenanceConfig(
        db_path=tmp_path / "x.db", store_roundtrip_fn=raising, store_roundtrip_path=receipt_path
    )
    attempt = maintenance._store_roundtrip(config)  # does not raise: maintenance never fails on it
    on_disk = json.loads(receipt_path.read_text())
    assert attempt == on_disk and on_disk["ok"] is False and "new probe failed" in on_disk["reason"]
    assert on_disk["checked_at"] > old["checked_at"]
    assert hf.store_canary_fact(receipt_path, datetime.now(UTC), None)[0] is False


def test_an_unrecordable_attempt_leaves_no_stale_success_standing(tmp_path, monkeypatch) -> None:
    from brainlayer import maintenance

    receipt_path = tmp_path / "store-roundtrip.json"
    receipt_path.write_text(json.dumps({"checked_at": datetime.now(UTC).isoformat(), "ok": True}))
    monkeypatch.setattr(hf, "write_store_roundtrip_receipt", lambda *_a, **_kw: 1 / 0)
    config = maintenance.MaintenanceConfig(
        db_path=tmp_path / "x.db", store_roundtrip_fn=lambda _db: (True, 0.1, "ok"), store_roundtrip_path=receipt_path
    )
    attempt = maintenance._store_roundtrip(config)
    assert attempt["ok"] is None and "not recorded" in attempt["reason"]
    assert hf.store_canary_fact(receipt_path, datetime.now(UTC), None)[0] is UNOBSERVED


def test_disk_timeout_reuses_measurement_only_within_thirty_minutes(tmp_path, monkeypatch):
    config = _config(tmp_path)
    monkeypatch.setattr(hf, "disk_fact", lambda *_: (100_000_000_000, 10, 5, 2))
    result = HealthCheckResult(checked_at="", ok=True)
    _, state = hf.collect_facts(config, result, None, NOW)
    monkeypatch.setattr(hf, "disk_fact", lambda *_: UNOBSERVED)
    for minute in (5, 10, 15):
        facts, state = hf.collect_facts(config, result, state, NOW + timedelta(minutes=minute))
        assert evaluate(facts).check("disk").state == "green"
        assert "earlier measurement" in facts.check_notes["disk"]
    facts, _ = hf.collect_facts(config, result, state, NOW + timedelta(minutes=31))
    assert facts.disk_free_bytes is UNOBSERVED


@pytest.mark.parametrize(
    "mode, encoded, batch_minutes, expected",
    [
        ("running", 16, 480, "amber"),
        ("high load", 0, 2, "amber"),
        ("high load", 0, 480, "amber"),
        ("waiting", 0, 480, "red"),
    ],
)
@pytest.mark.parametrize("count,age_hours", [(1, 0.1), (3500, 2), (3500, 8)])
def test_embedding_completed_batches_and_load_pause_are_not_net_stalls(
    tmp_path, mode, encoded, batch_minutes, expected, count, age_hours
):
    config = _config(tmp_path)
    with sqlite3.connect(config.db_path) as conn:
        conn.executemany(
            "INSERT INTO chunks VALUES (?, 'text', NULL, NULL, NULL, ?)",
            [(f"pending-{i}", NOW.isoformat()) for i in range(count)],
        )
    prior = {
        "embedding": {
            "count": count,
            "since": (NOW - timedelta(hours=age_hours)).isoformat(),
            "window_start": (NOW - timedelta(days=1)).isoformat(),
        }
    }
    payload = {
        "status": "measured",
        "db": str(config.db_path.resolve()),
        "measured_at": NOW.isoformat(),
        "backlog_encoded": encoded,
        "backlog_cpu_seconds": 2,
        "backlog_state": mode,
        "backlog_last_batch_at": (NOW - timedelta(minutes=batch_minutes)).isoformat(),
    }
    (tmp_path / "hotlane-cpu.json").write_text(json.dumps(payload))
    (tmp_path / "hotlane-budget.json").write_text(json.dumps({"backlog_max_load": 10}))
    facts, _ = _collect(config, prior)
    check = evaluate(facts).check("embedding")
    assert facts.embedding_stalled_since == NOW - timedelta(hours=age_hours)
    if age_hours < 1:
        expected = "green"
    elif age_hours == 2:
        expected = "amber"
    if expected == "red":
        from dataclasses import replace

        baseline = evaluate(replace(facts, embedding_activity=None)).check("embedding")
        assert check.state == baseline.state == "red"
        assert "not shrunk" in check.note
    elif expected == "green":
        assert check.state == "green"
    else:
        assert check.state == "amber"
        assert "not shrunk" in check.note
        if facts.embedding_activity:
            assert f"{count} queued" in check.note
            assert "last batch" in check.note
            assert ("paused by machine load" if mode == "high load" else "progressing") in check.note


@pytest.mark.parametrize("stale", [False, True])
def test_load_gate_requires_fresh_evidence_even_without_completed_batch(tmp_path, stale):
    cpu = {
        "status": "measured",
        "db": str(tmp_path.resolve()),
        "measured_at": (NOW - timedelta(minutes=3) if stale else NOW).isoformat(),
        "backlog_encoded": 0,
        "backlog_cpu_seconds": 0,
        "backlog_state": "high load",
    }
    _, text = hf.embedding_activity(cpu, {"backlog_max_load": 10}, tmp_path, NOW)
    assert (text is None) == stale
    if text:
        assert "paused by machine load" in text
        assert "time unavailable" in text


def test_r2_collector_pairs_tick_with_boot_and_awake_clock(tmp_path, monkeypatch):
    from types import SimpleNamespace

    import brainlayer.health_check as hc

    class Clock(datetime):
        @classmethod
        def now(cls, tz=None):
            return NOW + timedelta(seconds=5)

    calls = []
    monkeypatch.setattr(hf, "sys", SimpleNamespace(platform="darwin"))
    monkeypatch.setattr(hf, "datetime", Clock)
    monkeypatch.setattr(hf.time, "monotonic", lambda: 1000.0)
    monkeypatch.setattr(
        hc,
        "_default_command_runner",
        lambda args: calls.append(args) or SimpleNamespace(returncode=0, stdout="test-boot\n"),
    )
    facts, _ = _collect(_config(tmp_path))
    assert calls == [["/usr/sbin/sysctl", "-n", "kern.bootsessionuuid"]]
    assert facts.boot_session == "test-boot"
    assert facts.awake_clock_s == 995.0


@pytest.mark.parametrize("deadline", [None, "permanent", "2026-11-30T00:00:00+00:00"])
def test_old_consent_does_not_expire_without_reported_deadline(deadline):
    from brainlayer.health_facts import drive_auth_fact

    token = {"refresh_token": "fixture", "consented_at": "2000-01-01T00:00:00+00:00"}
    if deadline is not None:
        token["refresh_expires_at"] = deadline
    assert drive_auth_fact(token, NOW, []) == "ok"


def test_reported_deadline_and_receipts_determine_offline_auth():
    from brainlayer.health_facts import drive_auth_fact

    token = {
        "refresh_token": "fixture",
        "consented_at": (NOW - timedelta(hours=1)).isoformat(),
        "refresh_expires_at": NOW.isoformat(),
    }
    assert drive_auth_fact(token, NOW, []) == "consent_needed"
    token["refresh_expires_at"] = "permanent"
    assert drive_auth_fact(token, NOW, [(NOW - timedelta(hours=2), "consent_needed")]) == "ok"
    assert drive_auth_fact(token, NOW, [(NOW, "refresh_failed")]) == "refresh_failed"
    assert drive_auth_fact(token, NOW, [(NOW, "ok")]) == "ok"
