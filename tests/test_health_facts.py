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
            '{"refresh_token": "r", "consented_at": "2026-10-20T00:00:00+00:00"}',
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
    assert after_facts.backup_revoked and after_facts.backup_local_verified_at is None
    assert after.check("backup_local").state == "red" and "revoked" in after.check("backup_local").note


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
    assert hf.embedding_fact(0, {}, NOW) == (None, {"count": 0, "since": None})
    since, state = hf.embedding_fact(5, {}, NOW)
    assert since == NOW
    later = NOW + timedelta(hours=2)
    assert hf.embedding_fact(7, state, later)[0] == NOW  # growing: still not shrinking since NOW
    assert hf.embedding_fact(4, state, later)[0] == later  # shrank: progress
    assert hf.embedding_fact(UNOBSERVED, state, later) == (UNOBSERVED, state)


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
    assert failed["ok"] is None and "vec0 unavailable" in failed["reason"]  # never fails maintenance
    assert maintenance.MaintenanceConfig().store_roundtrip_fn is None  # library callers opt in
