import datetime as dt
import hashlib
import json
import os

import pytest

from brainlayer import jsonl_backup as backup


def test_keep_report_first_reason_and_next_age_gate(tmp_path):
    now = 2_000_000_000
    old = tmp_path / "claude-jsonl-2026-01-01.tar.gz"
    missing = tmp_path / "claude-jsonl-2026-01-02.tar.gz"
    newest = tmp_path / "claude-jsonl-2026-01-03.tar.gz"
    for p in (old, missing, newest):
        p.write_bytes(b"old")
        os.utime(p, (now - 30 * 86400, now - 30 * 86400))
    os.utime(newest, (now - 20 * 86400, now - 20 * 86400))
    state = tmp_path / "state"
    state.write_text(
        json.dumps(
            {
                "drive_archives": {
                    old.name: {
                        "verified": True,
                        "verified_at": now - 8 * 86400,
                        "name": old.name,
                        "size": 3,
                        "archive_id": "old",
                        "archive_md5": hashlib.md5(b"old").hexdigest(),
                        "sha256": hashlib.sha256(b"old").hexdigest(),
                    }
                }
            }
        )
    )
    before = state.read_bytes()
    log = tmp_path / "log"
    assert backup.prune_local_archives(tmp_path, state_path=state, log_path=log, now=now, surviving_archives={}) == []
    events = [json.loads(line) for line in log.read_text().splitlines()]
    assert len(events) == 1
    kept = {e["name"]: e for e in events[0]["kept"]}
    date = dt.datetime.fromtimestamp(now + 6 * 86400, dt.UTC).isoformat()
    assert kept[old.name]["first_reason"] == "verification_age"
    assert kept[old.name]["eligible_at"] == date
    assert kept[missing.name]["first_reason"] == "no_receipt"
    assert kept[missing.name]["eligible_at"] is None
    assert kept[newest.name]["first_reason"] == "newest_sibling"
    assert kept[newest.name]["eligible_at"] is None
    summary = next(e for e in events if e["event"] == "local_retention_summary")
    assert summary["bytes_held"] == 9
    assert summary["bytes_eligible_next"] == 3
    assert summary["next_eligible_at"] == date
    assert "subject to" in summary["message"]
    assert state.read_bytes() == before
    assert all(p.exists() for p in (old, missing, newest))


def test_keep_report_live_drive_gate_before_local_hash(tmp_path, monkeypatch):
    now = 2_000_000_000
    old = tmp_path / "claude-jsonl-2026-01-01.tar.gz"
    old.write_bytes(b"old")
    os.utime(old, (0, 0))
    (tmp_path / "claude-jsonl-2026-02-01.tar.gz").write_bytes(b"new")
    state = tmp_path / "state"
    state.write_text(
        json.dumps(
            {
                "drive_archives": {
                    old.name: {
                        "verified": True,
                        "verified_at": 0,
                        "archive_id": "gone",
                        "archive_md5": "expected",
                    }
                }
            }
        )
    )

    def refuse_hash(*args):
        raise AssertionError("Drive absence must short-circuit hashing")

    monkeypatch.setattr(backup, "_md5_file", refuse_hash)
    log = tmp_path / "log"
    backup.prune_local_archives(tmp_path, state_path=state, log_path=log, now=now, surviving_archives={})
    events = [json.loads(line) for line in log.read_text().splitlines()]
    kept = next(e for e in events[0]["kept"] if e["name"] == old.name)
    assert kept["first_reason"] == "drive_absent"
    assert kept["eligible_at"] is None
    assert old.exists()


@pytest.mark.parametrize(
    "fault,reason",
    [
        ("unverified", "unverified_receipt"),
        ("date", "verification_date_missing"),
        ("date_range", "verification_date_out_of_range"),
        ("md5_missing", "receipt_md5_missing"),
        ("remote_md5", "drive_md5_mismatch"),
        ("local_md5", "local_md5_mismatch"),
        ("sha256", "exact_receipt_mismatch"),
    ],
)
def test_first_reason_does_not_fabricate_an_eligibility_date(tmp_path, fault, reason):
    now = 2_000_000_000
    old = tmp_path / "claude-jsonl-2026-01-01.tar.gz"
    old.write_bytes(b"old")
    os.utime(old, (now - 30 * 86400, now - 30 * 86400))
    newest = tmp_path / "claude-jsonl-2026-02-01.tar.gz"
    newest.write_bytes(b"new")
    os.utime(newest, (now, now))
    receipt = dict(
        name=old.name,
        size=3,
        verified=True,
        verified_at=now - 30 * 86400,
        archive_id="old",
        archive_md5=hashlib.md5(b"old").hexdigest(),
        sha256=hashlib.sha256(b"old").hexdigest(),
    )
    live = {"old": receipt["archive_md5"]}
    if fault == "unverified":
        receipt["verified"] = False
    elif fault == "date":
        receipt.pop("verified_at")
    elif fault == "date_range":
        receipt["verified_at"] = 1e100
    elif fault == "md5_missing":
        receipt.pop("archive_md5")
    elif fault == "remote_md5":
        live["old"] = "wrong"
    elif fault == "local_md5":
        receipt["archive_md5"] = live["old"] = "wrong"
    else:
        receipt["sha256"] = "wrong"
    state = tmp_path / "state"
    state.write_text(json.dumps({"drive_archives": {old.name: receipt}}))
    log = tmp_path / "log"
    assert backup.prune_local_archives(tmp_path, state_path=state, log_path=log, now=now, surviving_archives=live) == []
    events = [json.loads(line) for line in log.read_text().splitlines()]
    kept = events[0]["kept"][0]
    assert kept["name"] == old.name
    assert kept["first_reason"] == reason
    assert kept["eligible_at"] is None
    summary = next(e for e in events if e["event"] == "local_retention_summary")
    assert summary["bytes_held"] == 3
    assert summary["bytes_eligible_next"] == 0
    assert summary["next_eligible_at"] is None
    assert old.exists()


def test_static_guard_accepts_reporting_but_rejects_an_unlink_before_continue():
    from pathlib import Path

    from brainlayer.backup_retention_invariant import inspect_jsonl_retention_invariant

    source = Path("src/brainlayer/jsonl_backup.py").read_text()
    other = Path("src/brainlayer/backup_daily.py").read_text()
    assert inspect_jsonl_retention_invariant(source, backup_daily_source=other) == []
    for replacement in ("path.unlink()", 'keep(path, "verification_age", path.unlink())'):
        unsafe = source.replace('keep(path, "verification_age")', replacement)
        assert "local unlink must follow" in " ".join(
            inspect_jsonl_retention_invariant(unsafe, backup_daily_source=other)
        )


@pytest.mark.parametrize("blocked,disabled", [("Drive inventory unavailable", False), (None, True)])
def test_blocked_sweep_reports_unknown_date_without_hashing_or_deleting(tmp_path, monkeypatch, blocked, disabled):
    old = tmp_path / "claude-jsonl-2026-01-01.tar.gz"
    old.write_bytes(b"old")
    os.utime(old, (0, 0))
    if disabled:
        monkeypatch.setenv("BRAINLAYER_JSONL_BACKUP_LOCAL_RETENTION", "0")
    monkeypatch.setattr(backup, "_md5_file", lambda *a: pytest.fail("blocked sweep must not hash"))
    log = tmp_path / "log"
    if disabled:
        assert (
            backup.prune_local_archives(
                tmp_path, state_path=tmp_path / "state", log_path=log, now=2_000_000_000, surviving_archives={}
            )
            == []
        )
    else:
        backup._report_blocked_local_archives(tmp_path, log_path=log, now=2_000_000_000, reason=blocked)
    events = [json.loads(line) for line in log.read_text().splitlines()]
    assert len(events) == 1
    assert events[0]["kept"][0]["first_reason"] == "sweep_blocked"
    assert events[0]["kept"][0]["eligible_at"] is None
    assert events[-1]["bytes_held"] == 3
    assert old.exists()


def test_empty_sweep_still_logs_one_summary(tmp_path):
    log = tmp_path / "log"
    assert (
        backup.prune_local_archives(
            tmp_path, state_path=tmp_path / "state", log_path=log, now=2_000_000_000, surviving_archives={}
        )
        == []
    )
    events = [json.loads(line) for line in log.read_text().splitlines()]
    assert len(events) == 1
    assert events[0]["bytes_held"] == 0


def test_reporting_failure_after_unlink_returns_deleted_list_and_finishes_sweep(tmp_path, monkeypatch):
    now = 2_000_000_000
    paths = [tmp_path / f"claude-jsonl-2026-01-0{i}.tar.gz" for i in range(1, 5)]
    for path in paths:
        path.write_bytes(b"archive")
        os.utime(path, (now - 30 * 86400, now - 30 * 86400))
    os.utime(paths[-1], (now, now))
    receipts = {
        p.name: dict(
            name=p.name,
            size=7,
            verified=True,
            verified_at=now - 30 * 86400,
            archive_id=p.name,
            archive_md5=hashlib.md5(b"archive").hexdigest(),
            sha256=hashlib.sha256(b"archive").hexdigest(),
        )
        for p in (paths[0], paths[2])
    }
    state = tmp_path / "state"
    state.write_text(json.dumps({"drive_archives": receipts}))
    monkeypatch.setattr(backup, "_local_archive_paths", lambda _: paths)
    append = backup._append_json_log

    def fail_reporting(log, entry):
        if entry["event"] == "local_retention_summary":
            raise OSError("reporting-only write failed")
        append(log, entry)

    monkeypatch.setattr(backup, "_append_json_log", fail_reporting)
    deleted = backup.prune_local_archives(
        tmp_path,
        state_path=state,
        log_path=tmp_path / "log",
        now=now,
        surviving_archives={r["archive_id"]: r["archive_md5"] for r in receipts.values()},
    )
    assert deleted == [paths[0].name, paths[2].name]
    assert not paths[0].exists() and not paths[2].exists()
    assert paths[1].exists() and paths[3].exists()
    rows = [json.loads(line) for line in (tmp_path / "log").read_text().splitlines()]
    assert [r["name"] for r in rows if r["event"] == "local_retention_deleted"] == deleted


@pytest.mark.parametrize("count", [1, 3])
def test_young_verified_files_have_an_honest_next_age_gate(tmp_path, count):
    now = 2_000_000_000
    paths = [tmp_path / f"claude-jsonl-2026-01-0{i}.tar.gz" for i in range(1, count + 1)]
    receipts = {}
    for i, path in enumerate(paths):
        path.write_bytes(b"new")
        age = 2 if i == 2 else 10
        os.utime(path, (now - age * 86400, now - age * 86400))
        receipts[path.name] = dict(verified=True, verified_at=now - (8 + i) * 86400)
    state = tmp_path / "state"
    state.write_text(json.dumps({"drive_archives": receipts}))
    log = tmp_path / "log"
    assert backup.prune_local_archives(tmp_path, state_path=state, log_path=log, now=now, surviving_archives={}) == []
    rows = [json.loads(line) for line in log.read_text().splitlines()]
    assert len(rows) == 1
    summary = rows[0]
    assert summary["bytes_held"] == 0
    assert summary["bytes_young"] == 3 * count
    assert summary["bytes_eligible_next"] == 3
    assert (
        summary["next_eligible_at"]
        == dt.datetime.fromtimestamp(now + (6 if count == 1 else 5) * 86400, dt.UTC).isoformat()
    )


@pytest.mark.parametrize("receipt", [None, [], "invalid"])
def test_young_file_with_malformed_receipt_reports_unknown_without_hashing(tmp_path, monkeypatch, receipt):
    now = 2_000_000_000
    path = tmp_path / "claude-jsonl-2026-01-01.tar.gz"
    path.write_bytes(b"new")
    os.utime(path, (now - 10 * 86400, now - 10 * 86400))
    state = tmp_path / "state"
    state.write_text(json.dumps({"drive_archives": {path.name: receipt}}))
    monkeypatch.setattr(backup, "_md5_file", lambda *a: pytest.fail("young archive must not hash"))
    log = tmp_path / "log"
    assert backup.prune_local_archives(tmp_path, state_path=state, log_path=log, now=now, surviving_archives={}) == []
    row = json.loads(log.read_text())
    assert row["bytes_young"] == 3
    assert row["next_eligible_at"] is None
    assert path.exists()


def test_blocked_reporting_has_one_row_with_all_old_archive_reasons(tmp_path):
    now = 2_000_000_000
    paths = [tmp_path / f"claude-jsonl-2026-01-0{i}.tar.gz" for i in range(1, 4)]
    for i, path in enumerate(paths):
        path.write_bytes(b"old")
        age = 30 if i < 2 else 2
        os.utime(path, (now - age * 86400, now - age * 86400))
    log = tmp_path / "log"
    backup._report_blocked_local_archives(tmp_path, log_path=log, now=now, reason="Drive unavailable")
    rows = [json.loads(line) for line in log.read_text().splitlines()]
    assert len(rows) == 1
    assert rows[0]["bytes_held"] == 6 and rows[0]["bytes_young"] == 3
    assert {e["name"] for e in rows[0]["kept"]} == {p.name for p in paths[:2]}
    assert all(set(e) == {"name", "bytes", "first_reason", "eligible_at"} for e in rows[0]["kept"])
    assert all(e["first_reason"] == "sweep_blocked" and e["eligible_at"] is None for e in rows[0]["kept"])
    assert not {"attempted_at", "uploaded", "verified", "status", "error"}.intersection(rows[0])
