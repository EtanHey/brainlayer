"""History-replay ratchet for the outcome verdict (S1).

Fixtures under ``tests/fixtures/verdict_replay/`` are real receipts from both Macs, 2026-10-12 to
2026-11-02 (``backup-daily.log``, ``jsonl-backup.log``, ``maintenance.log``), with resumable-upload
URLs dropped and home directories renamed. ``oracle.json`` lists the evidence events, built by
separate code (see the fixture README) with an explicit uncertainty window: an invocation's
completion time is not recorded, so its receipt is known somewhere between ``known_from`` and
``known_by`` (latest start seen in journal order + 8 h, the writers' run-limit upper bound; see ``build_oracle.py``). An hour
where the two bounds disagree is ``unknown`` and is not asserted, and so is an hour with no backup
evidence at all: missing history is not proof that no verified copy existed.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

from brainlayer import health_verdict as hv

FIXTURES = Path(__file__).parent / "fixtures" / "verdict_replay"
LOCAL = ZoneInfo("Asia/Jerusalem")
START = datetime(2026, 10, 19, 0, tzinfo=LOCAL)
END = datetime(2026, 11, 2, 0, tzinfo=LOCAL)
# Ticks are an hour apart here; widen the monitor-gap so the replay reads as continuous monitoring.
THRESHOLDS = hv.Thresholds(sleep_gap_s=2 * 3600)
REPLAYED = ("backup_local", "backup_offsite", "transcripts", "drive_auth", "integrity")
# The oracle's own copy of the plan §3 limits (amber, red), in hours. Deliberately not read from
# hv.Thresholds, so a change to a production default cannot move expected and actual together.
ORACLE_LIMITS_H = {"backup_local": (30, 54), "backup_offsite": (30, 78), "transcripts": (30, 54)}

# The ratchet row: every red hour (UTC, inclusive start, exclusive end) the receipts support.
# M1: Drive consent expired 2026-10-24T18:50Z; the first receipts to see it are the 10-25 02:01Z
# runs, certain once their 8 h completion bound passes (11:00Z); the next verified upload is the
# 10-29 02:00Z run. M4: no verified backup between the
# 10-17 nightly and 10-21 00:45Z (10-18 upload stalled, 10-19 disk short then upload timed out,
# 10-20 failed), and 10-24 01:15Z stayed the newest snapshot past 54 h on 10-26 (two failed
# nightlies; the 08:30Z manual run recovered it).
EXPECTED_RED: dict[str, dict[str, list[tuple[str, str]]]] = {
    "m1": {
        "drive_auth": [("2026-10-25T11:00Z", "2026-10-29T03:00Z")],
        "transcripts": [("2026-10-26T08:00Z", "2026-10-29T03:00Z")],
        "backup_offsite": [("2026-10-27T08:00Z", "2026-10-29T03:00Z")],
    },
    "m4": {
        "backup_local": [("2026-10-19T07:00Z", "2026-10-21T01:00Z"), ("2026-10-26T08:00Z", "2026-10-26T09:00Z")],
        "backup_offsite": [("2026-10-20T07:00Z", "2026-10-21T02:00Z")],
    },
}


def _load(name: str) -> list[dict]:
    return [json.loads(line) for line in (FIXTURES / name).read_text().splitlines() if line.strip()]


def _backup_prefix(rows: list[dict], at: datetime) -> list[dict]:
    """Only receipts written by this hour, including terminal copies with old attempt times."""
    known = []
    latest_start = None
    for row in rows:
        if row.get("attempted_at"):
            started = datetime.fromisoformat(row["attempted_at"])
            latest_start = max(latest_start or started, started)
        if latest_start is not None and latest_start > at:
            break
        known.append(row)
    return known


def _hours(start: datetime, end: datetime):
    t = start
    while t < end:
        yield t.astimezone(UTC)
        t += timedelta(hours=1)


def _job_alerts(rows: list[dict], at: datetime) -> dict[str, str]:
    """maintenance.log: an aborted run is an active ``maintenance-<mode>`` alert until a clean run."""
    active: dict[str, str] = {}
    for row in rows:
        ts = datetime.fromisoformat(row["ts"])
        if ts > at or not row.get("mode"):
            continue
        key = f"maintenance-{row['mode']}"
        if row.get("status") in ("aborted", "failed"):
            active[key] = f"BrainLayer {row['mode']} maintenance: {row.get('reason')}"
        elif row.get("status") is None and "checkpoint" in row:
            active.pop(key, None)
    return active


def _oracle_state(events: list[dict], check: str, at: datetime) -> str:
    mine = [e for e in events if e["check"] == check]
    lo = [e for e in mine if datetime.fromisoformat(e["known_by"]) <= at]
    hi = [e for e in mine if datetime.fromisoformat(e["known_from"]) <= at]

    def state(known: list[dict]) -> str:
        if check in ("drive_auth", "integrity"):
            if not known:
                return hv.UNKNOWN
            last = max(known, key=lambda e: e["known_from"])["value"]
            return {"ok": hv.GREEN, "consent_needed": hv.RED, "refresh_failed": hv.AMBER}.get(last, hv.RED)
        if not known:
            return hv.UNKNOWN  # no history for this period, not evidence that no copy existed
        if check == "backup_local":
            latest = max(known, key=lambda e: e["known_from"])
            if latest["value_at"] is None:
                return hv.RED
            newest = datetime.fromisoformat(latest["value_at"])
        else:
            newest = max(datetime.fromisoformat(e["value_at"]) for e in known)
        age = (at - newest).total_seconds()
        amber, red = (hours * 3600 for hours in ORACLE_LIMITS_H[check])
        return hv.RED if age > red else hv.AMBER if age > amber else hv.GREEN

    low, high = state(lo), state(hi)
    return low if low == high else hv.UNKNOWN


def _replay(mac: str):
    backups = _load(f"{mac}-backup-daily.log")
    transcripts = _load(f"{mac}-jsonl-backup.log")
    maintenance = _load(f"{mac}-maintenance.log")
    prior = None
    for at in _hours(START, END):
        facts_b = hv.backup_facts(_backup_prefix(backups, at), as_of=at)
        newest_transcript, drive_t = hv.transcript_facts(transcripts, as_of=at)
        drive = max((x for x in (facts_b.drive, drive_t) if x is not None), default=None, key=lambda x: x[0])
        verdict = hv.evaluate(
            hv.Facts(
                now=at,
                **facts_b.facts_kwargs(),
                transcripts_verified_at=newest_transcript,
                drive_auth=drive[1] if drive else hv.UNOBSERVED,
                notes=hv.notes_from_job_alerts(_job_alerts(maintenance, at)),
            ),
            prior,
            THRESHOLDS,
        )
        prior = verdict.state
        yield at, verdict


def _intervals(hours: list[datetime]) -> list[tuple[str, str]]:
    spans: list[list[datetime]] = []
    for hour in hours:
        if spans and hour - spans[-1][1] == timedelta(hours=1):
            spans[-1][1] = hour
        else:
            spans.append([hour, hour])
    fmt = "%Y-%m-%dT%H:%MZ"
    return [(a.strftime(fmt), (b + timedelta(hours=1)).strftime(fmt)) for a, b in spans]


@pytest.mark.parametrize("mac", ["m4", "m1"])
def test_replay_matches_the_evidence_oracle(mac):
    events = json.loads((FIXTURES / "oracle.json").read_text())[mac]["events"]
    checked = 0
    problems = []
    for at, verdict in _replay(mac):
        for name in REPLAYED:
            expected = _oracle_state(events, name, at)
            if expected == hv.UNKNOWN:
                continue
            checked += 1
            if verdict.check(name).state != expected:
                problems.append((at.isoformat(), name, verdict.check(name).state, expected))
    assert checked > 1000
    assert not problems, problems[:10]


@pytest.mark.parametrize("mac", ["m4", "m1"])
def test_replay_red_only_where_the_receipts_show_memory_at_risk(mac):
    events = json.loads((FIXTURES / "oracle.json").read_text())[mac]["events"]
    red: dict[str, list[datetime]] = {}
    for at, verdict in _replay(mac):
        active = hv.badge_alerts(verdict)
        # The 1.5.83 reader's invariant, and job alerts never light the dot.
        assert (verdict.dot == hv.RED) == any(item["severity"] != "warning" for item in active)
        assert all(item["severity"] == "warning" for item in active if item["code"].startswith("job_alert_"))
        assert verdict.dot != hv.RED or any(verdict.check(n).state == hv.RED for n in REPLAYED)
        for name in REPLAYED:
            if verdict.check(name).state == hv.RED and _oracle_state(events, name, at) != hv.UNKNOWN:
                red.setdefault(name, []).append(at)
    assert {name: _intervals(hours) for name, hours in red.items()} == EXPECTED_RED[mac]


def test_oracle_calls_a_period_without_backup_evidence_unknown_not_red():
    at = datetime(2026, 10, 20, tzinfo=UTC)
    for check in ("backup_local", "backup_offsite", "transcripts", "drive_auth", "integrity"):
        assert _oracle_state([], check, at) == hv.UNKNOWN


def test_oracle_limits_are_pinned_independently_of_production_defaults():
    assert ORACLE_LIMITS_H == {"backup_local": (30, 54), "backup_offsite": (30, 78), "transcripts": (30, 54)}
    event = {"check": "backup_local", "value": "verified", "value_at": "2026-10-20T00:00:00Z"}
    known = {**event, "known_from": "2026-10-20T00:00:00Z", "known_by": "2026-10-20T00:00:00Z"}
    assert _oracle_state([known], "backup_local", datetime(2026, 10, 22, 6, tzinfo=UTC)) == hv.AMBER  # 54 h
    assert _oracle_state([known], "backup_local", datetime(2026, 10, 22, 6, 1, tzinfo=UTC)) == hv.RED


@pytest.mark.parametrize("uploaded", [False, True])
def test_oracle_snapshot_removal_requires_its_own_verified_drive_copy(tmp_path, uploaded):
    import runpy

    builder = runpy.run_path(str(FIXTURES / "build_oracle.py"))
    rows = [
        {
            "backup_log_provenance": "real",
            "attempted_at": "2026-10-20T00:00:00Z",
            "snapshot": "/b/older.db.zst",
            "snapshot_created_at": "2026-10-19T00:00:00Z",
            "uploaded": True,
            "verified": True,
            "local_removed": True,
        },
        {
            "backup_log_provenance": "real",
            "attempted_at": "2026-10-21T00:00:00Z",
            "snapshot": "/b/newer.db.zst",
            "snapshot_created_at": "2026-10-20T00:00:00Z",
            "local_verified": True,
            "local_integrity": "ok",
            "uploaded": uploaded,
            "verified": uploaded,
        },
        {
            "backup_log_provenance": "real",
            "attempted_at": "2026-10-21T01:00:00Z",
            "snapshot": "/b/newer.db.zst",
            "local_removed": True,
        },
    ]
    (tmp_path / "m4-backup-daily.log").write_text("\n".join(json.dumps(row) for row in rows))
    (tmp_path / "m4-jsonl-backup.log").write_text("")
    events = builder["events_for"](tmp_path, "m4")
    snapshots = [e for e in events if e["check"] == "backup_local"]
    assert snapshots[-1]["value_at"] == ("2026-10-20T00:00:00Z" if uploaded else "2026-10-19T00:00:00Z")


def test_copied_terminal_receipt_cannot_rewrite_earlier_hours(tmp_path):
    import runpy

    original = {
        "backup_log_provenance": "real",
        "attempted_at": "2026-10-26T02:00:00Z",
        "snapshot": "/b/2026-10-26.db.zst",
        "snapshot_created_at": "2026-10-26T02:00:00Z",
        "local_verified": True,
        "local_integrity": "ok",
    }
    later_run = {"backup_log_provenance": "real", "attempted_at": "2026-10-29T02:00:00Z"}
    copied = {**original, "upload_state": "superseded", "local_verified": False}
    rows = [original, later_run, copied]
    before = datetime(2026, 10, 27, 12, tzinfo=UTC)
    after = datetime(2026, 10, 29, 12, tzinfo=UTC)
    assert hv.backup_facts(_backup_prefix(rows, before), as_of=before).local_verified_at == datetime.fromisoformat(
        original["snapshot_created_at"]
    )
    assert hv.backup_facts(_backup_prefix(rows, after), as_of=after).local_verified_at is None

    (tmp_path / "m1-backup-daily.log").write_text("\n".join(json.dumps(row) for row in rows))
    (tmp_path / "m1-jsonl-backup.log").write_text("")
    builder = runpy.run_path(str(FIXTURES / "build_oracle.py"))
    events = builder["events_for"](tmp_path, "m1")
    removal = next(e for e in events if e["check"] == "backup_local" and e["value"] == "none")
    assert removal["known_from"] == later_run["attempted_at"]
    assert removal["known_by"] == "2026-10-29T10:00:00Z"
    assert _oracle_state(events, "backup_local", before) == hv.AMBER
    assert _oracle_state(events, "backup_local", after) == hv.RED
