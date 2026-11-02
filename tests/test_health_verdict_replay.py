"""History-replay ratchet for the outcome verdict (S1).

Fixtures under ``tests/fixtures/verdict_replay/`` are real receipts from both Macs, 2026-10-12 to
2026-11-02 (``backup-daily.log``, ``jsonl-backup.log``, ``maintenance.log``), with resumable-upload
URLs dropped and home directories renamed. ``oracle.json`` lists the evidence events, built by
separate code (see the fixture README) with an explicit uncertainty window: an invocation's
completion time is not recorded, so its receipt is known somewhere between ``known_from`` and
``known_by``. An hour where the two bounds disagree is ``unknown`` and is not asserted.
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

# The ratchet row: every red hour (UTC, inclusive start, exclusive end) the receipts support.
# M1: Drive consent expired 2026-10-24T18:50Z; the first receipts to see it are the 10-25 02:01Z
# runs; the next verified upload is the 10-29 02:00Z run. M4: no verified backup between the
# 10-17 nightly and 10-21 00:45Z (10-18 upload stalled, 10-19 disk short then upload timed out,
# 10-20 failed), and 10-24 01:15Z stayed the newest snapshot past 54 h on 10-26 (two failed
# nightlies; the 08:30Z manual run recovered it).
EXPECTED_RED: dict[str, dict[str, list[tuple[str, str]]]] = {
    "m1": {
        "drive_auth": [("2026-10-25T06:00Z", "2026-10-29T03:00Z")],
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


def _oracle_state(events: list[dict], check: str, at: datetime, t: hv.Thresholds) -> str:
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
            return hv.RED
        newest = max(datetime.fromisoformat(e["value_at"]) for e in known)
        age = (at - newest).total_seconds()
        amber, red = getattr(t, f"{check}_amber_s"), getattr(t, f"{check}_red_s")
        return hv.RED if age > red else hv.AMBER if age > amber else hv.GREEN

    low, high = state(lo), state(hi)
    return low if low == high else hv.UNKNOWN


def _replay(mac: str):
    backups = _load(f"{mac}-backup-daily.log")
    transcripts = _load(f"{mac}-jsonl-backup.log")
    maintenance = _load(f"{mac}-maintenance.log")
    prior = None
    for at in _hours(START, END):
        facts_b = hv.backup_facts(backups, as_of=at)
        newest_transcript, drive_t = hv.transcript_facts(transcripts, as_of=at)
        drive = max((x for x in (facts_b.drive, drive_t) if x is not None), default=None, key=lambda x: x[0])
        verdict = hv.evaluate(
            hv.Facts(
                now=at,
                backup_local_verified_at=facts_b.local_verified_at,
                backup_offsite_verified_at=facts_b.offsite_verified_at,
                transcripts_verified_at=newest_transcript,
                drive_auth=drive[1] if drive else hv.UNOBSERVED,
                integrity=facts_b.integrity,
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
            expected = _oracle_state(events, name, at, THRESHOLDS)
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
            if verdict.check(name).state == hv.RED and _oracle_state(events, name, at, THRESHOLDS) != hv.UNKNOWN:
                red.setdefault(name, []).append(at)
    assert {name: _intervals(hours) for name, hours in red.items()} == EXPECTED_RED[mac]
