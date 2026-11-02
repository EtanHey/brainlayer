"""Unit table, clock/unknown/hysteresis rules, failure injections and 1.5.83 reader compatibility."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from brainlayer import health_verdict as hv

NOW = datetime(2026, 11, 2, 3, 0, tzinfo=UTC)
H = timedelta(hours=1)
M = timedelta(minutes=1)
EPS = timedelta(seconds=1)
T = hv.Thresholds()


CONTINUOUS = hv.Thresholds(sleep_gap_s=10**9)  # big time steps that are not a sleep


def run(prior=None, thresholds=None, **facts):
    return hv.evaluate(hv.Facts(now=facts.pop("now", NOW), **facts), prior, thresholds)


def state(name, prior=None, **facts):
    return run(prior, **facts).check(name).state


# --- Threshold edges: just under / just over -------------------------------------------------

AGE_EDGES = [
    ("backup_local", "backup_local_verified_at", 30 * H, 54 * H),
    ("backup_offsite", "backup_offsite_verified_at", 30 * H, 78 * H),
    ("transcripts", "transcripts_verified_at", 30 * H, 54 * H),
    ("ingest", "watcher_updated_at", 10 * M, 30 * M),
]


@pytest.mark.parametrize(("check", "fact", "amber", "red"), AGE_EDGES)
def test_age_threshold_edges(check, fact, amber, red):
    assert state(check, **{fact: NOW - amber}) == hv.GREEN
    assert state(check, **{fact: NOW - amber - EPS}) == hv.AMBER
    assert state(check, **{fact: NOW - red}) == hv.AMBER
    assert state(check, **{fact: NOW - red - EPS}) == hv.RED


@pytest.mark.parametrize(
    ("check", "fact", "amber", "red"),
    [("stores", "oldest_pending_store_at", 15 * M, 45 * M), ("embedding", "embedding_stalled_since", 1 * H, 6 * H)],
)
def test_stall_threshold_edges(check, fact, amber, red):
    assert state(check, **{fact: None}) == hv.GREEN
    assert state(check, **{fact: NOW - amber}) == hv.GREEN
    assert state(check, **{fact: NOW - amber - EPS}) == hv.AMBER
    assert state(check, **{fact: NOW - red}) == hv.AMBER
    assert state(check, **{fact: NOW - red - EPS}) == hv.RED


@pytest.mark.parametrize(("check", "fact"), [("search", "search_canary_ok"), ("stores", "store_canary_ok")])
def test_canary_counts(check, fact):
    prior, seen = None, []
    for _ in range(3):
        verdict = run(prior, **{fact: False}, oldest_pending_store_at=None)
        prior, seen = verdict.state, seen + [verdict.check(check).state]
    assert seen == [hv.AMBER, hv.AMBER, hv.RED]


def test_disk_edges():
    gib = 1024**3
    base = {"disk_peak_bytes": 50 * gib, "disk_floor_bytes": 2 * gib}
    assert state("disk", disk_free_bytes=54 * gib, **base) == hv.GREEN
    assert state("disk", disk_free_bytes=54 * gib - 1, **base) == hv.AMBER
    assert state("disk", disk_free_bytes=50 * gib, **base) == hv.AMBER
    assert state("disk", disk_free_bytes=50 * gib - 1, **base) == hv.RED


def test_drive_and_integrity_states():
    assert state("drive_auth", drive_auth="ok") == hv.GREEN
    assert state("drive_auth", drive_auth="refresh_failed") == hv.AMBER
    assert state("drive_auth", drive_auth="consent_needed") == hv.RED
    assert state("integrity", integrity="ok") == hv.GREEN
    assert state("integrity", integrity="corrupt") == hv.RED


def test_paused_watcher_is_a_note_not_a_dot():
    assert state("ingest", watcher_updated_at=NOW - 2 * H, ingest_paused=True) == hv.AMBER


def test_thresholds_from_env():
    t = hv.Thresholds.from_env(
        {"BRAINLAYER_VERDICT_BACKUP_LOCAL_RED_S": "3600", "BRAINLAYER_VERDICT_CANARY_RED_FAILURES": "5"}
    )
    assert (t.backup_local_red_s, t.canary_red_failures) == (3600.0, 5)
    with pytest.raises(ValueError):
        hv.Thresholds.from_env({"BRAINLAYER_VERDICT_INGEST_RED_S": "0"})


# --- Unknown never freezes the clocks; a known red holds -------------------------------------


def test_unreadable_receipts_keep_the_clock_aging_to_red():
    prior = run(backup_local_verified_at=NOW - 50 * H).state
    later = NOW + 5 * H
    check = run(prior, CONTINUOUS, now=later, backup_local_verified_at=hv.UNOBSERVED).check("backup_local")
    assert check.state == hv.RED and check.age_s == 55 * 3600


def test_missing_or_never_verified_receipts_are_red_but_unreadable_is_unknown():
    assert state("backup_local", backup_local_verified_at=None) == hv.RED
    assert state("backup_local", backup_local_verified_at=hv.UNOBSERVED) == hv.UNKNOWN


def test_clock_never_moves_backward():
    prior = run(backup_local_verified_at=NOW - 2 * H).state
    assert run(prior, backup_local_verified_at=NOW - 100 * H).check("backup_local").state == hv.GREEN


def test_unknown_keeps_last_known_then_amber_after_three_ticks():
    prior = run(search_canary_ok=True).state
    seen = []
    for tick in range(1, 4):
        verdict = run(prior, now=NOW + tick * 5 * M, search_canary_ok=hv.UNOBSERVED)
        prior, seen = verdict.state, seen + [verdict.check("search").state]
    assert seen == [hv.GREEN, hv.GREEN, hv.AMBER]


def test_known_red_holds_through_unknown_until_positive_evidence():
    prior = run(drive_auth="consent_needed").state
    for tick in range(1, 6):
        verdict = run(prior, now=NOW + tick * 5 * M)
        assert verdict.check("drive_auth").state == hv.RED
        prior = verdict.state
    assert run(prior, now=NOW + 30 * M, drive_auth="ok").check("drive_auth").state == hv.GREEN


def test_couldnt_look_codes_go_amber_after_three_ticks_and_never_red():
    prior, seen = None, []
    for tick in range(6):
        verdict = run(prior, now=NOW + tick * 5 * M, blind_codes=("process_snapshot_failed",))
        prior, seen = verdict.state, seen + [verdict.check("monitor").state]
    assert seen == [hv.UNKNOWN, hv.UNKNOWN, hv.AMBER, hv.AMBER, hv.AMBER, hv.AMBER]
    assert run(prior, now=NOW + H).check("monitor").state == hv.GREEN


def test_unobserved_pending_store_keeps_aging():
    prior = run(oldest_pending_store_at=NOW - 30 * M).state
    assert run(prior, CONTINUOUS, now=NOW + 20 * M).check("stores").state == hv.RED


# --- Recovery hysteresis ---------------------------------------------------------------------


def test_backups_recover_on_the_first_verified_receipt():
    prior = run(backup_local_verified_at=NOW - 60 * H).state
    assert run(prior, now=NOW + 5 * M, backup_local_verified_at=NOW).check("backup_local").state == hv.GREEN


@pytest.mark.parametrize(
    ("check", "bad", "good"),
    [
        ("search", {"search_canary_ok": False}, {"search_canary_ok": True}),
        ("ingest", {"watcher_updated_at": NOW - 40 * M}, {"watcher_updated_at": NOW}),
        ("stores", {"oldest_pending_store_at": NOW - 50 * M}, {"oldest_pending_store_at": None}),
    ],
)
def test_ingest_stores_search_need_two_good_ticks(check, bad, good):
    prior = None
    for _ in range(3):
        prior = run(prior, **bad).state
    assert prior["checks"][check]["state"] == hv.RED
    first = run(prior, now=NOW + 5 * M, **good)
    assert first.check(check).state == hv.RED and "recovering" in first.check(check).note
    assert run(first.state, now=NOW + 10 * M, **good).check(check).state == hv.GREEN


# --- Sleep/wake and restart -----------------------------------------------------------------


def test_time_asleep_gets_a_catch_up_grace_then_counts():
    prior = run(backup_local_verified_at=NOW - 50 * H).state
    woke = NOW + 10 * H  # lid closed for 10 h; the copy is now 60 h old
    first = run(prior, now=woke, backup_local_verified_at=NOW - 50 * H)
    assert first.check("backup_local").state == hv.AMBER and "catch-up" in first.check("backup_local").note
    prior, tick = first.state, woke
    while tick < woke + 2 * H + 5 * M:  # the monitor ticks every 5 min once awake
        tick += 5 * M
        verdict = run(prior, now=tick, backup_local_verified_at=NOW - 50 * H)
        prior = verdict.state
    assert verdict.check("backup_local").state == hv.RED


def test_ingest_age_counts_from_wake_not_from_before_sleep():
    prior = run(watcher_updated_at=NOW).state
    woke = NOW + 8 * H
    assert run(prior, now=woke, watcher_updated_at=NOW).check("ingest").state == hv.GREEN
    assert run(prior, now=woke, watcher_updated_at=NOW).state["resumed_at"] is not None


def test_restart_with_lost_state_reads_facts_afresh_and_kept_state_continues():
    assert state("backup_offsite", None, backup_offsite_verified_at=NOW - 80 * H) == hv.RED
    two = run(run(search_canary_ok=False).state, now=NOW + 5 * M, search_canary_ok=False)
    assert run(two.state, now=NOW + 10 * M, search_canary_ok=False).check("search").state == hv.RED
    assert run(None, now=NOW + 10 * M, search_canary_ok=False).check("search").state == hv.AMBER


# --- Independent failure injections ----------------------------------------------------------

HEALTHY = dict(
    backup_local_verified_at=NOW - 2 * H,
    backup_offsite_verified_at=NOW - 2 * H,
    transcripts_verified_at=NOW - 2 * H,
    drive_auth="ok",
    integrity="ok",
    watcher_updated_at=NOW,
    embedding_stalled_since=None,
    oldest_pending_store_at=None,
    store_canary_ok=True,
    search_canary_ok=True,
    disk_free_bytes=100,
    disk_peak_bytes=10,
    disk_floor_bytes=1,
)


@pytest.mark.parametrize(
    ("check", "fault"),
    [
        ("backup_local", {"backup_local_verified_at": NOW - 55 * H}),
        ("drive_auth", {"drive_auth": "consent_needed"}),
        ("ingest", {"watcher_updated_at": NOW - 31 * M}),
        ("integrity", {"integrity": "corrupt"}),
        ("transcripts", {"transcripts_verified_at": NOW - 55 * H}),
        ("stores", {"store_canary_ok": False}),
    ],
)
def test_each_injected_fault_is_red_and_alone(check, fault):
    healthy = run(**HEALTHY)
    assert healthy.dot == hv.GREEN and hv.badge_alerts(healthy) == []
    prior = None  # a stale clock is a fact from the start, not a jump back from a fresh one
    for tick in range(1, 4):  # the store canary needs three failed round trips
        verdict = run(prior, now=NOW + tick * 5 * M, **{**HEALTHY, **fault})
        prior = verdict.state
    assert verdict.dot == hv.RED
    assert [c.name for c in verdict.checks if c.state == hv.RED] == [check]
    assert [a["code"] for a in hv.badge_alerts(verdict) if a["severity"] == "critical"] == [f"outcome_{check}"]


def test_job_alerts_are_notes_and_never_light_the_dot():
    verdict = run(**HEALTHY, notes=hv.notes_from_job_alerts({"maintenance-full": "VACUUM skipped"}))
    assert verdict.dot == hv.GREEN
    assert hv.badge_alerts(verdict) == [
        {"code": "job_alert_maintenance-full", "severity": "warning", "message": "VACUUM skipped"}
    ]


# --- Receipt readers --------------------------------------------------------------------------


def test_integrity_counts_confirmed_results_only():
    assert hv.integrity_result({"local_integrity": "ok"}) == "ok"
    assert (
        hv.integrity_result(
            {"local_integrity": "*** in database main ***\nPage 7: btreeInitPage() returns error code 11"}
        )
        == "corrupt"
    )
    assert hv.integrity_result({"local_integrity": "DatabaseError: database disk image is malformed"}) == "corrupt"
    assert hv.integrity_result({"local_integrity": "OperationalError: database is locked"}) is None
    assert hv.integrity_result({"local_integrity": "skipped"}) is None
    assert hv.integrity_result({}) is None


def test_drive_events():
    assert (
        hv.drive_event({"error_type": "DriveCredentialError", "error": "Drive access expired: Reconnect in BrainBar"})
        == "consent_needed"
    )
    assert (
        hv.drive_event(
            {"error_type": "DriveCredentialError", "error": "Drive token refresh failed; run brainlayer backup auth"}
        )
        == "refresh_failed"
    )
    assert hv.drive_event({"uploaded": True, "verified": True}) == "ok"
    assert hv.drive_event({"error_type": "BackupTimeoutError"}) is None


def _receipt(**kw):
    return {
        "backup_log_provenance": "real",
        "attempted_at": NOW.isoformat(),
        "snapshot": "/b/2026-11-02.db.zst",
        "snapshot_created_at": (NOW - H).isoformat(),
        **kw,
    }


def test_backup_facts_revocation_and_supersession():
    verified = _receipt(local_verified=True, local_integrity="ok", uploaded=True, verified=True)
    facts = hv.backup_facts([verified])
    assert facts.local_verified_at == facts.offsite_verified_at == NOW - H
    assert (
        hv.backup_facts([verified, _receipt(local_verified=False, upload_state="superseded")]).local_verified_at
        == NOW - H
    )
    revoked = hv.backup_facts([verified, _receipt(uploaded=True, verified=True, drive_md5_match=False)])
    assert revoked.local_verified_at is None and revoked.offsite_verified_at is None
    assert hv.backup_facts([{**verified, "backup_log_provenance": "pytest"}]).local_verified_at is None
    assert hv.backup_facts([verified], as_of=NOW - EPS).local_verified_at is None


# --- BrainBar 1.5.83 reader compatibility ------------------------------------------------------


def swift_reader(document):
    """Port of BadgeStateReader.read's measured branch (Dashboard/ObservabilityView.swift)."""
    alerts = document["alerts"]
    if alerts["state"] != "measured" or alerts["reason"] != "":
        return {"badge_on": True, "failVisible": True}
    active = alerts["active"]
    if alerts["badge_on"] != any(item["severity"] != "warning" for item in active):
        return {"badge_on": True, "failVisible": True}
    return {
        "badge_on": alerts["badge_on"],
        "codes": [i["code"] for i in active],
        "messages": [i["message"] for i in active],
        "severities": [i["severity"] for i in active],
    }


def swift_reconciled(badge, job_alerts):
    """Port of BadgeStatePresentation.reconciled(with:) (BrainBarJobAlerts.swift)."""
    kept = []
    for code, message, severity in zip(badge["codes"], badge["messages"], badge["severities"]):
        if not code.startswith("job_alert_"):
            kept.append((code, message, severity))
        elif code[len("job_alert_") :] in job_alerts:
            kept.append((code, job_alerts[code[len("job_alert_") :]], severity))
    if [k[0] for k in kept] == badge["codes"] and [k[1] for k in kept] == badge["messages"]:
        return badge
    return {
        "badge_on": any(k[2] != "warning" for k in kept),
        "codes": [k[0] for k in kept],
        "messages": [k[1] for k in kept],
        "severities": [k[2] for k in kept],
    }


def _document(verdict):
    active = hv.badge_alerts(verdict)
    return {
        "schema_version": 1,
        "generated_at": "2026-11-02T03:00:00Z",
        "alerts": {
            "state": "measured",
            "reason": "",
            "inputs": [],
            "badge_on": verdict.dot == hv.RED,
            "active": active,
            "suppressed": [],
        },
    }


@pytest.mark.parametrize("red", [False, True])
@pytest.mark.parametrize("live", [{"maintenance-full": "VACUUM skipped"}, {"maintenance-full": "new reason"}, {}])
def test_old_reader_never_fails_visible_and_job_alerts_cannot_relight(red, live):
    facts = {**HEALTHY, "drive_auth": "consent_needed" if red else "ok"}
    verdict = run(**facts, notes=hv.notes_from_job_alerts({"maintenance-full": "VACUUM skipped"}))
    badge = swift_reader(_document(verdict))
    assert "failVisible" not in badge and badge["badge_on"] is red
    reconciled = swift_reconciled(badge, live)
    assert reconciled["badge_on"] is red  # a changed or cleared job alert never moves the dot
    assert ("outcome_drive_auth" in reconciled["codes"]) is red
    assert ("job_alert_maintenance-full" in reconciled["codes"]) is bool(live)
