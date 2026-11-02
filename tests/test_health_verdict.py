"""Unit table, clock/unknown/hysteresis rules, failure injections and 1.5.83 reader compatibility."""

from __future__ import annotations

import json
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


@pytest.mark.parametrize(("check", "fact"), [(check, fact) for check, fact, _, _ in AGE_EDGES[:3]])
def test_missing_or_never_verified_receipts_are_red_but_unreadable_is_unknown(check, fact):
    absent = run(**{fact: None})
    assert absent.check(check).state == hv.RED
    assert absent.check(check).note == (
        "no verified snapshot on record" if check == "backup_local" else "no verified copy on record"
    )
    alert = next(item for item in hv.badge_alerts(absent) if item["code"] == f"outcome_{check}")
    assert alert["message"] == f"{hv.CHECK_LABELS[check]}: {absent.check(check).note}"
    unreadable = run(**{fact: hv.UNOBSERVED}).check(check)
    assert unreadable.state == hv.UNKNOWN
    assert unreadable.note == "receipts could not be read"


def test_readable_receipts_can_replace_a_newer_persisted_clock():
    prior = run(backup_local_verified_at=NOW - 2 * H).state
    assert run(prior, backup_local_verified_at=NOW - 100 * H).check("backup_local").state == hv.RED


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


AGING_PIPES = [
    ("embedding", "embedding_stalled_since"),
    ("stores", "oldest_pending_store_at"),
    ("ingest", "watcher_updated_at"),
]
WAKE_ONCE = hv.Thresholds(
    embedding_amber_s=3600,
    embedding_red_s=7200,
    stores_amber_s=3600,
    stores_red_s=7200,
    ingest_amber_s=3600,
    ingest_red_s=7200,
    sleep_gap_s=3600,
    wake_grace_s=10800,
)


@pytest.mark.parametrize(("check", "fact"), AGING_PIPES)
@pytest.mark.parametrize("observed", [True, False], ids=["observed", "unobserved"])
def test_wake_once_repeated_gaps_do_not_renew_an_episode(check, fact, observed):
    prior, seen, ages = None, [], []
    for hour in (0, 1, 3, 5, 7):
        value = NOW if observed or hour == 0 else hv.UNOBSERVED
        verdict = run(prior, WAKE_ONCE, now=NOW + hour * H, **{fact: value})
        prior = verdict.state
        seen.append(verdict.check(check).state)
        ages.append(verdict.check(check).age_s)
    # The first resume at hour 3 earns a window through hour 6. Later gaps cannot move it.
    assert seen == [hv.GREEN, hv.GREEN, hv.GREEN, hv.AMBER, hv.RED]
    assert ages == [0, 3600, 0, 7200, 7 * 3600]


@pytest.mark.parametrize(("check", "fact"), AGING_PIPES)
def test_wake_once_fresh_clock_after_wake_gets_its_own_window(check, fact):
    prior = run(thresholds=WAKE_ONCE, **{fact: NOW if check == "ingest" else None}).state
    woke = NOW + 3 * H
    prior = run(prior, WAKE_ONCE, now=woke, **{fact: woke if check == "ingest" else None}).state
    fresh = woke + 5 * M
    prior = run(prior, WAKE_ONCE, now=fresh, **{fact: fresh}).state
    seen = []
    for at in (woke + 2 * H, woke + 4 * H, woke + 6 * H):
        verdict = run(prior, WAKE_ONCE, now=at, **{fact: fresh})
        prior = verdict.state
        seen.append(verdict.check(check).state)
    assert seen == [hv.GREEN, hv.AMBER, hv.RED]


@pytest.mark.parametrize(("check", "fact"), AGING_PIPES)
def test_wake_once_new_episode_after_recovery_is_eligible_again(check, fact):
    prior = None
    for hour in (0, 1, 3, 5, 7):
        prior = run(prior, WAKE_ONCE, now=NOW + hour * H, **{fact: NOW}).state
    recovered_at = NOW + 8 * H
    good = recovered_at if check == "ingest" else None
    for at in (recovered_at, recovered_at + 5 * M):
        recovered = run(prior, WAKE_ONCE, now=at, **{fact: good})
        prior = recovered.state
    assert recovered.check(check).state == hv.GREEN
    fresh = recovered_at + 10 * M
    prior = run(prior, WAKE_ONCE, now=fresh, **{fact: fresh}).state
    seen = []
    for hour in (10, 12, 14):
        verdict = run(prior, WAKE_ONCE, now=NOW + hour * H, **{fact: fresh})
        prior = verdict.state
        seen.append(verdict.check(check).state)
    assert seen == [hv.GREEN, hv.AMBER, hv.RED]


@pytest.mark.parametrize(("check", "fact"), AGING_PIPES)
def test_wake_once_deadline_survives_restart_and_expiry(check, fact, tmp_path):
    prior = run(thresholds=WAKE_ONCE, **{fact: NOW}).state
    first = run(prior, WAKE_ONCE, now=NOW + 3 * H, **{fact: NOW})
    path = tmp_path / "verdict-state.json"
    path.write_text(json.dumps(first.state))
    prior = json.loads(path.read_text())
    seen = []
    for hour in (5, 7, 9):
        verdict = run(prior, WAKE_ONCE, now=NOW + hour * H, **{fact: NOW})
        path.write_text(json.dumps(verdict.state))
        prior = json.loads(path.read_text())
        seen.append(verdict.check(check).state)
    assert seen == [hv.AMBER, hv.RED, hv.RED]
    assert prior["checks"][check]["grace_clock"] == hv._iso(NOW)
    assert prior["checks"][check]["grace_until"] == hv._iso(NOW + 6 * H)


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


# --- Review R1 regressions (PR #1583) --------------------------------------------------------


def _ticks(prior, steps, thresholds=None):
    """Run (now, facts) steps carrying state; return each step's verdict."""
    out = []
    for now, facts in steps:
        verdict = run(prior, thresholds, now=now, **facts)
        prior = verdict.state
        out.append(verdict)
    return out


def test_b1_failed_store_canary_latches_until_a_successful_round_trip():
    failed = {"oldest_pending_store_at": None, "store_canary_ok": False}
    quiet = {"oldest_pending_store_at": None}  # queue empty, canary not run this tick
    ok = {"oldest_pending_store_at": None, "store_canary_ok": True}
    steps = [failed] * 3 + [quiet, quiet, ok, quiet, quiet, ok, ok]
    seen = [v.check("stores").state for v in _ticks(None, [(NOW + i * 5 * M, f) for i, f in enumerate(steps)])]
    # Reviewer's executed repro was amber, amber, red, red, GREEN: the empty queue cleared a failed canary.
    # Unobserved ticks interrupt recovery; only the final adjacent successes clear red.
    assert seen == [hv.AMBER, hv.AMBER] + [hv.RED] * 7 + [hv.GREEN]


def test_b1_canary_and_queue_evidence_are_persisted_separately():
    verdict = run(oldest_pending_store_at=NOW - 20 * M, store_canary_ok=False)
    stores = verdict.state["checks"]["stores"]
    assert stores["fails"] == 1 and stores["canary_owed"] is True
    assert stores["clock"] == hv._iso(NOW - 20 * M)


@pytest.mark.parametrize(
    ("check", "bad", "stale", "good", "need"),
    [
        ("ingest", {"watcher_updated_at": NOW - H}, {"watcher_updated_at": NOW - H}, "watcher_updated_at", 2),
        (
            "embedding",
            {"embedding_stalled_since": NOW - 7 * H},
            {"embedding_stalled_since": NOW - 7 * H},
            None,
            1,
        ),
        (
            "stores",
            {"oldest_pending_store_at": NOW - 50 * M},
            {"oldest_pending_store_at": NOW - 50 * M},
            None,
            2,
        ),
    ],
)
@pytest.mark.parametrize("after_gap", ["unobserved", "same_stale_fact"])
def test_b2_resume_grace_never_fabricates_recovery(check, bad, stale, good, need, after_gap):
    fact = next(iter(bad))
    red = run(**bad)
    assert red.check(check).state == hv.RED
    woke = NOW + 8 * H
    gap_facts = {} if after_gap == "unobserved" else stale
    held = _ticks(red.state, [(woke, gap_facts), (woke + 5 * M, gap_facts), (woke + 10 * M, gap_facts)])
    # Reviewer's executed repro: ingest red -> red -> GREEN, embedding red -> GREEN, with no new fact.
    assert [v.check(check).state for v in held] == [hv.RED] * 3
    # Real recovery evidence still recovers: an advancing heartbeat, or an observed clearance.
    recovering = [(woke + (15 + 5 * i) * M, {fact: (woke + (15 + 5 * i) * M) if good else None}) for i in range(need)]
    assert [v.check(check).state for v in _ticks(held[-1].state, recovering)][-1] == hv.GREEN


def test_b2_wake_grace_still_spares_a_healthy_pipe():
    prior = run(watcher_updated_at=NOW, embedding_stalled_since=None, oldest_pending_store_at=NOW - M).state
    woke = run(prior, now=NOW + 8 * H, watcher_updated_at=NOW, oldest_pending_store_at=NOW - M)
    assert woke.check("ingest").state == hv.GREEN and woke.check("stores").state == hv.GREEN


BACKUPS = [
    ("backup_local", "backup_local_verified_at", 54 * H),
    ("backup_offsite", "backup_offsite_verified_at", 78 * H),
    ("transcripts", "transcripts_verified_at", 54 * H),
]


@pytest.mark.parametrize(("check", "fact", "red"), BACKUPS)
@pytest.mark.parametrize("receipts", ["unobserved", "same_copy"])
@pytest.mark.parametrize("microseconds", [0, 549877])
def test_b3_repeated_gaps_never_renew_the_catch_up_deadline(check, fact, red, receipts, microseconds):
    copy = (NOW - (red - 4 * H)).replace(microsecond=microseconds)
    prior = run(**{fact: copy}).state
    steps = [(NOW + k * H, {} if receipts == "unobserved" else {fact: copy}) for k in range(1, 49)]
    verdicts = _ticks(prior, steps)  # hourly ticks: every one follows a >15 min monitoring gap
    states = [v.check(check).state for v in verdicts]
    # Reviewer's executed repro ended 98 h old and amber, with no red tick at all.
    assert states[-1] == hv.RED
    # Red from the first hour past one absolute catch-up window, and never back to amber.
    first_red_tick = 4 + 1 + 2 + 1  # crosses red at +4 h, wakes at +5 h, deadline +7 h, next tick +8 h
    assert states[: first_red_tick - 1].count(hv.RED) == 0
    assert set(states[first_red_tick - 1 :]) == {hv.RED}


@pytest.mark.parametrize(("check", "fact", "red"), BACKUPS)
@pytest.mark.parametrize("microseconds", [0, 549877])
def test_b3_long_sleeps_get_one_catch_up_window(check, fact, red, microseconds):
    copy = (NOW - (red - 4 * H)).replace(microsecond=microseconds)
    prior = run(**{fact: copy}).state
    woke = NOW + 30 * H  # one long sleep: the copy is now red by 26 h
    first = run(prior, now=woke, **{fact: copy})
    assert first.check(check).state == hv.AMBER and "catch-up" in first.check(check).note
    deadline = first.state["checks"][check]["catch_up_until"]
    assert deadline == hv._iso(woke + 2 * H)
    # Another gap inside the window does not move the deadline...
    second = run(first.state, now=woke + H, **{fact: copy})
    assert second.check(check).state == hv.AMBER
    assert second.state["checks"][check]["catch_up_until"] == deadline
    # ...and a second long sleep past it wakes straight to red, with no new receipt.
    assert run(second.state, now=woke + 10 * H, **{fact: copy}).check(check).state == hv.RED
    # A new verified copy ends the episode.
    fresh = run(second.state, now=woke + 90 * M, **{fact: woke + 80 * M})
    assert fresh.check(check).state == hv.GREEN and "catch_up_until" not in fresh.state["checks"][check]


def test_b4_confirmed_drive_access_loss_latches_until_ok():
    steps = [
        (NOW, {"drive_auth": "consent_needed"}),
        (NOW + 5 * M, {"drive_auth": "refresh_failed"}),
        (NOW + 10 * M, {}),
        (NOW + 15 * M, {"drive_auth": "refresh_failed"}),
        (NOW + 20 * M, {"drive_auth": "ok"}),
        (NOW + 25 * M, {"drive_auth": "refresh_failed"}),
    ]
    seen = [v.check("drive_auth").state for v in _ticks(None, steps)]
    # Reviewer's executed repro: consent_needed -> refresh_failed went red -> amber.
    assert seen == [hv.RED, hv.RED, hv.RED, hv.RED, hv.GREEN, hv.AMBER]


SNAP_OLD, SNAP_NEW = "/b/2026-10-31.db.zst", "/b/2026-11-02.db.zst"


def _verified_receipt(snapshot, created, attempted):
    return _receipt(
        snapshot=snapshot,
        snapshot_created_at=created.isoformat(),
        attempted_at=attempted.isoformat(),
        local_verified=True,
        local_integrity="ok",
        uploaded=True,
        verified=True,
    )


def _reader_tick(receipts, prior, now):
    facts = hv.backup_facts(receipts, as_of=now)
    return run(prior, now=now, **facts.facts_kwargs())


def test_b5_revoking_the_only_copy_ends_the_persisted_clock():
    receipts = [_verified_receipt(SNAP_NEW, NOW - H, NOW - 50 * M)]
    first = _reader_tick(receipts, None, NOW)
    assert first.check("backup_local").state == first.check("backup_offsite").state == hv.GREEN
    receipts.append(_receipt(snapshot=SNAP_NEW, attempted_at=(NOW + M).isoformat(), drive_md5_match=False))
    second = _reader_tick(receipts, first.state, NOW + 5 * M)
    # Reviewer's executed repro: both stayed green on the persisted maximum.
    assert second.check("backup_local").state == second.check("backup_offsite").state == hv.RED
    # An unreadable log afterwards does not bring the revoked copy back.
    third = run(second.state, now=NOW + 10 * M, backup_local_verified_at=hv.UNOBSERVED)
    assert third.check("backup_local").state == hv.RED


def test_b5_revocation_falls_back_to_the_older_surviving_copy():
    receipts = [
        _verified_receipt(SNAP_OLD, NOW - 40 * H, NOW - 39 * H),
        _verified_receipt(SNAP_NEW, NOW - H, NOW - 50 * M),
    ]
    first = _reader_tick(receipts, None, NOW)
    assert first.check("backup_local").age_s == H.total_seconds()
    receipts.append(_receipt(snapshot=SNAP_NEW, attempted_at=(NOW + M).isoformat(), verification_error="md5"))
    second = _reader_tick(receipts, first.state, NOW + 5 * M)
    for name in ("backup_local", "backup_offsite"):
        check = second.check(name)
        assert check.state == hv.AMBER and check.age_s == (40 * H + 5 * M).total_seconds()
    # Re-verifying the same snapshot reinstates it.
    receipts.append(_verified_receipt(SNAP_NEW, NOW - H, NOW + 6 * M))
    assert _reader_tick(receipts, second.state, NOW + 10 * M).check("backup_local").state == hv.GREEN


def test_b5_no_verified_copy_overrides_legacy_state_without_identity():
    legacy = run(backup_local_verified_at=NOW - H).state
    legacy["checks"]["backup_local"].pop("clock_id", None)
    later = run(legacy, now=NOW + 5 * M, backup_local_verified_at=None)
    assert later.check("backup_local").state == hv.RED


# --- DeepSource fixes ------------------------------------------------------------------------


def test_verdict_check_names_a_missing_check():
    with pytest.raises(KeyError):
        run().check("nope")


def test_thresholds_from_env_rejects_nan_and_defaults_to_os_environ(monkeypatch):
    with pytest.raises(ValueError):
        hv.Thresholds.from_env({"BRAINLAYER_VERDICT_INGEST_RED_S": "nan"})
    monkeypatch.setenv("BRAINLAYER_VERDICT_INGEST_RED_S", "120")
    assert hv.Thresholds.from_env().ingest_red_s == 120.0


# --- Review R2 regressions: the readable receipt history is authoritative --------------------


@pytest.mark.parametrize("survivor_hours", [None, 100, 77.5])
def test_b5a_revocation_at_wake_is_immediately_red(survivor_hours):
    receipts = (
        []
        if survivor_hours is None
        else [_verified_receipt(SNAP_OLD, NOW - survivor_hours * H, NOW - (survivor_hours - 1) * H)]
    )
    receipts.append(_verified_receipt(SNAP_NEW, NOW - H, NOW - 50 * M))
    first = _reader_tick(receipts, None, NOW)
    receipts.append(_receipt(attempted_at=(NOW + H).isoformat(), drive_md5_match=False))
    prior = first.state
    for at in (NOW + H, NOW + 2 * H, NOW + 3 * H + M):
        verdict = _reader_tick(receipts, prior, at)
        for name in ("backup_local", "backup_offsite"):
            check = verdict.check(name)
            assert check.state == hv.RED
            assert check.age_s == (None if survivor_hours is None else (at - NOW + survivor_hours * H).total_seconds())
            assert "catch_up_until" not in verdict.state["checks"][name]
        prior = verdict.state


@pytest.mark.parametrize("gap", [5 * M, H], ids=["ordinary-ticks", "monitoring-gap"])
@pytest.mark.parametrize("survivor_hours", [None, 40])
def test_b5b_local_restoration_never_restores_offsite_for_either_monitor_schedule(gap, survivor_hours):
    receipts = []
    if survivor_hours is not None:
        receipts.append(_verified_receipt(SNAP_OLD, NOW - survivor_hours * H, NOW - 39 * H))
    receipts.append(_verified_receipt(SNAP_NEW, NOW - H, NOW - 50 * M))
    first = _reader_tick(receipts, None, NOW)
    receipts.append(_receipt(attempted_at=(NOW + M).isoformat(), drive_md5_match=False))
    saw_bad = _reader_tick(receipts, first.state, NOW + M)
    receipts.append(_receipt(attempted_at=(NOW + 2 * M).isoformat(), local_verified=True, local_integrity="ok"))
    results = [_reader_tick(receipts, prior, NOW + gap) for prior in (first.state, saw_bad.state)]
    for verdict in results:
        assert verdict.check("backup_local").state == hv.GREEN
        offsite = verdict.check("backup_offsite")
        assert offsite.state == (hv.RED if survivor_hours is None else hv.AMBER)
        assert offsite.age_s == (None if survivor_hours is None else (survivor_hours * H + gap).total_seconds())
    receipts.append(_verified_receipt(SNAP_NEW, NOW - H, NOW + gap + M))
    for verdict in results:
        restored = _reader_tick(receipts, verdict.state, NOW + gap + 5 * M)
        assert restored.check("backup_offsite").state == hv.GREEN


@pytest.mark.parametrize("creation_time", ["missing", "null"])
def test_b5c_legacy_state_and_filename_only_revocation(creation_time):
    receipts = [_verified_receipt(SNAP_NEW, NOW - H, NOW - 50 * M)]
    first = _reader_tick(receipts, None, NOW)
    for name in ("backup_local", "backup_offsite"):
        first.state["checks"][name].pop("clock_id", None)
    bad = _receipt(attempted_at=(NOW + M).isoformat(), drive_md5_match=False)
    bad.pop("snapshot_created_at")  # filename midnight differs from the exact 02:00 creation time
    receipts.append(bad)
    revoked = _reader_tick(receipts, first.state, NOW + 5 * M)
    for name in ("backup_local", "backup_offsite"):
        assert revoked.check(name).state == hv.RED
        assert revoked.check(name).age_s is None
    # A filename-only re-verification must retain the artifact's exact creation time, too.
    restored = _receipt(attempted_at=(NOW + 6 * M).isoformat(), local_verified=True, local_integrity="ok")
    if creation_time == "missing":
        restored.pop("snapshot_created_at")
    else:
        restored["snapshot_created_at"] = None
    receipts.append(restored)
    final = _reader_tick(receipts, revoked.state, NOW + 10 * M)
    assert final.check("backup_local").age_s == (H + 10 * M).total_seconds()
    assert final.check("backup_offsite").state == hv.RED


@pytest.mark.parametrize(("check", "fact", "red"), BACKUPS)
@pytest.mark.parametrize("observed", [None, NOW - 100 * H])
def test_readable_backup_facts_replace_persisted_clocks(check, fact, red, observed):
    prior = run(**{fact: NOW - H}).state
    verdict = run(prior, now=NOW + H, **{fact: observed})
    assert verdict.check(check).state == hv.RED
    assert verdict.check(check).age_s == (None if observed is None else 101 * 3600)


@pytest.mark.parametrize(("check", "fact", "red"), BACKUPS)
def test_no_copy_cancels_an_active_backup_catch_up_window(check, fact, red):
    copy = NOW - red + 4 * H
    prior = run(**{fact: copy}).state
    woke = run(prior, now=NOW + 5 * H, **{fact: copy})
    assert woke.check(check).state == hv.AMBER
    absent = run(woke.state, now=NOW + 5 * H + 5 * M, **{fact: None})
    assert absent.check(check).state == hv.RED
    assert "catch_up_until" not in absent.state["checks"][check]


@pytest.mark.parametrize("interrupted", [False, True])
def test_search_recovery_requires_consecutive_observed_good_ticks(interrupted):
    prior = None
    for _ in range(3):
        prior = run(prior, search_canary_ok=False).state
    first = run(prior, search_canary_ok=True)
    assert first.check("search").state == hv.RED
    if interrupted:
        failed = run(first.state, search_canary_ok=hv.UNOBSERVED, blind_codes=("search_canary_failed",))
        assert failed.state["checks"]["search"]["good_ticks"] == 0
        first = run(failed.state, search_canary_ok=True)
        assert first.check("search").state == hv.RED
        assert "recovering (1/2" in first.check("search").note
    assert run(first.state, search_canary_ok=True).check("search").state == hv.GREEN


@pytest.mark.parametrize("later_removal", [False, True])
@pytest.mark.parametrize("surviving_local", [False, True])
def test_removed_uploaded_snapshot_still_counts_as_verified(later_removal, surviving_local):
    receipts = []
    if surviving_local:
        receipts.append(
            _receipt(
                snapshot="/b/older.db.zst",
                snapshot_created_at=(NOW - 2 * H).isoformat(),
                local_verified=True,
                local_integrity="ok",
            )
        )
    receipts.append(_receipt(uploaded=True, verified=True, local_removed=not later_removal))
    if later_removal:
        receipts.append(_receipt(attempted_at=(NOW + M).isoformat(), local_removed=True))
    facts = hv.backup_facts(receipts)
    assert facts.local_verified_at == NOW - H
    verdict = run(**facts.facts_kwargs())
    assert hv.CHECK_LABELS["backup_local"] == "Database snapshot (verified)"
    assert "newest verified snapshot" in verdict.check("backup_local").note
    assert "local copy unavailable after Drive verification" in verdict.check("backup_local").note
    assert facts.offsite_verified_at == NOW - H


@pytest.mark.parametrize("surviving_local", [False, True])
def test_removed_snapshot_without_its_own_upload_drops_from_clock(surviving_local):
    receipts = (
        [
            _receipt(
                snapshot="/b/older.db.zst",
                snapshot_created_at=(NOW - 2 * H).isoformat(),
                uploaded=True,
                verified=True,
                local_removed=True,
            )
        ]
        if surviving_local
        else []
    )
    receipts.extend([_receipt(local_verified=True, local_integrity="ok"), _receipt(local_removed=True)])
    facts = hv.backup_facts(receipts)
    assert facts.local_verified_at == (NOW - 2 * H if surviving_local else None)
    assert facts.offsite_verified_at == (NOW - 2 * H if surviving_local else None)


@pytest.mark.parametrize(
    "invalidation",
    [
        {"local_verified": False},
        {"upload_state": "abandoned"},
        {"upload_state": "superseded"},
        {"upload_state": "abandoned", "local_verified": False, "error_code": "local_snapshot_invalid"},
    ],
)
@pytest.mark.parametrize("uploaded", [False, True])
def test_invalidated_local_snapshot_requires_surviving_verified_upload(invalidation, uploaded):
    receipts = [
        _receipt(local_verified=True, local_integrity="ok", uploaded=uploaded, verified=uploaded),
        _receipt(attempted_at=(NOW + M).isoformat(), **invalidation),
    ]
    facts = hv.backup_facts(receipts)
    assert facts.local_verified_at == (NOW - H if uploaded else None)
    assert facts.offsite_verified_at == (NOW - H if uploaded else None)
    assert run(**facts.facts_kwargs()).check("backup_local").state == (hv.GREEN if uploaded else hv.RED)


def test_superseded_stall_does_not_return_after_newer_upload_revoked():
    stall = _receipt(
        snapshot="/b/stall.db.zst",
        snapshot_created_at=(NOW - 3 * H).isoformat(),
        local_verified=True,
        local_integrity="ok",
    )
    uploaded = _receipt(uploaded=True, verified=True)
    facts = hv.backup_facts(
        [
            stall,
            uploaded,
            {
                **stall,
                "attempted_at": (NOW + M).isoformat(),
                "local_verified": False,
                "upload_state": "superseded",
                "local_removed": False,
            },
            {**uploaded, "attempted_at": (NOW + 2 * M).isoformat(), "drive_md5_match": False},
        ]
    )
    assert facts.local_verified_at is None
    assert facts.offsite_verified_at is None
    assert run(**facts.facts_kwargs()).check("backup_local").state == hv.RED


@pytest.mark.parametrize("clock", [None, hv.UNOBSERVED])
def test_removal_suffix_requires_observed_snapshot_clock(clock):
    check = run(backup_local_verified_at=clock, backup_snapshot_local_removed=True).check("backup_local")
    assert "local copy unavailable" not in check.note


@pytest.mark.parametrize(
    ("check", "bad", "amber"),
    [
        ("ingest", {"watcher_updated_at": NOW - H}, {"watcher_updated_at": NOW - 20 * M}),
        ("stores", {"oldest_pending_store_at": NOW - H}, {"oldest_pending_store_at": NOW - 20 * M}),
    ],
)
def test_f1_hysteresis_red_holds_on_observed_amber(check, bad, amber):
    prior = None
    for _ in range(3):
        prior = run(prior, **bad).state
    assert prior["checks"][check]["state"] == hv.RED
    result = run(prior, now=NOW + M, **amber)
    assert result.check(check).state == hv.RED
    assert result.state["checks"][check]["good_ticks"] == 0


def test_f2_unobserved_fresh_ingest_resets_recovery():
    red = run(watcher_updated_at=NOW - H)
    first = run(red.state, now=NOW + M, watcher_updated_at=NOW + M)
    assert first.state["checks"]["ingest"]["good_ticks"] == 1
    blind = run(first.state, now=NOW + 2 * M)
    assert blind.check("ingest").state == hv.RED
    assert blind.state["checks"]["ingest"]["good_ticks"] == 0
    next_good = run(blind.state, now=NOW + 3 * M, watcher_updated_at=NOW + 3 * M)
    assert next_good.check("ingest").state == hv.RED
    assert run(next_good.state, now=NOW + 4 * M, watcher_updated_at=NOW + 4 * M).check("ingest").state == hv.GREEN


def test_f3_graced_stale_ingest_resets_recovery():
    red = run(watcher_updated_at=NOW - H)
    first = run(red.state, now=NOW + M, watcher_updated_at=NOW + M)
    assert first.state["checks"]["ingest"]["good_ticks"] == 1
    woke = run(first.state, now=NOW + 3 * H, watcher_updated_at=NOW + M)
    assert woke.check("ingest").state == hv.RED
    assert woke.state["checks"]["ingest"]["good_ticks"] == 0
    next_good = run(woke.state, now=NOW + 3 * H + M, watcher_updated_at=NOW + 3 * H + M)
    assert next_good.check("ingest").state == hv.RED
    assert (
        run(next_good.state, now=NOW + 3 * H + 2 * M, watcher_updated_at=NOW + 3 * H + 2 * M).check("ingest").state
        == hv.GREEN
    )
