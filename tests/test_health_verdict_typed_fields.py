"""Seam PR-1: typed check fields (F3, F4, F5, F9, F19, F20, F21) exposed from existing verdict state.

Every field is display-only. ``test_typed_inputs_never_move_the_verdict`` pins that no new input
or field can change the dot, a check's state, its note or the persisted state.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta

import pytest

from brainlayer import health_verdict as hv

NOW = datetime(2026, 11, 2, 3, 0, tzinfo=UTC)
H = timedelta(hours=1)
M = timedelta(minutes=1)
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
LEGACY_KEYS = ("name", "state", "age_s", "threshold_s", "since", "note")


def run(prior=None, **facts):
    return hv.evaluate(hv.Facts(now=facts.pop("now", NOW), **{**HEALTHY, **facts}), prior)


def typed(verdict, name):
    return {k: v for k, v in verdict.check(name).to_dict().items() if k not in LEGACY_KEYS}


def test_f3_catch_up_until_on_backup_checks():
    prior = run(backup_local_verified_at=NOW - 50 * H).state
    assert typed(run(prior), "backup_local")["catch_up_until"] is None
    woke = NOW + 10 * H  # the copy crossed red while the Mac was not running
    first = run(prior, now=woke, backup_local_verified_at=NOW - 50 * H)
    assert first.check("backup_local").state == hv.AMBER
    assert typed(first, "backup_local")["catch_up_until"] == "2026-11-02T15:00:00Z"
    assert typed(first, "backup_offsite")["catch_up_until"] is None
    assert "catch_up_until" not in typed(first, "ingest")


def test_f4_good_ticks_count_toward_recovery_on_hysteresis_checks():
    prior = None
    for _ in range(3):
        prior = run(prior, search_canary_ok=False).state
    first = run(prior, now=NOW + 5 * M)
    assert first.check("search").state == hv.RED
    assert typed(first, "search").items() >= {"good_ticks": 1, "good_ticks_needed": 2}.items()
    second = run(first.state, now=NOW + 10 * M)
    assert second.check("search").state == hv.GREEN and typed(second, "search")["good_ticks"] == 0
    assert {"good_ticks", "good_ticks_needed"} <= set(typed(second, "ingest")) & set(typed(second, "stores"))
    assert "good_ticks" not in typed(second, "backup_local")


def test_f5_blind_tick_keeps_green_and_says_it_could_not_look():
    prior = run().state
    blind = run(prior, now=NOW + 5 * M, search_canary_ok=hv.UNOBSERVED)
    assert blind.check("search").state == hv.GREEN
    assert typed(blind, "search").items() >= {"blind": True, "unknown_ticks": 1}.items()
    assert typed(blind, "ingest").items() >= {"blind": False, "unknown_ticks": 0}.items()


def test_f5_held_red_is_blind():
    prior = run(drive_auth="consent_needed").state
    held = run(prior, now=NOW + 5 * M, drive_auth=hv.UNOBSERVED)
    assert held.check("drive_auth").state == hv.RED
    assert typed(held, "drive_auth").items() >= {"blind": True, "unknown_ticks": 1}.items()


def test_f9_maintenance_pause_and_its_end_on_ingest_and_stores():
    ends = NOW + 90 * M
    paused = run(ingest_paused=True, stores_paused=True, pause_ends_at=ends, watcher_updated_at=NOW - 40 * M)
    assert paused.check("ingest").state == hv.AMBER  # the existing demotion, unchanged
    for name in ("ingest", "stores"):
        assert (
            typed(paused, name).items()
            >= {"paused_by_maintenance": True, "pause_ends_at": "2026-11-02T04:30:00Z"}.items()
        )
    only_drain = run(stores_paused=True, pause_ends_at=ends)
    assert typed(only_drain, "ingest").items() >= {"paused_by_maintenance": False, "pause_ends_at": None}.items()
    assert "paused_by_maintenance" not in typed(only_drain, "search")


def test_f19_clock_is_the_persisted_positive_timestamp():
    verdict = run(watcher_updated_at=NOW - 3 * M, oldest_pending_store_at=NOW - 20 * M)
    assert typed(verdict, "ingest")["clock"] == "2026-11-02T02:57:00Z"
    assert typed(verdict, "backup_local")["clock"] == "2026-11-02T01:00:00Z"
    assert typed(verdict, "stores")["clock"] == "2026-11-02T02:40:00Z"
    assert typed(verdict, "search")["clock"] is None
    unread = run(verdict.state, now=NOW + 5 * M, backup_local_verified_at=hv.UNOBSERVED)
    assert typed(unread, "backup_local")["clock"] == "2026-11-02T01:00:00Z"


@pytest.mark.parametrize(
    ("facts", "cause", "fails"),
    [
        ({"oldest_pending_store_at": NOW - 20 * M}, "queue", 0),
        ({"store_canary_ok": False}, "round_trip", 1),
        ({"oldest_pending_store_at": NOW - 20 * M, "store_canary_ok": False}, "both", 1),
        ({"oldest_pending_store_at": hv.UNOBSERVED, "store_canary_ok": hv.UNOBSERVED}, "blind", 0),
        ({}, "none", 0),
    ],
)
def test_f20_stores_cause_and_canary_fails(facts, cause, fails):
    verdict = run(**facts)
    assert typed(verdict, "stores").items() >= {"cause": cause, "fails": fails}.items()
    assert "cause" not in typed(verdict, "search")


def test_f20_search_fails_count_consecutive_canary_failures():
    prior = run(search_canary_ok=False).state
    assert typed(run(prior, now=NOW + 5 * M, search_canary_ok=False), "search")["fails"] == 2


def test_f21_context_codes_on_each_check_and_blind_codes_on_monitor():
    verdict = run(
        check_codes={"embedding": ("hotlane_dead", "missing_embeddings_backlog", "hotlane_dead")},
        blind_codes=("slow_check", "process_snapshot_failed"),
    )
    assert typed(verdict, "embedding")["context_codes"] == ["hotlane_dead", "missing_embeddings_backlog"]
    assert all(typed(verdict, c.name)["context_codes"] == [] for c in verdict.checks if c.name != "embedding")
    assert typed(verdict, "monitor")["blind_codes"] == ["process_snapshot_failed", "slow_check"]
    assert "blind_codes" not in typed(verdict, "ingest")


def _outcome(verdict):
    checks = [{k: check.to_dict()[k] for k in LEGACY_KEYS} for check in verdict.checks]
    return json.dumps({"dot": verdict.dot, "checks": checks, "state": verdict.state}, sort_keys=True)


def test_typed_inputs_never_move_the_verdict():
    """The same sequence with and without the new inputs is byte-identical outside the typed keys."""
    ticks = [
        {},
        {"search_canary_ok": False, "watcher_updated_at": NOW - 40 * M},
        {"search_canary_ok": False, "store_canary_ok": False, "ingest_paused": True},
        {"search_canary_ok": hv.UNOBSERVED, "drive_auth": "consent_needed", "blind_codes": ("slow_check",)},
        {"oldest_pending_store_at": NOW - 50 * M, "stores_paused": True},
        {"backup_local_verified_at": NOW - 60 * H},
        {},
        {},
    ]
    extra = {
        "stores_paused": True,
        "pause_ends_at": NOW + 2 * H,
        "check_codes": {"stores": ("queue_backed_up",), "ingest": ("watcher_stalled",)},
    }
    outcomes = []
    for with_new in (False, True):
        prior, seen = None, []
        for step, tick in enumerate(ticks):
            facts = {**tick, **(extra if with_new else {})}
            if not with_new:
                facts.pop("stores_paused", None)
            verdict = run(prior, now=NOW + step * 5 * M, **facts)
            prior = verdict.state
            seen.append(_outcome(verdict))
        outcomes.append(seen)
    assert outcomes[0] == outcomes[1]
    assert "unknown_ticks" in json.dumps(run().check("search").to_dict())
    assert list(run().check("search").to_dict())[:6] == list(LEGACY_KEYS)
