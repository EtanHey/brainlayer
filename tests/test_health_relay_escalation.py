"""Tiered relay behavior on disposable inboxes, with an injected clock."""

import json
import os
from datetime import datetime, timezone

import pytest

from tests import test_health_relay as core
from tests.test_health_relay import EPOCH, EVENT, _alerts, _blocked, _status

env, relay = core.env, core.relay


@pytest.fixture
def tiered(env):
    env["escalation"] = env["tmp"] / "escalation.md"
    env["escalation"].write_text("# escalation\n")
    env["raw"].update(first_run_grace_seconds=1000000, escalation={"inbox": str(env["escalation"])})
    env["config"].write_text(json.dumps(env["raw"]))
    _alerts(env, {"backup-daily": "private reason"})
    return env


def _state(env):
    return json.loads((env["tmp"] / "state/state.json").read_text())


def _ack(env, line=None):
    with env["inbox"].open("a") as handle:
        handle.write(f"ACK {EVENT}\n" if line is None else line)


@pytest.fixture
def migrated_v3(tiered):
    path = tiered["tmp"] / "state/state.json"
    path.parent.mkdir()
    path.write_text(
        json.dumps(
            {
                "version": 3,
                "epoch": EPOCH,
                "hosts": {"m4-max": {"seq": 1, "episodes": {"job-alert:backup-daily": {"open": True, "seq": 1}}}},
                "outbox": [],
            }
        )
    )
    return tiered


LEGACY_EVENT = "bl-m4-max-1-job-alert.backup-daily"


@pytest.mark.parametrize("already_appended", [False, True])
def test_migrated_pending_escalation_survives_id_rebinding(relay, migrated_v3, monkeypatch, already_appended):
    env = migrated_v3
    archived = env["inbox"].with_suffix(".archived")
    env["inbox"].rename(archived)
    env["escalation"].unlink()
    assert relay.run(env["config"], clock=lambda: 1000) == 1
    assert relay.run(env["config"], clock=lambda: 2800) == 1
    queued = _state(env)["escalation_outbox"]
    assert len(queued) == 1 and queued[0]["id"] == EVENT + ".esc"
    # Also model append followed by failed state commit: the same event is still queued.
    env["escalation"].write_text(relay._owed((set(), ""), queued[0]) if already_appended else "# restored\n")
    archived.rename(env["inbox"])
    _ack(env, relay._marker(LEGACY_EVENT) + "\n")
    if not already_appended:
        real_append = relay._append

        def fail_escalation(path, records):
            if path == str(env["escalation"]):
                raise relay.RelayError("inbox-append-failed")
            real_append(path, records)

        with monkeypatch.context() as patch:
            patch.setattr(relay, "_append", fail_escalation)
            assert relay.run(env["config"], clock=lambda: 2801) == 1
        rebound = _state(env)["escalation_outbox"][0]
        assert rebound["id"] == LEGACY_EVENT + ".esc" and rebound["ts"] == queued[0]["ts"]
        assert rebound["line"].endswith("; event=" + rebound["id"])
    assert relay.run(env["config"], clock=lambda: 2801) == 0
    delivered = env["escalation"].read_text()
    expected_id = (EVENT if already_appended else LEGACY_EVENT) + ".esc"
    assert delivered.count("@orc BLOCKED:") == 1
    assert f"; event={expected_id}\n" in delivered and relay._marker(expected_id) in delivered
    state = _state(env)
    assert state["escalation_outbox"] == []
    assert state["hosts"]["m4-max"]["episodes"]["job-alert:backup-daily"]["event_id"] == LEGACY_EVENT
    assert relay.run(env["config"], clock=lambda: 10000) == 0
    assert env["escalation"].read_text() == delivered


@pytest.mark.parametrize("event_id", [EVENT, LEGACY_EVENT])
def test_migrated_id_resolves_after_snapshot_with_neither_marker(relay, migrated_v3, event_id):
    env = migrated_v3
    assert relay.run(env["config"], clock=lambda: 1000) == 0
    _ack(env, f"{relay._marker(event_id)}\nACK {event_id}\n")
    assert relay.run(env["config"], clock=lambda: 2800) == 0
    episode = _state(env)["hosts"]["m4-max"]["episodes"]["job-alert:backup-daily"]
    assert episode["event_id"] == event_id and episode["acknowledged"] is True
    assert "migrated" not in episode
    assert env["escalation"].read_text() == "# escalation\n"


def test_migrated_rebinding_finishes_appended_line_before_changing_dedup_id(relay, migrated_v3):
    env = migrated_v3
    archived = env["inbox"].with_suffix(".archived")
    env["inbox"].rename(archived)
    env["escalation"].unlink()
    assert relay.run(env["config"], clock=lambda: 1000) == 1
    assert relay.run(env["config"], clock=lambda: 2800) == 1
    pending = _state(env)["escalation_outbox"][0]
    env["escalation"].write_text(pending["line"])  # Complete line, interrupted before marker.
    archived.rename(env["inbox"])
    _ack(env, relay._marker(LEGACY_EVENT) + "\n")
    assert relay.run(env["config"], clock=lambda: 2801) == 0
    delivered = env["escalation"].read_text()
    assert delivered.count("@orc BLOCKED:") == 1 and relay._marker(EVENT + ".esc") in delivered
    assert _state(env)["escalation_outbox"] == []
    assert relay.run(env["config"], clock=lambda: 2802) == 0
    episode = _state(env)["hosts"]["m4-max"]["episodes"]["job-alert:backup-daily"]
    assert episode["event_id"] == LEGACY_EVENT and "migrated" not in episode
    assert env["escalation"].read_text() == delivered


@pytest.mark.parametrize("event_id", [EVENT, LEGACY_EVENT])
def test_unresolved_migrated_episode_accepts_either_ack(relay, migrated_v3, event_id):
    env = migrated_v3
    _ack(env, f"ACK {event_id}\n")
    assert relay.run(env["config"], clock=lambda: 1000) == 0
    episode = _state(env)["hosts"]["m4-max"]["episodes"]["job-alert:backup-daily"]
    assert episode.get("migrated") is True and episode["acknowledged"] is True
    assert _state(env)["acknowledged_event_ids"] == [event_id]
    env["inbox"].rename(env["inbox"].with_suffix(".archived"))
    assert relay.run(env["config"], clock=lambda: 2800) == 1
    assert env["escalation"].read_text() == "# escalation\n"


def test_ack_cache_prunes_absent_episodes_and_keeps_live_and_closed_acks(relay, tiered):
    assert relay.run(tiered["config"], clock=lambda: 1000) == 0
    _ack(tiered, f"ACK {EVENT}\nACK {EVENT}.esc\nACK unrelated\n")
    assert relay.run(tiered["config"], clock=lambda: 1100) == 0
    assert _state(tiered)["acknowledged_event_ids"] == [EVENT]
    _alerts(tiered, {})
    assert relay.run(tiered["config"], clock=lambda: 1200) == 0
    assert _state(tiered)["acknowledged_event_ids"] == [EVENT]
    _alerts(tiered, {"backup-daily": "x"})
    assert relay.run(tiered["config"], clock=lambda: 1300) == 0
    assert _state(tiered)["acknowledged_event_ids"] == []
    assert relay.run(tiered["config"], clock=lambda: 3100) == 0
    assert tiered["escalation"].read_text().count("@orc BLOCKED:") == 1


def test_primary_immediate_escalation_once_at_threshold_after_restart(relay, tiered):
    assert relay.run(tiered["config"], clock=lambda: 1000) == 0
    assert len(_blocked(tiered)) == 1 and tiered["escalation"].read_text() == "# escalation\n"
    assert relay.run(tiered["config"], clock=lambda: 2799) == 0
    assert tiered["escalation"].read_text() == "# escalation\n"
    assert relay.run(tiered["config"], clock=lambda: 2800) == 0
    delivered = tiered["escalation"].read_text()
    assert delivered.count("@orc BLOCKED:") == 1 and f"; event={EVENT}.esc\n" in delivered
    assert "private reason" not in delivered
    assert relay.run(tiered["config"], clock=lambda: 10000) == 0
    assert tiered["escalation"].read_text() == delivered and len(_blocked(tiered)) == 1


def test_primary_renamed_escalates_once_and_restored_delivery_resumes(relay, tiered):
    assert relay.run(tiered["config"], clock=lambda: 1000) == 0
    archived = tiered["inbox"].with_suffix(".archived")
    tiered["inbox"].rename(archived)
    for tick, expected in ((1300, 0), (2799, 0), (2800, 1), (4600, 1), (15400, 1)):
        assert relay.run(tiered["config"], clock=lambda: tick) == 1
        assert _status(tiered)["error"] == "inbox-unavailable"
        assert tiered["escalation"].read_text().count("@orc BLOCKED:") == expected
    delivered = tiered["escalation"].read_bytes()
    _alerts(tiered, {"backup-daily": "x", "jsonl-backup": "x"})
    assert relay.run(tiered["config"], clock=lambda: 15500) == 1
    assert _status(tiered)["pending"] == 1
    archived.rename(tiered["inbox"])
    assert relay.run(tiered["config"], clock=lambda: 15501) == 0
    assert len(_blocked(tiered)) == 2 and _status(tiered)["pending"] == 0
    assert relay.run(tiered["config"], clock=lambda: 15502) == 0
    assert len(_blocked(tiered)) == 2 and tiered["escalation"].read_bytes() == delivered


@pytest.mark.parametrize("delivery_failure", [False, True])
def test_ack_learned_before_primary_outage_still_suppresses(relay, tiered, monkeypatch, delivery_failure):
    assert relay.run(tiered["config"], clock=lambda: 1000) == 0
    _ack(tiered)
    with monkeypatch.context() as patch:
        if delivery_failure:
            _alerts(tiered, {"backup-daily": "x", "scan": "x"})
            patch.setattr(relay, "_append", lambda *args: (_ for _ in ()).throw(relay.RelayError("inbox-unavailable")))
        assert relay.run(tiered["config"], clock=lambda: 1100) == int(delivery_failure)
    tiered["inbox"].rename(tiered["inbox"].with_suffix(".archived"))
    for tick in (2800, 10000):
        assert relay.run(tiered["config"], clock=lambda: tick) == 1
        assert _status(tiered)["error"] == "inbox-unavailable"
        assert tiered["escalation"].read_text() == "# escalation\n"


@pytest.mark.parametrize("error", ["inbox-unreadable", "inbox-append-failed", "inbox-ack-missing"])
def test_primary_delivery_errors_preserved_while_escalating(relay, tiered, monkeypatch, error):
    assert relay.run(tiered["config"], clock=lambda: 1000) == 0
    real_read, real_deliver = relay._read_inbox, relay._deliver

    def read(path):
        if path == str(tiered["inbox"]) and error == "inbox-unreadable":
            raise relay.RelayError(error)
        return real_read(path)

    def deliver(cfg, *args):
        if cfg["inbox"] == str(tiered["inbox"]):
            raise relay.RelayError(error)
        return real_deliver(cfg, *args)

    monkeypatch.setattr(relay, "_read_inbox", read)
    monkeypatch.setattr(relay, "_deliver", deliver)
    for tick in (2800, 10000):
        assert relay.run(tiered["config"], clock=lambda: tick) == 1
        assert _status(tiered)["error"] == error
        assert tiered["escalation"].read_text().count("@orc BLOCKED:") == 1


@pytest.mark.parametrize("error", ["state-stale", "state-corrupt", "state-commit-failed", "state-unverified"])
def test_state_integrity_errors_never_escalate(relay, tiered, monkeypatch, error):
    assert relay.run(tiered["config"], clock=lambda: 1000) == 0

    def fail(*args):
        raise relay.RelayError(error)

    monkeypatch.setattr(relay, "_save_state" if error == "state-commit-failed" else "_verify_inbox", fail)
    assert relay.run(tiered["config"], clock=lambda: 2800) == (2 if error == "state-corrupt" else 1)
    assert _status(tiered)["error"] == error
    assert tiered["escalation"].read_text() == "# escalation\n"


def test_both_inboxes_missing_retains_primary_error_and_retries_escalation(relay, tiered):
    assert relay.run(tiered["config"], clock=lambda: 1000) == 0
    tiered["inbox"].rename(tiered["inbox"].with_suffix(".archived"))
    tiered["escalation"].unlink()
    assert relay.run(tiered["config"], clock=lambda: 2800) == 1
    assert _status(tiered)["error"] == "inbox-unavailable" and _status(tiered)["pending"] == 1
    tiered["escalation"].write_text("# restored\n")
    for tick in (2801, 10000):
        assert relay.run(tiered["config"], clock=lambda: tick) == 1
        assert _status(tiered)["error"] == "inbox-unavailable" and _status(tiered)["pending"] == 0
        assert tiered["escalation"].read_text().count("@orc BLOCKED:") == 1


@pytest.mark.parametrize("value", [None, "bad", [True], ["bad\n"]])
def test_corrupt_ack_cache_blocks_both_destinations(relay, tiered, value):
    assert relay.run(tiered["config"], clock=lambda: 1000) == 0
    path = tiered["tmp"] / "state/state.json"
    state = _state(tiered)
    state["acknowledged_event_ids"] = value
    original = json.dumps(state)
    path.write_text(original)
    assert relay.run(tiered["config"], clock=lambda: 2800) == 2
    assert _status(tiered)["error"] == "state-corrupt" and path.read_text() == original
    assert len(_blocked(tiered)) == 1 and tiered["escalation"].read_text() == "# escalation\n"


@pytest.mark.parametrize("key", ["mcp-transport", "brainbar-activation", "backup-snapshot"])
def test_default_user_visible_job_alerts_escalate(relay, tiered, key):
    _alerts(tiered, {key: "private reason"})
    assert relay.run(tiered["config"], clock=lambda: 1000) == 0
    assert f"] job-alert:{key} on host=" in _blocked(tiered)[0]
    assert relay.run(tiered["config"], clock=lambda: 2799) == 0
    assert tiered["escalation"].read_text() == "# escalation\n"
    assert relay.run(tiered["config"], clock=lambda: 2800) == 0
    assert tiered["escalation"].read_text().count("@orc BLOCKED:") == 1


@pytest.mark.parametrize("ack", [f"ACK {EVENT}\n", f"ACK {EVENT}"])
def test_lead_ack_before_threshold_is_persisted(relay, tiered, ack):
    assert relay.run(tiered["config"], clock=lambda: 1000) == 0
    _ack(tiered, ack)
    assert relay.run(tiered["config"], clock=lambda: 1100) == 0
    assert relay.run(tiered["config"], clock=lambda: 2800) == 0
    assert tiered["escalation"].read_text() == "# escalation\n"
    assert _state(tiered)["hosts"]["m4-max"]["episodes"]["job-alert:backup-daily"]["acknowledged"] is True


@pytest.mark.parametrize("ack", [f"> ACK {EVENT}\n", f" ACK {EVENT}\n", f"ACK {EVENT} extra\n", f'"ACK {EVENT}"\n'])
def test_quoted_indented_or_partial_acks_do_not_suppress(relay, tiered, ack):
    assert relay.run(tiered["config"], clock=lambda: 1000) == 0
    _ack(tiered, ack)
    assert relay.run(tiered["config"], clock=lambda: 2800) == 0
    assert tiered["escalation"].read_text().count("@orc BLOCKED:") == 1


def test_internal_issue_and_non_fullmatch_never_escalate(relay, tiered):
    _alerts(tiered, {"backup-daily-extra": "x", "scan": "x"})
    assert relay.run(tiered["config"], clock=lambda: 1000) == 0
    assert relay.run(tiered["config"], clock=lambda: 10000) == 0
    assert len(_blocked(tiered)) == 2 and tiered["escalation"].read_text() == "# escalation\n"


def test_close_before_threshold_then_reopen_gets_its_own_timer(relay, tiered):
    assert relay.run(tiered["config"], clock=lambda: 1000) == 0
    _ack(tiered)
    _alerts(tiered, {})
    assert relay.run(tiered["config"], clock=lambda: 1100) == 0
    assert relay.run(tiered["config"], clock=lambda: 2800) == 0
    assert tiered["escalation"].read_text() == "# escalation\n"
    _alerts(tiered, {"backup-daily": "x"})
    assert relay.run(tiered["config"], clock=lambda: 3000) == 0
    assert relay.run(tiered["config"], clock=lambda: 4799) == 0
    assert tiered["escalation"].read_text() == "# escalation\n"
    assert relay.run(tiered["config"], clock=lambda: 4800) == 0
    assert f"bl-m4-max-{EPOCH}-2-job-alert.backup-daily.esc" in tiered["escalation"].read_text()


@pytest.mark.parametrize("fault", ["missing", "unwritable", "fsync", "lost-ack"])
def test_escalation_failure_does_not_block_primary_and_retry_dedups(relay, tiered, monkeypatch, fault):
    assert relay.run(tiered["config"], clock=lambda: 1000) == 0
    real_append, real_save = relay._append, relay._save_state
    if fault == "missing":
        tiered["escalation"].unlink()
    elif fault == "unwritable":

        def append(path, records):
            if path == str(tiered["escalation"]):
                raise relay.RelayError("inbox-unavailable")
            real_append(path, records)

        monkeypatch.setattr(relay, "_append", append)
    elif fault == "fsync":
        real_fsync, ino = relay.os.fsync, tiered["escalation"].stat().st_ino

        def fsync(fd):
            if relay.os.fstat(fd).st_ino == ino:
                raise OSError("disk")
            real_fsync(fd)

        monkeypatch.setattr(relay.os, "fsync", fsync)
    else:

        def save(path, state):
            if tiered["escalation"].read_text().count("@orc BLOCKED:"):
                raise relay.RelayError("state-commit-failed")
            real_save(path, state)

        monkeypatch.setattr(relay, "_save_state", save)
    _alerts(tiered, {"backup-daily": "x", "jsonl-backup": "x"})
    assert relay.run(tiered["config"], clock=lambda: 2800) == 1
    assert len(_blocked(tiered)) == 2 and not _state(tiered)["outbox"]
    assert _status(tiered)["error"].startswith("escalation-")
    assert _status(tiered)["pending"] == 1
    monkeypatch.undo()
    if fault == "missing":
        assert not tiered["escalation"].exists()
        tiered["escalation"].write_text("# restored\n")
    assert relay.run(tiered["config"], clock=lambda: 2801) == 0
    assert tiered["escalation"].read_text().count("@orc BLOCKED:") == 1
    assert _status(tiered)["pending"] == 0


@pytest.mark.parametrize("action", ["ack", "close"])
def test_pending_escalation_is_cancelled_by_ack_or_recovery(relay, tiered, action):
    assert relay.run(tiered["config"], clock=lambda: 1000) == 0
    tiered["escalation"].unlink()
    assert relay.run(tiered["config"], clock=lambda: 2800) == 1
    if action == "ack":
        _ack(tiered)
    else:
        _alerts(tiered, {})
    tiered["escalation"].write_text("# restored\n")
    assert relay.run(tiered["config"], clock=lambda: 2801) == 0
    assert tiered["escalation"].read_text() == "# restored\n"


def test_v3_migration_starts_conservative_episode_timer_and_preserves_id(relay, tiered):
    path = tiered["tmp"] / "state/state.json"
    path.parent.mkdir()
    path.write_text(
        json.dumps(
            {
                "version": 3,
                "epoch": EPOCH,
                "hosts": {
                    "m4-max": {
                        "seq": 1,
                        "episodes": {"job-alert:backup-daily": {"open": True, "seq": 1}},
                    }
                },
                "outbox": [],
            }
        )
    )
    assert relay.run(tiered["config"], clock=lambda: 1000) == 0
    assert _state(tiered)["version"] == 4 and _blocked(tiered) == []
    assert relay.run(tiered["config"], clock=lambda: 2799) == 0
    assert tiered["escalation"].read_text() == "# escalation\n"
    assert relay.run(tiered["config"], clock=lambda: 2800) == 0
    assert f"event={EVENT}.esc" in tiered["escalation"].read_text()


@pytest.mark.parametrize(
    "field,value",
    [
        ("opened_at", True),
        ("opened_at", float("nan")),
        ("opened_at", 10**400),
        ("acknowledged", 1),
        ("escalated", "yes"),
        ("event_id", "bad\n"),
    ],
)
def test_invalid_v4_episode_preserves_state(relay, tiered, field, value):
    assert relay.run(tiered["config"], clock=lambda: 1000) == 0
    path = tiered["tmp"] / "state/state.json"
    state = _state(tiered)
    state["hosts"]["m4-max"]["episodes"]["job-alert:backup-daily"][field] = value
    original = json.dumps(state)
    path.write_text(original)
    assert relay.run(tiered["config"], clock=lambda: 2800) == 2
    assert path.read_text() == original and _status(tiered)["error"] == "state-corrupt"


@pytest.mark.parametrize(
    "value",
    [
        None,
        [],
        {},
        {"inbox": "relative"},
        {"after_seconds": True},
        {"after_seconds": -1},
        {"issues": "badge:.*"},
        {"issues": ["["]},
    ],
)
def test_bad_escalation_config_rejected_before_writes(relay, tiered, value):
    tiered["raw"]["escalation"] = (
        value if not isinstance(value, dict) or not value else {"inbox": str(tiered["escalation"]), **value}
    )
    tiered["config"].write_text(json.dumps(tiered["raw"]))
    assert relay.run(tiered["config"], clock=lambda: 1000) == 2
    assert not (tiered["tmp"] / "state").exists()


@pytest.mark.parametrize("target", ["primary", "source", "state", "status", "lock", "state-temp", "status-temp"])
@pytest.mark.parametrize("alias", ["direct", "symlink", "hardlink"])
def test_escalation_collision_rejected_before_writes(relay, tiered, target, alias):
    destinations = {"primary": tiered["inbox"], "source": tiered["data"] / "job-alerts.json"}
    destinations.update(
        {
            key: tiered["tmp"] / "state" / name
            for key, name in (
                ("state", "state.json"),
                ("status", "status.json"),
                ("lock", "relay.lock"),
                ("state-temp", ".state.json.tmp"),
                ("status-temp", ".status.json.tmp"),
            )
        }
    )
    destination = destinations[target]
    destination.parent.mkdir(exist_ok=True)
    if not destination.exists():
        destination.write_text("preserve\n")
    if alias == "direct":
        tiered["raw"]["escalation"]["inbox"] = str(destination)
    else:
        tiered["escalation"].unlink()
        if alias == "symlink":
            tiered["escalation"].symlink_to(destination)
        else:
            os.link(destination, tiered["escalation"])
    tiered["config"].write_text(json.dumps(tiered["raw"]))
    original = destination.read_bytes()
    assert relay.run(tiered["config"], clock=lambda: 1000) == 2
    assert destination.read_bytes() == original


@pytest.mark.parametrize(
    "code,expected",
    [("brain-search-canary-failed", 1), ("job-scan-failed", 0), ("slow-check", 0)]
    + [(code, 0) for code in ("lock_holder_wedge", "queue_backed_up", "hotlane_dead", "drain_no_progress")],
)
def test_default_badge_visibility_policy(relay, tiered, code, expected):
    _alerts(tiered, {})
    (tiered["data"] / "badge-state.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "alerts": {"state": "measured", "active": [{"code": code, "severity": "critical"}], "suppressed": []},
            }
        )
    )
    assert relay.run(tiered["config"], clock=lambda: 1000) == 0
    assert relay.run(tiered["config"], clock=lambda: 2800) == 0
    assert len(_blocked(tiered)) == 1
    assert tiered["escalation"].read_text().count("@orc BLOCKED:") == expected


def test_unknown_evidence_keeps_timer_and_recovered_episode_can_escalate_again(relay, tiered):
    assert relay.run(tiered["config"], clock=lambda: 1000) == 0
    _alerts(tiered, "not-json")
    assert relay.run(tiered["config"], clock=lambda: 2800) == 0
    assert tiered["escalation"].read_text().count("@orc BLOCKED:") == 1
    _alerts(tiered, {})
    assert relay.run(tiered["config"], clock=lambda: 3000) == 0
    _alerts(tiered, {"backup-daily": "x"})
    assert relay.run(tiered["config"], clock=lambda: 4000) == 0
    assert relay.run(tiered["config"], clock=lambda: 5800) == 0
    assert tiered["escalation"].read_text().count("@orc BLOCKED:") == 2


def test_configured_threshold_and_fullmatch_patterns(relay, tiered):
    tiered["raw"]["escalation"].update(after_seconds=7, issues=["job-alert:backup-.*"])
    tiered["config"].write_text(json.dumps(tiered["raw"]))
    _alerts(tiered, {"backup-other": "x", "other-backup-daily": "x"})
    assert relay.run(tiered["config"], clock=lambda: 1000) == 0
    assert relay.run(tiered["config"], clock=lambda: 1006) == 0
    assert tiered["escalation"].read_text() == "# escalation\n"
    assert relay.run(tiered["config"], clock=lambda: 1007) == 0
    assert tiered["escalation"].read_text().count("@orc BLOCKED:") == 1


def test_destination_change_refused_after_primary_delivery(relay, tiered):
    assert relay.run(tiered["config"], clock=lambda: 1000) == 0
    other = tiered["tmp"] / "other-inbox.md"
    other.write_text("# other\n")
    tiered["raw"]["escalation"]["inbox"] = str(other)
    tiered["config"].write_text(json.dumps(tiered["raw"]))
    _alerts(tiered, {"backup-daily": "x", "jsonl-backup": "x"})
    assert relay.run(tiered["config"], clock=lambda: 2800) == 1
    assert len(_blocked(tiered)) == 2 and other.read_text() == "# other\n"
    assert _status(tiered)["error"] == "escalation-destination-changed"


def test_v3_with_legacy_episode_id_honors_original_ack_after_inbox_outage(relay, tiered, monkeypatch):
    legacy = "bl-m4-max-1-job-alert.backup-daily"
    path = tiered["tmp"] / "state/state.json"
    path.parent.mkdir()
    path.write_text(
        json.dumps(
            {
                "version": 3,
                "epoch": EPOCH,
                "hosts": {
                    "m4-max": {
                        "seq": 1,
                        "episodes": {"job-alert:backup-daily": {"open": True, "seq": 1}},
                    }
                },
                "outbox": [],
            }
        )
    )
    _ack(tiered, f"@orc BLOCKED: legacy; event={legacy}\n<!-- brainlayer-health:{legacy} -->\nACK {legacy}\n")
    with monkeypatch.context() as patch:
        patch.setattr(relay, "_read_inbox", lambda _: (_ for _ in ()).throw(relay.RelayError("inbox-unavailable")))
        assert relay.run(tiered["config"], clock=lambda: 1000) == 1
    assert _state(tiered)["hosts"]["m4-max"]["episodes"]["job-alert:backup-daily"]["migrated"] is True
    assert relay.run(tiered["config"], clock=lambda: 2800) == 0
    assert tiered["escalation"].read_text() == "# escalation\n"
    episode = _state(tiered)["hosts"]["m4-max"]["episodes"]["job-alert:backup-daily"]
    assert episode["event_id"] == legacy and episode["acknowledged"] is True
