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
    "code,expected", [("brain-search-canary-failed", 1), ("job-scan-failed", 0), ("slow-check", 0)]
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
    assert relay.run(tiered["config"], clock=lambda: 2800) == 0
    assert tiered["escalation"].read_text() == "# escalation\n"
    episode = _state(tiered)["hosts"]["m4-max"]["episodes"]["job-alert:backup-daily"]
    assert episode["event_id"] == legacy and episode["acknowledged"] is True
