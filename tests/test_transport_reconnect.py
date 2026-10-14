"""Independent receipt inputs for aggregate idle-socket recovery alerts."""

import json
from datetime import UTC, datetime, timedelta

import pytest

from brainlayer.job_alerts import report
from brainlayer.observability_surface import InputRecorder
from brainlayer.observability_transport import build_transport_section, report_transport_alert


@pytest.fixture(autouse=True)
def fixed_boot(monkeypatch):
    monkeypatch.setattr("brainlayer.supervisor_guard.boot_time", lambda: "100:0")


NOW = datetime(2030, 1, 1, tzinfo=UTC)


def receipt(tmp_path, *, recovered=0, elapsed=60):
    db, path, alerts = (tmp_path / name for name in ("private.db", "transport.json", "alerts.json"))
    value = {
        "schema_version": 1,
        "db_path": str(db),
        "state": "measured",
        "reason": "",
        "generated_at": NOW.isoformat(),
        "producer_started_at": (NOW - timedelta(seconds=900)).isoformat(),
        "window_seconds": 900,
        "observed_seconds": 900,
        "in_flight_requests": 0,
        "counts": {
            "connections_accepted": 0,
            "connections_closed": 0,
            "disconnects_with_in_flight": 0,
            "requests_received": 0,
            "requests_answered": 0,
            "requests_dropped": 0,
            "accept_errors": 0,
            "read_errors": 0,
            "write_errors": 0,
        },
        "incident": None,
        "reconnect": {
            "state": "measured",
            "reason": "",
            "db_path": str(db),
            "boot_time": "100:0",
            "severed_at": (NOW - timedelta(seconds=60)).isoformat(),
            "minimum_connection_age_seconds": 10,
            "grace_seconds": 60,
            "observed_seconds": elapsed,
            "severed_persistent": 12,
            "recovered_persistent": recovered,
        },
    }
    env = {
        "BRAINLAYER_DB": str(db),
        "BRAINLAYER_TRANSPORT_HEALTH_PATH": str(path),
        "BRAINLAYER_JOB_ALERT_PATH": str(alerts),
    }
    return value, path, alerts, env


def consume(value, path, env, *, now=NOW):
    path.write_text(json.dumps(value))
    path.chmod(0o600)
    section = build_transport_section(env=env, now=now, record_input=InputRecorder(root=None, trace_path=None, now=now))
    report_transport_alert(section, env=env)
    return section


def test_twelve_dead_idle_connections_raise_deficit_without_request_drops(tmp_path):
    value, path, alerts, env = receipt(tmp_path)
    section = consume(value, path, env)
    assert section["counts"]["requests_dropped"] == 0
    assert "12 persistent connections did not recover after restart" in alerts.read_text()


def test_twelve_healthy_idle_reconnects_do_not_alert(tmp_path):
    value, path, alerts, env = receipt(tmp_path, recovered=12)
    consume(value, path, env)
    assert not alerts.exists() or "mcp-transport" not in alerts.read_text()


def test_partial_grace_cannot_clear_an_existing_alert(tmp_path):
    value, path, alerts, env = receipt(tmp_path, elapsed=59)
    report("mcp-transport", "existing incident", path=alerts)
    consume(value, path, env)
    assert "existing incident" in alerts.read_text()


@pytest.mark.parametrize(
    "defect",
    ["foreign", "future", "boolean", "negative", "age", "grace", "window", "lifetime", "unknown"],
)
def test_bad_reconnect_evidence_is_nonnumeric_and_cannot_clear_alert(tmp_path, defect):
    value, path, alerts, env = receipt(tmp_path)
    frame = value["reconnect"]
    if defect == "foreign":
        frame["db_path"] += "-other"
    elif defect == "future":
        frame["severed_at"] = (NOW + timedelta(seconds=1)).isoformat()
    elif defect == "boolean":
        frame["severed_persistent"] = True
    elif defect == "negative":
        frame["recovered_persistent"] = -1
    elif defect == "age":
        frame["minimum_connection_age_seconds"] = 0
    elif defect == "grace":
        frame["grace_seconds"] = 0
    elif defect == "window":
        frame["observed_seconds"] = 61
    elif defect == "lifetime":
        value["producer_started_at"] = (NOW - timedelta(seconds=1)).isoformat()
        value["observed_seconds"] = 1
    else:
        value["reconnect"] = {"state": "unmeasurable", "reason": "unreadable previous recovery"}
    report("mcp-transport", "existing incident", path=alerts)
    section = consume(value, path, env)
    assert section["state"] == "unmeasurable"
    assert section["reason"]
    assert "counts" not in section
    assert "existing incident" in alerts.read_text()


def test_two_connection_churn_is_below_the_deficit_tripwire(tmp_path):
    value, path, alerts, env = receipt(tmp_path, recovered=10)
    consume(value, path, env)
    assert not alerts.exists() or "mcp-transport" not in alerts.read_text()


def test_unknown_recovery_does_not_hide_a_valid_historical_request_episode(tmp_path):
    value, path, alerts, env = receipt(tmp_path)
    value["incident"] = {
        k: value[k]
        for k in (
            "state",
            "reason",
            "db_path",
            "generated_at",
            "producer_started_at",
            "window_seconds",
            "observed_seconds",
            "counts",
        )
    }
    value["incident"]["counts"] = {**value["counts"], "requests_dropped": 3}
    value["reconnect"] = {"state": "unmeasurable", "reason": "unknown recovery"}
    section = consume(value, path, env)
    assert section["state"] == "unmeasurable" and "counts" not in section
    assert "historical transport episode" in alerts.read_text()


@pytest.mark.parametrize(
    "boot,age,measured",
    [
        ("99:0", 60, False),
        ("100:0", 172800, False),
        ("100:0", 300, True),
        ("100:0", 301, False),
        (None, 60, False),
        (True, 60, False),
    ],
)
def test_boot_and_age_bound_prevent_false_reconnect_alert(tmp_path, boot, age, measured):
    value, path, alerts, env = receipt(tmp_path)
    value["reconnect"].update(boot_time=boot, severed_at=(NOW - timedelta(seconds=age)).isoformat())
    section = consume(value, path, env)
    assert (section["state"] == "measured") == isinstance(boot, str)
    assert (alerts.exists() and "did not recover" in alerts.read_text()) == measured
    if not measured:
        if isinstance(boot, str):
            assert section["counts"]["requests_dropped"] == 0
            assert section.reconnect["state"] == "unmeasurable" and "severed_persistent" not in section.reconnect
        else:
            assert "counts" not in section and section["reason"]


@pytest.mark.parametrize(
    "reason",
    [
        "baseline from a previous boot; reconnect deficit not measured",
        "baseline expired; reconnect deficit not measured",
    ],
)
def test_discard_preserves_counters_drop_alert_and_clean_window(tmp_path, reason):
    value, path, alerts, env = receipt(tmp_path)
    value["reconnect"] = {"state": "unmeasurable", "reason": reason}
    value["producer_started_at"] = NOW.isoformat()
    for elapsed, drops in [(1, 0), (86400, 3), (87300, 0)]:
        wall = NOW + timedelta(seconds=elapsed)
        value.update(generated_at=wall.isoformat(), observed_seconds=min(elapsed, 900))
        value["counts"]["requests_dropped"] = drops
        section = consume(value, path, env, now=wall)
        assert section["state"] == "measured" and section["counts"]["requests_dropped"] == drops
        assert (alerts.exists() and "mcp-transport" in alerts.read_text()) == bool(drops)
