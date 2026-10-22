from __future__ import annotations

import json
import os
from datetime import UTC, datetime, timedelta

import pytest

from brainlayer.observability_surface import InputRecorder

NOW = datetime(2026, 10, 13, 12, tzinfo=UTC)
FIELDS = (
    "connections_accepted",
    "connections_closed",
    "disconnects_with_in_flight",
    "requests_received",
    "requests_answered",
    "requests_dropped",
    "accept_errors",
    "read_errors",
    "write_errors",
)


def sample(tmp_path, *, drops=0, observed=900):
    path = tmp_path / "transport-health.json"
    db = tmp_path / "fixture.db"
    payload = {
        "schema_version": 1,
        "state": "measured",
        "reason": "",
        "db_path": str(db),
        "generated_at": NOW.isoformat(),
        "producer_started_at": (NOW - timedelta(seconds=observed)).isoformat(),
        "window_seconds": 900,
        "observed_seconds": observed,
        "in_flight_requests": 0,
        "counts": {k: drops if k == "requests_dropped" else 0 for k in FIELDS},
    }
    path.write_text(json.dumps(payload))
    path.chmod(0o600)
    os.utime(path, (NOW.timestamp(), NOW.timestamp()))
    return path, db, payload


def read(path, db):
    from brainlayer.observability_transport import build_transport_section

    return build_transport_section(
        env={"BRAINLAYER_DB": str(db), "BRAINLAYER_TRANSPORT_HEALTH_PATH": str(path)},
        now=NOW,
        record_input=InputRecorder(root=None, trace_path=None, now=NOW),
    )


@pytest.mark.parametrize(
    "defect",
    [
        "missing",
        "malformed",
        "stale",
        "future",
        "negative",
        "bool",
        "foreign",
        "timestamp_type",
        "partial_lifetime",
        "schema_bool",
    ],
)
def test_unknown_transport_never_emits_zero_counts(tmp_path, defect):
    path, db, payload = sample(tmp_path)
    if defect == "missing":
        path.unlink()
    elif defect == "malformed":
        path.write_text("{")
    else:
        if defect == "stale":
            payload["generated_at"] = (NOW - timedelta(seconds=61)).isoformat()
        if defect == "future":
            payload["generated_at"] = (NOW + timedelta(seconds=3)).isoformat()
        if defect == "negative":
            payload["counts"]["requests_dropped"] = -1
        if defect == "bool":
            payload["counts"]["requests_dropped"] = False
        if defect == "foreign":
            payload["db_path"] = str(tmp_path / "other.db")
        if defect == "timestamp_type":
            payload["generated_at"] = 123
        if defect == "partial_lifetime":
            payload["producer_started_at"] = (NOW - timedelta(seconds=10)).isoformat()
        if defect == "schema_bool":
            payload["schema_version"] = True
        path.write_text(json.dumps(payload))
    os.utime(path, (NOW.timestamp(), NOW.timestamp())) if path.exists() else None
    result = read(path, db)
    assert result["state"] == "unmeasurable" and result["reason"]
    assert "counts" not in result and result["inputs"]


def test_drop_alert_survives_unknown_and_partial_restart_then_recovers(tmp_path):
    from brainlayer.observability_transport import report_transport_alert

    path, db, _ = sample(tmp_path, drops=3)
    env = {"BRAINLAYER_DB": str(db), "BRAINLAYER_JOB_ALERT_PATH": str(tmp_path / "alerts.json")}
    result = read(path, db)
    assert result["per_client_reconnects"]["state"] == "unmeasurable"
    assert result["per_client_reconnects"]["reason"] == "no stable client identity on the socket (fd reuse; proxy pid)"
    report_transport_alert(result, env=env)
    assert "mcp-transport" in json.loads((tmp_path / "alerts.json").read_text())
    path.unlink()
    report_transport_alert(read(path, db), env=env)
    path, db, _ = sample(tmp_path, observed=10)
    report_transport_alert(read(path, db), env=env)
    assert "mcp-transport" in json.loads((tmp_path / "alerts.json").read_text())
    path, db, _ = sample(tmp_path, drops=1)
    report_transport_alert(read(path, db), env=env)
    assert "mcp-transport" in json.loads((tmp_path / "alerts.json").read_text())
    path, db, _ = sample(tmp_path)
    report_transport_alert(read(path, db), env=env)
    assert json.loads((tmp_path / "alerts.json").read_text()) == {}


def test_transport_alert_reaches_the_health_badge(tmp_path, monkeypatch):
    from brainlayer.observability_transport import report_transport_alert
    from tests.test_badge_state import _run_minimal_health_check

    path, db, _ = sample(tmp_path, drops=3)
    alert = tmp_path / "alerts.json"
    monkeypatch.setenv("BRAINLAYER_JOB_ALERT_PATH", str(alert))
    report_transport_alert(read(path, db), env={"BRAINLAYER_DB": str(db), "BRAINLAYER_JOB_ALERT_PATH": str(alert)})
    badge = tmp_path / "badge-state.json"
    _run_minimal_health_check(tmp_path, badge, job_opt_path=tmp_path, job_alert_path=alert)
    value = json.loads(badge.read_text())["alerts"]
    assert value["badge_on"] is True
    issue = next(i for i in value["active"] if i["code"] == "job_alert_mcp-transport")
    assert "3 dropped requests" in issue["message"]


def test_default_receipt_path_matches_native_db_symlink_parent(tmp_path):
    from brainlayer.observability_transport import build_transport_section

    raw = tmp_path / "alias"
    raw.mkdir()
    path, db, _ = sample(raw)
    target = tmp_path / "real" / "fixture.db"
    target.parent.mkdir()
    target.write_bytes(b"fixture")
    db.symlink_to(target)
    payload = json.loads(path.read_text())
    payload["db_path"] = str(target)
    path.write_text(json.dumps(payload))
    os.utime(path, (NOW.timestamp(), NOW.timestamp()))
    section = build_transport_section(
        env={"BRAINLAYER_DB": str(db)}, now=NOW, record_input=InputRecorder(root=None, trace_path=None, now=NOW)
    )
    assert section["state"] == "measured"
    assert section["inputs"][0]["path"] == str(path)


@pytest.mark.parametrize(
    "defect", [None, "foreign", "stale", "future", "schema_bool", "window_bool", "timestamp", "state"]
)
def test_overflow_alert_requires_valid_current_producer_evidence(tmp_path, defect):
    from brainlayer.observability_transport import report_transport_alert

    path, db, payload = sample(tmp_path)
    payload.update(state="unmeasurable", reason="transport event buffer overflow")
    for key in ("counts", "observed_seconds", "in_flight_requests"):
        del payload[key]
    if defect == "foreign":
        payload["db_path"] = str(tmp_path / "other.db")
    elif defect == "stale":
        payload["generated_at"] = (NOW - timedelta(seconds=61)).isoformat()
    elif defect == "future":
        payload["generated_at"] = (NOW + timedelta(seconds=3)).isoformat()
    elif defect == "schema_bool":
        payload["schema_version"] = True
    elif defect == "window_bool":
        payload["window_seconds"] = True
    elif defect == "timestamp":
        payload["producer_started_at"] = "invalid"
    elif defect == "state":
        payload["state"] = "measured"
    path.write_text(json.dumps(payload))
    os.utime(path, (NOW.timestamp(), NOW.timestamp()))
    section = read(path, db)
    assert section["state"] == "unmeasurable" and "counts" not in section
    assert "overflow" not in json.loads(json.dumps(section))
    alert = tmp_path / "alerts.json"
    report_transport_alert(section, env={"BRAINLAYER_DB": str(db), "BRAINLAYER_JOB_ALERT_PATH": str(alert)})
    if defect is None:
        assert "buffer overflow" in json.loads(alert.read_text())["mcp-transport"]
    else:
        assert not alert.exists()
