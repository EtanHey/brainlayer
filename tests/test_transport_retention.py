import json
import os
from datetime import timedelta

import pytest

from brainlayer.job_alerts import report
from brainlayer.observability_transport import report_transport_alert
from tests.test_observability_transport import NOW, read, sample


def episode(db):
    return {
        "state": "measured",
        "reason": "",
        "db_path": str(db),
        "generated_at": (NOW - timedelta(hours=1)).isoformat(),
        "producer_started_at": (NOW - timedelta(hours=1, minutes=5)).isoformat(),
        "observed_seconds": 300,
        "window_seconds": 900,
        "counts": {**sample_counts(), "requests_dropped": 3},
    }


def sample_counts():
    from tests.test_observability_transport import FIELDS

    return {k: 0 for k in FIELDS}


def test_historical_episode_alerts_even_when_current_server_is_unmeasurable(tmp_path):
    path, db, payload = sample(tmp_path)
    payload.update(state="unmeasurable", reason="BrainBar server stopped", incident=episode(db))
    for key in ("counts", "observed_seconds", "in_flight_requests"):
        payload.pop(key)
    path.write_text(json.dumps(payload))
    os.utime(path, (NOW.timestamp(), NOW.timestamp()))
    section = read(path, db)
    assert section["state"] == "unmeasurable" and "counts" not in section
    alert = tmp_path / "alerts.json"
    report_transport_alert(section, env={"BRAINLAYER_DB": str(db), "BRAINLAYER_JOB_ALERT_PATH": str(alert)})
    assert "historical" in json.loads(alert.read_text())["mcp-transport"]
    assert "incident" not in json.loads(json.dumps(section))


@pytest.mark.parametrize("defect", ["foreign", "future", "negative", "bool", "missing", "unknown"])
def test_unknown_historical_evidence_never_clears_an_incident(tmp_path, defect):
    path, db, payload = sample(tmp_path)
    old = episode(db)
    if defect == "foreign":
        old["db_path"] = str(tmp_path / "foreign.db")
    if defect == "future":
        old["generated_at"] = (NOW + timedelta(seconds=1)).isoformat()
    if defect == "negative":
        old["counts"]["requests_dropped"] = -1
    if defect == "bool":
        old["counts"]["requests_dropped"] = False
    if defect == "missing":
        old.pop("counts")
    if defect == "unknown":
        old = {"state": "unmeasurable", "reason": "previous receipt unreadable"}
    payload["incident"] = old
    path.write_text(json.dumps(payload))
    os.utime(path, (NOW.timestamp(), NOW.timestamp()))
    alert = tmp_path / "alerts.json"
    section = read(path, db)
    assert section["state"] == "unmeasurable" and "counts" not in section
    report_transport_alert(section, env={"BRAINLAYER_DB": str(db), "BRAINLAYER_JOB_ALERT_PATH": str(alert)})
    assert not alert.exists()
    report("mcp-transport", "existing incident", path=alert, notify=lambda _: None)
    report_transport_alert(read(path, db), env={"BRAINLAYER_DB": str(db), "BRAINLAYER_JOB_ALERT_PATH": str(alert)})
    assert json.loads(alert.read_text())["mcp-transport"] == "existing incident"


def test_confirmed_clean_window_and_no_episode_clear_the_alert(tmp_path):
    path, db, payload = sample(tmp_path)
    payload["incident"] = None
    path.write_text(json.dumps(payload))
    os.utime(path, (NOW.timestamp(), NOW.timestamp()))
    alert = tmp_path / "alerts.json"
    report("mcp-transport", "existing incident", path=alert, notify=lambda _: None)
    report_transport_alert(read(path, db), env={"BRAINLAYER_DB": str(db), "BRAINLAYER_JOB_ALERT_PATH": str(alert)})
    assert json.loads(alert.read_text()) == {}
