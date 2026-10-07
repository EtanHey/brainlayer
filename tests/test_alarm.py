from __future__ import annotations

import threading
import time
from datetime import UTC, datetime, timedelta

import pytest

from brainlayer.alarm import BrainLayerAlarm, raise_alarm
from brainlayer.drain_liveness import check_drain_liveness

NOW = datetime(2026, 6, 25, 12, 0, tzinfo=UTC)


def test_raise_alarm_emits_telemetry_and_escapes_broad_exception_handlers(monkeypatch, capsys):
    emitted: list[tuple[str, dict]] = []
    telemetry_emitted = threading.Event()

    def capture_emit(dataset, event):
        emitted.append((dataset, event))
        telemetry_emitted.set()
        return True

    monkeypatch.setattr("brainlayer.telemetry.emit", capture_emit)

    with pytest.raises(BrainLayerAlarm) as raised:
        try:
            raise_alarm("write_zero", "writer produced zero durable writes", {"active_entries": 4})
        except Exception:  # noqa: BLE001 - proves the alarm is not silently swallowed by broad exception handlers.
            pytest.fail("BrainLayerAlarm was swallowed by a broad Exception handler")

    assert raised.value.code == "write_zero"
    assert raised.value.exit_code == 1
    assert raised.value.details == {"active_entries": 4}
    assert "BRAINLAYER_ALARM write_zero: writer produced zero durable writes" in capsys.readouterr().err
    assert telemetry_emitted.wait(1)
    assert emitted == [
        (
            "brainlayer-alarms",
            {
                "_type": "alarm",
                "code": "write_zero",
                "severity": "fatal",
                "message": "writer produced zero durable writes",
                "context": {"active_entries": 4},
                "exit_code": 1,
            },
        )
    ]


def test_raise_alarm_does_not_block_on_stuck_telemetry(monkeypatch, capsys):
    telemetry_started = threading.Event()

    def slow_emit(_dataset, _event):
        telemetry_started.set()
        time.sleep(0.35)
        return True

    monkeypatch.setattr("brainlayer.telemetry.emit", slow_emit)

    started = time.monotonic()
    with pytest.raises(BrainLayerAlarm) as raised:
        raise_alarm("telemetry_stuck", "alarm telemetry should not block the caller", {"active_entries": 1})
    elapsed = time.monotonic() - started

    assert telemetry_started.wait(1)
    assert elapsed < 0.2
    assert raised.value.code == "telemetry_stuck"
    assert "BRAINLAYER_ALARM telemetry_stuck" in capsys.readouterr().err


def test_drain_liveness_stalled_uses_alarm_primitive():
    issue = check_drain_liveness(
        drain_label="com.brainlayer.drain",
        drain_loaded=True,
        queue_count=2,
        drain_health={"updated_at": (NOW - timedelta(minutes=10)).isoformat(), "drain_cycles": 3},
        now=NOW,
        stale_seconds=300,
    )

    assert isinstance(issue, BrainLayerAlarm)
    assert issue.code == "drain_liveness_stalled"
    assert issue.severity == "fatal"
    assert issue.details["queue_count"] == 2


def test_retired_enrichment_quota_does_not_affect_durable_queue(monkeypatch, tmp_path):
    monkeypatch.setenv("BRAINLAYER_ENRICH_DAILY_USD_CAP", "0")
    monkeypatch.setenv("BRAINLAYER_ENRICH_COST_DIR", str(tmp_path))
    issue = check_drain_liveness(
        drain_label="com.brainlayer.drain",
        drain_loaded=True,
        queue_count=2,
        drain_health={"updated_at": (NOW - timedelta(minutes=10)).isoformat()},
        now=NOW,
    )
    assert isinstance(issue, BrainLayerAlarm)
    assert issue.code == "drain_liveness_stalled"
    assert "enrichment_backlog" not in issue.details


def test_empty_queue_has_no_retired_producer_liveness_issue():
    issue = check_drain_liveness(
        drain_label="com.brainlayer.drain",
        drain_loaded=True,
        queue_count=0,
        drain_health={"updated_at": (NOW - timedelta(minutes=10)).isoformat()},
        now=NOW,
    )
    assert issue is None
