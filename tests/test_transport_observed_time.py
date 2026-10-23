"""Transport evidence uses its FD read time, not the producer's earlier start."""

import json
import os
import shutil
from datetime import datetime, timedelta

import pytest

from brainlayer import observability_surface as surface
from brainlayer import observability_transport as transport
from tests.test_observability_transport import NOW, sample


class Clock(datetime):
    current = NOW

    @classmethod
    def now(cls, tz=None):
        return cls.current


def rewrite(path, payload, *, generated, mtime):
    payload["generated_at"] = generated.isoformat()
    payload["producer_started_at"] = (generated - timedelta(seconds=900)).isoformat()
    path.write_text(json.dumps(payload))
    os.utime(path, (mtime.timestamp(), mtime.timestamp()))


@pytest.mark.parametrize(
    "generated,mtime,measured,reason",
    [
        (5, 5, True, ""),
        (8, 8, True, ""),
        (-54, 5, True, ""),
        (9, 5, False, "future"),
        (5, 9, False, "future"),
        (-55, 5, False, "stale"),
    ],
)
def test_transport_uses_post_snapshot_clock_and_bounded_skew(tmp_path, monkeypatch, generated, mtime, measured, reason):
    path, db, payload = sample(tmp_path, drops=3)
    rewrite(path, payload, generated=NOW + timedelta(seconds=generated), mtime=NOW + timedelta(seconds=mtime))
    Clock.current = NOW
    monkeypatch.setattr(transport, "datetime", Clock)
    read = transport._read_receipt

    def delayed_read(path):
        snapshot = read(path)
        Clock.current = NOW + timedelta(seconds=6)
        return snapshot

    monkeypatch.setattr(transport, "_read_receipt", delayed_read)
    section = transport.build_transport_section(
        env={"BRAINLAYER_DB": str(db)},
        now=None,
        record_input=surface.InputRecorder(root=None, trace_path=None, now=NOW),
    )
    assert (section["state"] == "measured") == measured
    if measured:
        assert section["counts"]["requests_dropped"] == 3 and section["inputs"][0]["status"] == "read"
    else:
        assert reason in section["reason"] and "counts" not in section


def test_full_document_reads_mid_run_receipt_with_fresh_time(tmp_path, monkeypatch):
    path, db, payload = sample(tmp_path, drops=3)
    fixtures = surface.Path(__file__).parent / "fixtures/observability"
    shutil.copy2(fixtures / "db/healthy-dev.sqlite", db)
    os.utime(db, (NOW.timestamp(), NOW.timestamp()))
    Clock.current = NOW
    monkeypatch.setattr(surface, "datetime", Clock)
    monkeypatch.setattr(transport, "datetime", Clock)
    collect = surface._bounded_db_sections

    def slow_collection(*args):
        result = collect(*args)
        written = NOW + timedelta(seconds=5)
        rewrite(path, payload, generated=written, mtime=written)
        Clock.current = NOW + timedelta(seconds=6)
        return result

    monkeypatch.setattr(surface, "_bounded_db_sections", slow_collection)
    env = {"BRAINLAYER_DB": str(db), "BRAINLAYER_JOB_ALERT_PATH": str(tmp_path / "alerts.json")}
    for name in ("JSONL_BACKUP_LOG", "BACKUP_DAILY_LOG", "LAUNCHD_OUTPUT", "DISABLED_DIR"):
        env["BRAINLAYER_OBSERVABILITY_" + name] = str(tmp_path / name)
    document, _ = surface.build_document(env=env, emit_alerts=True)
    assert document["generated_at"] == NOW.isoformat().replace("+00:00", "Z")
    assert document["stores"]["state"] == "measured"
    assert document["transport"]["state"] == "measured"
    assert document["transport"]["counts"]["requests_dropped"] == 3
    assert "mcp-transport" in (tmp_path / "alerts.json").read_text()


def test_reconnect_ceiling_uses_transport_read_time(tmp_path, monkeypatch):
    path, db, payload = sample(tmp_path)
    generated = NOW + timedelta(seconds=5)
    rewrite(path, payload, generated=generated, mtime=generated)
    payload["reconnect"] = {
        "state": "measured",
        "reason": "",
        "db_path": str(db),
        "boot_time": "100:0",
        "severed_at": (NOW - timedelta(seconds=295)).isoformat(),
        "minimum_connection_age_seconds": 10,
        "grace_seconds": 60,
        "observed_seconds": 60,
        "severed_persistent": 12,
        "recovered_persistent": 0,
    }
    rewrite(path, payload, generated=generated, mtime=generated)
    Clock.current = NOW + timedelta(seconds=6)
    monkeypatch.setattr(transport, "datetime", Clock)
    monkeypatch.setattr("brainlayer.supervisor_guard.boot_time", lambda: "100:0")
    section = transport.build_transport_section(
        env={"BRAINLAYER_DB": str(db)},
        now=None,
        record_input=surface.InputRecorder(root=None, trace_path=None, now=NOW),
    )
    assert section["state"] == "measured" and section["counts"]["requests_dropped"] == 0
    assert section.reconnect == {"state": "unmeasurable", "reason": "baseline expired; reconnect deficit not measured"}
