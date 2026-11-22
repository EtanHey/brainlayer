"""Regression coverage for the T3 source-selection alarm boundary (issue #1431)."""

import sqlite3
from pathlib import Path

import pytest

from brainlayer import t3_sources
from brainlayer.alarm import BrainLayerAlarm

SECRET = "sk-ant-" + "A1b2C3d4" * 5


@pytest.fixture
def captured(monkeypatch, tmp_path):
    """Raise alarms without emitting telemetry, and keep HOME synthetic."""
    monkeypatch.setenv("HOME", str(tmp_path))
    alarms = []

    def capture(code, message, context):
        alarms.append((code, context))
        raise BrainLayerAlarm(code, message, context)

    monkeypatch.setattr(t3_sources, "raise_alarm", capture)
    monkeypatch.setattr(
        t3_sources, "_tables", lambda _path: (_ for _ in ()).throw(sqlite3.OperationalError("near " + SECRET))
    )
    return alarms


def _root(tmp_path: Path) -> Path:
    root = tmp_path / ".t3/userdata"
    root.mkdir(parents=True)
    return root


def test_explicit_source_failure_scrubs_secret_and_stays_fatal(captured, tmp_path):
    state_db = tmp_path / "explicit.sqlite"
    state_db.write_bytes(b"")
    with pytest.raises(BrainLayerAlarm, match="t3_source_unavailable"):
        t3_sources.resolve_t3_sources(state_db)
    (code, context) = captured[0]
    assert code == "t3_source_unavailable"
    assert context["state_db_path"] == str(state_db)
    assert SECRET not in context["error"]
    assert "REDACTED" in context["error"]
    assert context["error"].startswith("near ")


def test_default_legacy_failure_without_v2_scrubs_secret(captured, tmp_path):
    root = _root(tmp_path)
    (root / "state.sqlite").write_bytes(b"")
    with pytest.raises(BrainLayerAlarm):
        t3_sources.resolve_t3_sources()
    (_, context) = captured[0]
    assert context["state_db_path"] == str(root / "state.sqlite")
    assert SECRET not in context["error"]


def test_tolerated_legacy_failure_with_v2_still_returns_sources(captured, tmp_path):
    root = _root(tmp_path)
    (root / "statev2.sqlite").write_bytes(b"")
    (root / "state.sqlite").write_bytes(b"")
    sources = t3_sources.resolve_t3_sources(tolerate_legacy_failure=True)
    assert sources == [(root / "statev2.sqlite", 2), (root / "state.sqlite", 1)]
    assert captured == []
    with pytest.raises(BrainLayerAlarm):
        t3_sources.resolve_t3_sources()
    assert SECRET not in captured[0][1]["error"]
