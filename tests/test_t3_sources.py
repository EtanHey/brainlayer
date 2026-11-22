"""Synthetic source failures must be scrubbed before reaching alarm transport."""

import sqlite3
from pathlib import Path
from unittest.mock import Mock

import pytest

from brainlayer import t3_sources
from brainlayer.alarm import BrainLayerAlarm


@pytest.mark.parametrize("explicit", [True, False])
@pytest.mark.parametrize("prefix", ["", "probe " * 1000])
def test_source_alarm_scrubs_complete_error(monkeypatch, tmp_path, explicit, prefix):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setattr(Path, "is_file", lambda self: True)
    secret = "sk-ant-" + "A1b2C3d4" * 5
    error = prefix + "sqlite fixture " + secret + " end"
    probe = Mock(side_effect=sqlite3.OperationalError(error))
    monkeypatch.setattr(t3_sources, "_tables", probe)

    def capture(code, message, context):
        raise BrainLayerAlarm(code, message, context)

    boundary = Mock(side_effect=capture)
    monkeypatch.setattr(t3_sources, "raise_alarm", boundary)
    selected = tmp_path / "explicit.sqlite" if explicit else tmp_path / ".t3/userdata/state.sqlite"
    with pytest.raises(BrainLayerAlarm) as caught:
        t3_sources.resolve_t3_sources(selected if explicit else None)
    alarm = caught.value
    assert alarm.code == "t3_source_unavailable"
    assert alarm.message == "could not inspect T3 source read-only"
    assert alarm.severity == "fatal"
    assert alarm.exit_code == 1
    assert alarm.context == {
        "state_db_path": str(selected),
        "error": prefix + "sqlite fixture [REDACTED:anthropic] end",
    }
    assert secret not in str(boundary.call_args)
    probe.assert_called_once_with(selected)


def test_tolerated_legacy_failure_retains_both_sources(monkeypatch, tmp_path):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setattr(Path, "is_file", lambda self: True)
    monkeypatch.setattr(t3_sources, "_tables", Mock(side_effect=sqlite3.OperationalError("fixture")))
    boundary = Mock()
    monkeypatch.setattr(t3_sources, "raise_alarm", boundary)
    root = tmp_path / ".t3/userdata"
    assert t3_sources.resolve_t3_sources(tolerate_legacy_failure=True) == [
        (root / "statev2.sqlite", 2),
        (root / "state.sqlite", 1),
    ]
    boundary.assert_not_called()
