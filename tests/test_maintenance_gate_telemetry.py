import json
import os

import pytest

from brainlayer.supervisor_guard import component_gate


def test_busy_names_shared_holder(tmp_path):
    with component_gate(tmp_path, operation="index"):
        with pytest.raises(BlockingIOError) as busy:
            with component_gate(tmp_path, exclusive=True, operation="calendar"):
                pass
        assert busy.value.holders[0]["operation"] == "index"
        assert busy.value.holders[0]["pid"] == os.getpid()
    rows = [json.loads(line) for line in (tmp_path / "maintenance-events.jsonl").read_text().splitlines()]
    assert [r["event"] for r in rows] == ["acquired", "busy", "released"]


def test_long_lived_holder_survives_log_cap(tmp_path):
    from brainlayer.supervisor_guard import gate_event, start_time

    identity = dict(
        pid=os.getpid(), start_time=start_time(os.getpid()), lease="long", operation="long-index", mode="shared"
    )
    gate_event(tmp_path, "acquired", identity)
    for _ in range(501):
        gate_event(tmp_path, "busy", dict(identity, lease="contender"))
    holders = gate_event(tmp_path, "busy", dict(identity, lease="contender"))
    assert holders[0]["operation"] == "long-index"
    assert len((tmp_path / "maintenance-events.jsonl").read_text().splitlines()) == 500
