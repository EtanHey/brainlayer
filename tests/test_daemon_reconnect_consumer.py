"""The live-daemon replay must tolerate a receipt replaced during secure FD validation."""

import ast
import json
import os
import subprocess
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "brain-bar/Tests/BrainBarTests/Scripts/daemon_reconnect_process.py"


def consumer_source(script: str) -> str:
    tree = ast.parse(script)
    return next(
        node.value.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "consumer" for target in node.targets)
    )


def run_consumer(source: str, root: Path, *, wrong_db: bool = False) -> dict:
    root.mkdir()
    receipt = root / "transport.json"
    db = root / "private.db"
    now = datetime.now(UTC)
    receipt.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "state": "measured",
                "reason": "",
                "db_path": str(root / "foreign.db" if wrong_db else db),
                "generated_at": now.isoformat(),
                "producer_started_at": (now - timedelta(seconds=60)).isoformat(),
                "window_seconds": 900,
                "observed_seconds": 60,
                "counts": dict.fromkeys(
                    ["connections_accepted", "connections_closed", "disconnects_with_in_flight", "requests_received",
                     "requests_answered", "requests_dropped", "accept_errors", "read_errors", "write_errors"], 0
                ),
                "in_flight_requests": 0,
                "incident": None,
                "reconnect": None,
            }
        )
    )
    receipt.chmod(0o600)
    injection = """
import brainlayer.observability_transport as transport
original_read=transport.os.read
replaced=False
def replace_once(fd, size):
    global replaced
    data=original_read(fd,size)
    if not replaced:
        replaced=True
        path=Path(os.environ['BRAINLAYER_TRANSPORT_HEALTH_PATH'])
        replacement=path.with_suffix('.next')
        replacement.write_bytes(path.read_bytes())
        replacement.chmod(0o600)
        replacement.replace(path)
    return data
transport.os.read=replace_once
"""
    marker = "from brainlayer.observability_transport import build_transport_section, report_transport_alert\n"
    assert marker in source
    code = source.replace(marker, marker + injection, 1)
    env = {
        **os.environ,
        "PYTHONPATH": str(ROOT / "src"),
        "BRAINLAYER_DB": str(db),
        "BRAINLAYER_TRANSPORT_HEALTH_PATH": str(receipt),
        "BRAINLAYER_JOB_ALERT_PATH": str(root / "alerts.json"),
        "TEST_CLOCK_OFFSET": "0",
    }
    return json.loads(subprocess.check_output([sys.executable, "-S", "-c", code], env=env, text=True, timeout=10))


def test_live_replay_waits_for_a_consumable_receipt_after_atomic_replacement(tmp_path):
    result = run_consumer(consumer_source(SCRIPT.read_text()), tmp_path / "consumer")
    assert result["state"] == "measured", result
    assert result["alerts"] == {}


def test_live_replay_does_not_turn_persistently_invalid_evidence_into_measured(tmp_path):
    result = run_consumer(consumer_source(SCRIPT.read_text()), tmp_path / "invalid", wrong_db=True)
    assert result["state"] == "unmeasurable", result
    assert "different database" in result["reason"]
