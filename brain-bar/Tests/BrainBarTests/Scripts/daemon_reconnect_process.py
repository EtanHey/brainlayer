"""Actual daemon + twelve idle bridges or dead clients; private paths/PIDs only."""

import json
import os
import select
import signal
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path

binary, library, mode = sys.argv[1:]
assert mode in ("DEAD", "BRIDGE")
root = Path(tempfile.mkdtemp(prefix="bb-recovery-"))
path, receipt, db = root / "mcp.sock", root / "transport.json", root / "private.db"
source = Path(__file__).resolve().parents[4]
base = {
    **os.environ,
    "HOME": str(root),
    "CFFIXED_USER_HOME": str(root),
    "PYTHONPATH": str(source / "src"),
    "BRAINLAYER_TEST_PATH_PROVENANCE": "pytest",
    "BRAINLAYER_FORBID_EMBEDDING_MODEL": "1",
}
daemon_env = {
    **base,
    "BRAINBAR_SOCKET_PATH": str(path),
    "BRAINBAR_DB_PATH": str(db),
    "BRAINLAYER_TRANSPORT_HEALTH_PATH": str(receipt),
    "BRAINBAR_PYTHON": str(root / "no-helper"),
    "BRAINBAR_DEBUG_LOG": "0",
    "DYLD_INSERT_LIBRARIES": library,
}
bridge_env = {**base, "BRAINLAYER_MCP_SOCKET": str(path)}
bridge_env.pop("DYLD_INSERT_LIBRARIES", None)
log = (root / "daemon.log").open("wb")
children, sockets = [], []


def interrupted(_number, _frame):
    raise KeyboardInterrupt("owned replay interrupted")


signal.signal(signal.SIGTERM, interrupted)
signal.signal(signal.SIGINT, interrupted)


def wait(predicate, seconds=8):
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.02)
    raise AssertionError("private recovery replay timed out: " + str(root))


def value():
    try:
        return json.loads(receipt.read_text())
    except (FileNotFoundError, ValueError):
        return {}


def start():
    offset = (root / "daemon.log").stat().st_size
    child = subprocess.Popen([binary], env=daemon_env, stdout=log, stderr=log, start_new_session=True)
    children.append(child)
    wait(
        lambda: (
            path.exists()
            and value().get("state") == "measured"
            and "[BrainBarDaemon] Database ready" in (root / "daemon.log").read_text()[offset:]
        )
    )
    return child


def connection():
    peer = socket.socket(socket.AF_UNIX)
    peer.settimeout(3)
    peer.connect(str(path))
    sockets.append(peer)
    return peer


def initialize(number):
    return (
        json.dumps(
            {
                "jsonrpc": "2.0",
                "id": number,
                "method": "initialize",
                "params": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {},
                    "clientInfo": {"name": "private-idle-control", "version": "0"},
                },
            }
        )
        + "\n"
    ).encode()


try:
    first = start()
    bridges = []
    for number in range(12):
        if mode == "DEAD":
            peer = connection()
            peer.sendall(initialize(number))
            with peer.makefile("rb", buffering=0) as wire:
                assert "result" in json.loads(wire.readline())
        else:
            child = subprocess.Popen(
                ["/usr/bin/env", "python3", "-m", "brainlayer.mcp_stdio_bridge"],
                env=bridge_env,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                start_new_session=True,
            )
            children.append(child)
            bridges.append(child)
            child.stdin.write(initialize(number))
            child.stdin.flush()
            assert select.select([child.stdout], [], [], 5)[0]
            assert "result" in json.loads(child.stdout.readline())
    time.sleep(10.2)  # Production age threshold; no accelerated clock in this process test.
    first.terminate()
    first.wait(timeout=4)
    stopped = value()
    assert stopped["incident"] is None and stopped["reconnect"]["severed_persistent"] == 12, stopped
    second = start()
    # Answered one-shots cross PR2's50-connection count but must neither count as
    # persistent recovery nor falsely trigger its unanswered-churn threshold.
    for number in range(50):
        peer = connection()
        peer.sendall((json.dumps({"jsonrpc": "2.0", "id": 100 + number, "method": "ping"}) + "\n").encode())
        with peer.makefile("rb", buffering=0) as wire:
            assert "result" in json.loads(wire.readline())
        peer.close()
    wait(lambda: value().get("observed_seconds", 0) >= 55, seconds=65)
    short = connection()  # Still live at grace, but too young to mask any deficit.
    wait(lambda: value().get("observed_seconds", 0) >= 60, seconds=10)
    final = value()
    assert final["counts"]["requests_dropped"] == 0 and final["counts"]["requests_answered"] == 50, final
    if mode == "DEAD":
        assert final["reconnect"]["observed_seconds"] == 60 and final["reconnect"]["recovered_persistent"] == 0, final
    else:
        assert final["reconnect"] is None and all(p.poll() is None for p in bridges), final
    consumer = """import json, os
from pathlib import Path
from datetime import UTC, datetime
from brainlayer.observability_surface import InputRecorder
from brainlayer.observability_transport import build_transport_section, report_transport_alert
s=build_transport_section(env=os.environ, now=datetime.now(UTC), record_input=InputRecorder(root=None,trace_path=None,now=datetime.now(UTC)))
report_transport_alert(s,env=os.environ)
p=Path(os.environ['BRAINLAYER_JOB_ALERT_PATH'])
print(json.dumps({'state':s['state'],'alerts':json.loads(p.read_text()) if p.exists() else {}}))
"""
    alerts_env = {
        **base,
        "BRAINLAYER_DB": str(db),
        "BRAINLAYER_TRANSPORT_HEALTH_PATH": str(receipt),
        "BRAINLAYER_JOB_ALERT_PATH": str(root / "alerts.json"),
    }
    result = json.loads(
        subprocess.check_output(["/usr/bin/env", "python3", "-c", consumer], env=alerts_env, text=True, timeout=10)
    )
    assert result["state"] == "measured", result
    if mode == "DEAD":
        assert "12 persistent connections did not recover after restart" in result["alerts"]["mcp-transport"], result
    else:
        assert not result["alerts"], result
    print(
        json.dumps(
            {
                "mode": mode,
                "stopped": stopped,
                "after_grace": final,
                "consumer": result,
                "bridge_processes_alive": len(bridges),
                "repeated_initialize_sent": False,
            }
        )
    )
finally:
    for peer in sockets:
        peer.close()
    for child in reversed(children):
        if child.poll() is None:
            child.terminate()
            try:
                child.wait(timeout=3)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait(timeout=3)
    log.close()
    # Retain private receipts/logs for diagnosis. Every exact owned PID is reaped.
