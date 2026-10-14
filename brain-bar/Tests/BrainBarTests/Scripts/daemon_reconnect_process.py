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
from datetime import datetime, timezone
from pathlib import Path

binary, library, mode = sys.argv[1:]
assert mode in ("DEAD", "BRIDGE", "RELEASE", "UNSAFE_CLOCK")
root = Path(tempfile.mkdtemp(prefix="bb-recovery-"))
path, receipt, db = root / "mcp.sock", root / "transport.json", root / "private.db"
clock, offset = root / "clock.json", 0.0


def advance(value):
    global offset
    offset = value
    temporary = root / "clock-next.json"
    temporary.write_text(json.dumps(value))
    temporary.chmod(0o600)
    temporary.replace(clock)


advance(0)
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
    "BRAINBAR_RECOVERY_TEST_CLOCK": str(clock),
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
    if mode == "UNSAFE_CLOCK":
        clock.chmod(0o644)
        child = subprocess.Popen([binary], env=daemon_env, stdout=log, stderr=log)
        children.append(child)
        wait(lambda: child.poll() is not None)
        assert child.returncode != 0 and not receipt.exists()
        print(json.dumps({"clock_refused": True}))
        sys.exit(0)
    first = start()
    bridges = []
    for number in range(12):
        if mode != "BRIDGE":
            peer = connection()
            peer.sendall(initialize(number))
            with peer.makefile("rb", buffering=0) as wire:
                assert "result" in json.loads(wire.readline())
        else:
            child = subprocess.Popen(
                ["/usr/bin/env", "python3", "-S", "-m", "brainlayer.mcp_stdio_bridge"],
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
    advance(120 if mode == "RELEASE" else 10.2)
    if mode == "RELEASE":
        clock.chmod(0o644)  # Release must ignore even unsafe fixture files.
    first.terminate()
    first.wait(timeout=4)
    stopped = value()
    if mode == "RELEASE":
        assert stopped["reconnect"] is None, stopped
        generated = datetime.fromisoformat(stopped["generated_at"].replace("Z", "+00:00"))
        assert abs((datetime.now(timezone.utc) - generated).total_seconds()) < 3, stopped
        print(json.dumps({"release_clock_ignored": True, "stopped": stopped}))
        sys.exit(0)
    assert stopped["incident"] is None and isinstance(stopped["reconnect"], dict), stopped
    assert stopped["reconnect"]["severed_persistent"] == 12, stopped
    second = start()
    # Answered one-shots cross PR2's50-connection count but must neither count as
    # persistent recovery nor falsely trigger its unanswered-churn threshold.
    for number in range(50):
        peer = connection()
        peer.sendall((json.dumps({"jsonrpc": "2.0", "id": 100 + number, "method": "ping"}) + "\n").encode())
        with peer.makefile("rb", buffering=0) as wire:
            assert "result" in json.loads(wire.readline())
        peer.close()
    if mode == "BRIDGE":
        wait(lambda: value().get("counts", {}).get("connections_accepted", 0) >= 62)
    advance(65.2)  # 55 elapsed seconds in the restarted producer's clock.
    wait(lambda: value().get("observed_seconds", 0) >= 55)
    assert value()["reconnect"]["observed_seconds"] < 60
    short = connection()  # Still live at grace, but too young to mask any deficit.
    wait(lambda: value().get("counts", {}).get("connections_accepted", 0) >= (63 if mode == "BRIDGE" else 51))
    advance(70.2)
    wait(lambda: value().get("observed_seconds", 0) >= 60)
    final = value()
    assert final["counts"]["requests_dropped"] == 0 and final["counts"]["requests_answered"] == 50, final
    if mode == "DEAD":
        assert final["reconnect"]["observed_seconds"] == 60 and final["reconnect"]["recovered_persistent"] == 0, final
    else:
        assert final["reconnect"] is None and all(p.poll() is None for p in bridges), final
    consumer = """import json, os
from pathlib import Path
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from brainlayer.observability_transport import build_transport_section, report_transport_alert
s=build_transport_section(env=os.environ, now=datetime.now(UTC)+timedelta(seconds=float(os.environ['TEST_CLOCK_OFFSET'])), record_input=SimpleNamespace(file_snapshot=lambda path,**snapshot:{"path":str(path),"status":snapshot["status"]}))
report_transport_alert(s,env=os.environ)
p=Path(os.environ['BRAINLAYER_JOB_ALERT_PATH'])
print(json.dumps({'state':s['state'],'alerts':json.loads(p.read_text()) if p.exists() else {}}))
"""
    alerts_env = {
        **base,
        "BRAINLAYER_DB": str(db),
        "BRAINLAYER_TRANSPORT_HEALTH_PATH": str(receipt),
        "BRAINLAYER_JOB_ALERT_PATH": str(root / "alerts.json"),
        "TEST_CLOCK_OFFSET": str(offset),
    }
    result = json.loads(
        subprocess.check_output(
            ["/usr/bin/env", "python3", "-S", "-c", consumer], env=alerts_env, text=True, timeout=10
        )
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
