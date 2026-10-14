"""Real SIGTERM/SIGINT replay; only owned private processes/DBs/sockets."""

import json
import os
import signal
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path

binary, library, signal_name = sys.argv[1:]
assert signal_name in ("SIGTERM", "SIGINT")
root = Path(tempfile.mkdtemp(prefix="bb-termination-"))
sock = root / "mcp.sock"
receipt = root / "transport-health.json"
entered, release = root / "entered", root / "release"
log = (root / "daemon.log").open("wb")
env = {
    **os.environ,
    "HOME": str(root),
    "CFFIXED_USER_HOME": str(root),
    "BRAINBAR_SOCKET_PATH": str(sock),
    "BRAINBAR_DB_PATH": str(root / "private.db"),
    "BRAINLAYER_TRANSPORT_HEALTH_PATH": str(receipt),
    "BRAINBAR_PYTHON": str(root / "no-helper"),
    "BRAINBAR_DEBUG_LOG": "0",
    "DYLD_INSERT_LIBRARIES": library,
    "BRAINBAR_SHUTDOWN_TEST_ENTERED": str(entered),
    "BRAINBAR_SHUTDOWN_TEST_RELEASE": str(release),
    "BRAINLAYER_FORBID_EMBEDDING_MODEL": "1",
}
processes = []


def interrupted(_number, _frame):
    raise KeyboardInterrupt("owned replay interrupted")


signal.signal(signal.SIGTERM, interrupted)
signal.signal(signal.SIGINT, interrupted)


def wait(predicate, seconds=7):
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.01)
    raise AssertionError("private replay timed out: " + str(root))


def value():
    try:
        return json.loads(receipt.read_text())
    except (FileNotFoundError, ValueError):
        return {}


def start():
    offset = (root / "daemon.log").stat().st_size
    p = subprocess.Popen([binary], env=env, stdout=log, stderr=log, start_new_session=True)
    processes.append(p)
    wait(
        lambda: (
            sock.exists()
            and value().get("state") == "measured"
            and "[BrainBarDaemon] Database ready" in (root / "daemon.log").read_text()[offset:]
        )
    )
    return p


def held_request(number):
    entered.unlink(missing_ok=True)
    release.unlink(missing_ok=True)
    client = socket.socket(socket.AF_UNIX)
    client.settimeout(2)
    client.connect(str(sock))
    request = json.dumps(
        {
            "jsonrpc": "2.0",
            "id": number,
            "method": "tools/call",
            "params": {
                "name": "brain_backup_vacuum_into",
                "arguments": {"target_path": str(root / f"backup-{number}.db")},
            },
        }
    ).encode()
    client.sendall(f"Content-Length: {len(request)}\r\n\r\n".encode() + request)
    wait(entered.exists)
    return client


try:
    first = start()
    for number in (1, 2):
        client = held_request(number)
        client.close()
        wait(lambda: value().get("counts", {}).get("requests_dropped") == number)
        assert value().get("incident") is None
        release.touch()
        # The next timer confirms backup completion before issuing another.
        time.sleep(0.1)
    client = held_request(3)
    began = time.monotonic()
    os.kill(first.pid, getattr(signal, signal_name))
    # The peer remains OPEN: only real signal-driven cleanup can add drop3.
    first.wait(timeout=4)
    duration = time.monotonic() - began
    stopped = value()
    client.close()
    release.touch()
    second = start()
    restarted = value()
    assert stopped.get("reason") == "BrainBar server stopped", stopped
    assert stopped.get("incident", {}).get("counts", {}).get("requests_dropped") == 3, stopped
    assert restarted.get("incident", {}).get("counts", {}).get("requests_dropped") == 3, restarted
    assert first.returncode == 0 and duration < 2, (first.returncode, duration)
    assert "counts" not in stopped and restarted["counts"]["requests_dropped"] == 0
    print(json.dumps({"signal": signal_name, "elapsed_seconds": duration, "stopped": stopped, "restarted": restarted}))
finally:
    release.touch()
    for p in processes:
        if p.poll() is None:
            os.kill(p.pid, signal.SIGTERM)
            try:
                p.wait(timeout=3)
            except subprocess.TimeoutExpired:
                os.kill(p.pid, signal.SIGKILL)
                p.wait(timeout=3)
    log.close()
    # Keep scratch receipt/logs for diagnosis; exact owned PIDs have been reaped.
