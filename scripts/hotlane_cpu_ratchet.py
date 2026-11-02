"""Read live daemon CPU evidence, or explicitly collect a ten-minute ps audit."""

import argparse
import json
import math
import os
import sqlite3
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROW = "hotlane backlog CPU"


def installed(command):
    return (
        any(prefix in command for prefix in ("/Cellar/brainlayer/", "/opt/brainlayer/"))
        and "hotlane_brainbar_daemon.py" in command
    )


def ps_sample(pid):
    output = subprocess.check_output(
        ["/bin/ps", "-p", str(pid), "-o", "lstart=", "-o", "time=", "-o", "command="], text=True, timeout=5
    ).strip()
    fields = output.split(None, 6)
    if len(fields) != 7 or not installed(fields[6]):
        raise ValueError("PID is not an installed Homebrew hotlane daemon")
    parts = fields[5].split(":")
    seconds = sum(float(value) * 60**index for index, value in enumerate(reversed(parts)))
    return (" ".join(fields[:5]), fields[6]), seconds


def backlog(db):
    with sqlite3.connect(Path(db).resolve().as_uri() + "?mode=ro", uri=True, timeout=1) as connection:
        return connection.execute(
            "SELECT count(*) FROM chunks c LEFT JOIN chunk_vectors_rowids r ON r.id=c.id "
            "WHERE r.id IS NULL AND c.content IS NOT NULL AND c.content != '' "
            "AND c.archived_at IS NULL AND c.superseded_by IS NULL AND c.aggregated_into IS NULL"
        ).fetchone()[0]


def measure(pid, db):
    if sys.platform != "darwin":
        return {"status": "unavailable", "reason": "requires an installed Mac; this host is not Darwin"}
    identity, cpu_before = ps_sample(pid)
    pending = [backlog(db)]
    started = time.monotonic()
    for _ in range(60):
        time.sleep(10)
        current, cpu_after = ps_sample(pid)
        if current != identity:
            raise ValueError("hotlane PID identity changed during measurement")
        pending.append(backlog(db))
    elapsed = time.monotonic() - started
    return dict(
        status="measured",
        platform="darwin",
        pid=pid,
        command=identity[1],
        measured_at=datetime.now(timezone.utc).isoformat(),
        window_seconds=elapsed,
        cpu_percent=100 * (cpu_after - cpu_before) / elapsed,
        backlog=pending,
    )


def read_report(path):
    if path is None:
        return "n/a", "n/a — no ten-minute installed-Mac hotlane ps measurement supplied"
    try:
        payload = json.loads(path.read_text())
        if not isinstance(payload, dict):
            raise ValueError("CPU report must be an object")
        if payload.get("source") == "daemon":
            return daemon_verdict(payload)
        if payload.get("status") == "unavailable" and payload.get("reason"):
            return "n/a", "n/a — " + payload["reason"]
        cpu, window, pending = payload["cpu_percent"], payload["window_seconds"], payload["backlog"]
        budget = float(os.environ.get("BRAINLAYER_HOTLANE_CPU_BUDGET_PERCENT", "60"))
        age = (datetime.now(timezone.utc) - datetime.fromisoformat(payload["measured_at"])).total_seconds()
        if (
            payload["status"] != "measured"
            or payload["platform"] != "darwin"
            or not installed(payload["command"])
            or not all(type(x) in (float, int) and math.isfinite(x) and x >= 0 for x in (cpu, window, budget))
            or window < 600
            or not 0 <= age <= 3600
            or len(pending) != 61
            or not all(type(x) is int and x >= 0 for x in pending)
        ):
            raise ValueError("invalid, short, or stale installed-Mac CPU evidence")
        if not any(pending):
            return "n/a", "n/a — backlog was empty throughout the measurement"
        # Without observed progress, low CPU may just mean the backlog yielded.
        # Arrivals can mask a real drain; that is unmeasured, never a guessed pass.
        if not any(after < before for before, after in zip(pending, pending[1:])):
            return "n/a", "n/a — no observed backlog drain; yielded/paused work is not a budget measurement"
        return ("RED" if cpu > budget else "GREEN"), (
            f"{cpu:.1f}% / {budget:g}% budget over {window:.1f}s; backlog {pending[0]} → {pending[-1]}"
        )
    except (OSError, ValueError, KeyError, TypeError) as error:
        return "RED", f"invalid hotlane CPU report: {error}"


def daemon_verdict(payload):
    cpu, window, encoded = payload["cpu_percent"], payload["window_seconds"], payload["backlog_encoded"]
    budget = float(os.environ.get("BRAINLAYER_HOTLANE_CPU_BUDGET_PERCENT", "60"))
    age = (datetime.now(timezone.utc) - datetime.fromisoformat(payload["measured_at"])).total_seconds()
    state = payload["backlog_state"]
    if (
        payload["status"] != "measured"
        or payload["platform"] != "darwin"
        or type(payload["pid"]) is not int
        or payload["pid"] <= 0
        or not all(type(x) in (int, float) and math.isfinite(x) and x >= 0 for x in (cpu, window, budget))
        or type(encoded) is not int
        or encoded < 0
        or not 0 <= age <= 60
        or state
        not in {
            "starting",
            "running",
            "waiting",
            "high load",
            "pause marker",
            "load unavailable",
            "store queue",
            "error",
        }
    ):
        raise ValueError("invalid or stale daemon CPU evidence")
    value = f"{cpu:.1f}% / {budget:g}% budget over {window:.1f}s; backlog encoded={encoded}; state={state}"
    if state in {"high load", "pause marker", "load unavailable", "store queue", "error"}:
        return "n/a", f"n/a — backlog yielded/paused or unavailable: {value}"
    if window < 600:
        return "n/a", f"n/a — warming the installed ten-minute window: {value}"
    if not encoded:
        return "n/a", f"n/a — no observed backlog encoding: {value}"
    return ("RED" if cpu > budget else "GREEN"), value


def read_live_report(db, os_name):
    if os_name != "Darwin":
        return "n/a", "n/a — live hotlane CPU evidence requires an installed Mac"
    path = db.parent / "hotlane-cpu.json"
    if not path.exists():
        return "n/a", f"n/a — installed daemon has not produced {path}; unmeasured until installed"
    try:
        payload = json.loads(path.read_text())
        if payload["source"] != "daemon" or payload["db"] != str(db.resolve()):
            raise ValueError("CPU evidence belongs to a different producer/database")
        ps_sample(payload["pid"])
        return daemon_verdict(payload)
    except (OSError, ValueError, KeyError, TypeError, subprocess.SubprocessError) as error:
        return "RED", f"invalid live hotlane CPU evidence: {error}"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pid", type=int, required=True)
    parser.add_argument("--db", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    payload = measure(args.pid, args.db)
    args.out.write_text(json.dumps(payload, indent=2) + "\n")
    status, value = read_report(args.out)
    print(value)
    return int(status == "RED")


if __name__ == "__main__":
    raise SystemExit(main())
