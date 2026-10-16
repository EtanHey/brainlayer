#!/usr/bin/env python3
"""Live release gate. Uses real user configs; writes only tagged probe memories.

Run from the BrainLayer checkout/worktree with the installed keg Python (3.11+).
Receipts contain private local paths; do not publish raw harness stdout.
"""

import argparse
import hashlib
import json
import os
import re
import selectors
import shutil
import signal
import socket
import subprocess
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

from harness_probe_contract import (
    HARNESSES,
    MANUAL,
    QUERY,
    command,
    discover_configs,
    exit_code,
    matching_hit,
    parse_calls,
    session_id,
    table,
    verdict,
)


def run(argv):
    return subprocess.run(argv, capture_output=True, text=True, timeout=8).stdout


def observe(pid):
    """Only descendants of this fresh session count; resolve Unix peer via lsof."""
    processes = {}
    for line in run(["ps", "-axo", "pid=,ppid=,comm="]).splitlines():
        fields = line.strip().split(None, 2)
        if len(fields) == 3:
            processes[int(fields[0])] = (int(fields[1]), fields[2])
    children = {pid}
    while True:
        expanded = children | {p for p, (parent, _) in processes.items() if parent in children}
        if children == expanded:
            break
        children = expanded
    bridges = []
    for p in children:
        comm = processes.get(p, (None, ""))[1]
        # Python/socat may hide the script in comm, so inspect argv only in memory.
        argv = run(["ps", "-p", str(p), "-o", "command="]).strip()
        if ("brainlayer-mcp-stdio-bridge" in argv or "brainlayer.mcp_stdio_bridge" in argv) or (
            "socat" in comm and "brain" in argv
        ):
            safe = re.findall(r"\S*brainlayer-mcp-stdio-bridge|UNIX-CONNECT:[^\s]+", argv)
            bridges.append((p, safe))
    if not bridges:
        return []
    # macOS clients show ->kernel-address, not the socket name. Join the peer
    # address to the server's lsof row instead of inferring from configuration.
    lines = run(["lsof", "-nP", "-U"]).splitlines()
    address_paths = {}
    for line in lines:
        fields = line.split()
        if len(fields) >= 8 and fields[-1].startswith("/"):
            address_paths[fields[5]] = fields[-1]
    evidence = []
    for p, argv in bridges:
        for line in lines:
            fields = line.split()
            if len(fields) >= 8 and fields[1] == str(p):
                endpoint = address_paths.get(fields[-1].removeprefix("->"), fields[-1])
                if endpoint.startswith("/"):
                    evidence.append({"pid": p, "command": argv, "socket": endpoint})
    return evidence


class Bridge:
    """Separate installed direct bridge, pinned to BrainBar, never a harness override."""

    def __enter__(self):
        executable = Path("/opt/homebrew/opt/brainlayer/bin/brainlayer-mcp-stdio-bridge")
        if not executable.exists():
            raise RuntimeError("Installed bridge absent")
        env = dict(os.environ, BRAINLAYER_MCP_SOCKET="/tmp/brainbar.sock")
        env.pop("MCPLAYER_BRAINLAYER_FRONT", None)
        self.proc = subprocess.Popen(
            [str(executable)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            env=env,
            bufsize=0,
            start_new_session=True,
        )
        self.buffer = b""
        self.seq = 0
        self.selector = selectors.DefaultSelector()
        self.selector.register(self.proc.stdout, selectors.EVENT_READ)
        try:
            self.rpc(
                "initialize",
                {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {},
                    "clientInfo": {"name": "harness-probe-verifier", "version": "1"},
                },
            )
            self.proc.stdin.write(b'{"jsonrpc":"2.0","method":"notifications/initialized"}\n')
        except BaseException:
            self.__exit__(None, None, None)
            raise
        return self

    def rpc(self, method, params):
        self.seq += 1
        self.proc.stdin.write(
            (json.dumps({"jsonrpc": "2.0", "id": self.seq, "method": method, "params": params}) + "\n").encode()
        )
        deadline = time.monotonic() + 20
        while time.monotonic() < deadline:
            while b"\n" in self.buffer:
                line, self.buffer = self.buffer.split(b"\n", 1)
                result = json.loads(line)
                if result.get("id") == self.seq:
                    if "error" in result:
                        raise RuntimeError("Direct bridge RPC error")
                    return result["result"]
            if self.selector.select(timeout=min(1, max(0, deadline - time.monotonic()))):
                data = os.read(self.proc.stdout.fileno(), 65536)
                if not data:
                    raise RuntimeError("Direct bridge EOF")
                self.buffer += data
        raise TimeoutError("Direct bridge timed out")

    def search(self, nonce, tagged=False):
        args = {"query": nonce, "num_results": 20, "detail": "full"}
        if tagged:
            args["tag"] = "harness-probe"
        result = self.rpc("tools/call", {"name": "brain_search", "arguments": args})
        if result.get("isError"):
            raise RuntimeError("Direct search returned isError")
        return result

    def __exit__(self, *_):
        self.selector.close()
        self.proc.stdin.close()
        try:
            self.proc.wait(timeout=3)
        except subprocess.TimeoutExpired:
            self.proc.terminate()
            self.proc.wait(timeout=3)


def probe(harness, args):
    started = time.monotonic()
    nonce, ingest_nonce = "hp" + uuid.uuid4().hex, "hi" + uuid.uuid4().hex
    content = f"Harness probe {harness} {nonce}"
    prompt = (
        f"Run this bounded BrainLayer connectivity probe only. Transcript marker {ingest_nonce}. "
        f'Call the native brain_search tool with query "{QUERY}". Then call native brain_store exactly once '
        f'with content "{content}", tags ["harness-probe"], project "brainlayer", importance 1. '
        "Use your real configured MCP tools. No shell, file edits, config changes, delegation, or other stores. "
        "If tools are unavailable report that; do not emulate them. Stop after the tool results."
    )
    row = {
        "harness": harness,
        "nonce": nonce,
        "ingest_nonce": ingest_nonce,
        "prompt": prompt,
        "configs": discover_configs(harness, Path.home(), Path.cwd()),
        "legs": {},
        "transport": "NOT_PROVEN",
    }
    cmd = command(harness, prompt)
    if not cmd or not shutil.which(cmd[0]):
        row.update(status="FAIL", reason=MANUAL.get(harness, "Harness executable missing"), seconds=0)
        return row
    transports = []
    stdout_path, stderr_path = args.output / f"{harness}.stdout.jsonl", args.output / f"{harness}.stderr"
    with stdout_path.open("w") as stdout, stderr_path.open("w") as stderr:
        # Keep HOME, MCP config, auth, hooks and model selection intact.
        proc = subprocess.Popen(cmd, stdout=stdout, stderr=stderr, stdin=subprocess.DEVNULL, start_new_session=True)
        deadline = time.monotonic() + args.timeout
        try:
            while proc.poll() is None and time.monotonic() < deadline:
                transports.extend(observe(proc.pid))
                time.sleep(1)
        finally:
            if proc.poll() is None:
                # This is our new session group, never a pattern matching kill.
                os.killpg(proc.pid, signal.SIGTERM)
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(proc.pid, signal.SIGKILL)
                    proc.wait(timeout=5)
    output = stdout_path.read_text()
    calls = parse_calls(output)
    row["session_id"] = session_id(output)
    legs = row["legs"]
    legs["search"] = any(c["ok"] and c["tool"] == "brain_search" and c["args"].get("query") == QUERY for c in calls)
    legs["store"] = any(
        c["ok"]
        and c["tool"] == "brain_store"
        and c["args"].get("content") == content
        and c["args"].get("tags") == ["harness-probe"]
        and c["args"].get("project") == "brainlayer"
        and c["args"].get("importance") == 1
        for c in calls
    )
    row["calls"] = calls
    row["exit_code"] = proc.returncode
    row["transport_evidence"] = list({json.dumps(t, sort_keys=True): t for t in transports}.values())
    legs["transport"] = bool(transports)
    if transports:
        row["transport"] = ", ".join(sorted({str(t["command"]) + " → " + t["socket"] for t in transports}))
    legs["refound"], legs["ingest"] = False, False
    # agy owns its own transcript format; BrainLayer has no agy watcher adapter.
    if harness == "agy":
        legs["ingest"] = "NOT_APPLICABLE"
    try:
        with Bridge() as bridge:
            deadline = time.monotonic() + args.ingest_timeout
            while True:
                if legs["store"] and not legs["refound"]:
                    result = bridge.search(nonce, tagged=True)
                    legs["refound"] = matching_hit(result, nonce)
                    if legs["refound"]:
                        row["refound_result"] = result
                if legs["ingest"] is False:
                    result = bridge.search(ingest_nonce)
                    legs["ingest"] = bool(row["session_id"]) and matching_hit(
                        result, ingest_nonce, transcript=True, session_id=row["session_id"]
                    )
                    if legs["ingest"]:
                        row["ingest_result"] = result
                if (legs["refound"] and legs["ingest"]) or time.monotonic() >= deadline:
                    break
                time.sleep(3)
    except (OSError, RuntimeError, TimeoutError, ValueError) as exc:
        row["verification_error"] = str(exc)
    row["status"] = verdict(legs) if proc.returncode == 0 else "FAIL"
    row["reason"] = ", ".join(k + " not proven" for k, v in legs.items() if v is False)
    if proc.returncode != 0:
        row["reason"] += f"; harness exit {proc.returncode}; inspect private stderr"
    row["seconds"] = round(time.monotonic() - started, 2)
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--harnesses", nargs="+", choices=HARNESSES, default=list(HARNESSES))
    parser.add_argument("--timeout", type=float, default=180)
    parser.add_argument("--ingest-timeout", type=float, default=120)
    parser.add_argument("--output", type=Path, required=True, help="New private directory for receipts")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False, mode=0o700)
    metadata = {
        "host": socket.gethostname(),
        "started_at": datetime.now(timezone.utc).isoformat(),
        "cwd": str(Path.cwd()),
        "commit": run(["git", "rev-parse", "HEAD"]).strip(),
        "script_sha256": {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (Path(__file__), Path(__file__).with_name("harness_probe_contract.py"))
        },
    }
    (args.output / "run.json").write_text(json.dumps(metadata, indent=2))
    rows = []
    for harness in args.harnesses:
        print(f"Probing {harness}...", flush=True)
        try:
            rows.append(probe(harness, args))
        except (
            OSError,
            RuntimeError,
            subprocess.SubprocessError,
            ValueError,
            KeyError,
            TypeError,
            AttributeError,
        ) as exc:
            rows.append({"harness": harness, "status": "FAIL", "reason": type(exc).__name__})
        (args.output / "results.json").write_text(json.dumps(rows, indent=2))
        (args.output / "table.md").write_text(table(rows) + "\n")
        print(table(rows[-1:]), flush=True)
    print(table(rows))
    if tuple(args.harnesses) != HARNESSES:
        print("PARTIAL RUN: not a release receipt (all harnesses required).")
        return 1
    return exit_code(rows)


if __name__ == "__main__":
    sys.exit(main())
