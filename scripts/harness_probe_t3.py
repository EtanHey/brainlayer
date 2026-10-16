"""T3 adapter through an existing app-injected authenticated MCP session only."""

import json
import os
import re
import selectors
import sqlite3
import subprocess
import sys
import time
from contextlib import closing
from pathlib import Path
from urllib.parse import urlsplit

from harness_probe import Bridge, observe


def events(data, identity):
    thread = data["thread"]["threadId"]
    lines = [json.dumps({"type": "thread.started", "thread_id": identity or thread})]
    for item in data.get("items", []):
        if item.get("sourceThreadId") != thread or item.get("type") != "dynamic_tool" or item.get("textTruncated"):
            continue
        try:
            value = json.loads(item["text"])
        except (KeyError, ValueError):
            continue
        name = value.get("toolName", "")
        if name.startswith("brainlayer."):
            lines.append(
                json.dumps(
                    {
                        "type": "item.completed",
                        "item": {
                            "type": "mcp_tool_call",
                            "tool": name.split(".")[-1],
                            "arguments": value.get("input", {}),
                            "status": item.get("status"),
                            "result": {"content": value.get("output", [])},
                        },
                    }
                )
            )
    return "\n".join(lines)


def native_id(thread):
    path = Path.home() / ".t3/userdata/statev2.sqlite"
    with closing(sqlite3.connect(path.as_uri() + "?mode=ro", uri=True, timeout=1)) as conn:
        rows = conn.execute(
            "SELECT payload_json FROM orchestration_v2_projection_provider_threads WHERE thread_id=? AND driver=?",
            (thread, "codex"),
        ).fetchall()
    identities = set()
    for row in rows:
        ref = json.loads(row[0]).get("nativeThreadRef", {})
        identity = ref.get("nativeId", "")
        if (
            ref.get("driver") == "codex"
            and ref.get("strength") == "strong"
            and re.fullmatch(r"[0-9a-f-]{36}", identity)
        ):
            identities.add(identity)
    return next(iter(identities)) if len(identities) == 1 else None


class ExistingMCP(Bridge):
    def __enter__(self):
        helper = Path.home() / ".t3/bin/t3"
        self.proc = subprocess.Popen(
            [str(helper), "acp-mcp-bridge"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=sys.stderr,
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
                    "protocolVersion": "2025-06-18",
                    "capabilities": {},
                    "clientInfo": {"name": "brainlayer-harness-probe", "version": "1"},
                },
            )
            self.proc.stdin.write(b'{"jsonrpc":"2.0","method":"notifications/initialized"}\n')
        except BaseException:
            self.__exit__(None, None, None)
            raise
        return self

    def call(self, name, args):
        result = self.rpc("tools/call", {"name": name, "arguments": args})
        if result.get("isError"):
            raise RuntimeError("T3 MCP tool failed: " + name)
        if result.get("structuredContent"):
            return result["structuredContent"]
        return json.loads(next(b["text"] for b in result["content"] if b.get("type") == "text"))


def probe(prompt, deadline):
    if not all(os.environ.get(k) for k in ("T3_ACP_MCP_ENDPOINT", "T3_ACP_MCP_AUTHORIZATION")):
        raise RuntimeError("FAIL(auth): existing T3 MCP environment unavailable; no token minted")
    record = json.loads((Path.home() / ".t3/userdata/server-runtime.json").read_text())
    endpoint = urlsplit(os.environ["T3_ACP_MCP_ENDPOINT"])
    if endpoint.hostname not in ("127.0.0.1", "localhost", "::1") or endpoint.port != record["port"]:
        raise RuntimeError("Existing MCP endpoint does not match this host installed T3 runtime")
    root = record["pid"]
    baseline = {e["pid"] for e in observe(root)}
    with ExistingMCP() as mcp:
        # Inherit the existing native caller's provider/model/modes and MCP config.
        launched = mcp.call(
            "t3_thread_launch", {"scratch": True, "title": "BrainLayer harness probe", "message": prompt}
        )
        thread = launched["threadId"]
        complete = False
        print(json.dumps({"type": "harness.t3_thread", "thread_id": thread}), flush=True)
        try:
            while time.monotonic() < deadline:
                found = [e for e in observe(root) if e["pid"] not in baseline]
                if found:
                    print(json.dumps({"type": "harness.transport", "evidence": found}), flush=True)
                data = mcp.call(
                    "t3_thread_read", {"threadId": thread, "view": "activity", "limit": 100, "maxCharsPerItem": 50000}
                )
                if data["thread"]["status"] in ("completed", "failed", "cancelled", "interrupted"):
                    complete = True
                    if data.get("hasMore"):
                        raise RuntimeError("T3 timeline truncated; proof incomplete")
                    identity = native_id(thread)
                    print(events(data, identity), flush=True)
                    return 0 if data["thread"]["status"] == "completed" else 1
                time.sleep(2)
            raise TimeoutError("Own T3 probe timed out")
        finally:
            if not complete:
                try:
                    mcp.call(
                        "t3_thread_interrupt",
                        {"threadId": thread, "runId": launched.get("runId"), "clientRequestId": "probe-stop-" + thread},
                    )
                except (RuntimeError, OSError, TimeoutError):
                    print("Own T3 thread cleanup unproven: " + thread, file=sys.stderr)
