"""Native app probe adapters. Never start servers, mint tokens, or initiate login."""

import base64
import hashlib
import json
import os
import signal
import socket
import struct
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

from harness_probe import observe


def agy_auth_error(home):
    path = home / ".gemini/antigravity-cli/antigravity-oauth-token"
    if not path.is_file():
        return "FAIL(auth): existing agy CLI credential file missing; no login attempted"
    try:
        token = json.loads(path.read_text())["token"]
        expiry = datetime.fromisoformat(token["expiry"].replace("Z", "+00:00"))
        if not token.get("access_token") or expiry <= datetime.now(timezone.utc) + timedelta(minutes=5):
            return "FAIL(auth): existing agy CLI bearer expired/near expiry; no refresh or login attempted"
    except (OSError, ValueError, KeyError, TypeError):
        return "FAIL(auth): existing agy credential cannot be verified; no login attempted"
    return None


def event(item):
    return json.dumps({"type": "item.completed", "item": item})


def app_events(thread):
    lines = [json.dumps({"type": "thread.started", "thread_id": thread["id"]})]
    for turn in thread.get("turns", []):
        for item in turn.get("items", []):
            if item.get("type") == "mcpToolCall":
                lines.append(event(dict(item, type="mcp_tool_call")))
    return "\n".join(lines)


# Unix transport: https://learn.chatgpt.com/docs/app-server#protocol
class AppSocket:
    """RFC6455 client over the existing app daemon's Unix control socket."""

    def __init__(self, path):
        self.sock = socket.socket(socket.AF_UNIX)
        self.sock.settimeout(20)
        self.sock.connect(str(path))
        key = base64.b64encode(os.urandom(16)).decode()
        self.buffer = b""
        self.seq = 0
        self.sock.sendall(
            (
                "GET / HTTP/1.1\r\nHost: localhost\r\nUpgrade: websocket\r\nConnection: Upgrade\r\nSec-WebSocket-Version: 13\r\nSec-WebSocket-Key: "
                + key
                + "\r\n\r\n"
            ).encode()
        )
        while b"\r\n\r\n" not in self.buffer:
            data = self.sock.recv(4096)
            if not data:
                raise RuntimeError("App socket closed during handshake")
            self.buffer += data
            if len(self.buffer) > 65536:
                raise RuntimeError("App handshake too large")
        headers, self.buffer = self.buffer.split(b"\r\n\r\n", 1)
        expected = base64.b64encode(hashlib.sha1((key + "258EAFA5-E914-47DA-95CA-C5AB0DC85B11").encode()).digest())
        if not headers.startswith(b"HTTP/1.1 101") or expected not in headers:
            raise RuntimeError("App WebSocket upgrade refused")
        self.rpc(
            "initialize",
            {
                "clientInfo": {"name": "brainlayer-harness-probe", "version": "1"},
                "capabilities": {"experimentalApi": True},
            },
        )
        self.send(json.dumps({"method": "initialized"}).encode())

    def read(self, n):
        while len(self.buffer) < n:
            data = self.sock.recv(min(65536, n - len(self.buffer)))
            if not data:
                raise RuntimeError("App socket closed")
            self.buffer += data
        data, self.buffer = self.buffer[:n], self.buffer[n:]
        return data

    def send(self, payload, opcode=1):
        mask = os.urandom(4)
        n = len(payload)
        size = (
            bytes([n | 128])
            if n < 126
            else bytes([126 | 128]) + struct.pack("!H", n)
            if n < 65536
            else bytes([127 | 128]) + struct.pack("!Q", n)
        )
        self.sock.sendall(bytes([128 | opcode]) + size + mask + bytes(c ^ mask[i % 4] for i, c in enumerate(payload)))

    def recv(self):
        payload = b""
        while True:
            first, second = self.read(2)
            n = second & 127
            if n == 126:
                n = struct.unpack("!H", self.read(2))[0]
            if n == 127:
                n = struct.unpack("!Q", self.read(8))[0]
            if n + len(payload) > 16 * 1024 * 1024 or second & 128:
                raise RuntimeError("Invalid app frame")
            data = self.read(n)
            opcode = first & 15
            if opcode == 8:
                raise RuntimeError("App sent close")
            if opcode == 9:
                self.send(data, 10)
                continue
            if opcode == 10:
                continue
            if opcode not in (0, 1):
                raise RuntimeError("App sent non-text frame")
            payload += data
            if first & 128:
                return json.loads(payload)

    def rpc(self, method, params):
        self.seq += 1
        deadline = time.monotonic() + 20
        self.send(json.dumps({"id": self.seq, "method": method, "params": params}).encode())
        while time.monotonic() < deadline:
            response = self.recv()
            if response.get("id") == self.seq:
                if "error" in response:
                    raise RuntimeError("App RPC " + method + " rejected: " + str(response["error"].get("message", "")))
                return response["result"]
        raise TimeoutError("App RPC timed out")

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.sock.close()


def app_probe(prompt, deadline):
    home = Path(os.environ.get("CODEX_HOME", Path.home() / ".codex"))
    sock = home / "app-server-control/app-server-control.sock"
    if not sock.is_socket():
        raise RuntimeError("Existing app daemon control socket missing; no server started")
    record = json.loads((home / "app-server-daemon/daemon.pid").read_text())
    root = record["pid"]
    baseline = {e["pid"] for e in observe(root)}
    with AppSocket(sock) as app:
        thread = app.rpc("thread/start", {"cwd": str(Path.cwd()), "threadSource": "brainlayer-harness-probe"})["thread"]
        print(json.dumps({"type": "thread.started", "thread_id": thread["id"]}), flush=True)
        started = app.rpc("turn/start", {"threadId": thread["id"], "input": [{"type": "text", "text": prompt}]})
        turn_id = started["turn"]["id"]
        complete = False
        try:
            while time.monotonic() < deadline:
                found = [e for e in observe(root) if e["pid"] not in baseline]
                if found:
                    print(json.dumps({"type": "harness.transport", "evidence": found}), flush=True)
                thread = app.rpc("thread/read", {"threadId": thread["id"], "includeTurns": True})["thread"]
                turns = thread.get("turns", [])
                if turns and turns[-1].get("status") != "inProgress":
                    complete = True
                    print(app_events(thread), flush=True)
                    return 0 if turns[-1].get("status") == "completed" else 1
                time.sleep(2)
            raise TimeoutError("Native app probe timed out")
        finally:
            if not complete:
                app.sock.settimeout(2)
                try:
                    app.rpc("turn/interrupt", {"threadId": thread["id"], "turnId": turn_id})
                except (OSError, RuntimeError, TimeoutError):
                    print("Own app turn cleanup unproven: " + thread["id"], file=sys.stderr)


def main():
    harness, prompt = sys.argv[1:3]
    deadline = time.monotonic() + float(os.environ.get("BRAINLAYER_NATIVE_PROBE_TIMEOUT", "150"))
    if harness == "codex-app":
        return app_probe(prompt, deadline)
    from harness_probe_t3 import probe

    return probe(prompt, deadline)


if __name__ == "__main__":

    def stop_owned_probe(*_):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, stop_owned_probe)
    try:
        sys.exit(main())
    except (RuntimeError, OSError, ValueError, TimeoutError) as error:
        print(str(error), file=sys.stderr)
        sys.exit(1)
