"""Real requests over loopback; no Drive, credentials, or production data."""

import hashlib
import json
import socket
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace

import pytest

from brainlayer import backup_daily as backup

MIB = 1024 * 1024


@pytest.fixture
def upload_server():
    state = SimpleNamespace(
        rtt=0.3, rate=64 * MIB, confirmed=0, puts=0, probes=0, peers=set(), sockets=[], bodies=[], errors=[]
    )

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def setup(self):
            super().setup()
            state.sockets.append(self.connection)

        def log_message(self, *_args):
            pass

        def do_PUT(self):
            state.peers.add(self.client_address)
            content_range = self.headers["Content-Range"]
            total = int(content_range.rsplit("/", 1)[1])
            length = int(self.headers["Content-Length"])
            if length:
                state.puts += 1
                digest = hashlib.sha256()
                remaining = length
                started = time.monotonic()
                while remaining:
                    block = self.rfile.read(min(256 * 1024, remaining))
                    if not block:
                        state.errors.append("truncated request")
                        return
                    digest.update(block)
                    remaining -= len(block)
                    time.sleep(max(0, started + (length - remaining) / state.rate - time.monotonic()))
                start = int(content_range.split()[1].split("-", 1)[0])
                if start != state.confirmed:
                    state.errors.append("unexpected offset")
                state.confirmed = start + length
                state.bodies.append((length, digest.hexdigest()))
            else:
                state.probes += 1
            time.sleep(state.rtt)
            payload = json.dumps({"id": "loopback-file", "size": str(total)}).encode()
            complete = state.confirmed == total
            self.send_response(200 if complete else 308)
            self.send_header("Content-Length", str(len(payload) if complete else 0))
            if state.confirmed:
                self.send_header("Range", f"bytes=0-{state.confirmed - 1}")
            self.end_headers()
            if complete:
                try:
                    self.wfile.write(payload)
                except (BrokenPipeError, ConnectionResetError):
                    pass  # A deadline can abandon the response; the probe still sees committed bytes.

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server.daemon_threads = False
    thread = threading.Thread(target=server.serve_forever)
    thread.start()
    state.url = f"http://127.0.0.1:{server.server_port}/session"
    try:
        yield state
    finally:
        server.shutdown()
        for connection in state.sockets:
            try:
                connection.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
        server.server_close()
        thread.join()


def _upload(snapshot, server, **kwargs):
    return backup.upload_file_to_drive_raw(
        snapshot,
        "local-folder",
        SimpleNamespace(token="fake"),
        machine_id="test-machine",
        resume_session_url=server.url,
        **kwargs,
    )


def test_upload_throughput_with_300ms_rtt(tmp_path, upload_server, capsys):
    snapshot = tmp_path / "fixture.zst"
    snapshot.write_bytes(b"x" * (64 * MIB))
    started = time.monotonic()
    result = _upload(snapshot, upload_server)
    elapsed = time.monotonic() - started
    rate = snapshot.stat().st_size / elapsed / MIB
    assert result["id"] == "loopback-file"
    assert upload_server.confirmed == snapshot.stat().st_size
    assert sum(n for n, _ in upload_server.bodies) == snapshot.stat().st_size
    assert all(digest == hashlib.sha256(b"x" * n).hexdigest() for n, digest in upload_server.bodies)
    assert not upload_server.errors
    print(
        json.dumps(
            {
                "mib_per_second": rate,
                "seconds": elapsed,
                "puts": upload_server.puts,
                "probes": upload_server.probes,
                "connections": len(upload_server.peers),
            }
        )
    )
    with capsys.disabled():
        print(f"loopback 300ms RTT: {rate:.2f} MiB/s, {upload_server.puts} PUTs, {elapsed:.3f}s")
    assert rate >= 30, "RTT-bound chunking regression (64 MiB/s link, 300 ms per request)"
    assert upload_server.puts == 1
    assert len(upload_server.peers) == 1, "successful chunks/probes must reuse the TCP connection"


def test_slow_link_completes_without_deadline_probe_loop(tmp_path, upload_server, monkeypatch):
    # Accelerate both link and deadline 100x: an 8 MiB PUT at 45 KiB/s takes
    # ~182s, above the old 120s watchdog, below the fixed 256s byte budget.
    upload_server.rtt = 0.003
    upload_server.rate = 45 * 1024 * 100
    snapshot = tmp_path / "fixture.zst"
    snapshot.write_bytes(b"x" * (8 * MIB))
    real_put = backup._drive_put_with_deadline

    def accelerated(*args, deadline, **kwargs):
        return real_put(*args, deadline=deadline / 100, **kwargs)

    monkeypatch.setattr(backup, "_drive_put_with_deadline", accelerated)
    result = _upload(snapshot, upload_server, max_attempts=1)
    assert result["id"] == "loopback-file"
    assert upload_server.confirmed == snapshot.stat().st_size
    assert upload_server.bodies == [(snapshot.stat().st_size, hashlib.sha256(snapshot.read_bytes()).hexdigest())]
    assert not upload_server.errors
    assert upload_server.puts == upload_server.probes == 1
