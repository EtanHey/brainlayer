"""Real requests against a loopback resumable server; no Drive or credentials."""

import json
import socket
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace

import pytest

from brainlayer import backup_daily as backup
from tests.test_backup_daily import _DriveResponse
from tests.test_backup_upload_resilience import _last_run_receipt, lane  # noqa: F401


@pytest.fixture
def server(monkeypatch):
    state = SimpleNamespace(committed=0, calls=[], failures=[], probe_failures=[], always_fail=False)

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_PUT(self):
            content_range = self.headers["Content-Range"]
            body = self.rfile.read(int(self.headers["Content-Length"]))
            state.calls.append((content_range, body))
            is_probe = content_range.startswith("bytes */")
            total = int(content_range.rsplit("/", 1)[1])
            failures = state.probe_failures if is_probe else state.failures
            failure = failures.pop(0) if failures else (503, 0) if state.always_fail else None
            if not is_probe:
                start = int(content_range.split()[1].split("-", 1)[0])
                if start != state.committed:
                    self.send_response(409)
                    self.end_headers()
                    return
                state.committed += failure[1] if failure else len(body)
            if failure and failure[0] == "reset":
                self.connection.shutdown(socket.SHUT_RDWR)
                self.connection.close()
                return
            status = failure[0] if failure else 200 if state.committed == total else 308
            payload = json.dumps({"id": "done"}).encode() if status == 200 else b""
            self.send_response(status)
            self.send_header("Content-Length", str(len(payload)))
            if state.committed:
                self.send_header("Range", f"bytes=0-{state.committed - 1}")
            self.end_headers()
            self.wfile.write(payload)

    httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    state.url = f"http://127.0.0.1:{httpd.server_port}/session"
    monkeypatch.setattr(backup.requests, "post", lambda *a, **kw: _DriveResponse(200, headers={"Location": state.url}))
    state.sleeps = []
    monkeypatch.setattr(backup, "_sleep", state.sleeps.append)
    yield state
    httpd.shutdown()
    httpd.server_close()
    thread.join(timeout=2)


@pytest.mark.parametrize("failure", [408, 429, 500, 501, 502, 503, 504, 599, "reset"])
def test_ambiguous_chunk_queries_committed_offset_before_resending(tmp_path, server, failure):
    unit = 256 * 1024
    data = bytes(range(256)) * (3 * 1024) + b"tail"
    artifact = tmp_path / "snapshot.db.zst"
    artifact.write_bytes(data)
    server.failures = [(failure, unit)]
    result = backup.upload_file_to_drive_raw(
        artifact, "fake-folder", SimpleNamespace(token="fake"), machine_id="fake", chunk_size=2 * unit
    )
    assert result == {"id": "done"}
    assert [c[0] for c in server.calls] == [
        f"bytes 0-{2 * unit - 1}/{len(data)}",
        f"bytes */{len(data)}",
        f"bytes {unit}-{3 * unit - 1}/{len(data)}",
        f"bytes {3 * unit}-{len(data) - 1}/{len(data)}",
    ]
    assert b"".join(body for _, body in server.calls[2:]) == data[unit:]


@pytest.mark.parametrize("failure", [408, 429, 503, "reset"])
def test_failed_probe_retries_probe_with_backoff(tmp_path, server, failure):
    artifact = tmp_path / "snapshot.db.gz"
    artifact.write_bytes(b"abcdef")
    server.failures = [(408, 3)]
    server.probe_failures = [(failure, 0)]
    result = backup.upload_file_to_drive_raw(
        artifact, "fake-folder", SimpleNamespace(token="fake"), machine_id="fake", max_attempts=4
    )
    assert result["id"] == "done"
    assert [c[0] for c in server.calls] == ["bytes 0-5/6", "bytes */6", "bytes */6", "bytes 3-5/6"]
    assert server.sleeps == [60]


@pytest.mark.parametrize("budget", ["attempts", "elapsed"])
def test_retry_exhaustion_persists_failed_receipt_and_session(lane, server, monkeypatch, budget):  # noqa: F811
    monkeypatch.setattr(backup, "_drive_put_with_deadline", lane.real_put)
    server.always_fail = True
    if budget == "elapsed":
        monkeypatch.setenv(backup.DRIVE_UPLOAD_STALL_MAX_ATTEMPTS_ENV, "30")
        monkeypatch.setenv(backup.DRIVE_UPLOAD_RETRY_BUDGET_ENV, "90")
    with pytest.raises(backup.DriveUploadStalledError, match="confirmed 0/6 bytes"):
        lane.run()
    receipt = _last_run_receipt(lane.log)
    assert receipt["uploaded"] is receipt["verified"] is False
    assert receipt["upload_state"] == "failed"
    assert receipt["error_code"] == "drive_upload_stalled"
    assert receipt["bytes_confirmed"] == 0
    assert receipt["resumable_upload"]["url"] == server.url
    assert receipt["resumable_upload"]["size"] == 6
    assert lane.snapshot.read_bytes() == b"abcdef"
    assert server.sleeps == ([60, 30] if budget == "elapsed" else [])
    assert [c[0] for c in server.calls] == ["bytes 0-5/6"] + ["bytes */6"] * (3 if budget == "elapsed" else 1)


def test_repeated_no_progress_uses_exponential_capped_backoff(tmp_path, server):
    artifact = tmp_path / "snapshot.db.gz"
    artifact.write_bytes(b"abcdef")
    server.failures = [(503, 0)] * 7
    with pytest.raises(backup.DriveUploadStalledError):
        backup.upload_file_to_drive_raw(
            artifact, "fake-folder", SimpleNamespace(token="fake"), machine_id="fake", max_attempts=7
        )
    assert server.sleeps == [30, 60, 120, 240, 300, 300]
    assert [c[0] for c in server.calls] == ["bytes 0-5/6", "bytes */6"] * 7


def test_probe_of_completed_upload_returns_metadata_without_resending(tmp_path, server):
    artifact = tmp_path / "snapshot.db.gz"
    artifact.write_bytes(b"abcdef")
    server.failures = [(408, 6)]
    result = backup.upload_file_to_drive_raw(artifact, "fake-folder", SimpleNamespace(token="fake"), machine_id="fake")
    assert result["id"] == "done"
    assert [c[0] for c in server.calls] == ["bytes 0-5/6", "bytes */6"]
