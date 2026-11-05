"""#1413: one structured BrainBar busy contract, and safe reclamation of aged unmarked attempts.

Every BrainBar here is a scripted fake on a private Unix socket. The backup client talks real
JSON-RPC over the wire, with synthetic files only.
"""

import json
import os
import socket
import sqlite3
import tempfile
import threading
import time
from pathlib import Path

import pytest

from brainlayer import backup_daily

BUSY_TEXT = backup_daily.BRAINBAR_SNAPSHOT_BUSY_MESSAGE
# The wire contract, spelled literally so a rename on either side fails here.
STRUCTURED = {"brainlayer_error_code": "backup_snapshot_busy"}

BUSY_SHAPES = {
    # What BrainBarServer.handleBackupToolCallAsync sends now.
    "structured_jsonrpc": {"error": {"code": -32000, "message": BUSY_TEXT, "data": {**STRUCTURED, "admitted": False}}},
    # Older BrainBars: exact text only (legacy compatibility).
    "legacy_jsonrpc": {"error": {"code": -32000, "message": BUSY_TEXT}},
    # Sol's #1411 shape: an intermediary re-wrapped the rejection as a tool error.
    "wrapped_is_error": {"result": {"content": [{"type": "text", "text": f"Error: {BUSY_TEXT}"}], "isError": True}},
    "wrapped_meta_code": {
        "result": {"content": [{"type": "text", "text": "Error: busy"}], "isError": True, "_meta": STRUCTURED}
    },
    "wrapped_structured_content": {"result": {"content": [], "isError": True, "structuredContent": STRUCTURED}},
}

NOT_BUSY = {
    "terminal_error": {"error": {"code": -32000, "message": "vacuum failed"}},
    "terminal_is_error": {"result": {"content": [{"type": "text", "text": "Error: disk I/O error"}], "isError": True}},
    "success": {"result": {"content": [{"type": "text", "text": "{}"}]}},
    "busy_text_in_success": {"result": {"content": [{"type": "text", "text": BUSY_TEXT}]}},
    "other_code": {"error": {"code": -32000, "message": "x", "data": {"brainlayer_error_code": "other"}}},
    "malformed_content": {"result": {"content": "Error: nope", "isError": True}},
}


@pytest.mark.parametrize("shape", sorted(NOT_BUSY))
def test_terminal_and_success_shapes_are_not_busy(shape):
    assert not backup_daily._is_brainbar_snapshot_busy(NOT_BUSY[shape])


def test_structured_busy_code_matches_the_brainbar_admission_guard():
    server = Path(__file__).resolve().parents[1] / "brain-bar/Sources/BrainBar/BrainBarServer.swift"
    assert '"brainlayer_error_code": "backup_snapshot_busy"' in server.read_text()


class ScriptedBrainBar:
    """A fake BrainBar on a private Unix socket; `respond(request)` returns a reply or None."""

    def __init__(self, respond):
        self._dir = tempfile.mkdtemp(prefix="bl-1413-")  # AF_UNIX paths stay short.
        self.path = Path(self._dir) / "bb.sock"
        self.respond = respond
        self.requests: list[dict] = []
        self._server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self._server.bind(str(self.path))
        self._server.listen(8)
        self._thread = threading.Thread(target=self._serve, daemon=True)
        self._thread.start()

    def _serve(self):
        while True:
            try:
                conn, _ = self._server.accept()
            except OSError:
                return
            with conn:
                data = b""
                while not data.endswith(b"\n"):
                    chunk = conn.recv(65_536)
                    if not chunk:
                        break
                    data += chunk
                request = json.loads(data)
                self.requests.append(request)
                reply = self.respond(request)
                if reply is not None:
                    conn.sendall(json.dumps({"jsonrpc": "2.0", "id": request["id"], **reply}).encode() + b"\n")

    def close(self):
        self._server.close()
        self.path.unlink(missing_ok=True)
        os.rmdir(self._dir)


@pytest.mark.parametrize("shape", sorted(BUSY_SHAPES))
def test_busy_rejection_over_the_wire_never_unlinks_a_live_earlier_target(tmp_path, monkeypatch, shape):
    # Sol's #1413 repro: request 1 loses its answer while BrainBar keeps writing; request 2 is
    # rejected as busy. Before the fix, the wrapped shape was "terminal" and unlinked request 1.
    live, stop, writes = {}, threading.Event(), []

    def keep_writing(handle):  # The VACUUM INTO BrainBar keeps running after the lost answer.
        while not stop.is_set():
            handle.write(b"x" * 512)
            handle.flush()
            writes.append(1)
            time.sleep(0.005)

    def respond(request):
        if not live:
            live["path"] = Path(request["params"]["arguments"]["target_path"])
            live["handle"] = live["path"].open("wb")
            threading.Thread(target=keep_writing, args=(live["handle"],), daemon=True).start()
            return None  # Connection closes without an answer.
        return BUSY_SHAPES[shape]

    monkeypatch.setattr(backup_daily, "_sleep", lambda _seconds: None)
    bar = ScriptedBrainBar(respond)
    try:
        with pytest.raises(backup_daily.BrainBarSnapshotBusyError) as raised:
            backup_daily.request_brainbar_vacuum_into(
                tmp_path / "2026-11-05.db", socket_path=bar.path, timeout_seconds=5, retry_backoff_seconds=0
            )
        assert len(bar.requests) == 2  # No third request beside the unresolved first one.
        writes_at_rejection = len(writes)
        time.sleep(0.05)
        assert len(writes) > writes_at_rejection  # The writer was still running after the rejection.
        assert live["path"].exists()
        assert os.fstat(live["handle"].fileno()).st_nlink == 1  # Never unlinked under the writer.
        assert str(live["path"]) in str(raised.value)
    finally:
        stop.set()
        bar.close()
        if live:
            live["handle"].close()


def _status_bar(*server_infos):
    """Answer successive initialize probes with the given serverInfo dicts (last one repeats)."""
    answers = list(server_infos)

    def respond(request):
        assert request["method"] == "initialize"
        info = answers.pop(0) if len(answers) > 1 else answers[0]
        return {"result": {"serverInfo": info}}

    return ScriptedBrainBar(respond)


NOW = 1_000_000.0
EPOCH = NOW - 200_000  # The writer started long before the aged attempt was created.


def _attempt(directory: Path, name: str, age: float) -> Path:
    path = directory / f".2026-11-05.db.attempt-{name}"
    path.write_bytes(b"partial")
    os.utime(path, (NOW - age, NOW - age))
    return path


def _sweep(directory: Path, socket_path: Path | None, writer_started_at: float = EPOCH):
    probe = (
        None
        if socket_path is None
        else (lambda: backup_daily._brainbar_snapshot_idle_in_epoch(socket_path, writer_started_at))
    )
    return backup_daily._sweep_stale_backup_attempts(
        directory, max_age_seconds=86_400, now=NOW, writer_started_at=writer_started_at, writer_snapshot_idle=probe
    )


def test_aged_unmarked_attempt_in_the_same_epoch_survives_without_status(tmp_path):
    # The #1413 gap: age alone never proves the writer stopped, so without a status it stays.
    aged = _attempt(tmp_path, "aged", 90_000)
    deleted, surviving = _sweep(tmp_path, None)
    assert deleted == [] and surviving == [aged]


def test_idle_status_in_the_same_epoch_reclaims_only_the_aged_unmarked_attempt(tmp_path):
    aged = _attempt(tmp_path, "aged", 90_000)
    recent = _attempt(tmp_path, "recent", 60)
    bar = _status_bar({"backupWriterStartedAtUnix": EPOCH, "backupSnapshotInProgress": False})
    try:
        deleted, surviving = _sweep(tmp_path, bar.path)
    finally:
        bar.close()
    assert deleted == [aged.name]
    assert surviving == [recent]
    assert not aged.exists() and recent.exists()
    assert len(bar.requests) == 1  # Asked once, on the wire.


@pytest.mark.parametrize(
    "server_info",
    [
        pytest.param({"backupWriterStartedAtUnix": EPOCH, "backupSnapshotInProgress": True}, id="in_progress"),
        pytest.param({"backupWriterStartedAtUnix": EPOCH}, id="older_brainbar_without_status"),
        pytest.param({"backupWriterStartedAtUnix": EPOCH, "backupSnapshotInProgress": "false"}, id="unknown_status"),
        pytest.param({"backupWriterStartedAtUnix": EPOCH, "backupSnapshotInProgress": None}, id="null_status"),
        pytest.param({"backupWriterStartedAtUnix": EPOCH + 5, "backupSnapshotInProgress": False}, id="restarted"),
        pytest.param({"backupSnapshotInProgress": False}, id="missing_epoch"),
    ],
)
def test_aged_unmarked_attempt_survives_unless_idle_is_proven_in_the_same_epoch(tmp_path, server_info):
    aged = _attempt(tmp_path, "aged", 90_000)
    bar = _status_bar(server_info)
    try:
        deleted, surviving = _sweep(tmp_path, bar.path)
    finally:
        bar.close()
    assert deleted == [] and surviving == [aged] and aged.exists()


def test_probe_failure_fails_closed(tmp_path):
    aged = _attempt(tmp_path, "aged", 90_000)
    assert _sweep(tmp_path, tmp_path / "no-brainbar.sock") == ([], [aged])


def test_stale_epoch_attempt_is_still_reclaimed_while_a_new_snapshot_runs(tmp_path):
    # Restart: the attempt predates this writer epoch, so its writer is gone whatever runs now;
    # a current-epoch aged attempt beside it is preserved because a snapshot is in progress.
    prior = _attempt(tmp_path, "prior", 110_000)
    epoch = NOW - 100_000
    current = _attempt(tmp_path, "current", 90_000)  # Aged past the cutoff, but after the restart.
    bar = _status_bar({"backupWriterStartedAtUnix": epoch, "backupSnapshotInProgress": True})
    try:
        deleted, surviving = _sweep(tmp_path, bar.path, writer_started_at=epoch)
    finally:
        bar.close()
    assert deleted == [prior.name]
    assert surviving == [current]
    assert len(bar.requests) == 1  # Kept because BrainBar said a snapshot runs, not by age.


def test_create_snapshot_wires_the_epoch_status_into_the_sweep(tmp_path, monkeypatch):
    # The live entry point asks BrainBar for status over the socket and reclaims the aged
    # unmarked attempt; a refused snapshot request then ends the run.
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    source = tmp_path / "brainlayer.db"
    with sqlite3.connect(source) as db:
        db.execute("CREATE TABLE chunks (id TEXT)")
    aged = output_dir / ".2026-11-04.db.attempt-1-aged"
    aged.write_bytes(b"partial")
    old = time.time() - backup_daily.DEFAULT_BACKUP_ATTEMPT_MAX_AGE_SECONDS - 60
    os.utime(aged, (old, old))
    epoch = old - 10  # Same writer epoch: started before the attempt.

    def respond(request):
        if request["method"] == "initialize":
            return {"result": {"serverInfo": {"backupWriterStartedAtUnix": epoch, "backupSnapshotInProgress": False}}}
        return {"result": {"content": [{"type": "text", "text": "Error: stop here"}], "isError": True}}

    bar = ScriptedBrainBar(respond)
    monkeypatch.setenv("BRAINBAR_SOCKET_PATH", str(bar.path))
    monkeypatch.setattr(backup_daily, "_probe_raw_reclamation", lambda _path: (False, "fixture"))
    monkeypatch.setattr(backup_daily, "_sleep", lambda _seconds: None)
    try:
        with pytest.raises(RuntimeError, match="stop here"):
            backup_daily.create_sqlite_backup_artifact(source, output_dir, date_stamp="2026-11-05")
    finally:
        bar.close()
    assert not aged.exists()
    assert [r["method"] for r in bar.requests][:2] == ["initialize", "initialize"]
