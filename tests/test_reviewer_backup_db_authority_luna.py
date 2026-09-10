"""Independent generated review probes for backup socket attempt lifetime."""

from pathlib import Path
import socket

import pytest

from brainlayer import backup_run_once


def test_reviewer_terminal_error_reclaims_attempt_and_marker(tmp_path, monkeypatch):
    seen = {}

    def fake_send(_socket, request, timeout_seconds):
        path = Path(request["params"]["arguments"]["target_path"])
        path.write_bytes(b"synthetic partial")
        path.with_name(path.name + ".complete").write_text("complete\n")
        seen["path"] = path
        return {"result": {"isError": True, "content": [{"text": "synthetic failure"}]}}

    monkeypatch.setattr(backup_run_once.backup_daily, "_send_brainbar_json_request", fake_send)
    generation = tmp_path / "generation"
    generation.mkdir()
    with pytest.raises(backup_run_once.BackupRunOnceError, match="synthetic failure"):
        backup_run_once._socket_snapshot(Path("/tmp/unused.sock"), generation / "db", generation)

    # A terminal response proves the writer has stopped; the old helper reclaimed both.
    assert not seen["path"].exists()
    assert not seen["path"].with_name(seen["path"].name + ".complete").exists()


def test_reviewer_lost_response_preserves_isolated_attempt(tmp_path, monkeypatch):
    seen = {}

    def fake_send(_socket, request, timeout_seconds):
        path = Path(request["params"]["arguments"]["target_path"])
        path.write_bytes(b"synthetic in-flight")
        seen["path"] = path
        raise socket.timeout("synthetic timeout")

    monkeypatch.setattr(backup_run_once.backup_daily, "_send_brainbar_json_request", fake_send)
    generation = tmp_path / "generation"
    generation.mkdir()
    with pytest.raises(TimeoutError, match="synthetic timeout"):
        backup_run_once._socket_snapshot(Path("/tmp/unused.sock"), generation / "db", generation)

    assert seen["path"].is_file()
    assert not (generation / "db").exists()
