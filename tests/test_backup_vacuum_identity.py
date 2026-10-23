"""A backup aimed at one DB must never VACUUM a different DB into its folder; no production I/O.

Incident 2026-10-19 00:00 IDT: a run with a temp 8 KB db_path and temp staging reached the live
BrainBar socket, and BrainBar VACUUMed the 20 GB production DB into the temp folder, where the raw
copy was orphaned.
"""

import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from brainlayer import backup_daily as backup
from tests.test_backup_daily import _create_source_db


def _ok(payload: dict) -> dict:
    return {"result": {"content": [{"type": "text", "text": json.dumps({"status": "ok", **payload})}]}}


def _identity(path: Path) -> dict:
    info = os.stat(path)
    return {"db_dev": info.st_dev, "db_ino": info.st_ino}


def _fake_brainbar(served_db: Path, requests: list, *, enforce: bool = True, echo: bool = True):
    """Mimic BrainBar: refuse a mismatched expected_db_path before VACUUM, else vacuum served_db."""

    def send(socket_path, request, timeout_seconds):  # noqa: ARG001
        requests.append(request)
        args = request["params"]["arguments"]
        expected = args.get("expected_db_path")
        if enforce and expected is not None and os.path.realpath(expected) != os.path.realpath(served_db):
            return {
                "result": {
                    "isError": True,
                    "content": [{"type": "text", "text": f"{backup.BRAINBAR_DB_IDENTITY_MISMATCH}: {expected}"}],
                }
            }
        target = Path(args["target_path"])
        target.write_bytes(served_db.read_bytes())
        Path(f"{target}.complete").write_text("complete\n")
        return _ok(
            {"target_path": str(target), **({"db_path": str(served_db), **_identity(served_db)} if echo else {})}
        )

    return send


@pytest.fixture
def dbs(tmp_path):
    configured, production = tmp_path / "configured" / "brainlayer.db", tmp_path / "production" / "brainlayer.db"
    for path in (configured, production):
        path.parent.mkdir()
        _create_source_db(path)
    return configured, production


def _no_sleep(monkeypatch):
    monkeypatch.setattr(backup, "_sleep", lambda _s: pytest.fail("identity refusal must not retry"))


def test_request_sends_resolved_expected_db_path(tmp_path, dbs, monkeypatch):
    configured, _ = dbs
    link = tmp_path / "link.db"
    link.symlink_to(configured)
    requests: list = []
    monkeypatch.setattr(backup, "_send_brainbar_json_request", _fake_brainbar(configured, requests))

    backup.request_brainbar_vacuum_into(tmp_path / "out" / "snap.db", socket_path="/x.sock", expected_db_path=link)

    assert requests[0]["params"]["arguments"]["expected_db_path"] == os.path.realpath(configured)
    assert (tmp_path / "out" / "snap.db").exists()


def test_brainbar_identity_refusal_fails_closed_without_retry(tmp_path, dbs, monkeypatch):
    configured, production = dbs
    requests: list = []
    monkeypatch.setattr(backup, "_send_brainbar_json_request", _fake_brainbar(production, requests))
    _no_sleep(monkeypatch)
    out = tmp_path / "out"

    with pytest.raises(backup.BrainBarDatabaseIdentityError, match="identity"):
        backup.request_brainbar_vacuum_into(out / "snap.db", socket_path="/x.sock", expected_db_path=configured)

    assert len(requests) == 1
    assert list(out.iterdir()) == []


@pytest.mark.parametrize("echo", ["missing", "other"])
def test_unconfirmed_or_foreign_snapshot_is_deleted_and_refused(tmp_path, dbs, monkeypatch, echo):
    """An older BrainBar ignores expected_db_path and vacuums its own DB: never keep that copy."""
    configured, production = dbs
    requests: list = []
    monkeypatch.setattr(
        backup,
        "_send_brainbar_json_request",
        _fake_brainbar(production, requests, enforce=False, echo=echo == "other"),
    )
    _no_sleep(monkeypatch)
    out = tmp_path / "out"

    with pytest.raises(backup.BrainBarDatabaseIdentityError):
        backup.request_brainbar_vacuum_into(out / "snap.db", socket_path="/x.sock", expected_db_path=configured)

    assert len(requests) == 1
    assert list(out.iterdir()) == []


def test_artifact_refuses_mismatched_brainbar_before_any_raw(tmp_path, dbs, monkeypatch):
    configured, production = dbs
    requests: list = []
    monkeypatch.setattr(backup, "_send_brainbar_json_request", _fake_brainbar(production, requests))
    monkeypatch.setattr(backup, "_brainbar_writer_started_at", lambda *a: 1)
    monkeypatch.setattr(backup, "_important_usage_capacity_bytes", lambda *a: None)
    monkeypatch.setattr(backup.shutil, "disk_usage", lambda *a: SimpleNamespace(free=10**12))
    _no_sleep(monkeypatch)
    out = tmp_path / "backups"

    with pytest.raises(backup.BrainBarDatabaseIdentityError):
        backup.create_sqlite_backup_artifact(configured, out, date_stamp="2026-10-18")

    assert requests[0]["params"]["arguments"]["expected_db_path"] == os.path.realpath(configured)
    assert sorted(p.name for p in out.iterdir()) == []


@pytest.fixture
def raw_lane(tmp_path, monkeypatch):
    source, output = tmp_path / "source.db", tmp_path / "backups"
    _create_source_db(source)
    monkeypatch.setattr(backup, "_brainbar_writer_started_at", lambda *a: 1)
    monkeypatch.setattr(backup, "_important_usage_capacity_bytes", lambda *a: None)
    monkeypatch.setattr(backup.shutil, "disk_usage", lambda *a: SimpleNamespace(free=10**12))

    def promote(target, **_kw):
        target.write_bytes(source.read_bytes())
        return os.stat(target)

    monkeypatch.setattr(backup, "request_brainbar_vacuum_into", promote)
    return source, output


@pytest.mark.parametrize(
    "error",
    [KeyboardInterrupt(), backup.BackupTimeoutError("deadline"), SystemExit(1), RuntimeError("validator crashed")],
    ids=["interrupt", "timeout", "exit", "error"],
)
def test_noncanonical_staging_leaves_no_raw_when_run_exits_before_verification(raw_lane, monkeypatch, error):
    source, output = raw_lane

    def die(*_a, **_kw):
        raise error

    monkeypatch.setattr(backup, "_validate_backup_target", die)

    with pytest.raises(type(error)):
        backup.create_sqlite_backup_artifact(source, output, date_stamp="2026-10-18")

    assert not (output / "2026-10-18.db").exists()


def test_noncanonical_staging_keeps_raw_retained_on_purpose(raw_lane, monkeypatch):
    source, output = raw_lane
    monkeypatch.setattr(backup, "_sqlite_pragma_check", lambda *_a: "corrupt page")

    with pytest.raises(RuntimeError, match="retaining raw snapshot"):
        backup.create_sqlite_backup_artifact(source, output, date_stamp="2026-10-18")

    assert (output / "2026-10-18.db").exists()


def test_canonical_staging_retention_semantics_unchanged(raw_lane, monkeypatch):
    source, output = raw_lane
    monkeypatch.setattr(backup, "DEFAULT_STAGING_DIR", output)

    def die(*_a, **_kw):
        raise KeyboardInterrupt

    monkeypatch.setattr(backup, "_validate_backup_target", die)

    with pytest.raises(KeyboardInterrupt):
        backup.create_sqlite_backup_artifact(source, output, date_stamp="2026-10-18")

    assert (output / "2026-10-18.db").exists()


def test_noncanonical_staging_never_removes_a_raw_this_run_did_not_create(raw_lane, monkeypatch):
    source, output = raw_lane
    output.mkdir()
    (output / "2026-10-18.db").write_bytes(source.read_bytes())

    with pytest.raises(FileExistsError):
        backup.create_sqlite_backup_artifact(source, output, date_stamp="2026-10-18")

    assert (output / "2026-10-18.db").exists()


def test_foreign_db_on_production_socket_refuses_before_any_file_or_request(tmp_path, dbs, monkeypatch):
    """The 2026-10-19 caller: an ad-hoc probe outside pytest passed a temp db_path and no socket."""
    configured, production = dbs
    monkeypatch.delenv("BRAINBAR_SOCKET_PATH", raising=False)
    monkeypatch.setenv("BRAINLAYER_DB", str(production))
    monkeypatch.setattr(backup, "_send_brainbar_json_request", lambda *a, **kw: pytest.fail("socket reached"))
    out = tmp_path / "backups"

    with pytest.raises(backup.BrainBarDatabaseIdentityError, match="production BrainBar socket"):
        backup.create_sqlite_backup_artifact(configured, out, date_stamp="2026-10-18")

    assert not out.exists()


def test_configured_db_on_production_socket_passes_the_preflight(tmp_path, dbs, monkeypatch):
    _, production = dbs
    monkeypatch.delenv("BRAINBAR_SOCKET_PATH", raising=False)
    monkeypatch.setenv("BRAINLAYER_DB", str(production))
    requests: list = []
    monkeypatch.setattr(backup, "_send_brainbar_json_request", _fake_brainbar(production, requests))
    monkeypatch.setattr(backup, "_brainbar_writer_started_at", lambda *a: 1)
    monkeypatch.setattr(backup, "_important_usage_capacity_bytes", lambda *a: None)
    monkeypatch.setattr(backup.shutil, "disk_usage", lambda *a: SimpleNamespace(free=10**12))

    artifact = backup.create_sqlite_backup_artifact(production, tmp_path / "backups", date_stamp="2026-10-18")

    assert artifact.gzip_path.exists()
    assert len(requests) == 1


def test_echo_with_the_expected_pathname_but_another_inode_is_refused(tmp_path, dbs, monkeypatch):
    """Sol, #1414 round 1: a pathname echo cannot prove which file the open connection read."""
    configured, production = dbs
    requests: list = []

    def send(socket_path, request, timeout_seconds):  # noqa: ARG001
        requests.append(request)
        target = Path(request["params"]["arguments"]["target_path"])
        target.write_bytes(production.read_bytes())
        return _ok({"target_path": str(target), "db_path": str(configured), **_identity(production)})

    monkeypatch.setattr(backup, "_send_brainbar_json_request", send)
    _no_sleep(monkeypatch)
    out = tmp_path / "out"

    with pytest.raises(backup.BrainBarDatabaseIdentityError):
        backup.request_brainbar_vacuum_into(out / "snap.db", socket_path="/x.sock", expected_db_path=configured)

    assert len(requests) == 1
    assert list(out.iterdir()) == []


def test_echo_without_the_open_inode_is_refused(tmp_path, dbs, monkeypatch):
    configured, _ = dbs

    def send(socket_path, request, timeout_seconds):  # noqa: ARG001
        target = Path(request["params"]["arguments"]["target_path"])
        target.write_bytes(configured.read_bytes())
        return _ok({"target_path": str(target), "db_path": str(configured)})

    monkeypatch.setattr(backup, "_send_brainbar_json_request", send)
    _no_sleep(monkeypatch)
    out = tmp_path / "out"

    with pytest.raises(backup.BrainBarDatabaseIdentityError):
        backup.request_brainbar_vacuum_into(out / "snap.db", socket_path="/x.sock", expected_db_path=configured)

    assert list(out.iterdir()) == []


@pytest.mark.parametrize("echo", ["legacy", "missing_inode", "foreign_inode"])
@pytest.mark.parametrize("prior", ["completed", "completed_during_backoff", "unresolved", "peer_replacement"])
def test_lost_response_then_identity_failure_reconciles_only_owned_completed_attempts(
    tmp_path, dbs, monkeypatch, echo, prior
):
    """A loses its response; B completes without confirming the requested DB. Never retry B."""
    configured, foreign = dbs
    out = tmp_path / "attempts"
    out.mkdir()
    target = out / "snapshot.db"
    peer = out / ".snapshot.db.attempt-peer"
    peer.write_bytes(b"peer attempt")
    peer_marker = backup._backup_attempt_completion_marker(peer)
    peer_marker.write_text("complete\n")
    calls = []
    sleeps = []
    live_writer = None

    def complete(path):
        path.write_bytes(foreign.read_bytes())
        info = path.stat()
        backup._backup_attempt_completion_marker(path).write_text(
            json.dumps(
                {"schema_version": 1, "target_path": str(path), "target_dev": info.st_dev, "target_ino": info.st_ino}
            )
        )

    def backoff(seconds):
        sleeps.append(seconds)
        if prior == "completed_during_backoff":
            complete(calls[0])

    def send(socket_path, request, timeout_seconds):  # noqa: ARG001
        nonlocal live_writer
        path = Path(request["params"]["arguments"]["target_path"])
        calls.append(path)
        if len(calls) == 1:
            if prior == "unresolved":
                live_writer = path.open("wb")
                live_writer.write(b"writer still active")
                live_writer.flush()
            elif prior != "completed_during_backoff":
                complete(path)
            raise OSError("A response lost after admission")
        assert len(calls) == 2, "identity failure must not retry"
        if prior == "peer_replacement":
            # Keep the original inode allocated: this is a different file at A's old name.
            calls[0].rename(tmp_path / "displaced-owned-attempt")
            calls[0].write_bytes(b"replacement owned by peer")
        complete(path)
        target.write_bytes(b"peer dated raw")
        payload = {"target_path": str(path)}
        if echo != "legacy":
            payload["db_path"] = str(configured)
        if echo == "foreign_inode":
            payload.update(_identity(foreign))
        return _ok(payload)

    monkeypatch.setattr(backup, "_send_brainbar_json_request", send)
    monkeypatch.setattr(backup, "_sleep", backoff)
    try:
        with pytest.raises(backup.BrainBarDatabaseIdentityError):
            backup.request_brainbar_vacuum_into(
                target, socket_path="/isolated-backup.sock", expected_db_path=configured, retry_backoff_seconds=1
            )

        assert len(calls) == 2
        assert sleeps == [1]
        assert target.read_bytes() == b"peer dated raw"
        assert peer.read_bytes() == b"peer attempt"
        assert peer_marker.read_text() == "complete\n"
        assert not calls[1].exists()
        assert not backup._backup_attempt_completion_marker(calls[1]).exists()
        if prior == "unresolved":
            live_writer.write(b"; continued writing")
            live_writer.flush()
            assert calls[0].read_bytes() == b"writer still active; continued writing"
        elif prior == "peer_replacement":
            assert calls[0].read_bytes() == b"replacement owned by peer"
            assert backup._backup_attempt_completion_marker(calls[0]).exists()
        else:
            assert not calls[0].exists(), "identity failure stranded A's owned completed raw"
            assert not backup._backup_attempt_completion_marker(calls[0]).exists()
    finally:
        if live_writer is not None:
            live_writer.close()


@pytest.mark.parametrize("receipt", ["legacy", "bound"])
@pytest.mark.parametrize("echo", ["legacy", "missing_inode", "foreign_inode"])
def test_backoff_replacement_before_capture_never_becomes_owned(tmp_path, dbs, monkeypatch, receipt, echo):
    configured, foreign = dbs
    target = tmp_path / "out" / "snapshot.db"
    attempts = []
    sleeps = []

    def send(_socket, request, timeout_seconds):  # noqa: ARG001
        path = Path(request["params"]["arguments"]["target_path"])
        attempts.append(path)
        path.write_bytes(foreign.read_bytes())
        info = path.stat()
        marker = backup._backup_attempt_completion_marker(path)
        marker.write_text(
            "complete\n"
            if receipt == "legacy"
            else json.dumps(
                {"schema_version": 1, "target_path": str(path), "target_dev": info.st_dev, "target_ino": info.st_ino}
            )
        )
        if len(attempts) == 1:
            raise OSError("A response lost")
        assert len(attempts) == 2, "B identity refusal must remain terminal"
        payload = {} if echo == "legacy" else {"db_path": str(configured)}
        if echo == "foreign_inode":
            payload.update(_identity(foreign))
        return _ok(payload)

    def backoff(seconds):
        sleeps.append(seconds)
        attempts[0].rename(tmp_path / "original-A-kept-allocated")
        attempts[0].write_bytes(b"peer replacement BEFORE capture")

    monkeypatch.setattr(backup, "_send_brainbar_json_request", send)
    monkeypatch.setattr(backup, "_sleep", backoff)

    with pytest.raises(backup.BrainBarDatabaseIdentityError):
        backup.request_brainbar_vacuum_into(
            target, socket_path="/isolated.sock", expected_db_path=configured, retry_backoff_seconds=1
        )

    assert len(attempts) == 2
    assert sleeps == [1]
    assert attempts[0].exists(), "pre-capture peer raw was adopted and deleted"
    assert attempts[0].read_bytes() == b"peer replacement BEFORE capture"
    assert backup._backup_attempt_completion_marker(attempts[0]).exists()
    assert (tmp_path / "original-A-kept-allocated").read_bytes() == foreign.read_bytes()
    assert not attempts[1].exists()
    assert not backup._backup_attempt_completion_marker(attempts[1]).exists()


@pytest.mark.parametrize("invalid", ["legacy", "version", "path", "bool_inode", "inode", "truncated", "symlink"])
def test_unbound_or_invalid_prior_receipt_is_retained(tmp_path, dbs, monkeypatch, capsys, invalid):
    configured, _ = dbs
    requests = []
    vacuum = _fake_brainbar(configured, requests, echo=False)
    paths = []

    def send(socket_path, request, timeout_seconds):
        response = vacuum(socket_path, request, timeout_seconds)
        path = Path(request["params"]["arguments"]["target_path"])
        paths.append(path)
        if len(paths) == 1:
            info = path.stat()
            payload = {
                "schema_version": 1,
                "target_path": str(path),
                "target_dev": info.st_dev,
                "target_ino": info.st_ino,
            }
            changes = {
                "version": {"schema_version": 2},
                "path": {"target_path": "/another-attempt"},
                "bool_inode": {"target_ino": True},
                "inode": {"target_ino": info.st_ino + 1},
            }
            payload.update(changes.get(invalid, {}))
            marker = backup._backup_attempt_completion_marker(path)
            marker.write_text(
                "complete\n" if invalid == "legacy" else "{" if invalid == "truncated" else json.dumps(payload)
            )
            if invalid == "symlink":
                receipt = tmp_path / "peer-receipt"
                marker.rename(receipt)
                marker.symlink_to(receipt)
            raise OSError("A response lost")
        return response

    monkeypatch.setattr(backup, "_send_brainbar_json_request", send)
    monkeypatch.setattr(backup, "_sleep", lambda _s: None)
    with pytest.raises(backup.BrainBarDatabaseIdentityError):
        backup.request_brainbar_vacuum_into(
            tmp_path / "snapshot.db", socket_path="/isolated.sock", expected_db_path=configured
        )
    assert len(paths) == 2
    assert paths[0].read_bytes() == configured.read_bytes()
    assert backup._backup_attempt_completion_marker(paths[0]).exists()
    assert "retaining attempt without writer-bound ownership" in capsys.readouterr().out
    assert not paths[1].exists()


def test_peer_raw_that_appears_mid_run_survives_this_runs_early_exit(raw_lane, monkeypatch):
    """Sol + Cursor, #1414 round 1: a peer promotes after our absence check; we fail busy."""
    source, output = raw_lane
    peer = b"peer-snapshot"

    def busy_after_peer_promotes(target, **_kw):
        target.write_bytes(peer)
        raise backup.BrainBarSnapshotBusyError(backup.BRAINBAR_SNAPSHOT_BUSY_MESSAGE)

    monkeypatch.setattr(backup, "request_brainbar_vacuum_into", busy_after_peer_promotes)

    with pytest.raises(backup.BrainBarSnapshotBusyError):
        backup.create_sqlite_backup_artifact(source, output, date_stamp="2026-10-18")

    assert (output / "2026-10-18.db").read_bytes() == peer


def test_peer_raw_that_replaced_ours_survives_this_runs_early_exit(raw_lane, monkeypatch):
    source, output = raw_lane
    raw = output / "2026-10-18.db"
    peer = output / "peer.tmp"

    def peer_replaces_then_interrupt(*_a, **_kw):
        peer.write_bytes(b"peer-snapshot")
        os.replace(peer, raw)
        raise KeyboardInterrupt

    monkeypatch.setattr(backup, "_validate_backup_target", peer_replaces_then_interrupt)

    with pytest.raises(KeyboardInterrupt):
        backup.create_sqlite_backup_artifact(source, output, date_stamp="2026-10-18")

    assert raw.read_bytes() == b"peer-snapshot"


def test_promotion_never_replaces_a_peer_raw(tmp_path, dbs, monkeypatch):
    configured, _ = dbs
    out = tmp_path / "out"
    target = out / "2026-10-18.db"
    requests: list = []
    vacuum = _fake_brainbar(configured, requests)

    def send(socket_path, request, timeout_seconds):
        response = vacuum(socket_path, request, timeout_seconds)
        target.write_bytes(b"peer-snapshot")  # a peer promoted while our VACUUM ran
        return response

    monkeypatch.setattr(backup, "_send_brainbar_json_request", send)
    _no_sleep(monkeypatch)

    with pytest.raises(backup.BackupTargetConflictError):
        backup.request_brainbar_vacuum_into(target, socket_path="/x.sock", expected_db_path=configured)

    assert len(requests) == 1
    assert target.read_bytes() == b"peer-snapshot"
    assert sorted(p.name for p in out.iterdir()) == ["2026-10-18.db"]


def test_promotion_returns_the_promoted_inode(tmp_path, dbs, monkeypatch):
    configured, _ = dbs
    target = tmp_path / "out" / "2026-10-18.db"
    monkeypatch.setattr(backup, "_send_brainbar_json_request", _fake_brainbar(configured, []))

    promoted = backup.request_brainbar_vacuum_into(target, socket_path="/x.sock", expected_db_path=configured)

    assert (promoted.st_dev, promoted.st_ino) == (os.stat(target).st_dev, os.stat(target).st_ino)
