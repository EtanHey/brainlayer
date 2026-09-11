import fcntl
import json
import multiprocessing
import os
import shlex
import sqlite3
from pathlib import Path

import pytest

from brainlayer import backup_run_once


def _fixture_db(path: Path) -> None:
    with sqlite3.connect(path) as db:
        db.execute("CREATE TABLE chunks (id INTEGER PRIMARY KEY, content TEXT)")
        db.execute("INSERT INTO chunks(content) VALUES ('synthetic fixture')")


def _root(path: Path) -> list[Path]:
    root = path / "projects"
    root.mkdir()
    (root / "synthetic.jsonl").write_text('{"synthetic":true}\n')
    return [root]


def _hold_destination_directory_lock(destination: str, ready, release) -> None:
    fd = os.open(destination, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        ready.put(os.getpid())
        release.wait(10)
    finally:
        os.close(fd)


def test_run_once_creates_verified_db_and_jsonl_generation_without_remote_claim(tmp_path):
    db = tmp_path / "fixture.db"
    _fixture_db(db)
    destination = tmp_path / "backups"
    destination.mkdir()
    result = backup_run_once.run_backup_once(
        destination=destination,
        offline_db_path=db,
        transcript_roots=_root(tmp_path),
        expected_destination_identity=(destination.stat().st_dev, destination.stat().st_ino),
        generation_id="one",
        now=__import__("datetime").datetime(2026, 1, 1, tzinfo=__import__("datetime").timezone.utc),
    )
    generation = destination / result["generation"]
    assert (
        result["status"] == "local_verified"
        and result["local_verified"] is True
        and result["remote_verified"] is False
        and result["cleanup_permitted"] is False
    )
    assert (
        (generation / "brainlayer.db").is_file()
        and (generation / "receipt.complete.json").is_file()
        and not (generation / "receipt.incomplete.json").exists()
    )


def test_run_once_requires_existing_destination_and_explicit_source_mode(tmp_path):
    with pytest.raises(backup_run_once.BackupRunOnceError, match="already exist"):
        backup_run_once.run_backup_once(
            destination=tmp_path / "missing", offline_db_path=tmp_path / "missing.db", transcript_roots=[],
            expected_destination_identity=(0, 0),
        )
    destination = tmp_path / "backups"
    destination.mkdir()
    with pytest.raises(backup_run_once.BackupRunOnceError, match="exactly one"):
        backup_run_once.run_backup_once(
            destination=destination,
            socket_path=tmp_path / "socket",
            offline_db_path=tmp_path / "fixture.db",
            transcript_roots=_root(tmp_path),
            expected_destination_identity=(destination.stat().st_dev, destination.stat().st_ino),
        )


def test_run_once_rejects_corrupt_db_before_complete_receipt_with_real_quick_check(tmp_path):
    db = tmp_path / "corrupt.db"
    with sqlite3.connect(db) as connection:
        connection.execute("CREATE TABLE chunks (id INTEGER PRIMARY KEY, content TEXT)")
        connection.execute("CREATE TABLE junk (payload BLOB)")
        connection.execute("INSERT INTO chunks(content) VALUES ('synthetic fixture')")
        connection.execute("INSERT INTO junk(payload) VALUES (zeroblob(200000))")
    with db.open("r+b") as handle:
        handle.seek(-4096, 2)
        handle.write(b"\xff" * 4096)
    destination = tmp_path / "backups"
    destination.mkdir()
    with pytest.raises(backup_run_once.BackupRunOnceError, match="quick_check failed"):
        backup_run_once.run_backup_once(
            destination=destination,
            offline_db_path=db,
            transcript_roots=_root(tmp_path),
            expected_destination_identity=(destination.stat().st_dev, destination.stat().st_ino),
            generation_id="real-corrupt",
            now=__import__("datetime").datetime(2026, 1, 1, tzinfo=__import__("datetime").timezone.utc),
        )
    generation = next(destination.glob(".brainlayer-backup-*-real-corrupt"))
    receipt = json.loads((generation / "receipt.incomplete.json").read_text())
    assert receipt["status"] == "failed"
    assert not (generation / "receipt.complete.json").exists()


def test_run_once_failure_keeps_incomplete_receipt_and_partial_artifacts(tmp_path, monkeypatch):
    db = tmp_path / "fixture.db"
    _fixture_db(db)
    destination = tmp_path / "backups"
    destination.mkdir()
    monkeypatch.setattr(
        backup_run_once.backup_daily,
        "_validate_backup_target",
        lambda _: (_ for _ in ()).throw(RuntimeError("synthetic validation failure")),
    )
    with pytest.raises(RuntimeError, match="synthetic validation failure"):
        backup_run_once.run_backup_once(
            destination=destination,
            offline_db_path=db,
            transcript_roots=_root(tmp_path),
            expected_destination_identity=(destination.stat().st_dev, destination.stat().st_ino),
            generation_id="failed",
            now=__import__("datetime").datetime(2026, 1, 1, tzinfo=__import__("datetime").timezone.utc),
        )
    generation = next(destination.glob(".brainlayer-backup-*-failed"))
    receipt = json.loads((generation / "receipt.incomplete.json").read_text())
    assert (
        receipt["status"] == "failed"
        and not (generation / "receipt.complete.json").exists()
        and (generation / "brainlayer.db").is_file()
    )


def test_run_once_rejects_missing_root_and_outside_symlink(tmp_path):
    db = tmp_path / "fixture.db"
    _fixture_db(db)
    destination = tmp_path / "backups"
    destination.mkdir()
    missing = tmp_path / "missing"
    with pytest.raises(backup_run_once.BackupRunOnceError, match="transcript root"):
        backup_run_once.run_backup_once(
            destination=destination, offline_db_path=db, transcript_roots=[tmp_path / "projects", missing],
            expected_destination_identity=(destination.stat().st_dev, destination.stat().st_ino),
        )
    root = tmp_path / "projects"
    root.mkdir()
    outside = tmp_path / "outside.jsonl"
    outside.write_text("outside")
    (root / "escape.jsonl").symlink_to(outside)
    with pytest.raises(backup_run_once.BackupRunOnceError, match="symlink"):
        backup_run_once.run_backup_once(
            destination=destination, offline_db_path=db, transcript_roots=[root], generation_id="escape",
            expected_destination_identity=(destination.stat().st_dev, destination.stat().st_ino),
        )


def test_run_once_binds_destination_identity_and_records_digests(tmp_path):
    db = tmp_path / "fixture.db"
    _fixture_db(db)
    destination = tmp_path / "backups"
    destination.mkdir()
    roots = _root(tmp_path)
    identity = (destination.stat().st_dev, destination.stat().st_ino)
    result = backup_run_once.run_backup_once(
        destination=destination,
        offline_db_path=db,
        transcript_roots=roots,
        generation_id="identity",
        offline_source_open_generation="daemon-open-test",
        expected_offline_source_identity=(db.stat().st_dev, db.stat().st_ino),
        expected_destination_identity=identity,
    )
    assert result["destination_identity"] == {"st_dev": identity[0], "st_ino": identity[1]}
    assert result["destination_path"] == destination.resolve().as_posix()
    assert result["db_source_open_generation"] == "daemon-open-test"
    assert result["scheduled_offline_source_binding"]["source_open_generation"] == "daemon-open-test"
    assert len(result["db_snapshot_sha256"]) == 64 and len(result["jsonl_bundle_sha256"]) == 64
    with pytest.raises(backup_run_once.BackupRunOnceError, match="identity changed"):
        backup_run_once.run_backup_once(
            destination=destination,
            offline_db_path=db,
            transcript_roots=roots,
            generation_id="mismatch",
            expected_destination_identity=(identity[0], identity[1] + 1),
        )


def test_run_once_preserves_primary_error_when_failure_receipt_fails(tmp_path, monkeypatch):
    db = tmp_path / "fixture.db"
    _fixture_db(db)
    destination = tmp_path / "backups"
    destination.mkdir()
    original = backup_run_once._atomic_json

    def fail_failed_receipt(path, payload):
        if payload.get("status") == "failed":
            raise OSError("receipt finalizer failed")
        return original(path, payload)

    monkeypatch.setattr(
        backup_run_once.backup_daily,
        "_validate_backup_target",
        lambda _: (_ for _ in ()).throw(RuntimeError("primary failure")),
    )
    monkeypatch.setattr(backup_run_once, "_atomic_json", fail_failed_receipt)
    with pytest.raises(RuntimeError, match="primary failure"):
        backup_run_once.run_backup_once(
            destination=destination, offline_db_path=db, transcript_roots=_root(tmp_path), generation_id="primary",
            expected_destination_identity=(destination.stat().st_dev, destination.stat().st_ino),
        )


def test_run_once_rejects_source_replacement_during_bundle(tmp_path, monkeypatch):
    db = tmp_path / "fixture.db"
    _fixture_db(db)
    destination = tmp_path / "backups"
    destination.mkdir()
    root = tmp_path / "projects"
    root.mkdir()
    source = root / "synthetic.jsonl"
    source.write_text('{"safe":true}\n')
    outside = tmp_path / "outside.jsonl"
    outside.write_text('{"outside":true}\n')
    original = backup_run_once.jsonl_backup.create_jsonl_bundle_with_digests

    def swap_then_bundle(candidates, staging_dir, **kwargs):
        source.unlink()
        source.symlink_to(outside)
        return original(candidates, staging_dir, **kwargs)

    monkeypatch.setattr(backup_run_once.jsonl_backup, "create_jsonl_bundle_with_digests", swap_then_bundle)
    with pytest.raises(backup_run_once.BackupRunOnceError, match="symlink|no-follow|transcript candidate"):
        backup_run_once.run_backup_once(
            destination=destination,
            offline_db_path=db,
            transcript_roots=[root],
            expected_destination_identity=(destination.stat().st_dev, destination.stat().st_ino),
            generation_id="source-swap",
        )
    generation = next(destination.glob(".brainlayer-backup-*-source-swap"))
    receipt = json.loads((generation / "receipt.incomplete.json").read_text())
    assert receipt["status"] == "failed"


def test_cli_reads_literal_destination_and_requires_explicit_roots(tmp_path, monkeypatch, capsys):
    db = tmp_path / "fixture.db"
    _fixture_db(db)
    destination = tmp_path / "backup $literal`path`"
    destination.mkdir()
    root = tmp_path / "projects"
    root.mkdir()
    (root / "synthetic.jsonl").write_text('{"synthetic":true}\n')
    identity = destination.stat()
    config = tmp_path / "brainlayer.env"
    config.write_text(
        "\n".join(
            [
                "BRAINLAYER_BACKUP_ENABLED=1",
                f"BRAINLAYER_BACKUP_DESTINATION={shlex.quote(str(destination))}",
                f"BRAINLAYER_BACKUP_DESTINATION_ST_DEV={identity.st_dev}",
                f"BRAINLAYER_BACKUP_DESTINATION_ST_INO={identity.st_ino}",
            ]
        )
        + "\n"
    )
    monkeypatch.setattr(
        backup_run_once,
        "_socket_snapshot",
        lambda _socket, target, _generation, **_kwargs: {
            "chunks": backup_run_once._offline_snapshot(db, target),
            "source_db_filename": str(db),
            "source_open_generation": "test-generation",
            "source_observed_path_identity_before": {"kind": "pathname_observation_not_fd_proof"},
            "source_observed_path_identity_after": {"kind": "pathname_observation_not_fd_proof"},
        },
    )
    assert backup_run_once.main(["--config", str(config), "--transcript-root", str(root), "--generation-id", "cli"]) == 0
    output = json.loads(capsys.readouterr().out)
    assert output["status"] == "local_verified"
    assert output["db_source_mode"] == "brainbar_socket"
    assert output["transcript_roots"] == [str(root.resolve())]
    assert "backup $literal`path`" in output["receipt"]


def test_cli_rejects_implicit_optional_provider_roots(tmp_path, capsys):
    config = tmp_path / "brainlayer.env"
    config.write_text("BRAINLAYER_BACKUP_ENABLED=0\n")
    with pytest.raises(SystemExit) as exit_info:
        backup_run_once.main(["--config", str(config)])
    assert exit_info.value.code == 2
    assert "required" in capsys.readouterr().err


def test_literal_config_rejects_multiple_tokens(tmp_path):
    config = tmp_path / "brainlayer.env"
    config.write_text("BRAINLAYER_BACKUP_DESTINATION='/correct' trailing\n")
    from brainlayer.config import configured_brainlayer_literal_value

    with pytest.raises(RuntimeError, match="exactly one literal value"):
        configured_brainlayer_literal_value("BRAINLAYER_BACKUP_DESTINATION", config)


def test_destination_swap_after_bound_generation_keeps_failed_receipt(tmp_path, monkeypatch):
    destination = tmp_path / "backups"
    destination.mkdir()
    replacement = tmp_path / "replacement"
    replacement.mkdir()
    original = backup_run_once._run_backup_once_bound

    def swap_after_generation(**kwargs):
        generation = ".brainlayer-backup-race"
        Path(generation).mkdir()
        (Path(generation) / "brainlayer.db").write_bytes(b"db")
        (Path(generation) / "claude-jsonl-race.tar.gz").write_bytes(b"archive")
        (Path(generation) / "receipt.complete.json").write_text("{}")
        selected = destination.with_name("selected-original")
        os.rename(destination, selected)
        os.rename(replacement, destination)
        return {"generation": generation}

    monkeypatch.setattr(backup_run_once, "_run_backup_once_bound", swap_after_generation)
    identity = (destination.stat().st_dev, destination.stat().st_ino)
    with pytest.raises(backup_run_once.BackupRunOnceError, match="after generation"):
        backup_run_once.run_backup_once(
            destination=destination,
            offline_db_path=tmp_path / "offline.db",
            transcript_roots=[],
            expected_destination_identity=identity,
            generation_id="race",
        )
    assert not list(destination.rglob("*"))
    selected_generation = next(tmp_path.joinpath("selected-original").glob(".brainlayer-backup-race"))
    assert not (selected_generation / "receipt.complete.json").exists()
    assert (selected_generation / "brainlayer.db").read_bytes() == b"db"
    assert (selected_generation / "claude-jsonl-race.tar.gz").read_bytes() == b"archive"
    receipt = json.loads((selected_generation / "receipt.incomplete.json").read_text())
    assert receipt["status"] == "failed"
    assert "identity changed after generation" in receipt["error"]


def test_socket_snapshot_serializes_absolute_paths_and_promotes_real_helper(tmp_path, monkeypatch):
    captured = {}
    served_db = tmp_path / "served-by-daemon.db"

    def fake_send(socket_path, request, timeout_seconds):
        captured.update(socket=socket_path, request=request, timeout=timeout_seconds)
        target = Path(request["params"]["arguments"]["target_path"])
        _fixture_db(target)
        return {
            "result": {
                "content": [{
                    "text": json.dumps({
                        "status": "ok",
                        "source_db_filename": str(served_db),
                        "source_open_generation": "daemon-open-1",
                        "source_observed_path_identity_before": {"kind": "pathname_observation_not_fd_proof"},
                        "source_observed_path_identity_after": {"kind": "pathname_observation_not_fd_proof"},
                    })
                }]
            }
        }

    monkeypatch.setattr(backup_run_once.backup_daily, "_send_brainbar_json_request", fake_send)
    monkeypatch.chdir(tmp_path)
    local_target = Path("generation/db")
    generation = Path("generation")
    generation.mkdir()
    result = backup_run_once._socket_snapshot(Path("/tmp/unused.sock"), local_target, generation)
    assert result["chunks"] == 1
    assert result["source_db_filename"] == str(served_db)
    sent = Path(captured["request"]["params"]["arguments"]["target_path"])
    assert sent.is_absolute()
    assert sent.parent == (tmp_path / "generation").resolve()
    assert (tmp_path / "generation" / "db").exists()


def test_socket_backup_records_served_connection_and_refuses_optional_mismatch(tmp_path, monkeypatch):
    destination = tmp_path / "backups"
    destination.mkdir()
    served_db = tmp_path / "served-by-daemon.db"
    _fixture_db(served_db)
    caller_guess = tmp_path / "caller-guess.db"

    def fake_send(_socket_path, request, timeout_seconds):
        target = Path(request["params"]["arguments"]["target_path"])
        _fixture_db(target)
        return {
            "result": {
                "content": [{
                    "text": json.dumps({
                        "status": "ok",
                        "source_db_filename": str(served_db),
                        "source_open_generation": "served-open-generation",
                        "source_observed_path_identity_before": {
                            "kind": "pathname_observation_not_fd_proof", "path": str(served_db),
                        },
                        "source_observed_path_identity_after": {
                            "kind": "pathname_observation_not_fd_proof", "path": str(served_db),
                        },
                    })
                }]
            }
        }

    monkeypatch.setattr(backup_run_once.backup_daily, "_send_brainbar_json_request", fake_send)
    identity = (destination.stat().st_dev, destination.stat().st_ino)
    roots = _root(tmp_path)
    result = backup_run_once.run_backup_once(
        destination=destination,
        socket_path=tmp_path / "brainbar.sock",
        transcript_roots=roots,
        expected_destination_identity=identity,
        generation_id="served",
    )
    assert result["db_source_path"] == str(served_db)
    assert result["db_transport_socket"] == str((tmp_path / "brainbar.sock").resolve())
    assert result["db_source_open_generation"] == "served-open-generation"

    with pytest.raises(backup_run_once.BackupRunOnceError, match="source mismatch"):
        backup_run_once.run_backup_once(
            destination=destination,
            socket_path=tmp_path / "brainbar.sock",
            transcript_roots=roots,
            expected_destination_identity=identity,
            expected_source_path=caller_guess,
            generation_id="mismatch",
        )
    generation = next(destination.glob(".brainlayer-backup-*-mismatch"))
    assert json.loads((generation / "receipt.incomplete.json").read_text())["status"] == "failed"


@pytest.mark.parametrize(
    ("response", "error_match"),
    [
        ({"error": {"message": "top-level terminal error"}}, "top-level terminal error"),
        ({"result": {"content": [{"text": "not JSON"}]}}, "source authority"),
    ],
)
def test_socket_snapshot_reclaims_attempt_for_terminal_error_or_malformed_authority(
    tmp_path, monkeypatch, response, error_match
):
    seen = {}

    def fake_send(_socket_path, request, timeout_seconds):
        attempt = Path(request["params"]["arguments"]["target_path"])
        attempt.write_bytes(b"synthetic partial")
        attempt.with_name(attempt.name + ".complete").write_text("complete\n")
        seen["attempt"] = attempt
        return response

    monkeypatch.setattr(backup_run_once.backup_daily, "_send_brainbar_json_request", fake_send)
    generation = tmp_path / "generation"
    generation.mkdir()
    with pytest.raises(backup_run_once.BackupRunOnceError, match=error_match):
        backup_run_once._socket_snapshot(Path("/tmp/unused.sock"), generation / "db", generation)

    assert not seen["attempt"].exists()
    assert not seen["attempt"].with_name(seen["attempt"].name + ".complete").exists()


def test_cli_explicit_offline_path_uses_only_offline_helper_and_socket_remains_default(tmp_path, monkeypatch, capsys):
    db = tmp_path / "fixture.db"
    _fixture_db(db)
    destination = tmp_path / "backups"
    destination.mkdir()
    root = _root(tmp_path)
    identity = destination.stat()
    config = tmp_path / "brainlayer.env"
    config.write_text(
        "\n".join(
            [
                "BRAINLAYER_BACKUP_ENABLED=1",
                f"BRAINLAYER_BACKUP_DESTINATION={shlex.quote(str(destination))}",
                f"BRAINLAYER_BACKUP_DESTINATION_ST_DEV={identity.st_dev}",
                f"BRAINLAYER_BACKUP_DESTINATION_ST_INO={identity.st_ino}",
            ]
        )
        + "\n"
    )
    seen = []

    def offline_only(source, target):
        seen.append(source)
        _fixture_db(target)
        return 1

    monkeypatch.setattr(backup_run_once, "_offline_snapshot", offline_only)
    monkeypatch.setattr(backup_run_once, "_socket_snapshot", lambda *_args, **_kwargs: pytest.fail("socket fallback"))
    assert backup_run_once.main([
        "--config", str(config), "--transcript-root", str(root[0]), "--offline-db-path", str(db), "--generation-id", "offline-cli",
    ]) == 0
    output = json.loads(capsys.readouterr().out)
    assert seen == [db]
    assert output["db_source_mode"] == "explicit_offline_db"
    assert output["db_source_path"] == str(db.resolve())
    assert output["db_transport_socket"] is None


def test_cli_rejects_relative_or_ambiguous_offline_authority(tmp_path, capsys):
    config = tmp_path / "brainlayer.env"
    config.write_text("BRAINLAYER_BACKUP_ENABLED=0\n")
    with pytest.raises(SystemExit) as exit_info:
        backup_run_once.main([
            "--config", str(config), "--transcript-root", str(tmp_path), "--socket-path", "/tmp/socket", "--offline-db-path", "relative.db",
        ])
    assert exit_info.value.code == 2
    assert "not allowed with argument" in capsys.readouterr().err

    destination = tmp_path / "backups"
    destination.mkdir()
    identity = destination.stat()
    config.write_text(
        "\n".join(
            [
                "BRAINLAYER_BACKUP_ENABLED=1",
                f"BRAINLAYER_BACKUP_DESTINATION={shlex.quote(str(destination))}",
                f"BRAINLAYER_BACKUP_DESTINATION_ST_DEV={identity.st_dev}",
                f"BRAINLAYER_BACKUP_DESTINATION_ST_INO={identity.st_ino}",
            ]
        )
        + "\n"
    )
    root = _root(tmp_path)
    assert backup_run_once.main([
        "--config", str(config), "--transcript-root", str(root[0]), "--offline-db-path", "relative.db",
    ]) == 1
    failure = json.loads(capsys.readouterr().out)
    assert "absolute explicit source path" in failure["error"]
    assert not list(destination.glob(".brainlayer-backup-*"))


@pytest.mark.parametrize("socket_error", ["returned authority error", "lost socket response"])
def test_socket_errors_never_automatically_fall_back_to_offline(tmp_path, monkeypatch, socket_error):
    destination = tmp_path / "backups"
    destination.mkdir()
    offline_called = False

    def reject_socket(*_args, **_kwargs):
        raise backup_run_once.BackupRunOnceError(socket_error)

    def reject_offline(*_args, **_kwargs):
        nonlocal offline_called
        offline_called = True
        pytest.fail("offline helper must not be reached")

    monkeypatch.setattr(backup_run_once, "_socket_snapshot", reject_socket)
    monkeypatch.setattr(backup_run_once, "_offline_snapshot", reject_offline)
    with pytest.raises(backup_run_once.BackupRunOnceError, match=socket_error):
        backup_run_once.run_backup_once(
            destination=destination,
            socket_path=tmp_path / "brainbar.sock",
            transcript_roots=_root(tmp_path),
            expected_destination_identity=(destination.stat().st_dev, destination.stat().st_ino),
            generation_id="no-fallback",
        )
    assert offline_called is False
    generation = next(destination.glob(".brainlayer-backup-*-no-fallback"))
    assert json.loads((generation / "receipt.incomplete.json").read_text())["status"] == "failed"


def test_cli_refuses_subprocess_destination_lock_without_generation_then_releases(tmp_path, capsys):
    db = tmp_path / "fixture.db"
    _fixture_db(db)
    destination = tmp_path / "backups"
    destination.mkdir()
    root = _root(tmp_path)
    identity = destination.stat()
    config = tmp_path / "brainlayer.env"
    config.write_text(
        "\n".join(
            [
                "BRAINLAYER_BACKUP_ENABLED=1",
                f"BRAINLAYER_BACKUP_DESTINATION={shlex.quote(str(destination))}",
                f"BRAINLAYER_BACKUP_DESTINATION_ST_DEV={identity.st_dev}",
                f"BRAINLAYER_BACKUP_DESTINATION_ST_INO={identity.st_ino}",
            ]
        )
        + "\n"
    )
    context = multiprocessing.get_context("spawn")
    ready = context.Queue()
    release = context.Event()
    holder = context.Process(target=_hold_destination_directory_lock, args=(str(destination), ready, release))
    holder.start()
    assert ready.get(timeout=10)
    try:
        assert backup_run_once.main([
            "--config", str(config), "--transcript-root", str(root[0]), "--offline-db-path", str(db), "--generation-id", "busy",
        ]) == 2
        refusal = json.loads(capsys.readouterr().out)
        assert refusal["status"] == "refused"
        assert refusal["reason"] == "destination_busy"
        assert refusal["generation"] is None
        assert not list(destination.glob(".brainlayer-backup-*-busy"))
    finally:
        release.set()
        holder.join(timeout=10)
    assert holder.exitcode == 0
    result = backup_run_once.run_backup_once(
        destination=destination / ".",
        offline_db_path=db,
        transcript_roots=root,
        expected_destination_identity=(identity.st_dev, identity.st_ino),
        generation_id="released",
    )
    assert result["status"] == "local_verified"


def test_destination_process_lock_releases_after_failed_generation(tmp_path, monkeypatch):
    db = tmp_path / "fixture.db"
    _fixture_db(db)
    destination = tmp_path / "backups"
    destination.mkdir()
    roots = _root(tmp_path)
    identity = (destination.stat().st_dev, destination.stat().st_ino)
    original = backup_run_once._offline_snapshot
    monkeypatch.setattr(
        backup_run_once,
        "_offline_snapshot",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(backup_run_once.BackupRunOnceError("synthetic failure")),
    )
    with pytest.raises(backup_run_once.BackupRunOnceError, match="synthetic failure"):
        backup_run_once.run_backup_once(
            destination=destination, offline_db_path=db, transcript_roots=roots,
            expected_destination_identity=identity, generation_id="failed-lock",
        )
    monkeypatch.setattr(backup_run_once, "_offline_snapshot", original)
    result = backup_run_once.run_backup_once(
        destination=destination, offline_db_path=db, transcript_roots=roots,
        expected_destination_identity=identity, generation_id="after-failure",
    )
    assert result["status"] == "local_verified"
