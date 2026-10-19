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
        return _ok({"target_path": str(target), **({"db_path": str(served_db)} if echo else {})})

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
    monkeypatch.setattr(
        backup, "request_brainbar_vacuum_into", lambda target, **kw: target.write_bytes(source.read_bytes())
    )
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
