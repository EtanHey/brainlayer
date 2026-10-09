from pathlib import Path
from types import SimpleNamespace

from typer.testing import CliRunner

from brainlayer import backup_daily as daily
from brainlayer import backup_recovery as recovery
from brainlayer.cli import app

ARGS = ["--snapshot-date", "2026-10-09", "--source-receipt-line", "1", "--expected-raw-bytes", "42", "--no-retention"]


def test_cli_routes_through_installed_wrapper_and_pinned_python(tmp_path, monkeypatch):
    from brainlayer import hook_python

    runner = tmp_path / ".local/lib/brainlayer/brainlayer-env-run.sh"
    runner.parent.mkdir(parents=True)
    runner.write_text("synthetic wrapper")
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setattr(hook_python, "resolve_hook_python", lambda: "/synthetic/venv/bin/python")
    monkeypatch.setenv("PYTHONPATH", "/wrong/source")
    calls = []
    monkeypatch.setattr(
        "subprocess.run", lambda command, **kwargs: calls.append((command, kwargs)) or SimpleNamespace(returncode=7)
    )
    result = CliRunner().invoke(app, ["backup", "recover-existing", *ARGS])
    assert result.exit_code == 7
    command, options = calls[0]
    assert command[:5] == [str(runner), "/synthetic/venv/bin/python", "-I", "-m", "brainlayer.backup_recovery"]
    assert "--no-retention" in command
    assert "PYTHONPATH" not in options["env"]


def test_cli_missing_no_retention_refuses_before_subprocess(monkeypatch):
    monkeypatch.setattr(
        "subprocess.run", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("must not start"))
    )
    result = CliRunner().invoke(app, ["backup", "recover-existing", *ARGS[:-1]])
    assert result.exit_code != 0
    assert "requires --no-retention" in result.output


def test_module_help_is_safe_and_supervisor_routes_to_recovery(monkeypatch):
    import pytest

    calls = []
    monkeypatch.delenv(daily.BACKUP_SUPERVISED_CHILD_ENV, raising=False)
    monkeypatch.setattr(
        daily, "_supervise_backup_process", lambda timeout, **kwargs: calls.append(kwargs["command"]) or 0
    )
    with pytest.raises(SystemExit) as result:
        recovery.main(["--help"])
    assert result.value.code == 0 and calls == []
    assert recovery.main(ARGS) == 0
    assert calls[0][1:4] == ["-I", "-m", "brainlayer.backup_recovery"]


def test_child_routes_env_configuration_and_restores_deadline(monkeypatch, tmp_path):
    monkeypatch.setenv(daily.BACKUP_SUPERVISED_CHILD_ENV, "1")
    monkeypatch.setenv("BRAINLAYER_BACKUP_STAGING_DIR", str(tmp_path))
    monkeypatch.setenv("BRAINLAYER_BACKUP_DRIVE_FOLDER", "original/folder")
    calls = []
    timers = []
    monkeypatch.setattr(
        recovery, "recover_existing_artifact", lambda **kwargs: calls.append(kwargs) or {"verified": True}
    )
    monkeypatch.setattr(recovery.signal, "setitimer", lambda *args: timers.append(args))
    assert recovery.main(ARGS) == 0
    assert calls[0]["folder_parts"] == ["original", "folder"]
    assert calls[0]["staging_dir"] == tmp_path
    assert timers[-1][1] == 0


def test_uploader_keeps_verified_descriptor_when_path_is_replaced(tmp_path, monkeypatch):
    import json
    import os

    path = tmp_path / "2026-10-09.db.gz"
    path.write_bytes(b"verified bytes")
    descriptor, _ = daily._open_stable_regular_file(path)
    metadata, sent = [], []

    def post(*args, **kwargs):
        metadata.append(json.loads(kwargs["data"]))
        path.rename(tmp_path / "preserved-original.gz")
        path.write_bytes(b"unverified replacement")
        return SimpleNamespace(raise_for_status=lambda: None, headers={"Location": "https://synthetic.invalid/session"})

    class Session:
        def put(self, *args, **kwargs):
            sent.append(kwargs["data"])
            return SimpleNamespace(status_code=200, json=lambda: {"id": "file"})

        def close(self):
            pass

    monkeypatch.setattr(daily.requests, "post", post)
    monkeypatch.setattr(daily.requests, "Session", Session)
    try:
        daily.upload_file_to_drive_raw(
            path,
            "folder",
            SimpleNamespace(token="synthetic-token"),
            machine_id="test-machine",
            source_descriptor=descriptor,
            drive_name="2026-10-09-recovery-attempt.db.gz",
            attempt_id="attempt",
        )
        assert sent == [b"verified bytes"]
        assert metadata[0]["name"] == "2026-10-09-recovery-attempt.db.gz"
        assert metadata[0]["appProperties"]["brainlayer_recovery_attempt"] == "attempt"
    finally:
        os.close(descriptor)
