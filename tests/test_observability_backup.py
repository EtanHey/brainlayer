from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import jsonschema
import pytest

from brainlayer import backup_daily, observability_backup, observability_surface
from brainlayer.observability_backup import build_backups_section

FIXTURES = Path(__file__).parent / "fixtures/observability"
NOW = datetime(2026, 9, 13, 12, tzinfo=UTC)


def _env(case: str) -> dict[str, str]:
    return {
        "BRAINLAYER_OBSERVABILITY_JSONL_BACKUP_LOG": str(FIXTURES / f"logs/{case}/jsonl-backup.log"),
        "BRAINLAYER_OBSERVABILITY_BACKUP_DAILY_LOG": str(FIXTURES / f"logs/{case}/backup-daily.log"),
        "BRAINLAYER_OBSERVABILITY_LAUNCHD_OUTPUT": str(FIXTURES / f"launchd/{case}.txt"),
        "BRAINLAYER_OBSERVABILITY_DISABLED_DIR": str(FIXTURES / f"launchd/{case}.disabled"),
    }


def _recorder(*, future: bool = False):
    opened: list[str] = []

    def record_input(
        path: Path,
        *,
        status: str = "read",
        rows_or_bytes: int | None = None,
        skipped_lines: int = 0,
        in_section_inputs: bool = True,
    ) -> dict[str, object]:
        relative = path.relative_to(FIXTURES).as_posix()
        opened.append(relative)
        size = len(list(path.iterdir())) if path.is_dir() else path.stat().st_size if path.exists() else None
        return {
            "path": relative,
            "status": "future" if future and path.name == "jsonl-backup.log" else status,
            "mtime": None,
            "rows_or_bytes": size if rows_or_bytes is None else rows_or_bytes,
            "sha256_first_64kb": None,
            "skipped_lines": skipped_lines,
        }

    return opened, record_input


def _build(case: str, *, future: bool = False) -> tuple[dict[str, object], list[str]]:
    opened, recorder = _recorder(future=future)
    return build_backups_section(env=_env(case), record_input=recorder, now=NOW), opened


@pytest.mark.parametrize("case", ["healthy-dev", "empty-db-dev", "missing-source-class-dev"])
def test_healthy_fixture_matches_frozen_backup_contract(case: str) -> None:
    result, opened = _build(case)

    assert {key: value for key, value in result.items() if key != "inputs"} == {
        "state": "measured",
        "reason": "",
        "last_verified_upload": {
            "at": "2026-09-13T10:00:00Z",
            "age_hours": 2.0,
            "archive_id": "synthetic-archive-815",
            "verified": True,
        },
        "freshness": "fresh",
        "threshold_hours": 36,
        "retention_invariant": "PASS",
        "surviving_archives_30d": 1,
        "launchd": {"label": "com.brainlayer.jsonl-backup", "bootstrapped": True, "disabled_dir_present": True},
        "db_snapshot": {"last_at": "2026-09-13T00:00:00Z", "destination": "synthetic-drive", "verified": True},
        "error_type": None,
    }
    assert opened == [
        f"logs/{case}/jsonl-backup.log",
        f"logs/{case}/backup-daily.log",
        f"launchd/{case}.txt",
        f"launchd/{case}.disabled",
    ]
    assert result["inputs"][1]["skipped_lines"] == 2


def test_active_job_alert_enters_backup_health_surface_from_supplied_env(tmp_path):
    alerts = tmp_path / "alerts.json"
    alerts.write_text(json.dumps({"drive-consent": "Drive access expired: Reconnect in BrainBar"}))
    env = {**_env("healthy-dev"), "BRAINLAYER_JOB_ALERT_PATH": str(alerts), "BRAINLAYER_DB": str(tmp_path / "db")}
    seen = []

    def record(path, **kwargs):
        seen.append(path)
        return {"path": str(path), "status": kwargs.get("status", "read")}

    result = build_backups_section(env=env, record_input=record, now=NOW)
    assert result["error_type"] == "job_alert:Drive access expired: Reconnect in BrainBar"
    assert alerts in seen

    def future_record(path, **kwargs):
        row = record(path, **kwargs)
        if path == alerts:
            row["status"] = "future"
        return row

    assert build_backups_section(env=env, record_input=future_record, now=NOW)["state"] == "unmeasurable"


@pytest.mark.parametrize(
    ("case", "freshness", "error_type"),
    [
        ("no-op-dev", "fresh", None),
        ("legacy-no-op-dev", "unknown", "jsonl_backup_attempt_invalid"),
    ],
)
def test_no_op_attempt_time_controls_freshness(
    case: str,
    freshness: str,
    error_type: str | None,
) -> None:
    result, _ = _build(case)

    assert result["last_verified_upload"] is None
    assert result["freshness"] == freshness
    assert result["error_type"] == error_type
    assert result["retention_invariant"] == "PASS"
    assert result["surviving_archives_30d"] == 0
    assert result["launchd"]["bootstrapped"] is False


def test_all_error_daily_log_is_measured_stale() -> None:
    result, _ = _build("backup-errors-dev")

    assert result["state"] == "measured"
    assert result["freshness"] == "stale"
    assert result["error_type"] == "FileNotFoundError"
    assert result["db_snapshot"] is None


@pytest.mark.parametrize("history", ["none", "older-than-30-days"])
def test_verified_idle_noop_is_fresh_without_a_recent_upload(tmp_path: Path, history: str) -> None:
    env, recorder = _command_env(tmp_path)
    launchd = tmp_path / "launchd.txt"
    launchd.write_text("service = com.brainlayer.jsonl-backup\n")
    env["BRAINLAYER_OBSERVABILITY_LAUNCHD_OUTPUT"] = str(launchd)
    env["BRAINLAYER_JOB_ALERT_PATH"] = str(tmp_path / "absent-alerts.json")
    records = []
    if history != "none":
        records.append(
            {
                "attempted_at": "2026-08-01T10:00:00Z",
                "status": "uploaded",
                "archive_id": "synthetic-old-archive",
                "verified": True,
                "uploaded": True,
            }
        )
    records.append(
        {
            "attempted_at": "2026-09-13T10:00:00Z",
            "status": "no-op",
            "verified": True,
            "uploaded": False,
            "message": "no new transcripts; already covered",
        }
    )
    Path(env["BRAINLAYER_OBSERVABILITY_JSONL_BACKUP_LOG"]).write_text(
        "".join(json.dumps(record) + "\n" for record in records)
    )

    result = build_backups_section(env=env, record_input=recorder, now=NOW)

    assert result["state"] == "measured"
    assert result["freshness"] == "fresh"
    assert result["error_type"] is None
    assert result["surviving_archives_30d"] == 0
    upload = result["last_verified_upload"]
    if history == "none":
        assert upload is None  # Never fabricate an upload to appease a UI check.
    else:
        assert upload["at"] == "2026-08-01T10:00:00Z"
        assert upload["age_hours"] > 30 * 24


@pytest.mark.parametrize(
    ("case", "reason", "status", "future"),
    [
        ("missing-log-dev", "required input missing: logs/missing-log-dev/jsonl-backup.log", "missing", False),
        (
            "malformed-log-dev-1",
            "malformed JSON line in logs/malformed-log-dev-1/jsonl-backup.log",
            "malformed",
            False,
        ),
        ("missing-launchd-dev", "launchd output is empty: launchd/missing-launchd-dev.txt", "empty", False),
        (
            "clock-skew-dev",
            "input mtime is later than generated_at: logs/clock-skew-dev/jsonl-backup.log",
            "future",
            True,
        ),
    ],
)
def test_invalid_required_input_fails_closed(case: str, reason: str, status: str, future: bool) -> None:
    result, opened = _build(case, future=future)

    assert result == {"state": "unmeasurable", "reason": reason, "inputs": result["inputs"]}
    assert len(opened) == 4
    assert status in {item["status"] for item in result["inputs"]}
    assert not any(isinstance(value, (int, float)) for key, value in result.items() if key != "inputs")


def test_drive_token_file_not_found_maps_to_credentials_error() -> None:
    _, error_type, all_errors = observability_backup._daily_snapshot(
        [
            {
                "backup_log_provenance": "real",
                "error": "Google Drive token file not found: /synthetic/.config/google-drive-mcp/tokens.json",
            }
        ]
    )

    assert error_type == "drive_credentials_missing"
    assert all_errors is True


@pytest.mark.parametrize(
    ("token_text", "expected_error"),
    [
        (
            json.dumps({"access_token": "restored-access", "refresh_token": "restored-refresh"}),
            "drive_credentials_restored_backup_pending",
        ),
        (None, "drive_credentials_missing"),
        ("not-json", "drive_credentials_missing"),
        ("[]", "drive_credentials_missing"),
        (json.dumps({"access_token": "restored-access"}), "drive_credentials_missing"),
    ],
)
def test_current_drive_credentials_replace_only_the_stale_missing_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    token_text: str | None,
    expected_error: str,
) -> None:
    token_path = tmp_path / "google-drive-mcp" / "tokens.json"
    if token_text is not None:
        token_path.parent.mkdir()
        token_path.write_text(token_text, encoding="utf-8")
    monkeypatch.setattr(backup_daily, "DEFAULT_TOKEN_PATH", token_path)
    monkeypatch.setattr(
        observability_backup,
        "_daily_snapshot",
        lambda records: (None, "drive_credentials_missing", True),
    )

    result, _ = _build("healthy-dev")

    assert result["error_type"] == expected_error
    assert result["freshness"] == "stale"
    assert result["last_verified_upload"]["at"] == "2026-09-13T10:00:00Z"


def test_legacy_daily_snapshot_derives_attempt_time_from_snapshot_date() -> None:
    snapshot, error_type, all_errors = observability_backup._daily_snapshot(
        [
            {
                "backup_log_provenance": "real",
                "snapshot": "/synthetic/backups/2026-09-13.db.gz",
                "drive_file": "synthetic-drive",
                "uploaded": True,
                "verified": True,
                "drive_md5_match": True,
            }
        ]
    )
    expected = "2026-09-13T00:00:00Z"
    assert snapshot == {
        "last_at": expected,
        "destination": "synthetic-drive",
        "verified": True,
    }
    assert error_type is None
    assert all_errors is False


def test_daily_snapshot_fits_current_drive_file_receipt_and_md5_verification() -> None:
    snapshot, error_type, all_errors = observability_backup._daily_snapshot(
        [
            {
                "backup_log_provenance": "real",
                "attempted_at": "2026-09-13T02:00:00Z",
                "snapshot_created_at": "2026-09-13T02:00:00Z",
                "db": "/synthetic/brainlayer.db",
                "drive_file": {"id": "drive-id", "name": "2026-09-13.db.gz", "size": "123"},
                "snapshot": "/synthetic/2026-09-13.db.gz",
                "uploaded": True,
                "verified": True,
                "drive_md5_match": True,
            }
        ]
    )
    assert snapshot == {"last_at": "2026-09-13T02:00:00Z", "destination": "2026-09-13.db.gz", "verified": True}
    assert error_type is None
    assert all_errors is False


def test_daily_snapshot_skips_failed_latest_attempt_for_last_success() -> None:
    snapshot, error_type, all_errors = observability_backup._daily_snapshot(
        [
            {
                "backup_log_provenance": "real",
                "attempted_at": "2026-09-09T02:00:00Z",
                "snapshot": "/synthetic/2026-09-09.db.gz",
                "uploaded": False,
                "verified": True,
            },
            {
                "backup_log_provenance": "real",
                "attempted_at": "2026-09-13T09:00:00Z",
                "snapshot_created_at": "2026-09-13T09:00:00Z",
                "drive_file": {"name": "2026-09-13.db.gz"},
                "snapshot": "/synthetic/2026-09-13.db.gz",
                "uploaded": True,
                "verified": True,
            },
            {
                "backup_log_provenance": "real",
                "attempted_at": "2026-09-14T02:00:00Z",
                "db": "/synthetic/brainlayer.db",
                "snapshot": "/synthetic/2026-09-14.db.gz",
                "uploaded": False,
                "verified": False,
                "error_type": "RuntimeError",
            },
        ]
    )
    assert snapshot == {"last_at": "2026-09-13T09:00:00Z", "destination": "2026-09-13.db.gz", "verified": True}
    assert error_type == "RuntimeError"
    assert all_errors is False


@pytest.mark.parametrize(("state", "code"), [("stale", "stale_receipt"), ("failed", "failed_receipt")])
def test_health_error_type_is_propagated(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, state: str, code: str
) -> None:
    monkeypatch.setattr(
        observability_backup,
        "inspect_jsonl_backup_health",
        lambda *args, **kwargs: (type("Health", (), {"state": state})(), type("Issue", (), {"code": code})()),
    )
    result, _ = _build("healthy-dev")
    assert result["error_type"] == code


def test_unbootstrapped_service_with_disabled_directory_is_measured(tmp_path: Path) -> None:
    env = _env("healthy-dev")
    disabled = tmp_path / "disabled"
    disabled.mkdir()
    env["BRAINLAYER_OBSERVABILITY_DISABLED_DIR"] = str(disabled)
    env["BRAINLAYER_OBSERVABILITY_LAUNCHD_OUTPUT"] = str(tmp_path / "launchd.txt")
    Path(env["BRAINLAYER_OBSERVABILITY_LAUNCHD_OUTPUT"]).write_text(
        'Could not find service "com.brainlayer.jsonl-backup" in domain for user gui: 501\n', encoding="utf-8"
    )

    def recorder(path: Path, **kwargs: object) -> dict[str, object]:
        return {
            "path": str(path),
            "status": str(kwargs.get("status", "read")),
            "mtime": None,
            "rows_or_bytes": kwargs.get("rows_or_bytes"),
            "sha256_first_64kb": None,
            "skipped_lines": kwargs.get("skipped_lines", 0),
        }

    result = build_backups_section(env=env, record_input=recorder, now=NOW)
    assert result["state"] == "measured"
    assert result["launchd"] == {
        "label": "com.brainlayer.jsonl-backup",
        "bootstrapped": False,
        "disabled_dir_present": True,
    }


def test_production_defaults_are_db_relative(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    db = tmp_path / "snapshot" / "brainlayer.db"
    db.parent.mkdir(parents=True)
    source = FIXTURES / "logs" / "healthy-dev"
    logs = db.parent / "logs"
    logs.mkdir()
    for name in ("jsonl-backup.log", "backup-daily.log"):
        (logs / name).write_bytes((source / name).read_bytes())
    monkeypatch.setattr(observability_backup.jsonl_backup, "DEFAULT_LOG_PATH", logs / "jsonl-backup.log")
    disabled = tmp_path / "Library" / "LaunchAgents" / ".disabled-retention-P0"
    disabled.mkdir(parents=True)
    launchd = tmp_path / "launchd.txt"
    launchd.write_bytes((FIXTURES / "launchd" / "healthy-dev.txt").read_bytes())
    monkeypatch.setattr(Path, "home", lambda: tmp_path)

    env = {
        "BRAINLAYER_DB": str(db),
        "BRAINLAYER_OBSERVABILITY_LAUNCHD_OUTPUT": str(launchd),
    }
    result, _ = _build_with_env(env)

    assert result["state"] == "measured"
    assert [item["path"] for item in result["inputs"]] == [
        str(logs / "jsonl-backup.log"),
        str(logs / "backup-daily.log"),
        str(launchd),
    ]


def test_producer_without_observability_wiring_measures_all_sections(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    case = FIXTURES / "db" / "healthy-dev.sqlite"
    db = tmp_path / "snapshot" / "brainlayer.db"
    db.parent.mkdir(parents=True)
    db.write_bytes(case.read_bytes())
    logs = db.parent / "logs"
    logs.mkdir()
    for name in ("jsonl-backup.log", "backup-daily.log"):
        (logs / name).write_bytes((FIXTURES / "logs" / "healthy-dev" / name).read_bytes())
    monkeypatch.setattr(observability_backup.jsonl_backup, "DEFAULT_LOG_PATH", logs / "jsonl-backup.log")
    disabled = tmp_path / "Library" / "LaunchAgents" / ".disabled-retention-P0"
    disabled.mkdir(parents=True)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    launchd_text = (FIXTURES / "launchd" / "healthy-dev.txt").read_text(encoding="utf-8")
    monkeypatch.setattr(
        observability_backup.subprocess,
        "run",
        lambda argv, **kwargs: subprocess.CompletedProcess(argv, 0, launchd_text.encode(), b""),
    )

    document, _ = observability_surface.build_document(env={"BRAINLAYER_DB": str(db)})

    for section in ("stores", "emitters", "author_unknown", "backups"):
        assert document[section]["state"] == "measured"
    assert document["transport"]["state"] == "unmeasurable"
    schema = json.loads((FIXTURES / "observability-schema.v2.json").read_text(encoding="utf-8"))
    jsonschema.validate(document, schema)
    assert [item["path"] for item in document["backups"]["inputs"][:2]] == [
        str(logs / "jsonl-backup.log"),
        str(logs / "backup-daily.log"),
    ]
    assert document["backups"]["inputs"][2]["path"].startswith("command:launchctl print gui/")
    assert document["backups"]["inputs"][2]["status"] == "read"


def test_backup_daily_empty_env_does_not_fall_back_to_process_environment(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    db = tmp_path / "snapshot" / "brainlayer.db"
    monkeypatch.setenv("BRAINLAYER_BACKUP_LOG_PATH", str(tmp_path / "process-env.log"))

    assert backup_daily._backup_log_path(None, db_path=db, env={}) == db.parent / "logs" / "backup-daily.log"


def test_nonzero_unrecognized_launchd_exit_is_unmeasurable(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    env, recorder = _command_env(tmp_path)
    output = f"service = {observability_backup.LABEL}\n"
    monkeypatch.setattr(
        observability_backup.subprocess,
        "run",
        lambda argv, **kwargs: subprocess.CompletedProcess(argv, 1, output.encode(), b""),
    )

    result = build_backups_section(env=env, record_input=recorder, now=NOW)

    assert result["state"] == "unmeasurable"
    assert "exit_code=1" in result["reason"]
    assert "['launchctl', 'print', 'gui/" in result["reason"]


def test_jsonl_writer_override_controls_reader_default(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    env, recorder = _command_env(tmp_path)
    writer_path = tmp_path / "writer-jsonl.log"
    writer_path.write_bytes((FIXTURES / "logs" / "healthy-dev" / "jsonl-backup.log").read_bytes())
    env.pop("BRAINLAYER_OBSERVABILITY_JSONL_BACKUP_LOG")
    env["BRAINLAYER_JSONL_BACKUP_LOG_PATH"] = str(writer_path)

    result = build_backups_section(env=env, record_input=recorder, now=NOW)

    assert result["inputs"][0]["path"] == str(writer_path)


def test_backup_process_uses_db_relative_default_log(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    db = tmp_path / "snapshot" / "brainlayer.db"
    captured: dict[str, Path] = {}
    monkeypatch.setenv("BRAINLAYER_DB", str(db))
    monkeypatch.delenv("BRAINLAYER_BACKUP_LOG_PATH", raising=False)

    def fake_run_backup(**kwargs):
        captured["log_path"] = kwargs["log_path"]
        return {"verified": True}

    monkeypatch.setattr(backup_daily, "run_backup", fake_run_backup)

    assert backup_daily._run_backup_process(1) == 0
    assert captured["log_path"] == db.parent / "logs" / "backup-daily.log"


def test_subprocess_error_keeps_document_measured_except_for_backups(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    case = FIXTURES / "db" / "healthy-dev.sqlite"
    db = tmp_path / "snapshot" / "brainlayer.db"
    db.parent.mkdir(parents=True)
    db.write_bytes(case.read_bytes())
    logs = db.parent / "logs"
    logs.mkdir()
    for name in ("jsonl-backup.log", "backup-daily.log"):
        (logs / name).write_bytes((FIXTURES / "logs" / "healthy-dev" / name).read_bytes())
    monkeypatch.setattr(observability_backup.jsonl_backup, "DEFAULT_LOG_PATH", logs / "jsonl-backup.log")
    disabled = tmp_path / "Library" / "LaunchAgents" / ".disabled-retention-P0"
    disabled.mkdir(parents=True)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setattr(
        observability_backup.subprocess,
        "run",
        lambda *args, **kwargs: (_ for _ in ()).throw(subprocess.SubprocessError("synthetic command failure")),
    )

    document, _ = observability_surface.build_document(env={"BRAINLAYER_DB": str(db)})

    assert document["backups"]["state"] == "unmeasurable"
    assert "SubprocessError" in document["backups"]["reason"]
    assert all(document[name]["state"] == "measured" for name in ("stores", "emitters", "author_unknown"))
    assert document["transport"]["state"] == "unmeasurable"
    schema = json.loads((FIXTURES / "observability-schema.v2.json").read_text(encoding="utf-8"))
    jsonschema.validate(document, schema)


def test_empty_jsonl_writer_override_is_unset(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    env, recorder = _command_env(tmp_path)
    env.pop("BRAINLAYER_OBSERVABILITY_JSONL_BACKUP_LOG")
    env["BRAINLAYER_JSONL_BACKUP_LOG_PATH"] = ""

    result = build_backups_section(env=env, record_input=recorder, now=NOW)

    assert result["inputs"][0]["path"] == str(observability_backup.jsonl_backup.DEFAULT_LOG_PATH)


def test_unset_launchd_input_uses_command_and_records_stdout(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    env, recorder = _command_env(tmp_path)
    output = "service = com.brainlayer.jsonl-backup\n"

    def run(argv, **kwargs):
        assert kwargs["timeout"] <= 5
        assert kwargs["shell"] is False
        return subprocess.CompletedProcess(argv, 0, output.encode(), b"")

    monkeypatch.setattr(observability_backup.subprocess, "run", run)
    result = build_backups_section(env=env, record_input=recorder, now=NOW)

    assert result["state"] == "measured"
    command = result["inputs"][2]
    assert (
        command["path"]
        == f"command:launchctl print gui/{observability_backup.os.getuid()}/{observability_backup.LABEL}"
    )
    assert command["sha256_first_64kb"] == hashlib.sha256(output.encode()).hexdigest()
    assert command["status"] == "read"


def test_unset_launchd_command_recognizes_not_loaded(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    env, recorder = _command_env(tmp_path)
    output = f'Could not find service "{observability_backup.LABEL}" in domain for user gui: 501\n'
    monkeypatch.setattr(
        observability_backup.subprocess,
        "run",
        lambda argv, **kwargs: subprocess.CompletedProcess(argv, 113, b"", output.encode()),
    )

    result = build_backups_section(env=env, record_input=recorder, now=NOW)

    assert result["state"] == "measured"
    assert result["launchd"]["bootstrapped"] is False
    assert result["inputs"][2]["status"] == "read"


def test_unset_launchd_command_failure_is_unmeasurable_with_argv(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    env, recorder = _command_env(tmp_path)
    monkeypatch.setattr(
        observability_backup.subprocess,
        "run",
        lambda argv, **kwargs: (_ for _ in ()).throw(FileNotFoundError("launchctl")),
    )

    result = build_backups_section(env=env, record_input=recorder, now=NOW)

    assert result["state"] == "unmeasurable"
    assert "['launchctl', 'print', 'gui/" in result["reason"]


def test_launchd_file_override_still_wins(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    env, recorder = _command_env(tmp_path)
    launchd = tmp_path / "override.txt"
    launchd.write_text("service = com.brainlayer.jsonl-backup\n", encoding="utf-8")
    env["BRAINLAYER_OBSERVABILITY_LAUNCHD_OUTPUT"] = str(launchd)
    monkeypatch.setattr(
        observability_backup.subprocess,
        "run",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("command fallback used")),
    )

    result = build_backups_section(env=env, record_input=recorder, now=NOW)

    assert result["state"] == "measured"
    assert result["inputs"][2]["path"] == str(launchd)


def _build_with_env(env: dict[str, str]) -> tuple[dict[str, object], list[str]]:
    opened: list[str] = []

    def recorder(path: Path, **kwargs: object) -> dict[str, object]:
        opened.append(str(path))
        return {
            "path": str(path),
            "status": str(kwargs.get("status", "read")),
            "mtime": None,
            "rows_or_bytes": kwargs.get("rows_or_bytes"),
            "sha256_first_64kb": None,
            "skipped_lines": kwargs.get("skipped_lines", 0),
        }

    return build_backups_section(env=env, record_input=recorder, now=NOW), opened


def _command_env(tmp_path: Path) -> tuple[dict[str, str], object]:
    source = FIXTURES / "logs" / "healthy-dev"
    logs = tmp_path / "logs"
    logs.mkdir()
    for name in ("jsonl-backup.log", "backup-daily.log"):
        (logs / name).write_bytes((source / name).read_bytes())
    disabled = tmp_path / "disabled"
    disabled.mkdir()
    env = {
        "BRAINLAYER_OBSERVABILITY_JSONL_BACKUP_LOG": str(logs / "jsonl-backup.log"),
        "BRAINLAYER_OBSERVABILITY_BACKUP_DAILY_LOG": str(logs / "backup-daily.log"),
        "BRAINLAYER_OBSERVABILITY_DISABLED_DIR": str(disabled),
    }

    def recorder(path: Path | None, **kwargs: object) -> dict[str, object]:
        if kwargs.get("kind") == "command":
            stdout = kwargs.get("stdout", b"")
            stderr = kwargs.get("stderr", b"")
            stdout_bytes = stdout if isinstance(stdout, bytes) else str(stdout).encode()
            stderr_bytes = stderr if isinstance(stderr, bytes) else str(stderr).encode()
            command_argv = kwargs["argv"]
            return {
                "path": "command:" + " ".join(command_argv),
                "status": kwargs.get("status", "read"),
                "mtime": None,
                "rows_or_bytes": len(stdout_bytes + stderr_bytes),
                "sha256_first_64kb": hashlib.sha256((stdout_bytes + stderr_bytes)[:65_536]).hexdigest(),
                "skipped_lines": 0,
            }
        assert path is not None
        return {
            "path": str(path),
            "status": str(kwargs.get("status", "read")),
            "mtime": None,
            "rows_or_bytes": kwargs.get("rows_or_bytes"),
            "sha256_first_64kb": None,
            "skipped_lines": kwargs.get("skipped_lines", 0),
        }

    return env, recorder


def test_verified_no_op_exposes_explicit_receipt_without_inventing_upload() -> None:
    result, _ = _build("no-op-dev")
    assert result["transcript_attempt"] == {
        "at": "2026-09-13T10:00:00Z",
        "status": "no-op",
        "verified": True,
        "uploaded": False,
    }
    assert result["last_verified_upload"] is None
    assert result["surviving_archives_30d"] == 0
    schema = json.loads((FIXTURES / "observability-schema.v2.json").read_text())
    jsonschema.validate(result, {"$defs": schema["$defs"], "$ref": "#/$defs/backups"})
    legacy, _ = _build("legacy-no-op-dev")
    assert "transcript_attempt" not in legacy


@pytest.mark.parametrize(
    "mutation",
    [
        {"attempted_at": "2026-09-13T13:00:00Z"},
        {"attempted_at": "2026-09-10T10:00:00Z"},
        {"verified": False},
        {"uploaded": True},
        {"status": "failed"},
    ],
)
def test_invalid_no_op_never_exposes_neutral_evidence(tmp_path, mutation) -> None:
    latest = json.loads((FIXTURES / "logs/no-op-dev/jsonl-backup.log").read_text())
    log = tmp_path / "jsonl-backup.log"
    log.write_text(json.dumps({**latest, **mutation}) + "\n")
    env = {**_env("no-op-dev"), "BRAINLAYER_OBSERVABILITY_JSONL_BACKUP_LOG": str(log)}
    result = build_backups_section(
        env=env,
        now=NOW,
        record_input=lambda path, **kwargs: {"path": str(path), "status": kwargs.get("status", "read")},
    )
    assert "transcript_attempt" not in result


def _two_copy_env(tmp_path: Path, *, transcript_at: str, daily: list[dict[str, object]]) -> tuple[dict, object]:
    env, recorder = _command_env(tmp_path)
    launchd = tmp_path / "launchd.txt"
    launchd.write_text("service = com.brainlayer.jsonl-backup\n")
    env["BRAINLAYER_OBSERVABILITY_LAUNCHD_OUTPUT"] = str(launchd)
    env["BRAINLAYER_JOB_ALERT_PATH"] = str(tmp_path / "absent-alerts.json")
    Path(env["BRAINLAYER_OBSERVABILITY_JSONL_BACKUP_LOG"]).write_text(
        json.dumps(
            {
                "archive_id": "synthetic-archive-12h",
                "attempted_at": transcript_at,
                "status": "uploaded",
                "uploaded": True,
                "verified": True,
            }
        )
        + "\n"
    )
    Path(env["BRAINLAYER_OBSERVABILITY_BACKUP_DAILY_LOG"]).write_text(
        "".join(json.dumps({"backup_log_provenance": "real", **record}) + "\n" for record in daily)
    )
    return env, recorder


def _db_receipt(created_at: str, name: str, **extra: object) -> dict[str, object]:
    return {
        "attempted_at": created_at,
        "snapshot_created_at": created_at,
        "snapshot": f"/synthetic/backups/{name}",
        "drive_file": {"id": "synthetic-id", "name": name},
        "drive_md5_match": True,
        "uploaded": True,
        "verified": True,
        **extra,
    }


def _assert_backups_schema(result: dict[str, object]) -> None:
    for version in ("v1", "v2"):
        schema = json.loads((FIXTURES / f"observability-schema.{version}.json").read_text())
        jsonschema.validate(result, {"$defs": schema["$defs"], "$ref": "#/$defs/backups"})


def test_stale_db_copy_makes_freshness_stale_even_when_transcripts_are_fresh(tmp_path: Path) -> None:
    """Production 2026-10-20: transcripts 12.6 h old, newest verified DB copy ~87 h old, UI said "fresh"."""
    env, recorder = _two_copy_env(
        tmp_path,
        transcript_at="2026-09-13T00:00:00Z",
        daily=[_db_receipt("2026-09-09T21:00:00+00:00", "2026-09-09.db.gz")],
    )

    result = build_backups_section(env=env, record_input=recorder, now=NOW)

    assert result["state"] == "measured"
    assert result["last_verified_upload"]["age_hours"] == 12.0
    assert result["db_snapshot"]["destination"] == "2026-09-09.db.gz"
    assert result["freshness"] == "stale"
    assert result["error_type"] == (
        "db_snapshot_stale: newest verified DB copy 2026-09-09.db.gz is 87.0 h old (threshold 36 h)"
    )
    _assert_backups_schema(result)


def test_both_copies_inside_threshold_stay_fresh(tmp_path: Path) -> None:
    env, recorder = _two_copy_env(
        tmp_path,
        transcript_at="2026-09-13T00:00:00Z",
        daily=[_db_receipt("2026-09-13T01:00:00+00:00", "2026-09-13.db.gz")],
    )

    result = build_backups_section(env=env, record_input=recorder, now=NOW)

    assert result["freshness"] == "fresh"
    assert result["error_type"] is None
    _assert_backups_schema(result)


@pytest.mark.parametrize(
    ("daily", "error_type"),
    [
        (
            [
                {
                    "attempted_at": "2026-09-13T01:00:00+00:00",
                    "snapshot_created_at": "2026-09-13T01:00:00+00:00",
                    "snapshot": "/synthetic/backups/2026-09-13.db.gz",
                    "upload_in_progress": True,
                    "uploaded": False,
                    "verified": False,
                }
            ],
            "db_snapshot_missing: no verified offsite DB copy on record (threshold 36 h)",
        ),
        (
            [_db_receipt("2026-09-13T01:00:00+00:00", "2026-09-13.db.gz", drive_md5_match=False)],
            "db_snapshot_unverified: newest offsite DB copy 2026-09-13.db.gz failed verification",
        ),
    ],
)
def test_missing_or_unverified_db_copy_is_never_fresh(
    tmp_path: Path, daily: list[dict[str, object]], error_type: str
) -> None:
    env, recorder = _two_copy_env(tmp_path, transcript_at="2026-09-13T00:00:00Z", daily=daily)

    result = build_backups_section(env=env, record_input=recorder, now=NOW)

    assert result["freshness"] == "stale"
    assert result["error_type"] == error_type
    _assert_backups_schema(result)


def test_future_db_copy_is_unknown_not_fresh(tmp_path: Path) -> None:
    env, recorder = _two_copy_env(
        tmp_path,
        transcript_at="2026-09-13T00:00:00Z",
        daily=[_db_receipt("2026-09-13T13:00:00+00:00", "2026-09-13.db.gz")],
    )

    result = build_backups_section(env=env, record_input=recorder, now=NOW)

    assert result["freshness"] == "unknown"
    assert result["error_type"] == (
        "db_snapshot_clock_skew: newest verified DB copy 2026-09-13.db.gz is dated after generated_at"
    )


def test_stale_db_copy_keeps_an_earlier_error_type(tmp_path: Path) -> None:
    env, recorder = _two_copy_env(
        tmp_path,
        transcript_at="2026-09-13T00:00:00Z",
        daily=[_db_receipt("2026-09-09T21:00:00+00:00", "2026-09-09.db.gz")],
    )
    alerts = tmp_path / "alerts.json"
    alerts.write_text(json.dumps({"backup-daily": "Database backup failed; check the backup log"}))
    env["BRAINLAYER_JOB_ALERT_PATH"] = str(alerts)

    result = build_backups_section(env=env, record_input=recorder, now=NOW)

    assert result["freshness"] == "stale"
    assert result["error_type"] == "job_alert:Database backup failed; check the backup log"
