from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
from brainlayer.observability_backup import build_backups_section

FIXTURES = Path(__file__).parent / "fixtures/observability"
NOW = datetime(2026, 9, 13, 12, tzinfo=UTC)


def _env(case: str) -> dict[str, str]:
    return {
        "BRAINLAYER_OBSERVABILITY_INPUT_ROOT": str(FIXTURES),
        "BRAINLAYER_OBSERVABILITY_JSONL_BACKUP_LOG": str(FIXTURES / f"logs/{case}/jsonl-backup.log"),
        "BRAINLAYER_OBSERVABILITY_BACKUP_DAILY_LOG": str(FIXTURES / f"logs/{case}/backup-daily.log"),
        "BRAINLAYER_OBSERVABILITY_LAUNCHD_OUTPUT": str(FIXTURES / f"launchd/{case}.txt"),
        "BRAINLAYER_OBSERVABILITY_DISABLED_DIR": str(FIXTURES / f"launchd/{case}.disabled"),
    }


def _recorder() -> tuple[list[str], Any]:
    opened: list[str] = []

    def record_input(
        path: Path,
        *,
        status: str,
        skipped_lines: int = 0,
        mtime: str | None = None,
    ) -> dict[str, object]:
        relative = path.relative_to(FIXTURES).as_posix()
        opened.append(relative)
        if path.is_dir():
            size = sum(item.stat().st_size for item in path.iterdir())
            digest = hashlib.sha256(b"".join(item.read_bytes() for item in sorted(path.iterdir()))).hexdigest()
        elif path.exists():
            payload = path.read_bytes()
            size = len(payload)
            digest = hashlib.sha256(payload[: 64 * 1024]).hexdigest()
        else:
            size = digest = None
        return {
            "path": relative,
            "status": status,
            "mtime": mtime,
            "rows_or_bytes": size,
            "sha256_first_64kb": digest,
            "skipped_lines": skipped_lines,
        }

    return opened, record_input


def _build(case: str) -> tuple[dict[str, object], list[str]]:
    opened, recorder = _recorder()
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
        "surviving_archives_30d": 7,
        "launchd": {
            "label": "com.brainlayer.jsonl-backup",
            "bootstrapped": True,
            "disabled_dir_present": True,
        },
        "db_snapshot": {
            "last_at": "2026-09-13T09:00:00Z",
            "destination": "synthetic-drive",
            "verified": True,
        },
        "error_type": None,
    }
    assert opened == [
        f"logs/{case}/jsonl-backup.log",
        f"logs/{case}/backup-daily.log",
        f"launchd/{case}.txt",
        f"launchd/{case}.disabled",
    ]
    assert result["inputs"][1]["skipped_lines"] == 2


def test_legacy_rows_are_not_coverage_and_no_op_is_healthy() -> None:
    result, _ = _build("no-op-dev")

    assert result["state"] == "measured"
    assert result["last_verified_upload"] is None
    assert result["freshness"] == "fresh"
    assert result["retention_invariant"] == "unknown"
    assert result["surviving_archives_30d"] == 0
    assert result["launchd"]["bootstrapped"] is False


def test_all_error_daily_log_is_measured_stale() -> None:
    result, _ = _build("backup-errors-dev")

    assert result["state"] == "measured"
    assert result["freshness"] == "stale"
    assert result["error_type"] == "FileNotFoundError"
    assert result["db_snapshot"] is None


@pytest.mark.parametrize(
    ("case", "reason", "status"),
    [
        ("missing-log-dev", "required input missing: logs/missing-log-dev/jsonl-backup.log", "missing"),
        (
            "malformed-log-dev-1",
            "malformed JSON line in logs/malformed-log-dev-1/jsonl-backup.log",
            "malformed",
        ),
        ("missing-launchd-dev", "launchd output is empty", "read"),
    ],
)
def test_invalid_required_input_fails_closed(case: str, reason: str, status: str) -> None:
    result, opened = _build(case)

    assert result == {"state": "unmeasurable", "reason": reason, "inputs": result["inputs"]}
    assert len(opened) == 4
    assert result["inputs"][0]["status"] == status
    assert not any(isinstance(value, (int, float)) for key, value in result.items() if key != "inputs")


def test_drive_token_file_not_found_maps_to_credentials_error() -> None:
    result, _ = _build("backup-errors-dev")

    assert result["error_type"] == "FileNotFoundError"
