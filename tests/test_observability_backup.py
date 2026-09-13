from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from brainlayer import observability_backup
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
        "db_snapshot": {"last_at": "2026-09-13T09:00:00Z", "destination": "synthetic-drive", "verified": True},
        "error_type": None,
    }
    assert opened == [
        f"logs/{case}/jsonl-backup.log",
        f"logs/{case}/backup-daily.log",
        f"launchd/{case}.txt",
        f"launchd/{case}.disabled",
    ]
    assert result["inputs"][1]["skipped_lines"] == 2


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
        ("missing-launchd-dev", "launchd output is empty", "empty", False),
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
