from __future__ import annotations

import fcntl
import json
import os
import sqlite3
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import brainlayer.health_check as health_check
from brainlayer.badge_state import (
    DATA_LOSS_CODES,
    SUPPRESSIBLE_CODES,
    badge_state_path,
    build_badge_state_document,
    build_pending_first_run_document,
    write_badge_state,
)
from brainlayer.health_check import HealthCheckConfig, HealthCheckResult, HealthIssue, run_health_check
from brainlayer.job_lifecycle_health import JobTick

FIXTURE = Path(__file__).parent / "fixtures/badge-state/badge-state-v1.json"
PENDING_FIXTURE = Path(__file__).parent / "fixtures/badge-state/badge-state-pending-v1.json"


def _result(*issues: HealthIssue) -> HealthCheckResult:
    return HealthCheckResult(
        checked_at="2026-09-15T08:00:00+00:00",
        ok=not issues,
        issues=list(issues),
    )


def _run_minimal_health_check(
    tmp_path: Path, badge_state_path: Path, *, job_opt_path: Path | None = None
) -> HealthCheckResult:
    db_path = tmp_path / "brainlayer.db"
    with sqlite3.connect(db_path) as connection:
        connection.executescript(
            """
            CREATE TABLE chunks (
                id TEXT PRIMARY KEY,
                content TEXT,
                archived_at TEXT,
                superseded_by TEXT,
                aggregated_into TEXT,
                archived INTEGER DEFAULT 0,
                status TEXT DEFAULT 'active',
                enriched_at TEXT,
                enrich_status TEXT,
                char_count INTEGER
            );
            CREATE TABLE chunk_vectors_rowids (id TEXT PRIMARY KEY, chunk_id INTEGER);
            INSERT INTO chunks (id, content) VALUES ('chunk-1', 'content');
            INSERT INTO chunk_vectors_rowids (id) VALUES ('chunk-1');
            """
        )
    return run_health_check(
        HealthCheckConfig(
            db_path=db_path,
            state_path=tmp_path / "health-state.json",
            badge_state_path=badge_state_path,
            job_opt_path=job_opt_path,
            source_jsonl_globs=[],
            queue_dir=tmp_path / "queue",
            offsets_path=tmp_path / "offsets.json",
            watcher_health_path=tmp_path / "watcher-health.json",
            drain_health_path=tmp_path / "drain-health.json",
            t3_health_path=tmp_path / "t3-health.json",
            jsonl_backup_log_path=tmp_path / "backup.log",
        ),
        ps_output_fn=lambda: "123 /usr/bin/python scripts/hotlane_brainbar_daemon.py --backlog-batch 4",
        socket_request_fn=lambda *_args: {"result": {"content": [{"type": "text", "text": "1 of 1 shown"}]}},
        command_runner=lambda _args: SimpleNamespace(returncode=0, stdout="state = running", stderr=""),
    )


def test_failed_job_heals_reach_unsuppressible_badge_and_incident_log(tmp_path: Path, monkeypatch, caplog) -> None:
    message = "com.brainlayer.watch: crashloop, last exit code 1; 3 heal attempts failed to restore a healthy job"
    monkeypatch.setattr(
        health_check,
        "scan_job_lifecycle",
        lambda *_args, **_kwargs: JobTick(
            {"com.brainlayer.watch": {"attempts": 3, "reason": "crashloop, last exit code 1"}}, [], [message]
        ),
    )
    caplog.set_level("INFO", logger="brainlayer.health_check")
    badge_path = tmp_path / "badge-state.json"
    (tmp_path / "health-state.json").write_text('{"job_lifecycle":{"com.brainlayer.watch":{"attempts":3}}}')
    result = _run_minimal_health_check(tmp_path, badge_path, job_opt_path=tmp_path)
    document = json.loads(badge_path.read_text(encoding="utf-8"))
    assert "job_failure" in [issue.code for issue in result.issues]
    assert document["alerts"]["active"][-1]["message"] == message
    assert document["alerts"]["badge_on"] is True
    assert "condition=job_failure" in caplog.text and "timestamp=" in caplog.text


def test_corrupt_health_state_keeps_badge_on_with_reason(tmp_path: Path) -> None:
    (tmp_path / "health-state.json").write_text("{broken", encoding="utf-8")
    badge_path = tmp_path / "badge-state.json"
    result = _run_minimal_health_check(tmp_path, badge_path)
    assert any(issue.code == "job_state_unknown" for issue in result.issues)
    assert json.loads(badge_path.read_text())["alerts"]["badge_on"] is True
    assert "state_corrupt" in json.loads((tmp_path / "health-state.json").read_text())


def test_overlapping_health_check_does_not_run_second_heal(tmp_path: Path) -> None:
    lock_path = tmp_path / "health-state.json.lock"
    with lock_path.open("a+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        result = _run_minimal_health_check(tmp_path, tmp_path / "badge-state.json")
    assert any(issue.code == "health_check_busy" for issue in result.issues)
    assert result.actions == []


def test_badge_state_path_is_db_relative_with_environment_override(tmp_path: Path) -> None:
    db_path = tmp_path / "nested" / "brainlayer.db"

    assert badge_state_path(db_path, env={}) == db_path.parent / "badge-state.json"
    assert badge_state_path(db_path, env={"BRAINLAYER_BADGE_STATE_PATH": str(tmp_path / "override.json")}) == (
        tmp_path / "override.json"
    )


def test_badge_document_uses_versioned_measured_section_shape() -> None:
    document = build_badge_state_document(_result())

    assert document["schema_version"] == 1
    assert document["generated_at"] == "2026-09-15T08:00:00Z"
    assert document["alerts"] == {
        "state": "measured",
        "reason": "",
        "inputs": [],
        "badge_on": False,
        "active": [],
        "suppressed": [],
    }


def test_versioned_contract_fixture_is_real_producer_output() -> None:
    result = _result(
        HealthIssue(
            "jsonl_backup_attempt_failed",
            "critical",
            "latest JSONL backup attempt failed verification",
        )
    )

    assert json.loads(FIXTURE.read_text(encoding="utf-8")) == build_badge_state_document(result)


def test_pending_fixture_is_real_install_state_and_install_writes_it_before_load() -> None:
    expected = build_pending_first_run_document(datetime.fromisoformat("2026-09-15T08:00:00+00:00"))
    install = Path("scripts/launchd/install.sh").read_text(encoding="utf-8")

    assert json.loads(PENDING_FIXTURE.read_text(encoding="utf-8")) == expected
    assert install.index("-m brainlayer.badge_state") < install.index('load_plist "$name"')


def test_pending_module_command_writes_contract(tmp_path: Path) -> None:
    output = tmp_path / "badge-state.json"
    env = {**os.environ, "BRAINLAYER_BADGE_STATE_PATH": str(output), "BRAINLAYER_DB": str(tmp_path / "brainlayer.db")}

    subprocess.run([sys.executable, "-m", "brainlayer.badge_state"], env=env, check=True)

    document = json.loads(output.read_text(encoding="utf-8"))
    assert document["schema_version"] == 1
    assert document["alerts"]["state"] == "pending_first_run"


def test_pending_module_command_preserves_existing_measured_alert(tmp_path: Path) -> None:
    output = tmp_path / "badge-state.json"
    measured_alert = build_badge_state_document(
        _result(HealthIssue("jsonl_backup_attempt_missing", "critical", "backup receipt missing"))
    )
    write_badge_state(output, measured_alert)
    original = output.read_bytes()
    env = {**os.environ, "BRAINLAYER_BADGE_STATE_PATH": str(output), "BRAINLAYER_DB": str(tmp_path / "brainlayer.db")}

    subprocess.run([sys.executable, "-m", "brainlayer.badge_state"], env=env, check=True)

    assert output.read_bytes() == original


def test_data_loss_codes_are_structurally_unsuppressible_even_when_marker_lists_them(
    tmp_path: Path, monkeypatch
) -> None:
    marker = tmp_path / "by-design-notifications.json"
    marker.write_text(
        json.dumps({"conditions": {code: "silence it" for code in DATA_LOSS_CODES}}),
        encoding="utf-8",
    )
    monkeypatch.setenv("BRAINLAYER_BY_DESIGN_REASON_FILE", str(marker))

    source = Path(__import__("brainlayer.badge_state", fromlist=["__file__"]).__file__).read_text(encoding="utf-8")
    assert "BRAINLAYER_BY_DESIGN_REASON_FILE" not in source

    for code in DATA_LOSS_CODES:
        document = build_badge_state_document(_result(HealthIssue(code, "critical", "backup verification failed")))
        assert document["alerts"]["badge_on"] is True
        assert [item["code"] for item in document["alerts"]["active"]] == [code]


def test_only_in_code_suppressible_allow_list_can_hide_critical_issue() -> None:
    code = next(iter(SUPPRESSIBLE_CODES))
    document = build_badge_state_document(_result(HealthIssue(code, "critical", "known noisy condition")))

    assert document["alerts"]["badge_on"] is False
    assert document["alerts"]["active"] == []
    assert [item["code"] for item in document["alerts"]["suppressed"]] == [code]


def test_unknown_critical_issue_fails_visible() -> None:
    document = build_badge_state_document(
        _result(HealthIssue("new_unclassified_failure", "critical", "unknown critical failure"))
    )

    assert document["alerts"]["badge_on"] is True
    assert [item["code"] for item in document["alerts"]["active"]] == ["new_unclassified_failure"]


def test_badge_state_write_is_atomic_and_round_trips(tmp_path: Path, monkeypatch) -> None:
    path = tmp_path / "state" / "badge-state.json"
    synced: list[int] = []
    monkeypatch.setattr(os, "fsync", lambda descriptor: synced.append(descriptor))
    expected = build_badge_state_document(
        _result(HealthIssue("jsonl_backup_attempt_failed", "critical", "backup verification failed"))
    )

    write_badge_state(path, expected)

    assert json.loads(path.read_text(encoding="utf-8")) == expected
    assert len(synced) == 2
    assert not list(path.parent.glob(".*.tmp"))


def test_concurrent_badge_writes_use_unique_temporary_files(tmp_path: Path, monkeypatch) -> None:
    path = tmp_path / "badge-state.json"
    documents = [
        build_badge_state_document(_result(HealthIssue(code, "critical", code)))
        for code in ("jsonl_backup_attempt_failed", "jsonl_backup_attempt_stale")
    ]
    barrier = threading.Barrier(2)
    sources: list[Path] = []
    real_replace = os.replace

    def synchronized_replace(source: str | bytes | os.PathLike, destination: str | bytes | os.PathLike) -> None:
        sources.append(Path(source))
        barrier.wait(timeout=5)
        real_replace(source, destination)

    monkeypatch.setattr("brainlayer.badge_state.os.replace", synchronized_replace)
    with ThreadPoolExecutor(max_workers=2) as executor:
        list(executor.map(lambda document: write_badge_state(path, document), documents))

    assert len(set(sources)) == 2
    assert json.loads(path.read_text(encoding="utf-8")) in documents
    assert not list(tmp_path.glob(".*.tmp"))


def test_completed_health_check_publishes_badge_contract(tmp_path: Path) -> None:
    output = tmp_path / "contract" / "badge-state.json"
    result = _run_minimal_health_check(tmp_path, output)

    document = json.loads(output.read_text(encoding="utf-8"))
    assert document == build_badge_state_document(result)


def test_failed_badge_publish_removes_prior_calm_document(tmp_path: Path, monkeypatch) -> None:
    output = tmp_path / "badge-state.json"
    write_badge_state(output, build_badge_state_document(_result()))

    def fail_write(_path: Path, _document: object) -> None:
        raise OSError(28, "No space left on device")

    monkeypatch.setattr("brainlayer.badge_state.write_badge_state", fail_write)
    result = _run_minimal_health_check(tmp_path, output)

    issue_codes = {issue.code for issue in result.issues}
    assert "jsonl_backup_attempt_missing" in issue_codes
    assert "badge_state_write_failed" in issue_codes
    assert not output.exists(), "A prior calm document would suppress the live failed-write alert."
