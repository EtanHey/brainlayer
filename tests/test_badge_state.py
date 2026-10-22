from __future__ import annotations

import fcntl
import json
import os
import sqlite3
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta
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
    tmp_path: Path,
    badge_state_path: Path,
    *,
    job_opt_path: Path | None = None,
    heal: bool = False,
    canary_response: dict | None = None,
    now_fn=lambda: datetime.now(UTC),
) -> HealthCheckResult:
    db_path = tmp_path / "brainlayer.db"
    with sqlite3.connect(db_path) as connection:
        connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS chunks (
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
            CREATE TABLE IF NOT EXISTS chunk_vectors_rowids (id TEXT PRIMARY KEY, chunk_id INTEGER);
            INSERT OR IGNORE INTO chunks (id, content) VALUES ('chunk-1', 'content');
            INSERT OR IGNORE INTO chunk_vectors_rowids (id) VALUES ('chunk-1');
            """
        )
    return run_health_check(
        HealthCheckConfig(
            db_path=db_path,
            state_path=tmp_path / "health-state.json",
            badge_state_path=badge_state_path,
            job_alert_path=tmp_path / "job-alerts.json",
            pending_stores_path=tmp_path / "pending-stores.jsonl",
            pause_sentinel_path=tmp_path / "pause.sentinel",
            job_plist_dir=tmp_path / "LaunchAgents",
            watch_plist_path=tmp_path / "watch.plist",
            drain_plist_path=tmp_path / "drain.plist",
            health_check_plist_path=tmp_path / "health-check.plist",
            job_opt_path=job_opt_path,
            heal=heal,
            source_jsonl_globs=[],
            queue_dir=tmp_path / "queue",
            offsets_path=tmp_path / "offsets.json",
            watcher_health_path=tmp_path / "watcher-health.json",
            drain_health_path=tmp_path / "drain-health.json",
            t3_health_path=tmp_path / "t3-health.json",
            jsonl_backup_log_path=tmp_path / "backup.log",
        ),
        ps_output_fn=lambda: "123 /usr/bin/python scripts/hotlane_brainbar_daemon.py --backlog-batch 4",
        socket_request_fn=lambda *_args: (
            canary_response or {"result": {"content": [{"type": "text", "text": "1 of 1 shown"}]}}
        ),
        command_runner=lambda _args: SimpleNamespace(returncode=0, stdout="state = running", stderr=""),
        now_fn=now_fn,
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


def test_corrupt_health_state_keeps_badge_on_with_reason(tmp_path: Path, monkeypatch) -> None:
    scans = []

    def scan(*_args, **_kwargs):
        scans.append(True)
        return JobTick({"com.brainlayer.watch": {"runs": 2, "consecutive": 0}}, [], [])

    monkeypatch.setattr(health_check, "scan_job_lifecycle", scan)
    (tmp_path / "health-state.json").write_text("{broken", encoding="utf-8")
    badge_path = tmp_path / "badge-state.json"
    result = _run_minimal_health_check(tmp_path, badge_path, job_opt_path=tmp_path)
    assert any(issue.code == "job_state_unknown" for issue in result.issues)
    assert json.loads(badge_path.read_text())["alerts"]["badge_on"] is True
    assert "state_corrupt" in json.loads((tmp_path / "health-state.json").read_text())
    recovered = _run_minimal_health_check(tmp_path, badge_path, job_opt_path=tmp_path)
    assert all(issue.code != "job_state_unknown" for issue in recovered.issues)
    saved = json.loads((tmp_path / "health-state.json").read_text())
    assert "state_corrupt" not in saved
    assert saved["job_lifecycle"]["com.brainlayer.watch"]["runs"] == 2
    assert len(scans) == 2


def test_non_object_nested_job_state_fails_closed(tmp_path: Path) -> None:
    (tmp_path / "health-state.json").write_text('{"job_lifecycle":[]}', encoding="utf-8")
    badge_path = tmp_path / "badge-state.json"
    result = _run_minimal_health_check(tmp_path, badge_path)
    assert any(issue.code == "job_state_unknown" for issue in result.issues)
    assert json.loads(badge_path.read_text())["alerts"]["badge_on"] is True


def test_prior_corruption_stays_visible_when_fresh_job_scan_fails(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(
        health_check, "scan_job_lifecycle", lambda *_args, **_kwargs: JobTick({}, [], [], "launchd unavailable")
    )
    state_path = tmp_path / "health-state.json"
    state_path.write_text('{"state_corrupt":"prior unreadable state"}', encoding="utf-8")
    badge_path = tmp_path / "badge-state.json"
    result = _run_minimal_health_check(tmp_path, badge_path, job_opt_path=tmp_path)
    assert {issue.code for issue in result.issues} >= {"job_state_unknown", "job_scan_failed"}
    assert json.loads(badge_path.read_text())["alerts"]["badge_on"] is True
    assert json.loads(state_path.read_text())["state_corrupt"] == "prior unreadable state"


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


def test_queue_685_oldest_1847_without_progress_is_badged() -> None:
    document = build_badge_state_document(
        _result(HealthIssue("queue_backed_up", "critical", "count=685 oldest_age=1847"))
    )
    assert document["alerts"]["badge_on"] is True
    assert document["alerts"]["suppressed"] == []


def _snapshot_episode() -> dict:
    return {
        "kind": "snapshot",
        "owner": "backup-daily:pid-123:start-456",
        "episode_id": "snapshot-1",
        "started_at": "2026-09-15T07:59:00Z",
        "observed_at": "2026-09-15T08:00:00Z",
        "deadline": "2026-09-15T08:05:00Z",
        "expected_codes": ["brain_search_canary_failed"],
    }


def test_owned_snapshot_suppresses_canary_only_until_deadline() -> None:
    episode = _snapshot_episode()
    result = _result(
        HealthIssue("brain_search_canary_failed", "critical", "snapshot refusal", expected_episode=episode)
    )
    document = build_badge_state_document(result)
    assert document["alerts"]["badge_on"] is False
    assert document["alerts"]["suppressed"][0]["episode"] == episode
    assert document == json.loads(FIXTURE.with_name("badge-state-episode-v1.json").read_text())
    result.checked_at = episode["deadline"]
    assert build_badge_state_document(result)["alerts"]["badge_on"] is True


def test_health_check_carries_owned_canary_episode_to_badge(tmp_path: Path) -> None:
    now = datetime.now(UTC)
    episode = _snapshot_episode() | {
        "started_at": (now - timedelta(seconds=60)).isoformat(),
        "observed_at": (now + timedelta(seconds=30)).isoformat(),
        "deadline": (now + timedelta(seconds=120)).isoformat(),
    }
    badge_path = tmp_path / "badge-state.json"
    response = {"error": {"code": -32000, "message": "snapshot refusal", "data": {"expected_episode": episode}}}
    clock = iter((now, now + timedelta(seconds=60)))
    result = _run_minimal_health_check(tmp_path, badge_path, canary_response=response, now_fn=lambda: next(clock))
    suppressed = json.loads(badge_path.read_text())["alerts"]["suppressed"]
    assert [issue["code"] for issue in suppressed] == ["brain_search_canary_failed"]
    assert suppressed[0]["episode"] == episode
    assert any(issue.code == "brain_search_canary_failed" for issue in result.issues)


def test_healthy_fixture_has_zero_alerts(tmp_path: Path) -> None:
    now = datetime(2026, 9, 15, 8, tzinfo=UTC)
    (tmp_path / "backup.log").write_text(
        json.dumps({"attempted_at": now.isoformat(), "status": "no-op", "verified": True}) + "\n"
    )
    badge_path = tmp_path / "badge-state.json"
    result = _run_minimal_health_check(tmp_path, badge_path, now_fn=lambda: now)
    assert result.issues == []
    assert result.canary_ok and result.hotlane_running and result.missing_vectors == 0
    alerts = json.loads(badge_path.read_text())["alerts"]
    assert alerts["active"] == alerts["suppressed"] == []
    assert alerts["badge_on"] is False


def test_warning_stays_visible_without_red_dot() -> None:
    issue = HealthIssue("missing_embeddings_climbing", "warning", "arrivals exceeded completions this tick")
    alerts = build_badge_state_document(_result(issue))["alerts"]
    assert alerts["active"] == [{"code": issue.code, "severity": issue.severity, "message": issue.message}]
    assert alerts["badge_on"] is False


def test_every_formerly_suppressed_critical_requires_evidence() -> None:
    for code in (
        "brain_search_canary_failed",
        "drain_no_progress",
        "drain_unloaded",
        "health_check_unloaded",
        "hotlane_backlog_disabled",
        "hotlane_dead",
        "lock_holder_wedge",
        "missing_embeddings_climbing",
        "missing_embeddings_not_draining",
        "observability_unloaded",
        "pause_sentinel_stale",
        "queue_backed_up",
        "watch_unloaded",
        "watcher_stalled",
    ):
        alerts = build_badge_state_document(_result(HealthIssue(code, "critical", "real fault")))["alerts"]
        assert alerts["badge_on"] is True, code
        assert alerts["suppressed"] == []


def test_invalid_or_unrelated_snapshot_evidence_fails_visible() -> None:
    invalid = [
        {"owner": ""},
        {"owner": "   "},
        {"observed_at": "2026-09-15T08:00:00"},
        {"episode_id": ""},
        {"kind": "recovering"},
        {"observed_at": "2026-09-15T07:54:59Z"},
        {"observed_at": "2026-09-15T08:00:01Z"},
        {"started_at": "2026-09-15T08:00:01Z"},
        {"deadline": "2026-09-16T08:00:00Z"},
        {"deadline": "invalid"},
        {"expected_codes": ["queue_backed_up"]},
    ]
    for change in invalid:
        episode = _snapshot_episode() | change
        assert build_badge_state_document(
            _result(
                HealthIssue("brain_search_canary_failed", "critical", "search unavailable", expected_episode=episode)
            )
        )["alerts"]["badge_on"]
    for code in (*DATA_LOSS_CODES, "queue_backed_up", "watcher_stalled", "lock_holder_wedge"):
        episode = _snapshot_episode() | {"expected_codes": [code]}
        assert build_badge_state_document(
            _result(HealthIssue(code, "critical", "fault during snapshot", expected_episode=episode))
        )["alerts"]["badge_on"]


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


def test_retired_service_json_codes_are_not_suppressible():
    compat = json.loads((Path(__file__).parent / "fixtures/service-json-retirement-v1.json").read_text())
    assert SUPPRESSIBLE_CODES.isdisjoint(compat["removed_codes"])
    unreleased = (Path(__file__).parents[1] / "CHANGELOG.md").read_text().split("## [Unreleased]", 1)[1]
    unreleased = unreleased.split("\n## [", 1)[0]
    for code in [*compat["removed_codes"], *compat["renamed_codes"], *compat["renamed_codes"].values()]:
        assert f"`{code}`" in unreleased


def test_partial_job_scan_keeps_critical_badge_and_persists_known_heal(tmp_path: Path, monkeypatch):
    import plistlib

    from brainlayer.job_lifecycle_health import scan_job_lifecycle

    current = tmp_path / "Cellar/brainlayer/1.5.56"
    current.mkdir(parents=True)
    opt = tmp_path / "opt/brainlayer"
    opt.parent.mkdir()
    opt.symlink_to(current)
    agents = tmp_path / "LaunchAgents"
    agents.mkdir()
    for name in ("aaa", "watch"):
        (agents / f"com.brainlayer.{name}.plist").write_bytes(
            plistlib.dumps({"Label": f"com.brainlayer.{name}", "ProgramArguments": [str(opt / "bin/brainlayer"), name]})
        )
    commands = []

    def runner(args):
        commands.append(args)
        output = ""
        if args[:2] == ["launchctl", "print"]:
            output = (
                "runs = 2\nlast exit code = 1\n"
                if args[-1].endswith(".watch")
                else "runs = 1\npid = 9\nlast exit code = 0\n"
            )
        elif args[0] == "lsof":
            output = "n/usr/lib/libSystem.B.dylib\n"
        elif args[0] == "ps":
            output = "/fixture/framework/python -m brainlayer daemon"
        else:
            assert args[:2] in (["launchctl", "print-disabled"], ["launchctl", "kickstart"])
        return SimpleNamespace(returncode=0, stdout=output, stderr="")

    def scan(_plist_dir, _opt, previous, **kwargs):
        return scan_job_lifecycle(agents, opt, previous, **{**kwargs, "command_runner": runner})

    monkeypatch.setattr(health_check, "scan_job_lifecycle", scan)
    state_path = tmp_path / "health-state.json"
    unknown = {"runs": 1, "attempts": 2, "next_retry_epoch": 900}
    state_path.write_text(
        json.dumps(
            {"job_lifecycle": {"com.brainlayer.aaa": unknown, "com.brainlayer.watch": {"runs": 1, "consecutive": 1}}}
        )
    )
    badge_path = tmp_path / "badge-state.json"
    result = _run_minimal_health_check(tmp_path, badge_path, job_opt_path=opt, heal=True)
    issue = next(issue for issue in result.issues if issue.code == "job_scan_failed")
    assert issue.severity == "critical" and "com.brainlayer.aaa" in issue.message
    assert "kickstart:com.brainlayer.watch" in result.actions
    saved = json.loads(state_path.read_text())["job_lifecycle"]
    assert saved["com.brainlayer.aaa"] == unknown
    assert saved["com.brainlayer.watch"]["attempts"] == 1
    document = json.loads(badge_path.read_text())
    assert document["alerts"]["badge_on"] is True
    assert any(alert["code"] == "job_scan_failed" for alert in document["alerts"]["active"])
    assert [args[-1] for args in commands if args[:2] == ["launchctl", "kickstart"]] == [
        f"gui/{os.getuid()}/com.brainlayer.watch"
    ]
