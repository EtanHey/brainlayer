from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from types import SimpleNamespace

from brainlayer.badge_state import (
    DATA_LOSS_CODES,
    SUPPRESSIBLE_CODES,
    badge_state_path,
    build_badge_state_document,
    write_badge_state,
)
from brainlayer.health_check import HealthCheckConfig, HealthCheckResult, HealthIssue, run_health_check

FIXTURE = Path(__file__).parent / "fixtures/badge-state/badge-state-v1.json"


def _result(*issues: HealthIssue) -> HealthCheckResult:
    return HealthCheckResult(
        checked_at="2026-09-15T08:00:00+00:00",
        ok=not issues,
        issues=list(issues),
    )


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


def test_badge_state_write_is_atomic_and_round_trips(tmp_path: Path) -> None:
    path = tmp_path / "state" / "badge-state.json"
    expected = build_badge_state_document(
        _result(HealthIssue("jsonl_backup_attempt_failed", "critical", "backup verification failed"))
    )

    write_badge_state(path, expected)

    assert json.loads(path.read_text(encoding="utf-8")) == expected
    assert not list(path.parent.glob(".*.tmp"))


def test_completed_health_check_publishes_badge_contract(tmp_path: Path) -> None:
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
    output = tmp_path / "contract" / "badge-state.json"
    result = run_health_check(
        HealthCheckConfig(
            db_path=db_path,
            state_path=tmp_path / "health-state.json",
            badge_state_path=output,
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

    document = json.loads(output.read_text(encoding="utf-8"))
    assert document == build_badge_state_document(result)
