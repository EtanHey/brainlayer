from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from brainlayer.cli import app


def _write_events(path: Path, events: list[dict]) -> None:
    path.write_text("".join(json.dumps(event) + "\n" for event in events), encoding="utf-8")


def test_writer_telemetry_tail_prints_newest_events_without_mutating_file(tmp_path):
    path = tmp_path / "writer-telemetry.jsonl"
    _write_events(path, [{"event": "txn_finished", "txn_id": str(index)} for index in range(3)])
    before = path.stat().st_mtime_ns

    result = CliRunner().invoke(app, ["writer-telemetry", "tail", "--lines", "2", "--path", str(path)])

    assert result.exit_code == 0, result.output
    assert [json.loads(line)["txn_id"] for line in result.output.splitlines()] == ["1", "2"]
    assert path.stat().st_mtime_ns == before


def test_writer_telemetry_summary_aggregates_finished_transactions(tmp_path):
    path = tmp_path / "writer-telemetry.jsonl"
    _write_events(
        path,
        [
            {"event": "txn_started", "producer": "drain", "lane": "interactive"},
            {
                "event": "txn_finished",
                "producer": "drain",
                "lane": "interactive",
                "outcome": "commit",
                "duration_ms": 10.0,
            },
            {
                "event": "txn_finished",
                "producer": "index",
                "lane": "batch",
                "outcome": "rollback",
                "duration_ms": 30.0,
            },
        ],
    )

    result = CliRunner().invoke(app, ["writer-telemetry", "summary", "--lines", "100", "--path", str(path)])

    assert result.exit_code == 0, result.output
    summary = json.loads(result.output)
    assert summary == {
        "counts_by_lane": {"batch": 1, "interactive": 1},
        "counts_by_outcome": {"commit": 1, "rollback": 1},
        "counts_by_producer": {"drain": 1, "index": 1},
        "duration_ms": {"max": 30.0, "p95": 30.0},
        "finished_transactions": 2,
        "lines_read": 3,
    }


def test_writer_telemetry_missing_file_is_an_empty_read_only_result(tmp_path):
    path = tmp_path / "missing.jsonl"
    runner = CliRunner()

    tail = runner.invoke(app, ["writer-telemetry", "tail", "--path", str(path)])
    summary = runner.invoke(app, ["writer-telemetry", "summary", "--path", str(path)])

    assert tail.exit_code == 0, tail.output
    assert tail.output == ""
    assert summary.exit_code == 0, summary.output
    assert json.loads(summary.output)["finished_transactions"] == 0
    assert not path.exists()


def test_writer_telemetry_rejects_unknown_action(tmp_path):
    result = CliRunner().invoke(
        app,
        ["writer-telemetry", "delete", "--path", str(tmp_path / "writer-telemetry.jsonl")],
    )

    assert result.exit_code != 0
