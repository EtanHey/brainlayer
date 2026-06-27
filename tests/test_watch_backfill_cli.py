import json

from typer.testing import CliRunner

from brainlayer.cli import app


def test_watch_backfill_dry_run_reports_cursor_agent_transcripts_without_writing(tmp_path):
    transcript = tmp_path / ".cursor" / "projects" / "repo" / "agent-transcripts" / "agent-session.jsonl"
    unrelated = tmp_path / ".cursor" / "projects" / "repo" / "state.jsonl"
    transcript.parent.mkdir(parents=True)
    transcript.write_text(
        json.dumps({"type": "message", "payload": {"role": "user", "content": "cursor agent transcript line"}}) + "\n"
    )
    unrelated.write_text(json.dumps({"role": "user", "content": "unrelated project state"}) + "\n")

    result = CliRunner().invoke(
        app,
        [
            "watch-backfill",
            "--home",
            str(tmp_path),
            "--registry",
            str(tmp_path / "offsets.json"),
            "--dry-run",
        ],
    )

    assert result.exit_code == 0
    assert "candidate_files=1" in result.output
    assert "cursor-agent-transcripts=1" in result.output
    assert "processed_entries=0" in result.output
    assert not (tmp_path / "offsets.json").exists()


def test_watch_backfill_flushes_final_partial_batch_and_is_idempotent(tmp_path):
    transcript = tmp_path / ".cursor" / "projects" / "repo" / "agent-transcripts" / "agent-session.jsonl"
    registry = tmp_path / "offsets.json"
    queue_dir = tmp_path / "queue"
    transcript.parent.mkdir(parents=True)
    transcript.write_text(
        json.dumps(
            {
                "type": "message",
                "payload": {
                    "role": "user",
                    "content": "Please remember this cursor agent transcript line for backfill replay.",
                },
                "timestamp": "2026-06-26T21:00:00Z",
            }
        )
        + "\n"
    )

    first = CliRunner().invoke(
        app,
        [
            "watch-backfill",
            "--home",
            str(tmp_path),
            "--registry",
            str(registry),
            "--max-cycles",
            "5",
        ],
        env={"BRAINLAYER_QUEUE_DIR": str(queue_dir)},
    )

    assert first.exit_code == 0, first.output
    assert "processed_entries=1" in first.output
    queue_files = list(queue_dir.glob("watcher-*.jsonl"))
    assert len(queue_files) == 1
    queued = [json.loads(line) for line in queue_files[0].read_text().splitlines()]
    assert queued[0]["kind"] == "watcher_chunk"
    assert "cursor agent transcript line" in queued[0]["content"]
    assert registry.exists()

    second = CliRunner().invoke(
        app,
        [
            "watch-backfill",
            "--home",
            str(tmp_path),
            "--registry",
            str(registry),
            "--max-cycles",
            "5",
        ],
        env={"BRAINLAYER_QUEUE_DIR": str(queue_dir)},
    )

    assert second.exit_code == 0, second.output
    assert "processed_entries=0" in second.output
    assert len(list(queue_dir.glob("watcher-*.jsonl"))) == 1
