import json

from typer.testing import CliRunner

from brainlayer.cli import app


def test_watch_backfill_dry_run_includes_cursor_agent_transcripts(tmp_path, monkeypatch):
    monkeypatch.delenv("BRAINLAYER_INGEST_DENYLIST", raising=False)
    transcript = (
        tmp_path / ".cursor" / "projects" / "repo" / "agent-transcripts" / "agent-session" / "agent-session.jsonl"
    )
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


def test_watch_backfill_dry_run_includes_codex_and_gemini_sessions(tmp_path, monkeypatch):
    monkeypatch.delenv("BRAINLAYER_INGEST_DENYLIST", raising=False)
    codex_transcript = tmp_path / ".codex" / "sessions" / "2026" / "07" / "worker.jsonl"
    gemini_transcript = tmp_path / ".gemini" / "sessions" / "worker.jsonl"
    for path in (codex_transcript, gemini_transcript):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"role": "user", "content": "worker transcript line"}) + "\n")

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
    assert "candidate_files=2" in result.output
    assert "codex=1" in result.output
    assert "gemini=1" in result.output
    assert "processed_entries=0" in result.output
    assert not (tmp_path / "offsets.json").exists()


def test_watch_backfill_indexes_cursor_agent_transcripts_once(tmp_path, monkeypatch):
    monkeypatch.delenv("BRAINLAYER_INGEST_DENYLIST", raising=False)
    transcript = (
        tmp_path / ".cursor" / "projects" / "repo" / "agent-transcripts" / "agent-session" / "agent-session.jsonl"
    )
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
    assert list(queue_dir.glob("watcher-*.jsonl")) == queue_files


def test_watch_backfill_keeps_polling_while_oversized_record_is_buffering(tmp_path, monkeypatch):
    monkeypatch.delenv("BRAINLAYER_INGEST_DENYLIST", raising=False)
    transcript = tmp_path / ".codex" / "sessions" / "2026" / "07" / "oversized.jsonl"
    registry = tmp_path / "offsets.json"
    queue_dir = tmp_path / "queue"
    transcript.parent.mkdir(parents=True)
    transcript.write_text(
        json.dumps(
            {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": "x" * 512}],
                "timestamp": "2026-07-31T21:00:00Z",
            }
        )
        + "\n"
    )

    result = CliRunner().invoke(
        app,
        [
            "watch-backfill",
            "--home",
            str(tmp_path),
            "--registry",
            str(registry),
            "--max-cycles",
            "10",
        ],
        env={
            "BRAINLAYER_QUEUE_DIR": str(queue_dir),
            "BRAINLAYER_WATCH_MAX_FILE_BYTES": "128",
        },
    )

    assert result.exit_code == 0, result.output
    assert "processed_entries=1" in result.output
    assert len(list(queue_dir.glob("watcher-*.jsonl"))) == 1
    persisted = json.loads(registry.read_text())
    assert persisted[str(transcript)]["offset"] == transcript.stat().st_size
