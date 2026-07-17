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


def test_watch_backfill_legacy_dry_run_counts_only_legacy_excluded_roots(tmp_path, monkeypatch):
    monkeypatch.delenv("BRAINLAYER_INGEST_DENYLIST", raising=False)
    direct = tmp_path / ".claude" / "projects" / "repo" / "direct.jsonl"
    legacy = tmp_path / ".codex" / "sessions" / "2026" / "07" / "worker.jsonl"
    for path in (direct, legacy):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"role": "user", "content": "transcript"}) + "\n")

    result = CliRunner().invoke(
        app,
        [
            "watch-backfill",
            "--home",
            str(tmp_path),
            "--since",
            "2026-07-10",
            "--until",
            "2026-07-16",
            "--legacy-excluded-only",
            "--dry-run",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "candidate_files=1" in result.output
    assert "codex=1" in result.output
    assert "claude=" not in result.output
    assert "scope=legacy-excluded-only" in result.output
    assert "-legacy-excluded-only.json" in result.output


def test_watch_backfill_legacy_scope_bypasses_current_subagent_denylist(tmp_path, monkeypatch):
    monkeypatch.delenv("BRAINLAYER_INGEST_DENYLIST", raising=False)
    transcript = tmp_path / ".claude" / "projects" / "repo" / "session" / "subagents" / "agent-worker.jsonl"
    transcript.parent.mkdir(parents=True)
    transcript.write_text(
        json.dumps(
            {
                "type": "assistant",
                "timestamp": "2026-07-12T12:00:00Z",
                "message": {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                "Legacy subagent backfill sentinel records the durable implementation decision, "
                                "the reason the previous indexing policy excluded this transcript, the exact "
                                "migration window, and the verification contract needed to replay it safely "
                                "without discarding the raw JSONL source or advancing offsets before persistence."
                            ),
                        }
                    ],
                },
            }
        )
        + "\n"
    )
    queue_dir = tmp_path / "queue"

    result = CliRunner().invoke(
        app,
        [
            "watch-backfill",
            "--home",
            str(tmp_path),
            "--since",
            "2026-07-10",
            "--until",
            "2026-07-16",
            "--legacy-excluded-only",
            "--max-cycles",
            "5",
        ],
        env={"BRAINLAYER_QUEUE_DIR": str(queue_dir)},
    )

    assert result.exit_code == 0, result.output
    assert "matched_entries=1" in result.output
    assert len(list(queue_dir.glob("watcher-*.jsonl"))) == 1


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


def test_watch_backfill_indexes_only_requested_window_and_is_idempotent(tmp_path, monkeypatch):
    monkeypatch.delenv("BRAINLAYER_INGEST_DENYLIST", raising=False)
    transcript = tmp_path / ".claude" / "projects" / "repo" / "session.jsonl"
    registry = tmp_path / "window-offsets.json"
    queue_dir = tmp_path / "queue"
    transcript.parent.mkdir(parents=True)

    def entry(timestamp: str, token: str) -> str:
        return json.dumps(
            {
                "type": "assistant",
                "timestamp": timestamp,
                "message": {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                f"{token} is a substantive backfill sentinel with enough durable technical "
                                "context to pass classification and chunking thresholds."
                            ),
                        }
                    ],
                },
            }
        )

    transcript.write_text(
        "\n".join(
            [
                entry("2026-07-09T23:59:59Z", "BEFOREWINDOW"),
                entry("2026-07-12T12:00:00Z", "INSIDEWINDOW"),
                entry("2026-07-16T00:00:00Z", "AFTERWINDOW"),
            ]
        )
        + "\n"
    )

    args = [
        "watch-backfill",
        "--home",
        str(tmp_path),
        "--registry",
        str(registry),
        "--since",
        "2026-07-10",
        "--until",
        "2026-07-16",
    ]
    first = CliRunner().invoke(
        app,
        args,
        env={"BRAINLAYER_QUEUE_DIR": str(queue_dir)},
    )

    assert first.exit_code == 0, first.output
    assert "matched_entries=1" in first.output
    assert len(list(queue_dir.glob("watcher-*.jsonl"))) == 1

    second = CliRunner().invoke(
        app,
        args,
        env={"BRAINLAYER_QUEUE_DIR": str(queue_dir)},
    )

    assert second.exit_code == 0, second.output
    assert "matched_entries=0" in second.output
    assert len(list(queue_dir.glob("watcher-*.jsonl"))) == 1


def test_watch_backfill_rejects_legacy_scope_without_window(tmp_path):
    result = CliRunner().invoke(
        app,
        [
            "watch-backfill",
            "--home",
            str(tmp_path),
            "--legacy-excluded-only",
            "--dry-run",
        ],
    )

    assert result.exit_code != 0
    assert "--legacy-excluded-only requires" in result.output
    assert "--since and --until" in result.output
