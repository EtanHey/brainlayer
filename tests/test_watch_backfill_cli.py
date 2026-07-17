import json

import pytest
from typer.testing import CliRunner

from brainlayer.cli import app
from brainlayer.paths import get_db_path


@pytest.fixture(autouse=True)
def _isolate_brainlayer_db(tmp_path, monkeypatch):
    monkeypatch.setenv("BRAINLAYER_DB", str(tmp_path / "brainlayer.db"))


def test_watch_backfill_tests_use_isolated_database(tmp_path):
    assert get_db_path() == tmp_path / "brainlayer.db"


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


def test_watch_backfill_legacy_scope_keeps_current_denylist(tmp_path, monkeypatch):
    monkeypatch.delenv("BRAINLAYER_INGEST_DENYLIST", raising=False)
    subagents = tmp_path / ".claude" / "projects" / "repo" / "session" / "subagents"
    allowed = subagents / "agent-ordinary.jsonl"
    unattributed = subagents / "agent-historical.jsonl"
    denied = subagents / "agent-brain-worker.jsonl"
    workflow = tmp_path / ".claude" / "projects" / "repo" / "wf_test" / "worker.jsonl"
    for path in (allowed, unattributed, denied, workflow):
        path.parent.mkdir(parents=True, exist_ok=True)

    def entry(attribution: str | None, label: str | None = None) -> str:
        worker = label or attribution or "historical-worker"
        return json.dumps(
            {
                "type": "assistant",
                "timestamp": "2026-07-12T12:00:00Z",
                **({"attributionAgent": attribution} if attribution else {}),
                "message": {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                f"Legacy {worker} backfill sentinel records the durable implementation "
                                "decision, why the retired blanket policy excluded this transcript, the exact "
                                "migration window, and the verification contract needed to replay it safely."
                            ),
                        }
                    ],
                },
            }
        )

    allowed.write_text(entry("repo-worker") + "\n")
    unattributed.write_text(entry(None, "historical-worker") + "\n")
    denied.write_text(entry("brain-worker") + "\n")
    workflow.write_text(entry("workflow-worker") + "\n")
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
    assert "matched_entries=2" in result.output
    queued = "".join(path.read_text() for path in queue_dir.glob("watcher-*.jsonl"))
    assert queued.count('"kind": "watcher_chunk"') == 2
    assert "repo-worker" in queued
    assert "historical-worker" in queued
    assert "brain-worker" not in queued
    assert "workflow-worker" not in queued


def test_watch_backfill_legacy_scope_honors_configured_denylist(tmp_path, monkeypatch):
    transcript = tmp_path / ".codex" / "sessions" / "2026" / "07" / "blocked" / "worker.jsonl"
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
                                "Configured denylist backfill sentinel records the durable implementation decision, "
                                "the exact migration window, and the verification contract for a safe replay."
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
        env={
            "BRAINLAYER_INGEST_DENYLIST": "~/.codex/sessions/**/blocked/**",
            "BRAINLAYER_QUEUE_DIR": str(queue_dir),
        },
    )

    assert result.exit_code == 0, result.output
    assert "processed_entries=0" in result.output
    assert "matched_entries=0" in result.output
    assert not list(queue_dir.glob("watcher-*.jsonl"))


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


def test_watch_backfill_persists_progress_for_rejected_and_malformed_lines(tmp_path, monkeypatch):
    monkeypatch.delenv("BRAINLAYER_INGEST_DENYLIST", raising=False)
    transcript = tmp_path / ".codex" / "sessions" / "2026" / "07" / "unsupported.jsonl"
    registry = tmp_path / "window-offsets.json"
    queue_dir = tmp_path / "queue"
    transcript.parent.mkdir(parents=True)
    transcript.write_text('{"type":"unsupported"}\nnot-json\n', encoding="utf-8")
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

    first = CliRunner().invoke(app, args, env={"BRAINLAYER_QUEUE_DIR": str(queue_dir)})

    assert first.exit_code == 0, first.output
    assert "processed_entries=2" in first.output
    assert "matched_entries=0" in first.output
    assert "queued_chunks=0" in first.output
    registry_payload = json.loads(registry.read_text())
    assert registry_payload[str(transcript)]["offset"] == transcript.stat().st_size

    second = CliRunner().invoke(app, args, env={"BRAINLAYER_QUEUE_DIR": str(queue_dir)})

    assert second.exit_code == 0, second.output
    assert "processed_entries=0" in second.output


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


def test_watch_backfill_legacy_scope_never_reads_or_advances_nonlegacy_files(tmp_path, monkeypatch):
    monkeypatch.delenv("BRAINLAYER_INGEST_DENYLIST", raising=False)
    direct = tmp_path / ".claude" / "projects" / "repo" / "direct.jsonl"
    legacy = tmp_path / ".codex" / "sessions" / "2026" / "07" / "worker.jsonl"
    direct.parent.mkdir(parents=True)
    legacy.parent.mkdir(parents=True)
    direct.write_text('{"type":"unsupported"}\n')
    legacy.write_text('{"type":"unsupported"}\n')
    registry = tmp_path / "shared-offsets.json"
    registry.write_text(
        json.dumps(
            {
                str(direct): {
                    "offset": 1,
                    "inode": direct.stat().st_ino,
                    "mtime": direct.stat().st_mtime,
                }
            }
        )
    )
    read_paths = []
    from brainlayer.watcher import JSONLTailer

    original_read = JSONLTailer.read_new_lines

    def tracking_read(self, *args, **kwargs):
        read_paths.append(self.filepath)
        return original_read(self, *args, **kwargs)

    monkeypatch.setattr(JSONLTailer, "read_new_lines", tracking_read)

    result = CliRunner().invoke(
        app,
        [
            "watch-backfill",
            "--home",
            str(tmp_path),
            "--registry",
            str(registry),
            "--since",
            "2026-07-10",
            "--until",
            "2026-07-16",
            "--legacy-excluded-only",
            "--max-cycles",
            "5",
        ],
        env={"BRAINLAYER_QUEUE_DIR": str(tmp_path / "queue")},
    )

    assert result.exit_code == 0, result.output
    assert str(direct) not in read_paths
    assert str(legacy) in read_paths
    assert json.loads(registry.read_text())[str(direct)]["offset"] == 1


def test_watch_backfill_max_cycles_exits_nonzero_while_complete_lines_remain(tmp_path, monkeypatch):
    monkeypatch.delenv("BRAINLAYER_INGEST_DENYLIST", raising=False)
    transcript = tmp_path / ".codex" / "sessions" / "2026" / "07" / "worker.jsonl"
    transcript.parent.mkdir(parents=True)
    transcript.write_text('{"type":"unsupported"}\n' * 101)

    result = CliRunner().invoke(
        app,
        [
            "watch-backfill",
            "--home",
            str(tmp_path),
            "--registry",
            str(tmp_path / "offsets.json"),
            "--since",
            "2026-07-10",
            "--until",
            "2026-07-16",
            "--max-cycles",
            "1",
        ],
        env={"BRAINLAYER_QUEUE_DIR": str(tmp_path / "queue")},
    )

    assert result.exit_code == 1, result.output
    assert "incomplete=true" in result.output


def test_watch_backfill_caps_each_file_read(tmp_path, monkeypatch):
    monkeypatch.delenv("BRAINLAYER_INGEST_DENYLIST", raising=False)
    transcript = tmp_path / ".codex" / "sessions" / "2026" / "07" / "worker.jsonl"
    transcript.parent.mkdir(parents=True)
    transcript.write_text('{"type":"unsupported"}\n')
    from brainlayer.watcher import JSONLTailer

    observed_read_limits = []
    original_read = JSONLTailer.read_new_lines

    def tracking_read(self, *args, **kwargs):
        observed_read_limits.append(kwargs.get("max_read_bytes"))
        return original_read(self, *args, **kwargs)

    monkeypatch.setattr(JSONLTailer, "read_new_lines", tracking_read)

    result = CliRunner().invoke(
        app,
        [
            "watch-backfill",
            "--home",
            str(tmp_path),
            "--registry",
            str(tmp_path / "offsets.json"),
            "--max-cycles",
            "2",
        ],
        env={"BRAINLAYER_QUEUE_DIR": str(tmp_path / "queue")},
    )

    assert result.exit_code == 0, result.output
    assert observed_read_limits
    assert set(observed_read_limits) == {1_048_576}


def test_watch_backfill_rejects_concurrent_run_for_same_registry(tmp_path):
    from brainlayer.backfill import backfill_run_lock

    registry = tmp_path / "offsets.json"
    with backfill_run_lock(registry):
        result = CliRunner().invoke(
            app,
            [
                "watch-backfill",
                "--home",
                str(tmp_path),
                "--registry",
                str(registry),
                "--max-cycles",
                "1",
            ],
        )

    assert result.exit_code == 2
    assert "another backfill is using registry" in result.output


def test_watch_backfill_exits_nonzero_when_flush_failure_is_quarantined(tmp_path, monkeypatch):
    monkeypatch.delenv("BRAINLAYER_INGEST_DENYLIST", raising=False)
    transcript = tmp_path / ".codex" / "sessions" / "2026" / "07" / "worker.jsonl"
    transcript.parent.mkdir(parents=True)
    transcript.write_text(
        json.dumps(
            {
                "role": "user",
                "content": "quarantine this backfill entry after a synthetic queue failure",
                "timestamp": "2026-07-12T12:00:00Z",
            }
        )
        + "\n"
    )
    monkeypatch.setenv("BRAINLAYER_WATCHER_FLUSH_RETAIN_LIMIT", "1")
    monkeypatch.setenv("BRAINLAYER_WATCHER_QUARANTINE_DIR", str(tmp_path / "quarantine"))

    flush_calls = []

    def failing_flush(entries):
        flush_calls.append(entries)
        raise RuntimeError("synthetic queue failure")

    monkeypatch.setattr("brainlayer.watcher_bridge.create_flush_callback", lambda *_args, **_kwargs: failing_flush)

    result = CliRunner().invoke(
        app,
        [
            "watch-backfill",
            "--home",
            str(tmp_path),
            "--registry",
            str(tmp_path / "offsets.json"),
            "--max-cycles",
            "2",
        ],
    )

    assert flush_calls, result.output
    assert list((tmp_path / "quarantine").glob("watcher-flush-*.jsonl"))
    assert result.exit_code == 1, result.output
    assert "incomplete=true" in result.output
