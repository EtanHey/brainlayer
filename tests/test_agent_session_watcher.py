"""Tests for the multi-agent session watcher."""

import json
from pathlib import Path

from brainlayer.agent_watch import AgentSessionRegistry, AgentSessionSource, AgentSessionWatcher


def _write_jsonl(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.write(json.dumps(payload) + "\n")


def _append_jsonl(path: Path, payload: dict) -> None:
    with open(path, "a") as f:
        f.write(json.dumps(payload) + "\n")


class TestAgentSessionRegistry:
    def test_round_trips_file_state(self, tmp_path):
        registry = AgentSessionRegistry(tmp_path / "registry.json")
        registry.set("/tmp/session.jsonl", mtime_ns=123, size=456)
        registry.flush()

        reloaded = AgentSessionRegistry(tmp_path / "registry.json")
        assert reloaded.get("/tmp/session.jsonl") == {"mtime_ns": 123, "size": 456}


class TestAgentSessionWatcher:
    def test_polls_multiple_sources_and_skips_unchanged_files(self, tmp_path):
        codex_file = tmp_path / "codex" / "sessions" / "2026" / "04" / "18" / "session-a.jsonl"
        cursor_file = tmp_path / "cursor" / "projects" / "brainlayer" / "agent-transcripts" / "abc" / "abc.jsonl"
        gemini_file = tmp_path / "gemini" / "tmp" / "brainlayer" / "chats" / "session-a.json"

        _write_jsonl(codex_file, {"type": "session_meta"})
        _write_jsonl(cursor_file, {"role": "user", "message": {"content": [{"type": "text", "text": "hello world"}]}})
        gemini_file.parent.mkdir(parents=True, exist_ok=True)
        gemini_file.write_text(json.dumps({"sessionId": "gem-1", "messages": []}))

        seen: list[tuple[str, str]] = []

        def _record(source_name: str):
            def _ingest(path: Path) -> int:
                seen.append((source_name, str(path)))
                return 1

            return _ingest

        watcher = AgentSessionWatcher(
            registry_path=tmp_path / "agent-registry.json",
            sources=[
                AgentSessionSource("codex_cli", ["**/*.jsonl"], _record("codex_cli"), root=tmp_path / "codex" / "sessions"),
                AgentSessionSource(
                    "cursor",
                    ["**/*.jsonl"],
                    _record("cursor"),
                    root=tmp_path / "cursor" / "projects",
                ),
                AgentSessionSource("gemini", ["**/session-*.json"], _record("gemini"), root=tmp_path / "gemini" / "tmp"),
            ],
        )

        assert watcher.poll_once() == 3
        assert len(seen) == 3

        assert watcher.poll_once() == 0
        assert len(seen) == 3

        _append_jsonl(codex_file, {"type": "event_msg", "payload": {"type": "user_message"}})
        assert watcher.poll_once() == 1
        assert seen[-1][0] == "codex_cli"
