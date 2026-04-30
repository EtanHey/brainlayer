"""Tests for the Cursor session ingestion adapter."""

import json
from pathlib import Path

from brainlayer.ingest.cursor import parse_cursor_session


def _write_jsonl(path: Path, lines: list[dict]) -> None:
    with open(path, "w") as f:
        for line in lines:
            f.write(json.dumps(line) + "\n")


def _cursor_user(text: str) -> dict:
    return {
        "role": "user",
        "message": {
            "content": [{"type": "text", "text": text}],
        },
    }


def _cursor_assistant(text: str) -> dict:
    return {
        "role": "assistant",
        "message": {
            "content": [{"type": "text", "text": text}],
        },
    }


class TestParseCursorSession:
    def test_extracts_user_query_and_assistant_reply(self, tmp_path):
        session_dir = tmp_path / ".cursor" / "projects" / "brainlayer" / "agent-transcripts" / "abc123"
        session_dir.mkdir(parents=True)
        session_file = session_dir / "abc123.jsonl"
        _write_jsonl(
            session_file,
            [
                _cursor_user("<user_query>\nWire the missing cursor ingestion adapter.\n</user_query>"),
                _cursor_assistant(
                    "I will inspect the parser wiring and add the missing adapter with tests."
                ),
            ],
        )

        entries = list(parse_cursor_session(session_file))

        assert len(entries) == 2
        assert entries[0]["content_type"] == "user_message"
        assert entries[0]["content"] == "Wire the missing cursor ingestion adapter."
        assert entries[0]["source"] == "cursor"
        assert entries[0]["project"] == "brainlayer"
        assert entries[0]["session_id"] == "abc123"

        assert entries[1]["content_type"] == "assistant_text"
        assert entries[1]["source"] == "cursor"
        assert entries[1]["project"] == "brainlayer"

    def test_classifies_code_block_as_ai_code(self, tmp_path):
        session_dir = tmp_path / ".cursor" / "projects" / "brainlayer" / "agent-transcripts" / "code-session"
        session_dir.mkdir(parents=True)
        session_file = session_dir / "code-session.jsonl"
        _write_jsonl(
            session_file,
            [
                _cursor_assistant(
                    "Here is the parser update:\n```python\n"
                    "def parse_cursor_session(path):\n    return []\n```\n"
                )
            ],
        )

        entries = list(parse_cursor_session(session_file))

        assert len(entries) == 1
        assert entries[0]["content_type"] == "ai_code"

    def test_skips_empty_and_short_messages(self, tmp_path):
        session_dir = tmp_path / ".cursor" / "projects" / "brainlayer" / "agent-transcripts" / "skip-session"
        session_dir.mkdir(parents=True)
        session_file = session_dir / "skip-session.jsonl"
        _write_jsonl(
            session_file,
            [
                _cursor_user("ok"),
                _cursor_assistant("short"),
            ],
        )

        assert list(parse_cursor_session(session_file)) == []
