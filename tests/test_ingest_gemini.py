"""Tests for the Gemini session ingestion adapter."""

import json
from pathlib import Path

from brainlayer.ingest.gemini import parse_gemini_session


def _write_json(path: Path, payload: dict) -> None:
    with open(path, "w") as f:
        json.dump(payload, f)


class TestParseGeminiSession:
    def test_extracts_user_and_gemini_messages(self, tmp_path):
        session_dir = tmp_path / ".gemini" / "tmp" / "brainlayer" / "chats"
        session_dir.mkdir(parents=True)
        session_file = session_dir / "session-2026-04-18T02-02-test.json"
        _write_json(
            session_file,
            {
                "sessionId": "gem-session-1",
                "startTime": "2026-04-18T02:02:10.184Z",
                "messages": [
                    {"id": "info-1", "timestamp": "2026-04-18T02:02:10.300Z", "type": "info", "content": "skip me"},
                    {
                        "id": "user-1",
                        "timestamp": "2026-04-18T02:02:11.000Z",
                        "type": "user",
                        "content": [{"text": "Backfill the missing Gemini sessions into BrainLayer."}],
                    },
                    {
                        "id": "assistant-1",
                        "timestamp": "2026-04-18T02:02:12.000Z",
                        "type": "gemini",
                        "content": "I will inspect the session format and backfill in batches.",
                    },
                ],
            },
        )

        entries = list(parse_gemini_session(session_file))

        assert len(entries) == 2
        assert entries[0]["content_type"] == "user_message"
        assert entries[0]["source"] == "gemini"
        assert entries[0]["project"] == "brainlayer"
        assert entries[0]["session_id"] == "gem-session-1"
        assert entries[0]["timestamp"] == "2026-04-18T02:02:11.000Z"

        assert entries[1]["content_type"] == "assistant_text"
        assert entries[1]["source"] == "gemini"
        assert entries[1]["project"] == "brainlayer"

    def test_classifies_code_block_as_ai_code(self, tmp_path):
        session_dir = tmp_path / ".gemini" / "tmp" / "brainlayer" / "chats"
        session_dir.mkdir(parents=True)
        session_file = session_dir / "session-code.json"
        _write_json(
            session_file,
            {
                "sessionId": "gem-session-2",
                "startTime": "2026-04-18T02:02:10.184Z",
                "messages": [
                    {
                        "id": "assistant-1",
                        "timestamp": "2026-04-18T02:02:12.000Z",
                        "type": "gemini",
                        "content": "```python\nprint('hello')\n```",
                    }
                ],
            },
        )

        entries = list(parse_gemini_session(session_file))

        assert len(entries) == 1
        assert entries[0]["content_type"] == "ai_code"
