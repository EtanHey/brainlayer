"""Tests for the Codex CLI session ingestion adapter."""

import json
from pathlib import Path

from brainlayer.ingest.codex import (
    _classify_tool_output,
    _extract_input_text,
    _extract_output_text,
    _extract_tool_output,
    _infer_project_from_cwd,
    _is_system_injection,
    parse_codex_session,
)

# ---------------------------------------------------------------------------
# Fixtures — minimal Codex JSONL session files
# ---------------------------------------------------------------------------


def _write_jsonl(path: Path, lines: list) -> None:
    with open(path, "w") as f:
        for line in lines:
            f.write(json.dumps(line) + "\n")


def _session_meta(session_id="test-uuid", cwd="/Users/test/Gits/myproject"):
    return {
        "timestamp": "2026-03-14T12:00:00.000Z",
        "type": "session_meta",
        "payload": {
            "id": session_id,
            "timestamp": "2026-03-14T12:00:00.000Z",
            "cwd": cwd,
            "originator": "codex_cli_rs",
            "cli_version": "0.114.0",
            "source": "cli",
            "model_provider": "openai",
        },
    }


def _response_item(payload, ts="2026-03-14T12:01:00.000Z"):
    return {"timestamp": ts, "type": "response_item", "payload": payload}


def _user_msg(text: str):
    return _response_item(
        {
            "type": "message",
            "role": "user",
            "content": [{"type": "input_text", "text": text}],
        }
    )


def _developer_msg(text: str):
    return _response_item(
        {
            "type": "message",
            "role": "developer",
            "content": [{"type": "input_text", "text": text}],
        }
    )


def _assistant_msg(text: str):
    return _response_item(
        {
            "type": "message",
            "role": "assistant",
            "content": [{"type": "output_text", "text": text}],
        }
    )


def _tool_output(text: str, call_id="call_123"):
    return _response_item(
        {
            "type": "function_call_output",
            "call_id": call_id,
            "output": text,
        }
    )


def _function_call(name: str, args: str):
    return _response_item(
        {
            "type": "function_call",
            "name": name,
            "arguments": args,
            "call_id": "call_xyz",
        }
    )


def _reasoning():
    return _response_item(
        {
            "type": "reasoning",
            "summary": [],
            "content": None,
            "encrypted_content": "gAAAA...",
        }
    )


# ---------------------------------------------------------------------------
# Unit tests — helper functions
# ---------------------------------------------------------------------------


class TestExtractInputText:
    def test_extracts_plain_text(self):
        blocks = [{"type": "input_text", "text": "Hello world"}]
        assert _extract_input_text(blocks) == "Hello world"

    def test_skips_non_input_text_blocks(self):
        blocks = [
            {"type": "output_text", "text": "should be skipped"},
            {"type": "input_text", "text": "kept"},
        ]
        assert _extract_input_text(blocks) == "kept"

    def test_empty_blocks(self):
        assert _extract_input_text([]) == ""

    def test_none_blocks(self):
        assert _extract_input_text(None) == ""


class TestExtractOutputText:
    def test_extracts_output_text(self):
        blocks = [{"type": "output_text", "text": "Assistant reply"}]
        assert _extract_output_text(blocks) == "Assistant reply"

    def test_skips_input_text_blocks(self):
        blocks = [
            {"type": "input_text", "text": "should be skipped"},
            {"type": "output_text", "text": "kept"},
        ]
        assert _extract_output_text(blocks) == "kept"


class TestExtractToolOutput:
    def test_plain_string(self):
        assert _extract_tool_output("plain output") == "plain output"

    def test_json_text_blocks(self):
        payload = json.dumps([{"type": "text", "text": "parsed output"}])
        assert _extract_tool_output(payload) == "parsed output"

    def test_invalid_json_returns_raw(self):
        raw = "[not valid json"
        assert _extract_tool_output(raw) == raw

    def test_empty(self):
        assert _extract_tool_output("") == ""

    def test_none(self):
        assert _extract_tool_output(None) == ""


class TestIsSystemInjection:
    def test_agents_md(self):
        assert _is_system_injection("# AGENTS.md instructions for /Users/test/Gits/foo\nsome content")

    def test_environment_context(self):
        assert _is_system_injection("<environment_context>\n<cwd>/foo</cwd>\n</environment_context>")

    def test_permissions_instructions(self):
        assert _is_system_injection("<permissions instructions>\nsome perms\n</permissions instructions>")

    def test_collaboration_mode(self):
        assert _is_system_injection("<collaboration_mode>\nDefault mode\n</collaboration_mode>")

    def test_real_user_prompt_not_filtered(self):
        assert not _is_system_injection("Fix the failing tests in auth.py")

    def test_long_real_prompt_not_filtered(self):
        prompt = "Please implement a new feature: " + "x" * 200
        assert not _is_system_injection(prompt)


class TestInferProjectFromCwd:
    def test_standard_gits_path(self):
        assert _infer_project_from_cwd("/Users/etan/Gits/golems") == "golems"

    def test_nested_path(self):
        assert _infer_project_from_cwd("/Users/etan/Gits/brainlayer/src") == "src"

    def test_none(self):
        assert _infer_project_from_cwd(None) is None

    def test_empty_string(self):
        assert _infer_project_from_cwd("") is None


class TestClassifyToolOutput:
    def test_stack_trace(self):
        text = "Traceback (most recent call last):\n  File 'foo.py', line 10\nAssertionError"
        assert _classify_tool_output(text) == "stack_trace"

    def test_build_log(self):
        text = "  3 passing\n  1 failing"
        assert _classify_tool_output(text) == "build_log"

    def test_git_diff(self):
        text = "diff --git a/foo.py b/foo.py\n@@ -1,3 +1,4 @@\n-old\n+new"
        assert _classify_tool_output(text) == "git_diff"

    def test_generic_file_read(self):
        text = "Some generic tool output without special patterns"
        assert _classify_tool_output(text) == "file_read"


# ---------------------------------------------------------------------------
# Integration tests — parse_codex_session
# ---------------------------------------------------------------------------


class TestParseCodexSession:
    def test_real_user_message_is_parsed(self, tmp_path):
        f = tmp_path / "session.jsonl"
        _write_jsonl(
            f,
            [
                _session_meta(),
                _user_msg("Fix the authentication bug in auth.py"),
            ],
        )
        entries = list(parse_codex_session(f))
        assert len(entries) == 1
        assert entries[0]["content_type"] == "user_message"
        assert "Fix the authentication bug" in entries[0]["content"]
        assert entries[0]["source"] == "codex_cli"
        assert entries[0]["project"] == "myproject"

    def test_developer_messages_skipped(self, tmp_path):
        f = tmp_path / "session.jsonl"
        _write_jsonl(
            f,
            [
                _session_meta(),
                _developer_msg("You are a coding agent. sandbox_mode is danger-full-access."),
            ],
        )
        entries = list(parse_codex_session(f))
        assert entries == []

    def test_system_injections_skipped(self, tmp_path):
        f = tmp_path / "session.jsonl"
        _write_jsonl(
            f,
            [
                _session_meta(),
                _user_msg("# AGENTS.md instructions for /Users/test/Gits/myproject\n\nYou are Codex..."),
            ],
        )
        entries = list(parse_codex_session(f))
        assert entries == []

    def test_environment_context_skipped(self, tmp_path):
        f = tmp_path / "session.jsonl"
        _write_jsonl(
            f,
            [
                _session_meta(),
                _user_msg("<environment_context>\n<cwd>/Users/test/Gits/myproject</cwd>\n</environment_context>"),
            ],
        )
        entries = list(parse_codex_session(f))
        assert entries == []

    def test_assistant_message_parsed(self, tmp_path):
        f = tmp_path / "session.jsonl"
        _write_jsonl(
            f,
            [
                _session_meta(),
                _assistant_msg(
                    "I found the bug in the authentication middleware. "
                    "The token expiry check was comparing timestamps incorrectly."
                ),
            ],
        )
        entries = list(parse_codex_session(f))
        assert len(entries) == 1
        assert entries[0]["content_type"] == "assistant_text"
        assert entries[0]["source"] == "codex_cli"

    def test_assistant_message_with_code_is_ai_code(self, tmp_path):
        f = tmp_path / "session.jsonl"
        _write_jsonl(
            f,
            [
                _session_meta(),
                _assistant_msg("Here's the fix:\n```python\ndef check_token(t):\n    return t.is_valid()\n```"),
            ],
        )
        entries = list(parse_codex_session(f))
        assert len(entries) == 1
        assert entries[0]["content_type"] == "ai_code"

    def test_tool_output_parsed(self, tmp_path):
        f = tmp_path / "session.jsonl"
        tool_out = "Process exited with code 0\n" + "some output " * 10
        _write_jsonl(
            f,
            [
                _session_meta(),
                _tool_output(tool_out),
            ],
        )
        entries = list(parse_codex_session(f))
        assert len(entries) == 1
        assert entries[0]["content_type"] == "file_read"

    def test_tool_json_output_decoded(self, tmp_path):
        f = tmp_path / "session.jsonl"
        text = "some tool result text that is definitely long enough to pass the minimum threshold"
        raw = json.dumps([{"type": "text", "text": text}])
        _write_jsonl(
            f,
            [
                _session_meta(),
                _tool_output(raw),
            ],
        )
        entries = list(parse_codex_session(f))
        assert len(entries) == 1
        assert "some tool result text" in entries[0]["content"]

    def test_function_call_skipped(self, tmp_path):
        f = tmp_path / "session.jsonl"
        _write_jsonl(
            f,
            [
                _session_meta(),
                _function_call("brain_search", '{"query": "auth bug"}'),
            ],
        )
        entries = list(parse_codex_session(f))
        assert entries == []

    def test_reasoning_skipped(self, tmp_path):
        f = tmp_path / "session.jsonl"
        _write_jsonl(
            f,
            [
                _session_meta(),
                _reasoning(),
            ],
        )
        entries = list(parse_codex_session(f))
        assert entries == []

    def test_short_user_message_filtered(self, tmp_path):
        f = tmp_path / "session.jsonl"
        _write_jsonl(
            f,
            [
                _session_meta(),
                _user_msg("ok"),  # too short
            ],
        )
        entries = list(parse_codex_session(f))
        assert entries == []

    def test_session_metadata_propagated(self, tmp_path):
        f = tmp_path / "session.jsonl"
        _write_jsonl(
            f,
            [
                _session_meta(session_id="abc-123", cwd="/Users/test/Gits/golems"),
                _user_msg("Implement the new feature for the dashboard"),
            ],
        )
        entries = list(parse_codex_session(f))
        assert len(entries) == 1
        e = entries[0]
        assert e["session_id"] == "abc-123"
        assert e["project"] == "golems"
        assert e["metadata"]["session_id"] == "abc-123"

    def test_stack_trace_in_tool_output(self, tmp_path):
        f = tmp_path / "session.jsonl"
        trace = (
            "Traceback (most recent call last):\n"
            '  File "auth.py", line 42, in check_token\n'
            "AssertionError: token expired\n"
        ) * 3  # ensure long enough
        _write_jsonl(
            f,
            [
                _session_meta(),
                _tool_output(trace),
            ],
        )
        entries = list(parse_codex_session(f))
        assert len(entries) == 1
        assert entries[0]["content_type"] == "stack_trace"

    def test_mixed_session(self, tmp_path):
        """Full session with system noise + real content."""
        f = tmp_path / "session.jsonl"
        _write_jsonl(
            f,
            [
                _session_meta(session_id="full-session", cwd="/Users/test/Gits/brainlayer"),
                _developer_msg("sandbox_mode instructions ..."),
                _user_msg("# AGENTS.md instructions for /Users/test/Gits/brainlayer\n..."),
                _user_msg("<environment_context>\n<cwd>/Users/test/Gits/brainlayer</cwd>\n</environment_context>"),
                _reasoning(),
                _user_msg("Add a test for the codex ingestion adapter"),
                _function_call("brain_search", '{"query": "codex adapter tests"}'),
                _tool_output("No results found in BrainLayer for this query. " * 3),
                _assistant_msg(
                    "I'll write the test. The key cases are: system injection filtering, "
                    "assistant message classification, and tool output parsing."
                ),
                _assistant_msg("Here's the implementation:\n```python\ndef test_codex():\n    pass\n```"),
            ],
        )
        entries = list(parse_codex_session(f))
        types = [e["content_type"] for e in entries]
        assert "user_message" in types
        assert "assistant_text" in types
        assert "ai_code" in types
        assert "file_read" in types
        # developer, system injections, reasoning, function_call → all filtered
        for e in entries:
            assert e["source"] == "codex_cli"
            assert e["project"] == "brainlayer"
