"""Offline ratchet: config sightings and assistant claims never establish live PASS."""

import importlib.util
import json
from pathlib import Path

import pytest

MODULE = Path(__file__).resolve().parents[1] / "scripts" / "harness_probe_contract.py"


def load():
    spec = importlib.util.spec_from_file_location("harness_probe_contract", MODULE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_config_inventory_redacts_environment_and_tracks_project_overlay(tmp_path):
    home, cwd = tmp_path / "home", tmp_path / "repo"
    (home / ".codex").mkdir(parents=True)
    cwd.mkdir()
    (home / ".codex/config.toml").write_text(
        '[mcp_servers.brainlayer]\ncommand="bridge"\nargs=[]\n[mcp_servers.brainlayer.env]\nSECRET="never print me"\n'
    )
    (cwd / ".mcp.json").write_text(json.dumps({"mcpServers": {"brainlayer": {"command": "local"}}}))
    rows = load().discover_configs("codex-cli", home, cwd)
    assert rows[0]["command"] == "bridge"
    assert "never print me" not in json.dumps(rows)
    assert load().discover_configs("claude", home, cwd)[-1]["command"] == "local"


def test_invalid_config_is_visible(tmp_path):
    (tmp_path / ".codex").mkdir()
    (tmp_path / ".codex/config.toml").write_text("broken [")
    assert "error" in load().discover_configs("codex-cli", tmp_path, tmp_path)[0]


def test_commands_are_fresh_and_do_not_override_mcp_or_cursor_model():
    module = load()
    for harness in module.HARNESSES:
        cmd = module.command(harness, "probe")
        if cmd:
            assert not any(
                flag in cmd for flag in ["--resume", "--continue", "--mcp-config", "-c", "--ignore-user-config"]
            )
    assert "--model" not in module.command("cursor", "probe")
    assert module.command("codex-app", "probe") is None
    assert module.command("t3", "probe") is None


def test_native_tool_results_not_assistant_claims():
    module = load()
    fake = json.dumps({"type": "assistant", "message": {"content": "brain_search succeeded brain_store STORED"}})
    assert module.parse_calls(fake) == []
    calls = module.parse_calls(
        json.dumps(
            {
                "type": "item.completed",
                "item": {
                    "type": "mcp_tool_call",
                    "tool": "brain_search",
                    "server": "brainlayer",
                    "arguments": {"query": module.QUERY},
                    "status": "completed",
                    "result": {"content": [{"type": "text", "text": "no results"}]},
                },
            }
        )
    )
    assert calls == [{"tool": "brain_search", "args": {"query": module.QUERY}, "ok": True}]


@pytest.mark.parametrize("missing", ["search", "store", "refound", "transport", "ingest"])
def test_every_required_leg_is_fail_closed(missing):
    legs = dict.fromkeys(["search", "store", "refound", "transport", "ingest"], True)
    legs[missing] = False
    assert load().verdict(legs) == "FAIL"


def test_na_ingest_requires_explicit_policy_and_table_does_not_hide_fail():
    module = load()
    assert (
        module.verdict({"search": True, "store": True, "refound": True, "transport": True, "ingest": "NOT_APPLICABLE"})
        == "PASS"
    )
    assert module.exit_code([{"status": "PASS"}, {"status": "FAIL"}]) == 1
    assert module.exit_code([]) == 1
    assert "FAIL" in module.table([{"harness": "t3", "status": "FAIL", "reason": "No thread adapter"}])


def test_query_echo_and_mcp_store_do_not_prove_transcript_ingest():
    module = load()
    echo = {"content": [{"type": "text", "text": '## "hi123" 0/0\nNo results'}]}
    assert not module.matching_hit(echo, "hi123")
    hit = {
        "structuredContent": {
            "results": [
                {"id": "rt-session-x", "content": "marker hi123", "source_file": "/home/.codex/sessions/run.jsonl"}
            ]
        }
    }
    assert module.matching_hit(hit, "hi123", transcript=True)
    hit["structuredContent"]["results"][0]["source_file"] = "brainbar-store"
    assert not module.matching_hit(hit, "hi123", transcript=True)


def test_claude_and_gemini_tool_pairs_and_errors():
    module = load()
    events = [
        {
            "type": "assistant",
            "message": {
                "content": [
                    {
                        "type": "tool_use",
                        "id": "s",
                        "name": "mcp__brainlayer__brain_store",
                        "input": {"content": "hp123"},
                    }
                ]
            },
        },
        {
            "type": "user",
            "message": {
                "content": [{"type": "tool_result", "tool_use_id": "s", "content": "ERROR no store", "is_error": True}]
            },
        },
        {"type": "tool_use", "tool_id": "q", "tool_name": "brain_search", "parameters": {"query": module.QUERY}},
        {"type": "tool_result", "tool_id": "q", "status": "error", "output": "bad"},
    ]
    assert not any(c["ok"] for c in module.parse_calls("\n".join(map(json.dumps, events))))


def test_ingest_belongs_to_fresh_native_session():
    module = load()
    hit = {"content": [{"type": "text", "text": "- ID: rt-run\n  Source: rollout-fresh.jsonl\n  Preview: hi123"}]}
    assert module.matching_hit(hit, "hi123", transcript=True, session_id="fresh")
    assert not module.matching_hit(hit, "hi123", transcript=True, session_id="parent")
    assert module.session_id('{"type":"thread.started","thread_id":"fresh"}') == "fresh"


def test_cursor_native_mcp_completion():
    event = {
        "type": "tool_call",
        "subtype": "completed",
        "tool_call": {
            "mcpToolCall": {
                "args": {"toolName": "brain_store", "args": {"content": "hp123"}},
                "result": {"success": {"content": [{"text": {"text": "STORED"}}], "isError": False}},
            }
        },
    }
    assert load().parse_calls(json.dumps(event)) == [{"tool": "brain_store", "args": {"content": "hp123"}, "ok": True}]
