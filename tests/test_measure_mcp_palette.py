"""Tests for deterministic MCP palette measurement."""

import json

import tiktoken

from scripts.measure_mcp_palette import extract_tools, main, measure_tools


def test_measure_tools_uses_compact_utf8_json_and_o200k_tokens():
    tools = [
        {
            "name": "memory_זיכרון",
            "inputSchema": {"properties": {}, "type": "object"},
        }
    ]
    canonical = json.dumps(tools, ensure_ascii=False, separators=(",", ":"), sort_keys=True)

    measurement = measure_tools(tools)

    assert measurement == {
        "tools": 1,
        "bytes": len(canonical.encode("utf-8")),
        "tokens": len(tiktoken.get_encoding("o200k_base").encode(canonical)),
    }


def test_extract_tools_accepts_jsonrpc_envelope():
    tools = [{"name": "brain_search", "inputSchema": {"type": "object"}}]

    assert extract_tools({"jsonrpc": "2.0", "id": 1, "result": {"tools": tools}}) == tools


def test_main_enforces_max_bytes(tmp_path, capsys):
    payload_path = tmp_path / "tools.json"
    payload_path.write_text(json.dumps({"tools": [{"name": "brain_search"}]}), encoding="utf-8")

    assert main([str(payload_path), "--max-bytes", "1"]) == 1
    output = json.loads(capsys.readouterr().out)
    assert output["bytes"] > 1

    assert main([str(payload_path), "--max-bytes", str(output["bytes"])]) == 0
