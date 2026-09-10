from brainlayer.native_targets import MAX_CALLS, extract_native_targets


def claude(*blocks):
    return {"type": "assistant", "message": {"role": "assistant", "content": list(blocks)}}


def test_claude_supported_target_preserves_block_pointer_and_id():
    report = extract_native_targets(claude({"type": "tool_use", "id": "call-1", "name": "mcp__cmuxlayer__send_to", "input": {"surface": "surface-7"}}))
    assert report.diagnostics == []
    assert report.targets == [{
        "tool_name": "mcp__cmuxlayer__send_to", "target": "surface-7", "target_kind": "surface",
        "declared_unverified": True, "call_id": "call-1",
        "raw_container_pointer": None, "decoded_pointer": "/message/content/0/input/surface",
        "pointer_kind": "direct_raw_json",
    }]


def test_codex_arguments_keep_raw_and_decoded_pointers_without_prompt_copy():
    entry = {"type": "response_item", "payload": {"type": "function_call", "name": "mcp__codex_app__send_message_to_thread", "call_id": "fc-2", "arguments": '{"threadId":"task-9","prompt":"private body"}'}}
    report = extract_native_targets(entry)
    assert report.targets[0]["target"] == "task-9"
    assert report.targets[0]["raw_container_pointer"] == "/payload/arguments"
    assert report.targets[0]["decoded_pointer"] == "/threadId"
    assert report.targets[0]["pointer_kind"] == "decoded_json_string"
    assert "private body" not in repr(report)


def test_multiple_calls_and_decoys_are_separate_and_prose_is_ignored():
    report = extract_native_targets(claude(
        {"type": "text", "text": "mcp__cmuxlayer__send_to agent-nope"},
        {"type": "tool_use", "name": "mcp__cmuxlayer__send_to", "input": {"agent_id": "agent-a"}},
        {"type": "tool_use", "name": "collaboration.send_message", "input": {"target": "agent-b"}},
        {"type": "tool_result", "content": [{"name": "mcp__cmuxlayer__send_to", "input": {"agent_id": "fake"}}]},
    ))
    assert [target["target"] for target in report.targets] == ["agent-a", "agent-b"]


def test_ambiguous_missing_invalid_and_oversize_are_diagnostics():
    report = extract_native_targets(claude(
        {"type": "tool_use", "name": "mcp__cmuxlayer__send_to", "input": {"agent_id": "a", "surface": "b"}},
        {"type": "tool_use", "name": "collaboration.send_message", "input": {}},
        {"type": "tool_use", "name": "mcp__codex_app__send_message_to_thread", "input": {"threadId": 3}},
    ))
    assert report.targets == []
    assert any("ambiguous" in item for item in report.diagnostics)
    assert any("missing target" in item for item in report.diagnostics)


def test_invalid_and_oversize_codex_arguments_are_omitted():
    bad = {"type": "response_item", "payload": {"type": "function_call", "name": "collaboration.send_message", "arguments": "[]"}}
    huge = {"type": "response_item", "payload": {"type": "function_call", "name": "collaboration.send_message", "arguments": "x" * 70000}}
    report = extract_native_targets(bad)
    report2 = extract_native_targets(huge)
    assert report.targets == [] and any("not an object" in item for item in report.diagnostics)
    assert any("byte cap" in item for item in report2.diagnostics)


def test_call_cap_is_explicit():
    blocks = [{"type": "tool_use", "name": "mcp__cmuxlayer__send_to", "input": {"agent_id": str(i)}} for i in range(MAX_CALLS + 2)]
    report = extract_native_targets(claude(*blocks))
    assert len(report.targets) == MAX_CALLS
    assert any("beyond cap" in item for item in report.diagnostics)



def test_equal_namespace_fields_are_ambiguous_and_limits_are_explicit():
    report = extract_native_targets(claude(
        {"type": "tool_use", "name": "mcp__cmuxlayer__send_to", "input": {"agent_id": "same", "surface": "same"}},
        {"type": "tool_use", "id": "x" * 5000, "name": "mcp__cmuxlayer__send_to", "input": {"agent_id": "a"}},
        {"type": "tool_use", "name": "mcp__cmuxlayer__send_to", "input": {"agent_id": "x" * 5000}},
    ))
    assert report.targets == []
    assert any("ambiguous" in item for item in report.diagnostics)
    assert any("call ID" in item for item in report.diagnostics)
    assert any("target" in item and "oversized" in item for item in report.diagnostics)


def test_claude_structured_argument_cap_and_malformed_tool_name_are_safe():
    huge_input = {"agent_id": "a", "padding": "x" * 70000}
    report = extract_native_targets(claude(
        {"type": "tool_use", "name": ["mcp__cmuxlayer__send_to"], "input": {"agent_id": "ignored"}},
        {"type": "tool_use", "name": "mcp__cmuxlayer__send_to", "input": huge_input},
        {"type": "tool_use", "name": "functions.exec", "input": {"agent_id": "quoted-only"}},
        {"type": "text", "text": "functions.exec mcp__cmuxlayer__send_to agent-prose"},
    ))
    assert report.targets == []
    assert any("arguments exceed" in item for item in report.diagnostics)


def test_diagnostics_are_bounded_during_scan():
    blocks = [{"type": "tool_use", "name": "mcp__cmuxlayer__send_to", "input": {}} for _ in range(100)]
    report = extract_native_targets(claude(*blocks))
    assert len(report.diagnostics) <= 32
    assert any("beyond cap" in item for item in report.diagnostics)


def codex(arguments):
    return {"type": "response_item", "payload": {"type": "function_call", "name": "mcp__codex_app__send_message_to_thread", "arguments": arguments}}


def test_codex_rejects_duplicate_keys_and_nonfinite_constants():
    duplicate = extract_native_targets(codex('{"threadId":"first","threadId":"second"}'))
    nonfinite = extract_native_targets(codex('{"threadId":NaN}'))
    assert duplicate.targets == [] and any("invalid arguments JSON" in item for item in duplicate.diagnostics)
    assert nonfinite.targets == [] and any("invalid arguments JSON" in item for item in nonfinite.diagnostics)


def test_codex_rejects_surrogate_and_deep_arguments_without_raising():
    surrogate = extract_native_targets(codex('{"threadId":"\\ud800"}'))
    deep = "{" * 140 + '"threadId":"task"' + "}" * 140
    deep_report = extract_native_targets(codex(deep))
    assert surrogate.targets == [] and any("invalid or oversized structured arguments" in item for item in surrogate.diagnostics)
    assert deep_report.targets == [] and any("invalid arguments JSON" in item for item in deep_report.diagnostics)


def test_malformed_call_id_and_duplicate_key_diagnostic_stay_bounded():
    surrogate_id = extract_native_targets(claude({"type": "tool_use", "id": "\ud800", "name": "mcp__cmuxlayer__send_to", "input": {"agent_id": "a"}}))
    duplicate_key = "{\"threadId\":\"" + ("x" * 20000) + "\",\"threadId\":\"last\"}"
    duplicate = extract_native_targets(codex(duplicate_key))
    assert surrogate_id.targets == [] and any("call ID" in item for item in surrogate_id.diagnostics)
    assert duplicate.targets == []
    assert all(len(item.encode("utf-8")) < 512 for item in duplicate.diagnostics)


def test_structured_size_bound_counts_escaped_keys_and_huge_integers():
    escaped_key = '"' * 50000
    escaped = extract_native_targets(claude({"type": "tool_use", "name": "mcp__cmuxlayer__send_to", "input": {"agent_id": "a", escaped_key: "v"}}))
    huge_integer = extract_native_targets(claude({"type": "tool_use", "name": "mcp__cmuxlayer__send_to", "input": {"agent_id": "a", "count": 10**100000}}))
    assert escaped.targets == [] and any("structured arguments" in item for item in escaped.diagnostics)
    assert huge_integer.targets == [] and any("structured arguments" in item for item in huge_integer.diagnostics)
