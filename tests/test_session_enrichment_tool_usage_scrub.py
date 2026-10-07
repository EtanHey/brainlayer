"""``tool_usage_stats`` is LLM output and is scrubbed before it is persisted.

The session-analysis model writes ``tool_usage_stats`` as a free-form list of
dicts, and ``parse_session_enrichment`` passes the items through unvalidated.
A model that copies a token into a tool name, a label, a nested value or a
dict KEY must not put it into ``session_enrichments`` at rest.

Tokens are assembled at runtime from obviously fake characters, so no
provider-shaped literal lives in this file.
"""

from __future__ import annotations

import json

import pytest

GROQ = "gsk_" + "0" * 52
GITHUB = "ghp_" + "0" * 36
OPENAI = "sk-" + "0" * 40
TOKENS = {"groq": GROQ, "github": GITHUB, "openai": OPENAI}


def _assert_no_token(blob: str, *, where: str) -> None:
    leaked = [name for name, token in TOKENS.items() if token in blob]
    assert not leaked, f"{where} carried unredacted synthetic token(s): {leaked}"


def _every_decoding(value, *, layers: int = 8):
    """Yield ``value`` and every string any reader could reach by JSON-decoding it again."""
    if isinstance(value, dict):
        for key, item in value.items():
            yield from _every_decoding(key, layers=layers)
            yield from _every_decoding(item, layers=layers)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _every_decoding(item, layers=layers)
    elif isinstance(value, str):
        yield value
        if layers:
            try:
                decoded = json.loads(value)
            except ValueError:
                return
            yield from _every_decoding(decoded, layers=layers - 1)


def _assert_not_recoverable(value, *, where: str) -> None:
    for text in _every_decoding(value):
        _assert_no_token(text, where=where)


def _persist(tmp_path, **fields):
    from brainlayer.vector_store import VectorStore

    store = VectorStore(tmp_path / "session.db")
    try:
        store.upsert_session_enrichment({"session_id": "s-1", "session_summary": "a plain summary", **fields})
        raw = list(store.conn.cursor().execute("SELECT * FROM session_enrichments WHERE session_id = 's-1'"))
        record = store.get_session_enrichment("s-1")
    finally:
        store.close()
    _assert_no_token(json.dumps([list(map(str, row)) for row in raw]), where="session_enrichments row")
    _assert_not_recoverable([[c for c in row if isinstance(c, str)] for row in raw], where="session_enrichments row")
    _assert_no_token(json.dumps(record, default=str), where="session enrichment record")
    return record


def test_tool_usage_stats_list_of_dicts_is_persisted_redacted(tmp_path):
    stats = [
        {"tool": f"Bash {GROQ}", "count": 3},
        {"tool": "Read", "count": 7, "label": GITHUB},
    ]

    record = _persist(tmp_path, tool_usage_stats=stats)

    stored = record["tool_usage_stats"]
    assert [sorted(item) for item in stored] == [["count", "tool"], ["count", "label", "tool"]]
    assert [item["count"] for item in stored] == [3, 7]
    assert stored[0]["tool"].startswith("Bash [REDACTED:")
    assert stored[1]["tool"] == "Read"
    assert stored[1]["label"].startswith("[REDACTED:")


def test_tool_usage_stats_json_string_is_persisted_redacted(tmp_path):
    stats = [{"tool": f"Edit {OPENAI}", "count": 2}]

    record = _persist(tmp_path, tool_usage_stats=json.dumps(stats))

    assert record["tool_usage_stats"][0]["count"] == 2
    assert record["tool_usage_stats"][0]["tool"].startswith("Edit [REDACTED:")


def test_tool_usage_stats_json_string_with_escaped_token_is_redacted(tmp_path):
    # JSON lets a model write any character as \\uXXXX. Scrubbing the raw text
    # would miss the escaped form; the decoded value is the token.
    escaped = "\\u" + format(ord(GROQ[0]), "04x") + GROQ[1:]
    raw = '[{"tool": "' + escaped + '", "count": 1}]'
    assert GROQ not in raw and json.loads(raw)[0]["tool"] == GROQ

    record = _persist(tmp_path, tool_usage_stats=raw)

    assert record["tool_usage_stats"] == [{"tool": "[REDACTED:groq]", "count": 1}]


def test_json_field_string_forms_round_trip_after_scrub(tmp_path):
    # A JSON string that decodes to a bare string stays valid JSON; text that is
    # not JSON is scrubbed and stored as the text it is.
    from brainlayer.vector_store import VectorStore

    record = _persist(tmp_path, tool_usage_stats=json.dumps(f"Bash {GROQ}"), patterns=f"not json {GITHUB}")

    assert record["tool_usage_stats"] == "Bash [REDACTED:groq]"
    store = VectorStore(tmp_path / "session.db")
    try:
        (raw_patterns,) = list(store.conn.cursor().execute("SELECT patterns FROM session_enrichments"))[0]
    finally:
        store.close()
    assert raw_patterns == "not json [REDACTED:github]"


def test_tool_usage_stats_nested_values_and_keys_are_redacted(tmp_path):
    stats = [
        {
            "tool": "Bash",
            "count": 2,
            "args": {"env": [GROQ, "plain"], "meta": {"depth": 1, "flag": True, "note": None, "ratio": 0.5}},
        },
        {OPENAI: 4},
    ]

    record = _persist(tmp_path, tool_usage_stats=stats)

    first, second = record["tool_usage_stats"]
    assert first["count"] == 2
    assert first["args"]["env"] == ["[REDACTED:groq]", "plain"]
    assert first["args"]["meta"] == {"depth": 1, "flag": True, "note": None, "ratio": 0.5}
    assert list(second.values()) == [4]
    assert list(second)[0].startswith("[REDACTED:")


def test_model_authored_dict_keys_in_other_json_fields_are_redacted(tmp_path):
    # Same class: decisions_made / corrections items are model-written dicts too.
    record = _persist(
        tmp_path,
        decisions_made=[{"decision": "ship it", GITHUB: "why"}],
        corrections=[{GROQ: {"nested": OPENAI}}],
    )

    assert record["decisions_made"][0]["decision"] == "ship it"
    assert "why" in record["decisions_made"][0].values()
    assert record["corrections"][0] == {"[REDACTED:groq]": {"nested": "[REDACTED:openai]"}}


def test_tool_usage_stats_scrub_failure_writes_nothing(tmp_path, monkeypatch):
    from brainlayer.pipeline import cloud_scrub
    from brainlayer.vector_store import VectorStore

    def boom(text):
        raise RuntimeError("scrubber down")

    monkeypatch.setattr(cloud_scrub, "scrub_secrets", boom)
    store = VectorStore(tmp_path / "session.db")
    try:
        with pytest.raises(cloud_scrub.CloudScrubError):
            store.upsert_session_enrichment({"session_id": "s-1", "tool_usage_stats": [{"tool": "Bash", "count": 1}]})
        count = list(store.conn.cursor().execute("SELECT COUNT(*) FROM session_enrichments"))[0][0]
    finally:
        store.close()
    assert count == 0


def test_saved_session_result_replay_redacts_tool_usage_stats(tmp_path):
    from brainlayer.pipeline.session_history import parse_session_enrichment
    from brainlayer.vector_store import VectorStore

    response = json.dumps(
        {
            "session_summary": "The agent fixed a bug in the watcher.",
            "tool_usage_stats": [{"tool": f"Bash {GROQ}", "count": 5}, {GITHUB: 1}],
        }
    )
    store = VectorStore(tmp_path / "session.db")
    try:
        store.conn.cursor().execute(
            "INSERT INTO chunks (id, content, metadata, source_file, content_type, created_at, char_count) "
            "VALUES ('c1', ?, '{}', '/p/sess-9.jsonl', 'user_message', '2026-09-28T00:00:00Z', 80)",
            ("user: please fix the watcher bug that drops lines on rotate, thanks a lot " * 2,),
        )
        enrichment = parse_session_enrichment(response)
        assert enrichment is not None
        store.upsert_session_enrichment({"session_id": "sess-9", **enrichment})
        record = store.get_session_enrichment("sess-9")
        rows = list(
            store.conn.cursor().execute(
                "SELECT session_id FROM session_enrichments_fts WHERE session_enrichments_fts MATCH ?",
                ("watcher",),
            )
        )
        assert rows == [("sess-9",)]
    finally:
        store.close()

    assert record is not None
    _assert_no_token(json.dumps(record, default=str), where="saved session replay record")
    assert record["tool_usage_stats"][0] == {"tool": "Bash [REDACTED:groq]", "count": 5}


def _escaped(token: str) -> str:
    """The token with its first character written as a JSON \\uXXXX escape."""
    return "\\u" + format(ord(token[0]), "04x") + token[1:]


def test_scrub_llm_output_always_scrubs_string_keys():
    from brainlayer.pipeline.cloud_scrub import scrub_llm_output

    value = {GROQ: [GITHUB, 3], "n": {OPENAI: None}, 7: "seven"}

    assert scrub_llm_output(value) == {
        "[REDACTED:groq]": ["[REDACTED:github]", 3],
        "n": {"[REDACTED:openai]": None},
        7: "seven",
    }


def test_scrub_llm_output_key_collision_drops_secret_keys_and_later_value_wins():
    # The contract: two secret-bearing keys that redact to the same placeholder
    # collapse into one entry. No secret survives; the later value is kept.
    from brainlayer.pipeline.cloud_scrub import scrub_llm_output

    other_groq = "gsk_" + "1" * 52
    value = {GROQ: "first", "plain": 0, other_groq: "second"}

    assert scrub_llm_output(value) == {"[REDACTED:groq]": "second", "plain": 0}


# ── B2: JSON nested inside JSON strings ──────────────────────────────────


def test_double_encoded_json_with_escaped_token_is_redacted(tmp_path):
    inner = '[{"tool": "' + _escaped(GROQ) + '", "count": 1}]'
    assert GROQ not in inner and GROQ not in json.dumps(inner)

    record = _persist(tmp_path, tool_usage_stats=json.dumps(inner))

    assert json.loads(record["tool_usage_stats"]) == [{"tool": "[REDACTED:groq]", "count": 1}]


def test_json_string_leaf_and_key_with_escaped_token_are_redacted(tmp_path):
    leaf = '{"env": ["' + _escaped(GITHUB) + '"]}'
    key = '"' + _escaped(OPENAI) + '"'

    record = _persist(tmp_path, tool_usage_stats=[{"tool": "Bash", "args": leaf, key: 2}])

    (item,) = record["tool_usage_stats"]
    assert json.loads(item["args"]) == {"env": ["[REDACTED:github]"]}
    assert [json.loads(k) for k in item if k not in ("tool", "args")] == ["[REDACTED:openai]"]
    assert list(item.values())[-1] == 2


def test_json_strings_up_to_the_depth_limit_are_redacted(tmp_path):
    from brainlayer.pipeline.cloud_scrub import MAX_JSON_STRING_DEPTH

    value = '[{"tool": "' + _escaped(GROQ) + '", "count": 4}]'
    for _ in range(MAX_JSON_STRING_DEPTH - 1):
        value = json.dumps(value)

    record = _persist(tmp_path, tool_usage_stats=value)

    decoded = record["tool_usage_stats"]
    while isinstance(decoded, str):
        decoded = json.loads(decoded)
    assert decoded == [{"tool": "[REDACTED:groq]", "count": 4}]


def test_json_strings_past_the_depth_limit_fail_closed(tmp_path):
    from brainlayer.pipeline.cloud_scrub import MAX_JSON_STRING_DEPTH, CloudScrubError
    from brainlayer.vector_store import VectorStore

    value = '[{"tool": "' + _escaped(GROQ) + '", "count": 4}]'
    for _ in range(MAX_JSON_STRING_DEPTH):
        value = json.dumps(value)

    store = VectorStore(tmp_path / "session.db")
    try:
        with pytest.raises(CloudScrubError, match="nested"):
            store.upsert_session_enrichment({"session_id": "s-1", "tool_usage_stats": value})
        count = list(store.conn.cursor().execute("SELECT COUNT(*) FROM session_enrichments"))[0][0]
    finally:
        store.close()
    assert count == 0


def test_json_string_too_deep_to_decode_fails_closed(tmp_path):
    from brainlayer.pipeline.cloud_scrub import CloudScrubError
    from brainlayer.vector_store import VectorStore

    deep = "[" * 100_000 + '"' + _escaped(GROQ) + '"' + "]" * 100_000

    store = VectorStore(tmp_path / "session.db")
    try:
        with pytest.raises(CloudScrubError, match="too deep"):
            store.upsert_session_enrichment({"session_id": "s-1", "tool_usage_stats": [{"tool": deep}]})
        count = list(store.conn.cursor().execute("SELECT COUNT(*) FROM session_enrichments"))[0][0]
    finally:
        store.close()
    assert count == 0


# ── B1: NER relation-property keys ───────────────────────────────────────


def _persist_ner_relation(tmp_path, properties):
    from brainlayer.pipeline.kg_extraction import extract_kg_from_chunk
    from brainlayer.vector_store import VectorStore

    response = json.dumps(
        {
            "entities": [{"text": "Etan", "type": "person"}, {"text": "brainlayer", "type": "project"}],
            "relations": [
                {
                    "source": "Etan",
                    "target": "brainlayer",
                    "type": "builds",
                    "fact": "Etan builds brainlayer",
                    "properties": properties,
                }
            ],
        }
    )
    store = VectorStore(tmp_path / "kg.db")
    try:
        store.upsert_chunks(
            [
                {
                    "id": "chunk-1",
                    "content": "Etan builds brainlayer every day",
                    "metadata": "{}",
                    "source_file": "t.jsonl",
                    "project": "brainlayer",
                    "content_type": "user_message",
                    "value_type": "HIGH",
                    "char_count": 32,
                }
            ],
            [[0.1] * 1024],
        )
        stats = extract_kg_from_chunk(store, "chunk-1", use_llm=True, llm_caller=lambda prompt: response)
        rows = list(store.conn.cursor().execute("SELECT relation_type, properties, fact FROM kg_relations"))
    finally:
        store.close()

    assert stats["relations_created"] >= 1
    _assert_not_recoverable([list(row) for row in rows], where="kg_relations rows")
    (builds,) = [row for row in rows if row[0] == "builds"]
    return json.loads(builds[1])


def test_ner_relation_property_keys_do_not_persist_a_token(tmp_path):
    properties = _persist_ner_relation(tmp_path, {GITHUB: "v", "meta": {GROQ: {"deeper": [OPENAI]}}})

    assert properties["meta"] == {"[REDACTED:groq]": {"deeper": ["[REDACTED:openai]"]}}
    assert properties["[REDACTED:github]"] == "v"


def test_ner_relation_property_json_strings_with_escaped_tokens_are_redacted(tmp_path):
    # B2's shape at the other free-form sink: a property value (or key) that is
    # itself JSON, hiding a token behind a \u escape.
    leaf = '{"env": ["' + _escaped(GROQ) + '"]}'
    key = '"' + _escaped(GITHUB) + '"'

    properties = _persist_ner_relation(tmp_path, {"args": leaf, key: "v"})

    assert json.loads(properties["args"]) == {"env": ["[REDACTED:groq]"]}
    assert '"[REDACTED:github]"' in properties


# ── B2 at the scalar columns ─────────────────────────────────────────────


def test_scalar_session_fields_holding_escaped_json_are_redacted(tmp_path):
    # A scalar column is not decoded by its readers, but the value must not
    # carry a token that one standards-compliant JSON decode reconstructs.
    record = _persist(
        tmp_path,
        session_summary='"' + _escaped(GROQ) + '"',
        what_worked='["' + _escaped(OPENAI) + '"]',
    )

    assert json.loads(record["session_summary"]) == "[REDACTED:groq]"
    assert json.loads(record["what_worked"]) == ["[REDACTED:openai]"]
