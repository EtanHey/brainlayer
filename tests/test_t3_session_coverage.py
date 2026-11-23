"""Decisions must follow real session storage, including the provider's raw shapes."""

import json
import sqlite3

import pytest

from brainlayer.ingest.t3 import T3Message
from brainlayer.ingest.t3_ownership import message_owner, ownership_facts, timestamp
from brainlayer.t3_thread_tables import T3ProviderThread, T3ThreadCatalogEntry
from brainlayer.vector_store import VectorStore
from brainlayer.watcher import normalize_provider_entry
from brainlayer.watcher_bridge import create_flush_callback
from tests.test_t3_thread_tables import _session_chunk

TEXT = "StorageEvidenceFixture: preserve the full steering instruction and its implementation details even when a later reply has already reached the session layer."
STAMP = "2026-11-20T12:00:00Z"


@pytest.mark.parametrize(
    "shape,expected_stored",
    [
        ("queue-operation", True),
        ("attachment", False),
        ("tool_result", False),
        ("image_text", True),
        ("codex_user", True),
        ("codex_assistant", True),
        ("thinking", True),
        ("managed_session", True),
        ("cursor", True),
    ],
)
def test_coverage_decision_matches_rows_the_real_watcher_stored(tmp_path, monkeypatch, shape, expected_stored):
    import brainlayer.watcher_bridge as bridge

    monkeypatch.setattr(bridge, "resolve_t3_sources", lambda *args: [])
    monkeypatch.setenv("BRAINLAYER_T3_PROJECTION_VERSION", "1")
    monkeypatch.setenv("BRAINLAYER_QUEUE_DIR", str(tmp_path / "queue"))
    db = tmp_path / "brain.db"
    VectorStore(db).close()
    codex = shape.startswith("codex")
    role = "assistant" if shape in {"codex_assistant", "thinking"} else "user"
    native = "01a1aaaa-0000-7000-8000-000000000099" if codex else "0a1b2c3d-1111-2222-3333-444455556666"
    driver = "codex" if codex else "cursor" if shape == "cursor" else "claudeAgent"
    if codex:
        source = tmp_path / ".codex" / "sessions" / f"rollout-2026-11-20T12-00-00-{native}.jsonl"
        conversation = source.stem
        raw = {
            "type": "response_item",
            "timestamp": STAMP,
            "payload": {
                "type": "message",
                "role": role,
                "content": [{"type": "input_text" if role == "user" else "output_text", "text": TEXT}],
            },
        }
    else:
        source = tmp_path / ".claude" / "projects" / "-Users-example-Gits-fixture" / f"{native}.jsonl"
        conversation = native
        raw = {"type": role, "timestamp": STAMP, "message": {"role": role, "content": [{"type": "text", "text": TEXT}]}}
        if shape == "queue-operation":
            raw = {"type": "queue-operation", "operation": "enqueue", "content": TEXT, "timestamp": STAMP}
        elif shape == "attachment":
            raw = {"type": "attachment", "attachment": {"type": "queued_command", "prompt": TEXT}, "timestamp": STAMP}
        elif shape == "tool_result":
            raw["message"]["content"] = [{"type": "tool_result", "tool_use_id": "call_fixture", "content": TEXT}]
        elif shape in {"image_text", "managed_session"}:
            raw["message"]["content"].insert(
                0, {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "synthetic"}}
            )
        elif shape == "thinking":
            raw["message"]["content"] = [{"type": "thinking", "thinking": TEXT}]
        elif shape == "cursor":
            source = tmp_path / ".cursor" / "sessions" / f"{native}.jsonl"
            raw = {"role": "user", "content": TEXT, "timestamp": STAMP}
    source.parent.mkdir(parents=True, exist_ok=True)
    header = {"type": "session_meta", "payload": {"id": native, "originator": "codex_cli_rs"}}
    source.write_text(json.dumps(header) + "\n" + json.dumps(raw) + "\n")
    # Claude entries flow through unchanged; Codex/Cursor are normalized by the
    # actual tailer normalizer before the real bridge persists them.
    entry = normalize_provider_entry(raw, "codex" if codex else "cursor") if codex or shape == "cursor" else raw
    entry["_source_file"] = str(source)
    create_flush_callback(db, arbitrated=False)([entry])
    with sqlite3.connect(db) as conn:
        rows = conn.execute(
            "SELECT content FROM chunks WHERE conversation_id=? AND source != 't3'", (conversation,)
        ).fetchall()
        stored = any(TEXT in row[0] for row in rows)
        assert stored is expected_stored
        if shape == "managed_session":
            conn.execute(
                "UPDATE chunks SET archived_at='2026-11-20T12:30:00Z' WHERE conversation_id=?", (conversation,)
            )
        _session_chunk(conn, "later-reply", conversation, "2026-11-20T12:00:30Z")
        catalog = T3ThreadCatalogEntry(
            "thread",
            "private",
            "fixture",
            None,
            None,
            None,
            STAMP,
            None,
            STAMP,
            (T3ProviderThread("provider", driver, driver, native, True),),
        )
        facts = ownership_facts(conn, [catalog])
        mid = "message:provider:claudeAgent:native-item:fixture%3Athinking%3A0" if shape == "thinking" else "message"
        message = T3Message(mid, "thread", role, TEXT, STAMP)
        assert message_owner(message, facts["thread"], now=1795270000) == ("covered" if stored else "session_late")
        assert message_owner(message, facts["thread"], now=timestamp(STAMP) + 60) == ("covered" if stored else "held")
    conn.close()


@pytest.mark.parametrize(
    "needle,texts,expected",
    [
        ("alpha beta gamma", ["extra alpha beta gamma tail"], True),
        ("alpha beta gamma", ["alpha beta", "gamma"], True),
        ("alpha beta gamma delta", ["alpha be", "ta gamma delta"], True),
        ("alpha beta middle gamma delta", ["alpha beta mid", "dle gamma delta"], True),
        ("alpha beta gamma delta", ["gamma delta", "alpha beta gamma"], True),
        ("alpha beta gamma delta", ["alpha beta", "delta"], False),
        ("cat", ["concatenate"], True),
        ("", [""], False),
        ("alpha beta gamma", ["alpha beta gamma"], True),
    ],
)
def test_candidate_index_preserves_full_partial_and_split_word_coverage(needle, texts, expected):
    from brainlayer.ingest.t3_audit import TextCoverageIndex, _text_covered

    assert _text_covered(needle, texts) is expected
    assert _text_covered(needle, TextCoverageIndex(texts)) is expected


def test_candidate_index_preserves_coverage_for_every_split_and_missing_character():
    from brainlayer.ingest.t3_audit import TextCoverageIndex, _text_covered

    needle = "alpha beta gamma delta epsilon"
    for split in range(1, len(needle)):
        for gap in (0, 1):
            texts = [needle[split + gap :], "unrelated stored text", needle[:split]]
            assert _text_covered(needle, TextCoverageIndex(texts)) == _text_covered(needle, iter(texts))
