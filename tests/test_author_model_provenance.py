"""#655: which model AUTHORED a chunk, for Claude Code and Codex ingestion alike.

All model names below are synthetic fixture strings, not claims about real models.
"""

import json
import sqlite3
from pathlib import Path

import pytest

from brainlayer.ingest.codex import parse_codex_session
from brainlayer.pipeline.classify import classify_content
from brainlayer.watcher_bridge import create_flush_callback

MODEL_A = "claude-fixture-model-a"
MODEL_B = "claude-fixture-model-b"


def _claude_assistant(source_file: Path, text: str, model, offset: int) -> dict:
    message = {"role": "assistant", "content": [{"type": "text", "text": text}]}
    if model is not ...:
        message["model"] = model
    return {
        "type": "assistant",
        "sessionId": "fixture-session",
        "message": message,
        "timestamp": f"2026-11-05T00:00:{offset:02d}Z",
        "_source_file": str(source_file),
        "_line_end_offset": offset * 100,
    }


def _claude_user(source_file: Path, text: str, offset: int) -> dict:
    return {
        "type": "user",
        "sessionId": "fixture-session",
        "message": {"role": "user", "content": [{"type": "text", "text": text}]},
        "timestamp": f"2026-11-05T00:00:{offset:02d}Z",
        "_source_file": str(source_file),
        "_line_end_offset": offset * 100,
    }


def _rows(db_path: Path) -> dict[str, tuple[dict, object, object]]:
    with sqlite3.connect(db_path) as conn:
        rows = conn.execute("SELECT content, metadata, enrichment_model, enrichment_backend FROM chunks").fetchall()
    return {content: (json.loads(metadata), em, eb) for content, metadata, em, eb in rows}


TEXT_A = "Decision: keep the retry budget at three attempts because the fixture backend flakes under load."
TEXT_B = "Root cause: the fixture cache key omitted the tenant id, so two tenants shared one cached answer."
TEXT_USER = "Why does the fixture cache return another tenant's answer when both tenants query at once?"


def test_mixed_model_claude_session_persists_per_turn_author_model_through_watcher(tmp_path):
    db_path = tmp_path / "scratch.db"
    source = tmp_path / "home" / ".claude" / "projects" / "-fixture-proj" / "fixture-session.jsonl"
    flush = create_flush_callback(db_path=db_path, arbitrated=False)

    result = flush(
        [
            _claude_user(source, TEXT_USER, 1),
            _claude_assistant(source, TEXT_A, MODEL_A, 2),
            _claude_assistant(source, TEXT_B, MODEL_B, 3),
        ]
    )

    assert result.inserted == 3
    rows = _rows(db_path)
    assert rows[TEXT_A][0]["model"] == MODEL_A
    assert rows[TEXT_B][0]["model"] == MODEL_B
    # A human turn has no authoring model; never borrow one from a neighbouring turn.
    assert "model" not in rows[TEXT_USER][0]
    # Authoring provenance is not enrichment provenance.
    for _metadata, enrichment_model, enrichment_backend in rows.values():
        assert enrichment_model is None
        assert enrichment_backend is None


@pytest.mark.parametrize(
    "raw_model",
    [..., None, "", "   ", "<synthetic>", 42, ["x"], "m" * 200, "bad\nmodel"],
    ids=["absent", "null", "empty", "blank", "synthetic", "int", "list", "overlong", "control-char"],
)
def test_untrustworthy_or_absent_transcript_model_is_left_unknown(tmp_path, raw_model):
    entry = _claude_assistant(tmp_path / "s.jsonl", TEXT_A, raw_model, 2)

    classified = classify_content(entry)

    assert classified is not None
    assert "model" not in classified.metadata


def test_transcript_model_is_stripped_but_not_rewritten(tmp_path):
    entry = _claude_assistant(tmp_path / "s.jsonl", TEXT_A, f"  {MODEL_A}  ", 2)

    assert classify_content(entry).metadata["model"] == MODEL_A


def test_user_entry_claiming_a_model_is_not_attributed(tmp_path):
    entry = _claude_user(tmp_path / "s.jsonl", TEXT_USER, 1)
    entry["message"]["model"] = MODEL_A

    assert "model" not in classify_content(entry).metadata


# ---------------------------------------------------------------------------
# Parity: the same "which model wrote this turn?" question against Codex.
# ---------------------------------------------------------------------------


def _codex_session(path: Path, lines: list[dict]) -> Path:
    path.write_text("".join(json.dumps(line) + "\n" for line in lines))
    return path


def _codex_meta() -> dict:
    return {
        "timestamp": "2026-11-05T00:00:00Z",
        "type": "session_meta",
        "payload": {"id": "codex-fixture", "cwd": "/fixture/Gits/proj", "model_provider": "openai"},
    }


def _codex_turn(model: str) -> dict:
    return {"timestamp": "2026-11-05T00:00:01Z", "type": "turn_context", "payload": {"model": model}}


def _codex_assistant(text: str) -> dict:
    return {
        "timestamp": "2026-11-05T00:00:02Z",
        "type": "response_item",
        "payload": {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": text}]},
    }


def test_codex_attributes_each_turn_to_its_turn_context_model_not_the_provider(tmp_path):
    session = _codex_session(
        tmp_path / "rollout.jsonl",
        [
            _codex_meta(),
            _codex_turn("codex-fixture-model-a"),
            _codex_assistant(TEXT_A),
            _codex_turn("codex-fixture-model-b"),
            _codex_assistant(TEXT_B),
        ],
    )

    chunks = {c["content"]: c["metadata"] for c in parse_codex_session(session)}

    assert chunks[TEXT_A]["model"] == "codex-fixture-model-a"
    assert chunks[TEXT_B]["model"] == "codex-fixture-model-b"
    assert chunks[TEXT_A]["model_provider"] == "openai"


def test_codex_without_turn_context_model_is_unknown_not_the_provider(tmp_path):
    session = _codex_session(tmp_path / "rollout.jsonl", [_codex_meta(), _codex_assistant(TEXT_A)])

    (chunk,) = list(parse_codex_session(session))

    assert chunk["metadata"]["model"] is None
    assert chunk["metadata"]["model_provider"] == "openai"


def test_claude_and_codex_answer_the_same_question_the_same_way(tmp_path):
    db_path = tmp_path / "scratch.db"
    source = tmp_path / "home" / ".claude" / "projects" / "-fixture-proj" / "fixture-session.jsonl"
    create_flush_callback(db_path=db_path, arbitrated=False)(
        [_claude_assistant(source, TEXT_A, MODEL_A, 2), _claude_assistant(source, TEXT_B, MODEL_B, 3)]
    )
    claude = {content: meta.get("model") for content, (meta, _em, _eb) in _rows(db_path).items()}

    codex_session = _codex_session(
        tmp_path / "rollout.jsonl",
        [_codex_meta(), _codex_turn(MODEL_A), _codex_assistant(TEXT_A), _codex_turn(MODEL_B), _codex_assistant(TEXT_B)],
    )
    codex = {c["content"]: c["metadata"]["model"] for c in parse_codex_session(codex_session)}

    assert claude == codex == {TEXT_A: MODEL_A, TEXT_B: MODEL_B}


def test_codex_user_and_tool_chunks_carry_provider_but_no_author_model(tmp_path):
    user = {
        "timestamp": "2026-11-05T00:00:03Z",
        "type": "response_item",
        "payload": {"type": "message", "role": "user", "content": [{"type": "input_text", "text": TEXT_USER}]},
    }
    tool = {
        "timestamp": "2026-11-05T00:00:04Z",
        "type": "response_item",
        "payload": {"type": "function_call_output", "call_id": "c1", "output": "fixture tool output " * 5},
    }
    session = _codex_session(tmp_path / "rollout.jsonl", [_codex_meta(), _codex_turn(MODEL_A), user, tool])

    chunks = list(parse_codex_session(session))

    assert [c["metadata"]["sender"] for c in chunks] == ["user", "tool"]
    for chunk in chunks:
        assert "model" not in chunk["metadata"]
        assert chunk["metadata"]["model_provider"] == "openai"
