"""Tests for session-aware follow-up query rewriting in the prompt hook."""

import importlib.util
import io
import sqlite3
from pathlib import Path

import pytest

from brainlayer.classify import classify_prompt

HOOK_PATH = Path(__file__).parent.parent / "hooks" / "brainlayer-prompt-search.py"


def load_prompt_search_module():
    spec = importlib.util.spec_from_file_location("brainlayer_prompt_search", HOOK_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_session_db(db_path: Path) -> None:
    conn = sqlite3.connect(db_path)
    conn.execute(
        """
        CREATE TABLE chunks (
            id TEXT PRIMARY KEY,
            content TEXT,
            importance REAL,
            project TEXT,
            tags TEXT,
            created_at TEXT,
            content_type TEXT,
            source_file TEXT
        )
        """
    )
    conn.execute("CREATE VIRTUAL TABLE chunks_fts USING fts5(chunk_id UNINDEXED, content)")
    conn.execute("CREATE TABLE kg_entities (id TEXT PRIMARY KEY, name TEXT, entity_type TEXT)")
    conn.execute("CREATE TABLE kg_entity_chunks (entity_id TEXT, chunk_id TEXT, relevance REAL)")

    rows = [
        (
            "chunk-auth",
            "Authentication was implemented with JWT middleware and route guards.",
            8,
            "brainlayer",
            '["auth"]',
            "2026-04-06T11:00:00Z",
            "memory",
            "/tmp/memory.jsonl",
        ),
        (
            "msg-1",
            "We should review the authentication rollout and JWT middleware next.",
            1,
            "brainlayer",
            "[]",
            "2026-04-06T10:00:00Z",
            "user_message",
            "/tmp/sessions/sess-123.jsonl",
        ),
        (
            "msg-2",
            "The route guards were added after the auth middleware landed.",
            1,
            "brainlayer",
            "[]",
            "2026-04-06T10:05:00Z",
            "user_message",
            "/tmp/sessions/sess-123.jsonl",
        ),
        (
            "msg-3",
            "Can you explain the middleware ordering again?",
            1,
            "brainlayer",
            "[]",
            "2026-04-06T10:10:00Z",
            "user_message",
            "/tmp/sessions/sess-123.jsonl",
        ),
        (
            "msg-other",
            "Unrelated session content should not be pulled in.",
            1,
            "brainlayer",
            "[]",
            "2026-04-06T10:15:00Z",
            "user_message",
            "/tmp/sessions/other-session.jsonl",
        ),
    ]
    conn.executemany(
        """
        INSERT INTO chunks (id, content, importance, project, tags, created_at, content_type, source_file)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        rows,
    )
    conn.execute(
        "INSERT INTO chunks_fts (chunk_id, content) VALUES (?, ?)",
        ("chunk-auth", "Authentication was implemented with JWT middleware and route guards."),
    )
    conn.commit()
    conn.close()


def run_hook(module, hook_input: dict, monkeypatch, capsys):
    monkeypatch.setattr(module.sys, "stdin", io.StringIO(__import__("json").dumps(hook_input)))
    with pytest.raises(SystemExit):
        module.main()
    return capsys.readouterr().out


def test_get_session_context_empty(tmp_path):
    module = load_prompt_search_module()
    db_path = tmp_path / "hook.db"
    make_session_db(db_path)

    conn = sqlite3.connect(db_path)
    try:
        assert module.get_session_context(conn, "") == []
        assert module.get_session_context(None, "sess-123") == []
    finally:
        conn.close()


def test_get_session_context_returns_recent(tmp_path):
    module = load_prompt_search_module()
    db_path = tmp_path / "hook.db"
    make_session_db(db_path)

    conn = sqlite3.connect(db_path)
    try:
        assert module.get_session_context(conn, "sess-123") == [
            "Can you explain the middleware ordering again?",
            "The route guards were added after the auth middleware landed.",
            "We should review the authentication rollout and JWT middleware next.",
        ]
    finally:
        conn.close()


def test_extract_context_keywords_filters_stopwords():
    module = load_prompt_search_module()

    assert module.extract_context_keywords(["tell me more about the authentication middleware"]) == [
        "authentication",
        "middleware",
    ]


def test_extract_context_keywords_deduplicates():
    module = load_prompt_search_module()

    assert module.extract_context_keywords(["authentication auth authentication middleware middleware"]) == [
        "authentication",
        "auth",
        "middleware",
    ]


def test_extract_context_keywords_respects_limit():
    module = load_prompt_search_module()

    assert module.extract_context_keywords(
        ["alpha beta gamma delta epsilon zeta eta theta iota"],
        max_keywords=4,
    ) == ["alpha", "beta", "gamma", "delta"]


def test_follow_up_classification():
    assert classify_prompt("tell me more") == "follow_up"


def test_follow_up_route_uses_session_context_for_search(tmp_path, monkeypatch, capsys):
    module = load_prompt_search_module()
    db_path = tmp_path / "hook.db"
    make_session_db(db_path)
    monkeypatch.setattr(module, "get_db_path", lambda: str(db_path))

    output = run_hook(
        module,
        {"prompt": "tell me more about that", "session_id": "sess-123"},
        monkeypatch,
        capsys,
    )

    assert "BrainLayer memory available" in output
    assert "Authentication was implemented with JWT middleware and route guards." in output
