"""Tests for prompt classification and hook routing behavior."""

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


def make_hook_db(db_path: Path) -> None:
    conn = sqlite3.connect(db_path)
    conn.execute(
        """
        CREATE TABLE chunks (
            id TEXT PRIMARY KEY,
            content TEXT,
            importance REAL,
            project TEXT,
            tags TEXT,
            created_at TEXT
        )
        """
    )
    conn.execute("CREATE VIRTUAL TABLE chunks_fts USING fts5(chunk_id UNINDEXED, content)")
    conn.execute("CREATE TABLE kg_entities (id TEXT PRIMARY KEY, name TEXT, entity_type TEXT)")
    conn.execute("CREATE TABLE kg_entity_chunks (entity_id TEXT, chunk_id TEXT, relevance REAL)")

    conn.execute(
        """
        INSERT INTO chunks (id, content, importance, project, tags, created_at)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (
            "chunk-auth",
            "Authentication was implemented with JWT middleware and route guards.",
            8,
            "brainlayer",
            '["auth"]',
            "2026-04-05T10:00:00Z",
        ),
    )
    conn.execute(
        "INSERT INTO chunks_fts (chunk_id, content) VALUES (?, ?)",
        ("chunk-auth", "Authentication was implemented with JWT middleware and route guards."),
    )

    conn.execute(
        """
        INSERT INTO chunks (id, content, importance, project, tags, created_at)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (
            "chunk-theo",
            "Theo Browne is linked to BrainLayer collaboration notes.",
            7,
            "brainlayer",
            '["people"]',
            "2026-04-04T10:00:00Z",
        ),
    )
    conn.execute(
        "INSERT INTO chunks_fts (chunk_id, content) VALUES (?, ?)",
        ("chunk-theo", "Theo Browne is linked to BrainLayer collaboration notes."),
    )
    conn.execute(
        "INSERT INTO kg_entities (id, name, entity_type) VALUES (?, ?, ?)",
        ("person-theo", "Theo Browne", "person"),
    )
    conn.execute(
        "INSERT INTO kg_entity_chunks (entity_id, chunk_id, relevance) VALUES (?, ?, ?)",
        ("person-theo", "chunk-theo", 0.9),
    )
    conn.commit()
    conn.close()


def run_hook(module, hook_input: dict, monkeypatch, capsys):
    monkeypatch.setattr(module.sys, "stdin", io.StringIO(__import__("json").dumps(hook_input)))
    with pytest.raises(SystemExit):
        module.main()
    return capsys.readouterr().out


def test_classify_slash_command():
    assert classify_prompt("/commit") == "command"


def test_classify_cli_command():
    assert classify_prompt("git status") == "command"


def test_classify_action_command():
    assert classify_prompt("run tests") == "command"


def test_classify_greeting():
    assert classify_prompt("hey") == "casual_chat"


def test_classify_thanks():
    assert classify_prompt("thanks!") == "casual_chat"


def test_classify_entity_lookup():
    detected = [{"id": "person-theo", "name": "Theo Browne", "entity_type": "person"}]
    assert classify_prompt("who is Theo Browne", detected_entities=detected) == "entity_lookup"


def test_classify_knowledge_question():
    assert classify_prompt("how did I implement authentication") == "knowledge_question"


def test_classify_follow_up():
    assert classify_prompt("tell me more") == "follow_up"


def test_classify_hebrew():
    assert classify_prompt("מה עשיתי אתמול") == "hebrew_query"


def test_classify_long_prompt_not_casual():
    prompt = "ok, can you summarize the authentication rollout and what changed in the hook?"
    assert classify_prompt(prompt) != "casual_chat"


def test_command_skips_retrieval(monkeypatch, capsys):
    module = load_prompt_search_module()

    def fail_connect(*args, **kwargs):
        raise AssertionError("sqlite3.connect should not be called for command prompts")

    monkeypatch.setattr(module.sqlite3, "connect", fail_connect)
    output = run_hook(module, {"prompt": "git status --short", "session_id": "sess-1"}, monkeypatch, capsys)
    assert output == ""


def test_casual_skips_retrieval(monkeypatch, capsys):
    module = load_prompt_search_module()

    def fail_connect(*args, **kwargs):
        raise AssertionError("sqlite3.connect should not be called for casual prompts")

    monkeypatch.setattr(module.sqlite3, "connect", fail_connect)
    output = run_hook(module, {"prompt": "sounds good, thanks", "session_id": "sess-1"}, monkeypatch, capsys)
    assert output == ""


def test_entity_route_injects_card(tmp_path, monkeypatch, capsys):
    module = load_prompt_search_module()
    db_path = tmp_path / "hook.db"
    make_hook_db(db_path)
    monkeypatch.setattr(module, "get_db_path", lambda: str(db_path))

    output = run_hook(module, {"prompt": "who is Theo Browne", "session_id": "sess-1"}, monkeypatch, capsys)

    assert "[Entity: Theo Browne" in output
    assert "Theo Browne is linked to BrainLayer collaboration notes." in output


def test_knowledge_route_injects_chunks(tmp_path, monkeypatch, capsys):
    module = load_prompt_search_module()
    db_path = tmp_path / "hook.db"
    make_hook_db(db_path)
    monkeypatch.setattr(module, "get_db_path", lambda: str(db_path))

    output = run_hook(
        module,
        {"prompt": "how did I implement authentication", "session_id": "sess-1"},
        monkeypatch,
        capsys,
    )

    assert "[BrainLayer" in output
    assert "Authentication was implemented with JWT middleware and route guards." in output
