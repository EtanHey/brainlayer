"""Tests for BrainLayer hook conditional activation (CC 2.1.85).

Validates the should_activate() gate in all three hook scripts:
- brainlayer-session-start.py
- brainlayer-prompt-search.py
- brainbar-stop-index.py

Env vars tested:
- BRAINLAYER_HOOKS_DISABLED=1 → skip all hooks
- CLAUDE_NON_INTERACTIVE=1 → skip (--print mode)
- BRAINLAYER_HOOKS_LIGHT=1 → reduced results for workers
"""

import importlib.util
import io
import os
import sqlite3
from pathlib import Path

import pytest

HOOKS_DIR = Path(__file__).parent.parent / "hooks"


def load_hook_module(filename):
    """Import a hook script as a module."""
    spec = importlib.util.spec_from_file_location(
        filename.replace("-", "_").replace(".py", ""),
        HOOKS_DIR / filename,
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def session_start():
    return load_hook_module("brainlayer-session-start.py")


@pytest.fixture
def prompt_search():
    return load_hook_module("brainlayer-prompt-search.py")


@pytest.fixture
def stop_index():
    return load_hook_module("brainbar-stop-index.py")


@pytest.fixture(autouse=True)
def clean_env():
    """Ensure relevant env vars are unset before each test."""
    env_vars = [
        "BRAINLAYER_HOOKS_DISABLED",
        "CLAUDE_NON_INTERACTIVE",
        "BRAINLAYER_HOOKS_LIGHT",
    ]
    saved = {k: os.environ.pop(k, None) for k in env_vars}
    yield
    for k, v in saved.items():
        if v is not None:
            os.environ[k] = v
        else:
            os.environ.pop(k, None)


class FakeCursor:
    def __init__(self, rows):
        self._rows = rows

    def fetchall(self):
        return self._rows


class FakeConn:
    def __init__(self, rows_by_query=None):
        self.rows_by_query = rows_by_query or {}
        self.executed = []
        self.closed = False

    def execute(self, query, params=()):
        self.executed.append((query, params))
        for needle, rows in self.rows_by_query.items():
            if needle in query:
                return FakeCursor(rows)
        return FakeCursor([])

    def close(self):
        self.closed = True


class TestSessionStartConditional:
    def test_default_activates(self, session_start):
        hook_input = {"session_id": "abc123", "cwd": "/tmp/test"}
        activate, light = session_start.should_activate(hook_input)
        assert activate is True
        assert light is False

    def test_disabled_env_var(self, session_start):
        os.environ["BRAINLAYER_HOOKS_DISABLED"] = "1"
        hook_input = {"session_id": "abc123"}
        activate, light = session_start.should_activate(hook_input)
        assert activate is False

    def test_non_interactive_skips(self, session_start):
        os.environ["CLAUDE_NON_INTERACTIVE"] = "1"
        hook_input = {"session_id": "abc123"}
        activate, light = session_start.should_activate(hook_input)
        assert activate is False

    def test_light_mode(self, session_start):
        os.environ["BRAINLAYER_HOOKS_LIGHT"] = "1"
        hook_input = {"session_id": "abc123"}
        activate, light = session_start.should_activate(hook_input)
        assert activate is True
        assert light is True

    def test_disabled_takes_precedence_over_light(self, session_start):
        os.environ["BRAINLAYER_HOOKS_DISABLED"] = "1"
        os.environ["BRAINLAYER_HOOKS_LIGHT"] = "1"
        hook_input = {"session_id": "abc123"}
        activate, light = session_start.should_activate(hook_input)
        assert activate is False

    def test_empty_input(self, session_start):
        activate, light = session_start.should_activate({})
        assert activate is True
        assert light is False

    def test_main_adds_hebrew_style_guidance_for_coach(self, session_start, monkeypatch, capsys):
        fake_conn = FakeConn({"FROM chunks_fts": []})

        monkeypatch.setattr(session_start, "get_db_path", lambda: "/tmp/brainlayer.db")
        monkeypatch.setattr(session_start, "load_scoped_projects", lambda: {})
        monkeypatch.setattr(
            session_start.sqlite3,
            "connect",
            lambda *args, **kwargs: fake_conn,
        )
        monkeypatch.setattr(
            session_start.sys,
            "stdin",
            io.StringIO('{"cwd":"/tmp/coach","session_id":"sess-1"}'),
        )

        with pytest.raises(SystemExit):
            session_start.main()

        output = capsys.readouterr().out
        assert "[Hebrew Style]" in output
        assert ("PRAGMA busy_timeout=1000", ()) in fake_conn.executed
        assert ("PRAGMA query_only=true", ()) in fake_conn.executed


class TestPromptSearchConditional:
    def test_default_activates(self, prompt_search):
        activate, light = prompt_search.should_activate()
        assert activate is True
        assert light is False

    def test_disabled_env_var(self, prompt_search):
        os.environ["BRAINLAYER_HOOKS_DISABLED"] = "1"
        activate, light = prompt_search.should_activate()
        assert activate is False

    def test_non_interactive_skips(self, prompt_search):
        os.environ["CLAUDE_NON_INTERACTIVE"] = "1"
        activate, light = prompt_search.should_activate()
        assert activate is False

    def test_light_mode_reduces_results(self, prompt_search):
        os.environ["BRAINLAYER_HOOKS_LIGHT"] = "1"
        activate, light = prompt_search.should_activate()
        assert activate is True
        assert light is True

    def test_extract_keywords_keeps_short_meaningful_terms(self, prompt_search):
        keywords = prompt_search.extract_keywords("T3 Code AI PR review")

        assert "t3" in keywords
        assert "code" in keywords
        assert "ai" in keywords
        assert "pr" in keywords

    def test_detect_entities_includes_project_and_tool_types(self, prompt_search):
        conn = sqlite3.connect(":memory:")
        conn.execute("CREATE TABLE kg_entities (id TEXT, name TEXT, entity_type TEXT)")
        conn.execute(
            "INSERT INTO kg_entities (id, name, entity_type) VALUES (?, ?, ?)",
            ("project-1", "T3 Code", "project"),
        )
        conn.execute(
            "INSERT INTO kg_entities (id, name, entity_type) VALUES (?, ?, ?)",
            ("tool-1", "Claude Code", "tool"),
        )

        matches = prompt_search.detect_entities_in_prompt(
            "Compare T3 Code with Claude Code",
            conn,
        )

        assert {match["name"] for match in matches} == {"T3 Code", "Claude Code"}

    def test_detect_entities_returns_empty_when_no_match(self, prompt_search):
        conn = sqlite3.connect(":memory:")
        conn.execute("CREATE TABLE kg_entities (id TEXT, name TEXT, entity_type TEXT)")

        matches = prompt_search.detect_entities_in_prompt("nothing relevant here", conn)

        assert matches == []

    def test_detect_entities_uses_db_cache_and_longest_multiword_match(self, prompt_search):
        conn = sqlite3.connect(":memory:")
        conn.execute("CREATE TABLE kg_entities (id TEXT, name TEXT, entity_type TEXT)")
        conn.executemany(
            "INSERT INTO kg_entities (id, name, entity_type) VALUES (?, ?, ?)",
            [
                ("project-1", "BrainLayer", "project"),
                ("person-1", "Etan Heyman", "person"),
                ("project-2", "BrainLayer MCP", "project"),
            ],
        )
        conn.commit()

        prompt_search._ENTITY_CACHE = None
        prompt_search._ENTITY_CACHE_DB_PATH = None

        first_matches = prompt_search.detect_entities_in_prompt(
            "Compare brainlayer mcp with EtAn HeYmAn",
            conn,
        )
        conn.execute("DELETE FROM kg_entities")
        conn.commit()
        second_matches = prompt_search.detect_entities_in_prompt(
            "Compare brainlayer mcp with EtAn HeYmAn",
            conn,
        )

        assert [(match["id"], match["name"]) for match in first_matches] == [
            ("project-2", "BrainLayer MCP"),
            ("person-1", "Etan Heyman"),
        ]
        assert [(match["id"], match["name"]) for match in second_matches] == [
            ("project-2", "BrainLayer MCP"),
            ("person-1", "Etan Heyman"),
        ]

    def test_detect_entities_opens_db_from_paths_when_conn_missing(self, prompt_search, monkeypatch, tmp_path):
        db_path = tmp_path / "entities.db"
        conn = sqlite3.connect(db_path)
        conn.execute("CREATE TABLE kg_entities (id TEXT, name TEXT, entity_type TEXT)")
        conn.execute(
            "INSERT INTO kg_entities (id, name, entity_type) VALUES (?, ?, ?)",
            ("person-1", "Etan Heyman", "person"),
        )
        conn.commit()
        conn.close()

        prompt_search._ENTITY_CACHE = None
        prompt_search._ENTITY_CACHE_DB_PATH = None
        monkeypatch.setattr(prompt_search.paths, "get_db_path", lambda: db_path)

        matches = prompt_search.detect_entities_in_prompt("What did etan heyman decide?")

        assert [(match["id"], match["name"]) for match in matches] == [("person-1", "Etan Heyman")]

    def test_load_entity_cache_retries_after_transient_sqlite_error(self, prompt_search):
        class ErrorConn:
            def execute(self, query, params=()):
                raise sqlite3.OperationalError("database is locked")

        prompt_search._ENTITY_CACHE = None
        prompt_search._ENTITY_CACHE_DB_PATH = None

        failed_cache = prompt_search._load_entity_cache(ErrorConn())

        assert failed_cache["entities_by_name"] == {}
        assert prompt_search._ENTITY_CACHE_DB_PATH is None

        conn = sqlite3.connect(":memory:")
        conn.execute("CREATE TABLE kg_entities (id TEXT, name TEXT, entity_type TEXT)")
        conn.execute(
            "INSERT INTO kg_entities (id, name, entity_type) VALUES (?, ?, ?)",
            ("person-1", "Etan Heyman", "person"),
        )
        conn.commit()

        recovered_cache = prompt_search._load_entity_cache(conn)

        assert "etan heyman" in recovered_cache["entities_by_name"]

    def test_load_entity_cache_filters_to_supported_injected_types(self, prompt_search):
        conn = sqlite3.connect(":memory:")
        conn.execute("CREATE TABLE kg_entities (id TEXT, name TEXT, entity_type TEXT)")
        conn.execute(
            """
            CREATE TABLE kg_entity_aliases (
                alias TEXT,
                entity_id TEXT,
                alias_type TEXT
            )
            """
        )
        conn.executemany(
            "INSERT INTO kg_entities (id, name, entity_type) VALUES (?, ?, ?)",
            [
                ("project-1", "BrainLayer", "project"),
                ("library-1", "SQLite", "library"),
            ],
        )
        conn.executemany(
            "INSERT INTO kg_entity_aliases (alias, entity_id, alias_type) VALUES (?, ?, ?)",
            [
                ("BL", "project-1", "handle"),
                ("sql", "library-1", "handle"),
            ],
        )
        conn.commit()

        prompt_search._ENTITY_CACHE = None
        prompt_search._ENTITY_CACHE_DB_PATH = None

        cache = prompt_search._load_entity_cache(conn)

        assert "brainlayer" in cache["entities_by_name"]
        assert "sqlite" not in cache["entities_by_name"]
        assert "bl" in cache["aliases_by_name"]
        assert "sql" not in cache["aliases_by_name"]

    def test_record_injection_event_persists_rows(self, prompt_search, tmp_path):
        db_path = tmp_path / "hook-events.db"
        conn = sqlite3.connect(db_path)
        conn.execute(
            """
            CREATE TABLE injection_events (
                session_id TEXT,
                query TEXT,
                chunk_ids TEXT,
                token_count INTEGER
            )
            """
        )
        conn.commit()
        conn.close()

        prompt_search.record_injection_event(
            str(db_path),
            "session-1",
            "Prompt text",
            ["chunk-1", "chunk-2"],
            42,
        )

        rows = (
            sqlite3.connect(db_path)
            .execute("SELECT session_id, query, chunk_ids, token_count FROM injection_events")
            .fetchall()
        )
        assert rows == [("session-1", "Prompt text", '["chunk-1", "chunk-2"]', 42)]

    def test_record_injection_event_records_latency(self, prompt_search, tmp_path):
        db_path = tmp_path / "hook-events.db"
        conn = sqlite3.connect(db_path)
        conn.execute(
            """
            CREATE TABLE injection_events (
                session_id TEXT,
                query TEXT,
                chunk_ids TEXT,
                token_count INTEGER
            )
            """
        )
        conn.commit()
        conn.close()

        prompt_search.record_injection_event(
            str(db_path),
            "session-1",
            "Prompt text",
            ["chunk-1"],
            42,
            latency_ms=123,
            mode="normal",
            entities_detected=2,
        )

        row = (
            sqlite3.connect(db_path)
            .execute("SELECT latency_ms, mode, entities_detected FROM injection_events")
            .fetchone()
        )
        assert row == (123, "normal", 2)

    def test_record_injection_event_records_mode(self, prompt_search, tmp_path):
        db_path = tmp_path / "hook-events.db"
        conn = sqlite3.connect(db_path)
        conn.execute(
            """
            CREATE TABLE injection_events (
                session_id TEXT,
                query TEXT,
                chunk_ids TEXT,
                token_count INTEGER
            )
            """
        )
        conn.commit()
        conn.close()

        prompt_search.record_injection_event(
            str(db_path),
            "session-1",
            "Prompt text",
            ["chunk-1"],
            42,
            latency_ms=55,
            mode="entity",
            entities_detected=1,
        )

        row = sqlite3.connect(db_path).execute("SELECT mode FROM injection_events").fetchone()
        assert row == ("entity",)

    def test_injection_event_skip_logged(self, prompt_search, tmp_path, monkeypatch, capsys):
        db_path = tmp_path / "hook-events.db"
        conn = sqlite3.connect(db_path)
        conn.execute(
            """
            CREATE TABLE injection_events (
                session_id TEXT,
                query TEXT,
                chunk_ids TEXT,
                token_count INTEGER
            )
            """
        )
        conn.commit()
        conn.close()

        monkeypatch.setattr(prompt_search, "get_db_path", lambda: str(db_path))
        monkeypatch.setattr(prompt_search, "classify_prompt", lambda prompt, detected_entities=None: "command")
        monkeypatch.setattr(
            prompt_search.sys,
            "stdin",
            io.StringIO('{"prompt":"show git status","session_id":"sess-skip"}'),
        )

        with pytest.raises(SystemExit):
            prompt_search.main()

        assert capsys.readouterr().out == ""
        row = (
            sqlite3.connect(db_path)
            .execute("SELECT session_id, chunk_ids, token_count, mode FROM injection_events")
            .fetchone()
        )
        assert row == ("sess-skip", "[]", 0, "skip")

    def test_main_prints_search_before_assume_warning(self, prompt_search, monkeypatch, capsys):
        fake_conn = FakeConn()

        monkeypatch.setattr(prompt_search, "get_db_path", lambda: "/tmp/brainlayer.db")
        monkeypatch.setattr(
            prompt_search.sqlite3,
            "connect",
            lambda *args, **kwargs: fake_conn,
        )
        monkeypatch.setattr(
            prompt_search.sys,
            "stdin",
            io.StringIO('{"prompt":"What hardware does Etan use?","session_id":"sess-1"}'),
        )

        with pytest.raises(SystemExit):
            prompt_search.main()

        output = capsys.readouterr().out
        assert "SEARCH-BEFORE-ASSUME" in output
        assert ("PRAGMA busy_timeout=1000", ()) in fake_conn.executed
        assert ("PRAGMA query_only=true", ()) in fake_conn.executed

    def test_detect_correction_categorizes_common_prompts(self, prompt_search):
        assert prompt_search.detect_correction("No, that's wrong. Avi works at Lightricks.") == "factual"
        assert prompt_search.detect_correction("Please don't do that again.") == "preference"
        assert prompt_search.detect_correction("This response is too verbose.") == "style"
        assert prompt_search.detect_correction("לא נכון, תתקן את זה") == "factual"
        assert prompt_search.detect_correction("טעות") == "factual"
        assert prompt_search.detect_correction("ok") is None
        assert prompt_search.detect_correction("I want to build auth with bun.") is None
        assert prompt_search.detect_correction("I like Python for backend work.") is None
        assert prompt_search.detect_correction("What is CSS style isolation?") is None
        assert prompt_search.detect_correction("Explain tone mapping in image pipelines.") is None

    def test_main_prints_correction_store_nudge(self, prompt_search, monkeypatch, capsys):
        fake_conn = FakeConn()

        monkeypatch.setattr(prompt_search, "get_db_path", lambda: "/tmp/brainlayer.db")
        monkeypatch.setattr(
            prompt_search.sqlite3,
            "connect",
            lambda *args, **kwargs: fake_conn,
        )
        monkeypatch.setattr(
            prompt_search.sys,
            "stdin",
            io.StringIO('{"prompt":"No, that\'s wrong. Avi works at Lightricks.","session_id":"sess-1"}'),
        )

        with pytest.raises(SystemExit):
            prompt_search.main()

        output = capsys.readouterr().out
        assert "Correction detected: factual" in output
        assert "brain_store" in output
        assert "correction:factual" in output

    def test_main_does_not_trigger_assume_warning_on_substring_matches(self, prompt_search, monkeypatch, capsys):
        fake_conn = FakeConn()

        monkeypatch.setattr(prompt_search, "get_db_path", lambda: "/tmp/brainlayer.db")
        monkeypatch.setattr(
            prompt_search.sqlite3,
            "connect",
            lambda *args, **kwargs: fake_conn,
        )
        monkeypatch.setattr(
            prompt_search.sys,
            "stdin",
            io.StringIO('{"prompt":"How should this package program work in practice?","session_id":"sess-1"}'),
        )

        with pytest.raises(SystemExit):
            prompt_search.main()

        output = capsys.readouterr().out
        assert "SEARCH-BEFORE-ASSUME" not in output


class TestStopIndexConditional:
    def test_default_activates(self, stop_index):
        assert stop_index.should_activate() is True

    def test_disabled_env_var(self, stop_index):
        os.environ["BRAINLAYER_HOOKS_DISABLED"] = "1"
        assert stop_index.should_activate() is False

    def test_non_interactive_skips(self, stop_index):
        os.environ["CLAUDE_NON_INTERACTIVE"] = "1"
        assert stop_index.should_activate() is False

    def test_disabled_value_must_be_1(self, stop_index):
        os.environ["BRAINLAYER_HOOKS_DISABLED"] = "0"
        assert stop_index.should_activate() is True

    def test_disabled_value_true_not_recognized(self, stop_index):
        os.environ["BRAINLAYER_HOOKS_DISABLED"] = "true"
        assert stop_index.should_activate() is True
