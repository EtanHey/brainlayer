"""#1824 fix round 1: hidden classes (agents' thinking) stay out of every default surface.

Etan, 2026-11-17, choice A: thinking is stored, kept out of default search, searchable only
on purpose. Most cases here are the red-team drivers (Fable, Opus), run on both writer
paths: direct (non-arbitrated) and production (the watcher enqueues, the drain writes).
"""

from __future__ import annotations

import importlib.util
import io
import json
import sqlite3
import sys
from pathlib import Path

import pytest

from brainlayer.dedupe import is_near_duplicate, normalized_exact_hash
from brainlayer.drain import drain_once
from brainlayer.source_visibility import DEFAULT_HIDDEN_SOURCE_CLASSES, hidden_source_class_sql
from brainlayer.vector_store import VectorStore
from brainlayer.watcher_bridge import create_flush_callback

REPO = Path(__file__).resolve().parent.parent
HOOKS = REPO / "hooks"
PROJECT_DIR = "-Users-example-Gits-myproject"
SESSION_A = "0a1b2c3d-1111-2222-3333-444455556666"
SESSION_B = "9f8e7d6c-aaaa-bbbb-cccc-ddddeeeeffff"

BASE = (
    "the watcher offset was never the cause of the stall because the tailer confirmed every line "
    "before the flush returned, so the real defect sits in the classifier which drops assistant "
    "entries whose text is shorter than the entry gate, and the fix therefore belongs in the "
    "classifier and not in the tailer loop or the offset registry or the drain worker at all"
)
THINK = "RedteamThinkNeedle " + BASE
SAID_EXACT = THINK  # the reply repeats the thinking word for word


def _session(tmp_path: Path, stem: str) -> str:
    return str(tmp_path / ".claude" / "projects" / PROJECT_DIR / f"{stem}.jsonl")


def _entry(*blocks, ts="2026-11-17T12:00:00Z"):
    return {"type": "assistant", "message": {"role": "assistant", "content": list(blocks)}, "timestamp": ts}


def _thinking(text):
    return {"type": "thinking", "thinking": text}


def _text(text):
    return {"type": "text", "text": text}


def _rows(db_path: Path):
    conn = sqlite3.connect(str(db_path))
    try:
        return conn.execute(
            "SELECT id, source_class, content FROM chunks WHERE source = 'realtime_watcher' ORDER BY rowid"
        ).fetchall()
    finally:
        conn.close()


def _default_hits(db_path: Path, needle: str) -> list[str]:
    store = VectorStore(db_path)
    try:
        return list(store.search(query_text=needle, n_results=20)["ids"][0])
    finally:
        store.close()


def _near_variant(base: str, marker: str) -> str:
    """A text that is NOT the same exact hash as base but is a simhash near-duplicate of it."""
    words = base.split()
    candidates = [f"{base} {marker}", f"{marker} {base}"]
    candidates += [" ".join([*words[:index], marker, *words[index:]]) for index in range(1, len(words))]
    candidates += [" ".join([*words[:index], marker, *words[index + 1 :]]) for index in range(1, len(words))]
    for candidate in candidates:
        if normalized_exact_hash(candidate) != normalized_exact_hash(base) and is_near_duplicate(
            candidate, base, created_at_a="2026-11-17T12:00:00Z", created_at_b="2026-11-17T12:00:00Z"
        ):
            return candidate
    raise AssertionError("could not build a simhash near-duplicate for this base text")


class _Writer:
    """One writer path: direct (non-arbitrated) or production (watcher enqueues, drain writes)."""

    def __init__(self, tmp_path: Path, monkeypatch, *, arbitrated: bool):
        self.db = tmp_path / "test.db"
        self.queue = tmp_path / "queue"
        self.arbitrated = arbitrated
        VectorStore(self.db).close()
        monkeypatch.setenv("BRAINLAYER_QUEUE_DIR", str(self.queue))
        self.tmp_path = tmp_path

    def write(self, entry: dict, stem: str = SESSION_A) -> None:
        entry = dict(entry)
        entry["_source_file"] = _session(self.tmp_path, stem)
        flush = create_flush_callback(self.db, arbitrated=self.arbitrated)
        try:
            flush([entry])
        finally:
            flush.close()
        if self.arbitrated:
            drain_once(db_path=self.db, queue_dir=self.queue, log_path=self.tmp_path / "drain.log")


@pytest.fixture(params=["direct", "drain"])
def writer(request, tmp_path, monkeypatch):
    return _Writer(tmp_path, monkeypatch, arbitrated=request.param == "drain")


def _load_hook(filename: str, *, module_name: str | None = None):
    spec = importlib.util.spec_from_file_location(module_name or filename.replace("-", "_")[:-3], HOOKS / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ── 1. The hooks are default search: they never inject a hidden class (blocks merge) ──


def test_prompt_hook_fts_paths_never_return_agent_thinking(writer):
    """The live path (fetch_matching_chunks) and the FTS fallback, on a DB the real writers built."""
    writer.write(_entry(_thinking(THINK)))
    assert [row[1] for row in _rows(writer.db)] == ["agent-thinking"]

    hook = _load_hook("brainlayer-prompt-search.py")
    conn = sqlite3.connect(f"file:{writer.db}?mode=ro", uri=True)
    try:
        live_path = hook.fetch_matching_chunks(conn, hook.build_fts_query(["RedteamThinkNeedle"]), 5)
    finally:
        conn.close()
    fallback = hook.run_fts_search(str(writer.db), ["RedteamThinkNeedle"], 5)

    assert live_path == []
    assert fallback == []


def _hidden_class_fixture_db() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.executescript(
        """
        CREATE TABLE chunks (id TEXT, content TEXT, importance REAL, project TEXT, tags TEXT, created_at TEXT,
            content_type TEXT, superseded_by TEXT, aggregated_into TEXT, archived_at TEXT, source_class TEXT);
        CREATE VIRTUAL TABLE chunks_fts USING fts5(chunk_id UNINDEXED, content, tags);
        CREATE TABLE kg_entity_chunks (entity_id TEXT, chunk_id TEXT, relevance REAL);
        """
    )
    for chunk_id, source_class in [
        ("said", "cli-agent"),
        ("legacy", None),
        ("thought", "agent-thinking"),
        ("desktop", "desktop"),
        ("worker", "brain-worker"),
    ]:
        content = f"brainlayer decision HiddenClassNeedle {chunk_id}"
        conn.execute(
            "INSERT INTO chunks (id, content, project, created_at, source_class) "
            "VALUES (?, ?, 'brainlayer', '2026-11-17', ?)",
            (chunk_id, content, source_class),
        )
        conn.execute("INSERT INTO chunks_fts VALUES (?, ?, '')", (chunk_id, content))
        conn.execute("INSERT INTO kg_entity_chunks VALUES ('entity-1', ?, 1.0)", (chunk_id,))
    return conn


def test_prompt_hook_live_and_entity_queries_hide_every_hidden_class():
    hook = _load_hook("brainlayer-prompt-search.py")
    conn = _hidden_class_fixture_db()

    rows = hook.fetch_matching_chunks(conn, '"HiddenClassNeedle"', 10)
    entity_rows = hook.get_entity_chunks("entity-1", conn, limit=10)

    assert sorted(row[0] for row in rows) == ["legacy", "said"]
    assert sorted(row[0].rsplit(" ", 1)[-1] for row in entity_rows) == ["legacy", "said"]


def test_prompt_hook_hybrid_fts_half_uses_the_class_gate_on_the_store_cursor(tmp_path):
    """run_hybrid_search's FTS half reads through VectorStore's apsw cursor."""
    hook = _load_hook("brainlayer-prompt-search.py")
    store = VectorStore(tmp_path / "hybrid.db")
    try:
        clause = hook.visible_source_class_sql(store._read_cursor())
    finally:
        store.close()

    assert clause == hidden_source_class_sql("c.source_class")


def test_session_start_hook_never_injects_a_hidden_class(monkeypatch, capsys):
    hook = _load_hook("brainlayer-session-start.py")
    conn = _hidden_class_fixture_db()
    monkeypatch.setattr(hook, "get_db_path", lambda: "scratch.db")
    monkeypatch.setattr(hook.sqlite3, "connect", lambda *a, **kw: conn)
    monkeypatch.setattr(hook, "load_scoped_projects", lambda: {})
    monkeypatch.setattr(hook.sys, "stdin", io.StringIO(json.dumps({"cwd": "/projects/brainlayer"})))

    with pytest.raises(SystemExit):
        hook.main()
    output = capsys.readouterr().out

    assert "HiddenClassNeedle said" in output
    for hidden in ("thought", "desktop", "worker"):
        assert f"HiddenClassNeedle {hidden}" not in output


@pytest.mark.parametrize("filename", ["brainlayer-prompt-search.py", "brainlayer-session-start.py"])
def test_hook_fallback_list_matches_the_shared_list(filename, monkeypatch):
    """The hooks are copies that can run against an older keg; their fallback must not drift."""
    monkeypatch.setitem(sys.modules, "brainlayer.source_visibility", None)  # forces the ImportError branch
    hook = _load_hook(filename, module_name=f"fallback_{filename.replace('-', '_')[:-3]}")

    assert hook.hidden_source_class_sql("c.source_class") == hidden_source_class_sql("c.source_class")
    assert DEFAULT_HIDDEN_SOURCE_CLASSES == ("desktop", "agent-thinking", "brain-worker")


# ── 2. Dedupe never folds across the hidden/visible line, in either direction ──


def test_said_text_equal_to_earlier_thinking_in_another_session_stays_visible(writer):
    """Thinking first (hidden row), then a reply with the same words in a second session, same project."""
    writer.write(_entry(_thinking(THINK)), SESSION_A)
    writer.write(_entry(_text(SAID_EXACT)), SESSION_B)

    assert _default_hits(writer.db, "RedteamThinkNeedle"), _rows(writer.db)


def test_said_text_near_duplicate_of_earlier_thinking_stays_visible(writer):
    """Thinking first, then a reply that restates it with one extra word (simhash near-duplicate)."""
    said = _near_variant(THINK, "RedteamSaidOnly")
    writer.write(_entry(_thinking(THINK)), SESSION_A)
    writer.write(_entry(_text(said)), SESSION_A)

    assert _default_hits(writer.db, "RedteamSaidOnly"), _rows(writer.db)


def test_thinking_near_duplicate_of_earlier_said_text_stays_hidden(writer):
    """Reply first (visible row), then thinking that restates it with one extra word."""
    said = "RedteamSaidNeedle " + BASE
    thinking = _near_variant(said, "RedteamThinkOnly")
    writer.write(_entry(_text(said)), SESSION_A)
    writer.write(_entry(_thinking(thinking)), SESSION_A)

    assert _default_hits(writer.db, "RedteamThinkOnly") == [], _rows(writer.db)


def test_thinking_never_folds_into_an_always_hidden_brain_worker_row(writer):
    """Re-check R2: brain-worker is a different tier from thinking; the opt-in could never reach it."""
    from brainlayer.dedupe import compute_dedupe_fields

    fields = compute_dedupe_fields(THINK, "2026-11-17T11:00:00Z")
    store = VectorStore(writer.db)
    store.conn.cursor().execute(
        """
        INSERT INTO chunks (id, content, metadata, source_file, project, content_type, char_count, created_at,
                            source_class, dedupe_hash, simhash, simhash_band_0, simhash_band_1, simhash_band_2,
                            simhash_band_3)
        VALUES ('worker-row', ?, '{}', '/subagents/agent-x.jsonl', 'myproject', 'assistant_text', ?,
                '2026-11-17T11:00:00Z', 'brain-worker', ?, ?, ?, ?, ?, ?)
        """,
        (THINK, len(THINK), fields.dedupe_hash, fields.simhash, *fields.bands),
    )
    store.close()

    writer.write(_entry(_thinking(THINK)))

    store = VectorStore(writer.db)
    try:
        opt_in = store.search(query_text="RedteamThinkNeedle", n_results=10, include_hidden_source_classes=True)
    finally:
        store.close()
    assert [row[1] for row in _rows(writer.db)] == ["agent-thinking"]
    assert opt_in["ids"][0] and "worker-row" not in opt_in["ids"][0]


# ── 3. Same chunk id: a thinking write never relabels a row that exists (both paths) ──


def test_thinking_write_never_relabels_an_existing_visible_row(writer):
    writer.write(_entry(_text(SAID_EXACT)), SESSION_A)
    assert [row[1] for row in _rows(writer.db)] == ["cli-agent"]

    writer.write(_entry(_thinking(THINK), ts="2026-11-17T12:05:00Z"), SESSION_A)

    assert [row[1] for row in _rows(writer.db)] == ["cli-agent"]
    assert _default_hits(writer.db, "RedteamThinkNeedle")


def test_thinking_write_never_hides_a_user_message_with_the_same_words(writer):
    user = {"type": "user", "message": {"role": "user", "content": THINK}, "timestamp": "2026-11-17T12:00:00Z"}
    writer.write(user, SESSION_A)
    assert [row[1] for row in _rows(writer.db)] == ["cli-agent"]

    writer.write(_entry(_thinking(THINK), ts="2026-11-17T12:05:00Z"), SESSION_A)

    assert _default_hits(writer.db, "RedteamThinkNeedle"), _rows(writer.db)


def test_one_entry_with_equal_thinking_and_text_keeps_the_text_visible(writer):
    writer.write(_entry(_thinking(THINK), _text(SAID_EXACT)), SESSION_A)

    assert _default_hits(writer.db, "RedteamThinkNeedle"), _rows(writer.db)


def test_said_text_after_equal_thinking_in_the_same_session_becomes_visible(writer):
    """Thinking first, then the same words said aloud: saying it makes the row visible."""
    writer.write(_entry(_thinking(THINK)), SESSION_A)
    writer.write(_entry(_text(SAID_EXACT), ts="2026-11-17T12:05:00Z"), SESSION_A)

    assert _default_hits(writer.db, "RedteamThinkNeedle"), _rows(writer.db)


# ── 4. Readers of source_class ──


def test_source_class_migration_accepts_a_db_that_holds_agent_thinking_rows(tmp_path):
    from brainlayer.source_class_migration import migrate_source_class

    db = tmp_path / "copy.db"
    store = VectorStore(db)
    store.conn.cursor().execute(
        """
        INSERT INTO chunks (id, content, metadata, source_file, project, content_type, char_count, source_class)
        VALUES ('thought', 'RedteamThinkNeedle body', '{}', '/session.jsonl', 'myproject', 'assistant_text', 23,
                'agent-thinking')
        """
    )
    store.close()

    receipt = migrate_source_class(db, git_sha="a" * 40, actor="test-1824")
    assert receipt["distribution"].get("agent-thinking") == 1


def test_expand_context_labels_thinking_rows_and_keeps_showing_them(tmp_path):
    """brain_expand's Python twin treats thinking like desktop (shown), with a thinking label."""
    store = VectorStore(tmp_path / "expand.db")
    try:
        for position, (chunk_id, source_class) in enumerate([("said", "cli-agent"), ("thought", "agent-thinking")]):
            store.conn.cursor().execute(
                """
                INSERT INTO chunks (id, content, metadata, source_file, project, content_type, char_count,
                                    conversation_id, position, source_class)
                VALUES (?, ?, '{}', '/session.jsonl', 'brainlayer', 'assistant_text', 30, 'conversation', ?, ?)
                """,
                (chunk_id, f"ExpandLabelNeedle {chunk_id} body text", position, source_class),
            )
        context = store.get_context("said", before=1, after=1)
    finally:
        store.close()

    labels = {row["id"]: (row["thinking"], row["source_class"]) for row in context["context"]}
    assert labels == {"said": (False, "cli-agent"), "thought": (True, "agent-thinking")}
    assert context["target"]["thinking"] is False
