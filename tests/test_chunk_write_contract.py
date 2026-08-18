"""Every production chunk writer must persist the same canonical insert columns."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

import pytest

from brainlayer.chunk_origin import detect_chunk_origin
from brainlayer.drain import _apply_hook, _apply_store, _apply_watcher
from brainlayer.store import store_memory
from brainlayer.vector_store import VectorStore

REPO_ROOT = Path(__file__).resolve().parents[1]
ISO_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(\.\d+)?(Z|[+-]\d{2}:\d{2})$")

CANONICAL_NONEMPTY = (
    "id",
    "content",
    "source_file",
    "content_type",
    "char_count",
    "source",
    "created_at",
    "chunk_origin",
    "content_hash",
    "ingested_at",
    "seen_count",
    "last_seen_at",
    "content_class",
    "preview_text",
    "brick_id",
    "source_uri",
    "status",
    "dedupe_hash",
    "simhash",
    "simhash_band_0",
    "simhash_band_1",
    "simhash_band_2",
    "simhash_band_3",
)

SWIFT_STORE_FILES = (
    REPO_ROOT / "brain-bar/Sources/BrainBar/BrainDatabase.swift",
    REPO_ROOT / "brain-bar/Sources/BrainBarDaemon/BrainDatabase.swift",
)

SWIFT_CANONICAL_COLUMNS = (
    "id",
    "content",
    "metadata",
    "source_file",
    "project",
    "tags",
    "importance",
    "source",
    "content_type",
    "value_type",
    "char_count",
    "created_at",
    "preview_text",
    "conversation_id",
    "position",
    "content_hash",
    "chunk_origin",
    "ingested_at",
    "seen_count",
    "last_seen_at",
    "content_class",
    "dedupe_hash",
    "simhash",
    "simhash_band_0",
    "simhash_band_1",
    "simhash_band_2",
    "simhash_band_3",
    "brick_id",
    "source_uri",
    "status",
)

INSERT_GUARD = re.compile(r"INSERT\s+(OR\s+\w+\s+)?INTO\s+chunks\b", re.IGNORECASE)

CONTENT = "Repair-d writer alignment contract payload about authentication decisions."
PROJECT = "brainlayer"


def _embed(text: str) -> list[float]:
    seed = sum(ord(c) for c in text[:50]) % 100
    return [float(seed + i) / 1000.0 for i in range(1024)]


def _row(store: VectorStore, chunk_id: str) -> dict:
    cursor = store.conn.cursor()
    columns = [info[1] for info in cursor.execute("PRAGMA table_info(chunks)")]
    values = cursor.execute("SELECT * FROM chunks WHERE id = ?", (chunk_id,)).fetchone()
    assert values is not None, chunk_id
    return dict(zip(columns, values))


def _assert_canonical_row(row: dict, *, content: str) -> None:
    for column in CANONICAL_NONEMPTY:
        assert row.get(column) not in (None, ""), column
    assert ISO_RE.match(str(row["created_at"])), row["created_at"]
    assert ISO_RE.match(str(row["last_seen_at"])), row["last_seen_at"]
    assert row["content"] == content
    assert row["chunk_origin"] == detect_chunk_origin(content)
    assert row["content_hash"] == hashlib.sha256(content.encode("utf-8")).hexdigest()
    assert int(row["char_count"]) == len(content)
    assert str(row["status"]) == "active"
    assert str(row["brick_id"])
    assert str(row["source_uri"])
    assert str(row["preview_text"]).strip()


@pytest.fixture
def store(tmp_path):
    db = VectorStore(tmp_path / "contract.db")
    yield db
    db.close()


def test_prepare_canonical_insert_fills_required_columns():
    from brainlayer.chunk_write import prepare_canonical_insert

    prepared = prepare_canonical_insert(
        {
            "id": "manual-contract-1",
            "content": CONTENT,
            "source_file": "brainlayer-store",
            "source": "manual",
            "project": PROJECT,
            "content_type": "learning",
        }
    )
    _assert_canonical_row(prepared, content=CONTENT)


def test_store_memory_writes_canonical_columns(store):
    result = store_memory(
        store=store,
        embed_fn=_embed,
        content=CONTENT,
        memory_type="learning",
        project=PROJECT,
        chunk_id="manual-contract-store",
    )
    _assert_canonical_row(_row(store, result["id"]), content=CONTENT)


def test_upsert_chunks_writes_canonical_columns(store):
    chunk_id = "idx-contract-upsert"
    store.upsert_chunks(
        [
            {
                "id": chunk_id,
                "content": CONTENT,
                "metadata": {},
                "source_file": "/Users/etanheyman/.claude/projects/-Users-etanheyman-Gits-brainlayer/session.jsonl",
                "project": PROJECT,
                "content_type": "assistant_text",
                "value_type": "high",
                "char_count": len(CONTENT),
                "source": "claude_code",
                "created_at": "2026-08-18T00:00:00Z",
            }
        ],
        [_embed(CONTENT)],
    )
    row = _row(store, chunk_id)
    _assert_canonical_row(row, content=CONTENT)
    assert row["source_class"] == "cli-agent"


def test_drain_watcher_writes_canonical_columns(store, monkeypatch):
    from brainlayer import drain as drain_mod

    monkeypatch.setattr(drain_mod, "_record_watcher_liveness", lambda *_args, **_kwargs: None)
    chunk_id = "rt-contract-watch"
    _apply_watcher(
        store.conn,
        {
            "chunk_id": chunk_id,
            "content": CONTENT,
            "source_file": "/Users/etanheyman/.claude/projects/-Users-etanheyman-Gits-brainlayer/session.jsonl",
            "project": PROJECT,
            "content_type": "assistant_text",
            "value_type": "high",
            "created_at": "2026-08-18T00:00:01Z",
            "conversation_id": "session-contract",
            "sender": "assistant",
            "source_end_offset": 42,
            "queued_at": 1780190105.85048,
        },
    )
    row = _row(store, chunk_id)
    _assert_canonical_row(row, content=CONTENT)
    assert row["source_end_offset"] == 42
    assert row["source_class"] == "cli-agent"


def test_drain_store_replay_writes_canonical_columns(store):
    chunk_id = "manual-contract-replay"
    _apply_store(
        store.conn,
        {
            "chunk_id": chunk_id,
            "content": CONTENT,
            "memory_type": "learning",
            "project": PROJECT,
            "source": "manual",
            "created_at": "2026-08-18T00:00:02Z",
        },
    )
    _assert_canonical_row(_row(store, chunk_id), content=CONTENT)


def test_drain_hook_writes_canonical_columns(store):
    result = _apply_hook(
        store.conn,
        {
            "session_id": "session-contract-hook",
            "chunk_id": "rt-contract-hook",
            "content": CONTENT,
            "source_file": "/Users/etanheyman/.claude/projects/-Users-etanheyman-Gits-brainlayer/session.jsonl",
            "created_at": "2026-08-18T00:00:03Z",
        },
    )
    _assert_canonical_row(_row(store, result.chunk_id), content=CONTENT)


def test_python_writers_agree_on_canonical_subset(store):
    variants = {
        "store": "Repair-d store writer unique payload about authentication decisions.",
        "upsert": "Repair-d upsert writer unique payload about authentication decisions.",
        "replay": "Repair-d replay writer unique payload about authentication decisions.",
    }
    store_id = store_memory(
        store=store,
        embed_fn=_embed,
        content=variants["store"],
        memory_type="learning",
        project=PROJECT,
        chunk_id="agree-store",
    )["id"]
    store.upsert_chunks(
        [
            {
                "id": "agree-upsert",
                "content": variants["upsert"],
                "metadata": {},
                "source_file": "brainlayer-store",
                "project": PROJECT,
                "content_type": "learning",
                "source": "manual",
                "created_at": "2026-08-18T00:00:04Z",
            }
        ],
        [_embed(variants["upsert"])],
    )
    _apply_store(
        store.conn,
        {
            "chunk_id": "agree-replay",
            "content": variants["replay"],
            "memory_type": "learning",
            "project": PROJECT,
            "source": "manual",
            "created_at": "2026-08-18T00:00:05Z",
        },
    )
    payloads = {
        "store": _row(store, store_id),
        "upsert": _row(store, "agree-upsert"),
        "replay": _row(store, "agree-replay"),
    }
    for name, row in payloads.items():
        _assert_canonical_row(row, content=variants[name])
        assert row["brick_id"] == row["id"], name
        assert row["source_uri"] == row["source_file"], name
        assert row["status"] == "active", name


def test_brainbar_store_sql_prepares_on_fresh_schema():
    import sqlite3

    type_map = {
        "importance": "INTEGER",
        "char_count": "INTEGER",
        "ingested_at": "INTEGER",
        "seen_count": "INTEGER",
    }
    for path in SWIFT_STORE_FILES:
        text = path.read_text(encoding="utf-8")
        match = re.search(
            r'let sql = """\s*(INSERT INTO chunks \([^)]+\)\s*VALUES \([^"]+\))',
            text,
            re.DOTALL,
        )
        assert match, path
        insert_sql = " ".join(match.group(1).split())
        columns = [
            part.strip() for part in re.search(r"INSERT INTO chunks \(([^)]+)\)", insert_sql).group(1).split(",")
        ]
        for column in SWIFT_CANONICAL_COLUMNS:
            assert column in columns, (path.name, column)
        column_defs = ", ".join(f"{column} {type_map.get(column, 'TEXT')}" for column in columns)
        conn = sqlite3.connect(":memory:")
        conn.execute(f"CREATE TABLE chunks ({column_defs})")
        placeholder_count = insert_sql.count("?")
        values = [1 if index in {5, 7, 13, 14} else "test" for index in range(placeholder_count)]
        conn.execute(insert_sql, values)
        conn.close()


def test_production_inserts_go_through_chunk_write():
    allowed = {
        REPO_ROOT / "src/brainlayer/chunk_write.py",
        REPO_ROOT / "src/brainlayer/vector_store.py",
    }
    offenders = []
    scan_roots = (
        REPO_ROOT / "src/brainlayer",
        REPO_ROOT / "scripts",
        REPO_ROOT / "hooks",
    )
    for root in scan_roots:
        for path in root.rglob("*.py"):
            if path in allowed:
                continue
            body = path.read_text(encoding="utf-8")
            if INSERT_GUARD.search(body):
                offenders.append(str(path.relative_to(REPO_ROOT)))
    assert offenders == []
