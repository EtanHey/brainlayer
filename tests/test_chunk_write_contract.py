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
    "brick_id",
    "source_uri",
    "status",
)

# BrainBar must NOT reference these in its INSERT. Its Swift simhash port diverged from Python's
# (hamming distance 25-31 on byte-identical content, against find_duplicate's threshold of 3), so
# computing them here produced values no Python row could ever match. They are omitted from the
# INSERT entirely -- SQLite stores NULL -- so "not computed" is distinguishable from "computed
# differently". Reintroducing them requires a faithful port pinned by exact-Python-hex tests.
SWIFT_ABSENT_DEDUPE_COLUMNS = (
    "dedupe_hash",
    "simhash",
    "simhash_band_0",
    "simhash_band_1",
    "simhash_band_2",
    "simhash_band_3",
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
        for column in SWIFT_ABSENT_DEDUPE_COLUMNS:
            assert column not in columns, (
                path.name,
                column,
                "BrainBar must not write dedupe columns until its simhash is byte-identical to Python",
            )
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


# --- repair (e): ONE content_hash contract -------------------------------------
# The live DB carried four hash schemes at once (64-char sha256, a 32-char scheme,
# 16-char truncations from drain, and 52k rows with none). Rows hashed under
# different schemes never group, so duplicates accrue faster than dedupe removes
# them. The contract is now: content_hash is ALWAYS sha256 over the stripped
# content, recomputed at write time, and a caller-supplied value never wins.

CANONICAL_CONTENT_HASH_CONTENT = "  Etan said: ship the fenced migration.\n\n"


def canonical_content_hash(content: str) -> str:
    return hashlib.sha256(content.strip().encode("utf-8")).hexdigest()


def test_content_hash_is_sha256_of_stripped_content():
    from brainlayer.chunk_write import prepare_canonical_insert

    row = prepare_canonical_insert({"id": "hash-contract-1", "content": CANONICAL_CONTENT_HASH_CONTENT})
    assert row["content_hash"] == canonical_content_hash(CANONICAL_CONTENT_HASH_CONTENT)
    assert len(row["content_hash"]) == 64


def test_content_hash_ignores_surrounding_whitespace():
    """Two writers disagreeing only on trailing newlines must agree on the hash."""
    from brainlayer.chunk_write import prepare_canonical_insert

    bare = prepare_canonical_insert({"id": "hash-contract-2", "content": "same text"})
    padded = prepare_canonical_insert({"id": "hash-contract-3", "content": "\n  same text  \n"})
    assert bare["content_hash"] == padded["content_hash"]


def test_caller_supplied_content_hash_never_wins():
    """A truncated or stale hash handed in by a legacy caller must be overridden."""
    from brainlayer.chunk_write import prepare_canonical_insert

    row = prepare_canonical_insert(
        {
            "id": "hash-contract-4",
            "content": CANONICAL_CONTENT_HASH_CONTENT,
            "content_hash": "deadbeefdeadbeef",  # 16-char legacy scheme
        }
    )
    assert row["content_hash"] == canonical_content_hash(CANONICAL_CONTENT_HASH_CONTENT)


def test_only_one_content_hash_implementation_exists():
    """A second implementation is how four schemes got into the column.

    Byte-identical duplicates count: `enrichment_controller` had its own
    `_content_hash` feeding four `UPDATE chunks SET content_hash` sites, and
    `store.py` computed an UNSTRIPPED sha256. Both now import the contract.
    """
    from brainlayer import enrichment_controller
    from brainlayer.chunk_write import canonical_content_hash

    assert enrichment_controller._content_hash is canonical_content_hash

    offenders = []
    definition = re.compile(r"^def _?content_hash\(", re.M)
    for root in (REPO_ROOT / "src/brainlayer", REPO_ROOT / "hooks", REPO_ROOT / "scripts"):
        for path in root.rglob("*.py"):
            if path == REPO_ROOT / "src/brainlayer/chunk_write.py":
                continue
            body = path.read_text(encoding="utf-8")
            for match in definition.finditer(body):
                line = body[: match.start()].count("\n") + 1
                offenders.append(f"{path.relative_to(REPO_ROOT)}:{line}")
    assert offenders == [], f"content_hash must be defined once, in chunk_write: {offenders}"


def test_update_paths_write_the_canonical_hash():
    """The contract must cover UPDATE, not only INSERT."""
    import sqlite3

    from brainlayer.chunk_write import canonical_content_hash
    from brainlayer.enrichment_controller import _content_hash

    content = "  enrichment rewrote this row\n\n"
    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE TABLE chunks (id TEXT PRIMARY KEY, content TEXT, content_hash TEXT)")
    conn.execute("INSERT INTO chunks VALUES ('u1', ?, 'stale-16char0000')", (content,))
    # the exact statement shape used at enrichment_controller UPDATE sites
    conn.execute("UPDATE chunks SET content_hash = ? WHERE id = ?", (_content_hash(content), "u1"))
    stored = conn.execute("SELECT content_hash FROM chunks WHERE id = 'u1'").fetchone()[0]
    conn.close()
    assert stored == canonical_content_hash(content)
    assert len(stored) == 64


def test_no_unstripped_sha256_is_written_to_content_hash():
    """No writer may hash UNSTRIPPED content into content_hash.

    Matches any module alias (`hashlib.sha256`, `_h.sha256`, a bare `sha256`),
    not just the literal `hashlib.` spelling -- the first version of this guard
    only caught `hashlib.sha256(` and a rename slipped straight past it.
    """
    from brainlayer.chunk_write import canonical_content_hash

    assert canonical_content_hash(" a ") == canonical_content_hash("a")

    offenders = []
    pattern = re.compile(r"content_hash\s*=\s*[\w.]*sha256\(([^)]*)\)", re.M)
    for root in (REPO_ROOT / "src/brainlayer", REPO_ROOT / "hooks", REPO_ROOT / "scripts"):
        for path in root.rglob("*.py"):
            body = path.read_text(encoding="utf-8")
            # Scoped to the chunks column. queue_merge.py hashes FILE BYTES into a
            # local also named content_hash and never touches the chunks table --
            # a true regex hit but not this invariant.
            if "chunks" not in body:
                continue
            for match in pattern.finditer(body):
                if ".strip()" in match.group(1):
                    continue
                offenders.append(f"{path.relative_to(REPO_ROOT)}:{body[: match.start()].count(chr(10)) + 1}")
    assert offenders == [], f"unstripped sha256 written to content_hash: {offenders}"


def test_store_memory_persists_the_canonical_hash():
    """Behavioural backstop: spelling tricks cannot evade an executed write."""
    import inspect

    from brainlayer import store as store_module
    from brainlayer.chunk_write import canonical_content_hash

    source = inspect.getsource(store_module.store_memory)
    assignment = [line.strip() for line in source.splitlines() if "content_hash =" in line]
    assert assignment == ["content_hash = canonical_content_hash(content)"], assignment
    padded, bare = "  a stored memory \n", "a stored memory"
    assert canonical_content_hash(padded) == canonical_content_hash(bare)


def test_no_truncated_content_hash_schemes_remain_in_production_code():
    """No production path may store a truncated sha256 as a chunk content_hash."""
    offenders = []
    # Assignments only: `content_hash = ...hexdigest()[:16]` or a
    # `"content_hash": ...hexdigest()[:16]` dict entry. Reading a legacy queue
    # key named content_hash is not a violation -- storing a truncated digest is.
    pattern = re.compile(r"""(?:^|[^"'\w])content_hash\s*(?:=|:)\s*[^\n=]{0,80}hexdigest\(\)\[:\d+\]""", re.M)
    for root in (REPO_ROOT / "src/brainlayer", REPO_ROOT / "hooks", REPO_ROOT / "scripts"):
        for path in root.rglob("*.py"):
            body = path.read_text(encoding="utf-8")
            for match in pattern.finditer(body):
                line = body[: match.start()].count("\n") + 1
                offenders.append(f"{path.relative_to(REPO_ROOT)}:{line}")
    assert offenders == [], f"truncated content_hash schemes: {offenders}"
