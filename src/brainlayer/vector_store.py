"""SQLite-vec based vector store for fast search.

Thin facade: VectorStore inherits from focused mixin modules.
See search_repo.py, kg_repo.py, session_repo.py for the extracted methods.
"""

import json
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

import apsw
import apsw.bestpractice
import sqlite_vec

from ._helpers import (  # noqa: I001
    _DEFAULT_MIN_CHARS as _DEFAULT_MIN_CHARS,
)
from ._helpers import (
    _SOURCE_MIN_CHARS as _SOURCE_MIN_CHARS,
)
from ._helpers import (
    _escape_fts5_query as _escape_fts5_query,
)
from ._helpers import (
    _safe_json_loads as _safe_json_loads,
)
from ._helpers import (
    serialize_f32 as serialize_f32,
)
from ._helpers import (
    source_aware_min_chars as source_aware_min_chars,
)
from .kg_repo import KGMixin
from .search_repo import SearchMixin
from .session_repo import SessionMixin


def _set_busy_timeout_hook(conn: apsw.Connection) -> None:
    """Set busy_timeout on every new connection before any other hooks.

    APSW bestpractice hooks (connection_optimize) run PRAGMA optimize inside
    the Connection() constructor. Without busy_timeout set first, this PRAGMA
    fails with BusyError when other processes hold the DB lock.
    """
    conn.setbusytimeout(30_000)  # 30 seconds — needed for parallel enrichment workers + MCP


# Register busy_timeout hook BEFORE bestpractice hooks so it fires first.
# bestpractice.apply() adds hooks that run PRAGMA optimize inside Connection(),
# which needs busy_timeout active or it crashes under contention.
apsw.connection_hooks.insert(0, _set_busy_timeout_hook)
apsw.bestpractice.apply(apsw.bestpractice.recommended)


class VectorStore(SearchMixin, KGMixin, SessionMixin):
    """SQLite-vec based vector store.

    Core chunk CRUD and schema management live here.
    Search, KG, and session methods are inherited from mixin classes.
    """

    # Retry settings for DB init under contention (multiple MCP instances + enrichment)
    _INIT_MAX_RETRIES = 5
    _INIT_BASE_DELAY = 0.5  # seconds

    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db_with_retry()

    def _init_db_with_retry(self) -> None:
        """Initialize DB with retry on BusyError.

        Multiple BrainLayer processes (MCP instances, daemon, enrichment) may
        contend for write locks during DDL. Retry with exponential backoff
        instead of crashing on the first BusyError.
        """
        import time

        last_err = None
        for attempt in range(self._INIT_MAX_RETRIES):
            try:
                self._init_db()
                return
            except apsw.BusyError as e:
                last_err = e
                delay = self._INIT_BASE_DELAY * (2**attempt)
                import sys

                print(
                    f"  DB init BusyError (attempt {attempt + 1}/{self._INIT_MAX_RETRIES}), "
                    f"retrying in {delay:.1f}s...",
                    file=sys.stderr,
                )
                time.sleep(delay)
        raise last_err  # type: ignore[misc]

    def _init_db(self) -> None:
        """Initialize database with vector extension."""
        self.conn = apsw.Connection(str(self.db_path))

        # Set busy timeout IMMEDIATELY via APSW native method — before any DDL.
        self.conn.setbusytimeout(10_000)  # 10 seconds

        self.conn.enableloadextension(True)
        self.conn.loadextension(sqlite_vec.loadable_path())
        self.conn.enableloadextension(False)

        cursor = self.conn.cursor()

        # WAL mode is persistent on the DB file — set it every time
        cursor.execute("PRAGMA journal_mode = WAL")

        # Create tables
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS chunks (
                id TEXT PRIMARY KEY,
                content TEXT NOT NULL,
                metadata TEXT NOT NULL,
                source_file TEXT NOT NULL,
                project TEXT,
                content_type TEXT,
                value_type TEXT,
                char_count INTEGER,
                source TEXT,
                sender TEXT,
                language TEXT,
                conversation_id TEXT,
                position INTEGER,
                context_summary TEXT
            )
        """)

        # Add columns if upgrading existing DB
        existing_cols = {row[1] for row in cursor.execute("PRAGMA table_info(chunks)")}
        for col, typ in [
            ("source", "TEXT"),
            ("sender", "TEXT"),
            ("language", "TEXT"),
            ("conversation_id", "TEXT"),
            ("position", "INTEGER"),
            ("context_summary", "TEXT"),
            ("tags", "TEXT"),
            ("tag_confidence", "REAL"),
            ("summary", "TEXT"),
            ("importance", "REAL"),
            ("intent", "TEXT"),
            ("enriched_at", "TEXT"),
            ("primary_symbols", "TEXT"),
            ("resolved_query", "TEXT"),
            ("epistemic_level", "TEXT"),
            ("version_scope", "TEXT"),
            ("debt_impact", "TEXT"),
            ("external_deps", "TEXT"),
            ("created_at", "TEXT"),
            ("sentiment_label", "TEXT"),
            ("sentiment_score", "REAL"),
            ("sentiment_signals", "TEXT"),
        ]:
            if col not in existing_cols:
                cursor.execute(f"ALTER TABLE chunks ADD COLUMN {col} {typ}")

        # Indexes for filtering
        for idx, col in [
            ("idx_chunks_source", "source"),
            ("idx_chunks_sender", "sender"),
            ("idx_chunks_conversation", "conversation_id"),
            ("idx_chunks_intent", "intent"),
            ("idx_chunks_importance", "importance"),
            ("idx_chunks_enriched", "enriched_at"),
            ("idx_chunks_created", "created_at"),
            ("idx_chunks_sentiment", "sentiment_label"),
            ("idx_chunks_project", "project"),
        ]:
            cursor.execute(f"CREATE INDEX IF NOT EXISTS {idx} ON chunks({col})")

        # Vector table (1024 dims for bge-large-en-v1.5)
        cursor.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS chunk_vectors USING vec0(
                chunk_id TEXT PRIMARY KEY,
                embedding FLOAT[1024]
            )
        """)

        # FTS5 full-text search
        cursor.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
                content, chunk_id UNINDEXED
            )
        """)

        # FTS5 sync triggers
        cursor.execute("""
            CREATE TRIGGER IF NOT EXISTS chunks_fts_insert AFTER INSERT ON chunks BEGIN
                INSERT INTO chunks_fts(content, chunk_id) VALUES (new.content, new.id);
            END
        """)
        cursor.execute("""
            CREATE TRIGGER IF NOT EXISTS chunks_fts_delete AFTER DELETE ON chunks BEGIN
                DELETE FROM chunks_fts WHERE chunk_id = old.id;
            END
        """)
        cursor.execute("""
            CREATE TRIGGER IF NOT EXISTS chunks_fts_update AFTER UPDATE OF content ON chunks BEGIN
                DELETE FROM chunks_fts WHERE chunk_id = old.id;
                INSERT INTO chunks_fts(content, chunk_id) VALUES (new.content, new.id);
            END
        """)

        # Session context table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS session_context (
                session_id TEXT PRIMARY KEY,
                project TEXT,
                branch TEXT,
                pr_number INTEGER,
                commit_shas TEXT,
                files_changed TEXT,
                started_at TEXT,
                ended_at TEXT,
                created_at TEXT
            )
        """)
        existing_sc_cols = {row[1] for row in cursor.execute("PRAGMA table_info(session_context)")}
        for col in ("plan_name", "plan_phase", "story_id"):
            if col not in existing_sc_cols:
                cursor.execute(f"ALTER TABLE session_context ADD COLUMN {col} TEXT")
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS file_interactions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                file_path TEXT NOT NULL,
                timestamp TEXT,
                session_id TEXT,
                action TEXT,
                chunk_id TEXT,
                project TEXT
            )
        """)
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_file_interactions_path ON file_interactions(file_path)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_file_interactions_session ON file_interactions(session_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_session_context_project ON session_context(project)")

        # Operations table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS operations (
                id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL,
                operation_type TEXT,
                chunk_ids TEXT,
                summary TEXT,
                outcome TEXT,
                started_at TEXT,
                ended_at TEXT,
                step_count INTEGER DEFAULT 0,
                created_at TEXT
            )
        """)
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_operations_session ON operations(session_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_operations_type ON operations(operation_type)")

        # Topic chains table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS topic_chains (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                file_path TEXT NOT NULL,
                session_a TEXT NOT NULL,
                session_b TEXT NOT NULL,
                shared_actions INTEGER DEFAULT 0,
                time_delta_hours REAL,
                project TEXT,
                created_at TEXT
            )
        """)
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_topic_chains_file ON topic_chains(file_path)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_topic_chains_session ON topic_chains(session_a)")

        # Session enrichment table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS session_enrichments (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL UNIQUE,
                file_path TEXT,
                enrichment_version TEXT NOT NULL DEFAULT '1.0',
                enrichment_model TEXT,
                enrichment_timestamp TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
                session_start_time TEXT,
                session_end_time TEXT,
                duration_seconds INTEGER,
                message_count INTEGER NOT NULL DEFAULT 0,
                user_message_count INTEGER NOT NULL DEFAULT 0,
                assistant_message_count INTEGER NOT NULL DEFAULT 0,
                tool_call_count INTEGER NOT NULL DEFAULT 0,
                session_summary TEXT,
                primary_intent TEXT,
                outcome TEXT CHECK(outcome IN ('success','partial_success','failure','abandoned','ongoing')),
                complexity_score INTEGER CHECK(complexity_score BETWEEN 1 AND 10),
                session_quality_score INTEGER CHECK(session_quality_score BETWEEN 1 AND 10),
                decisions_made TEXT DEFAULT '[]',
                corrections TEXT DEFAULT '[]',
                learnings TEXT DEFAULT '[]',
                mistakes TEXT DEFAULT '[]',
                patterns TEXT DEFAULT '[]',
                topic_tags TEXT DEFAULT '[]',
                tool_usage_stats TEXT DEFAULT '[]',
                what_worked TEXT,
                what_failed TEXT,
                summary_embedding BLOB
            )
        """)
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_session_enrichments_session ON session_enrichments(session_id)")
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_session_enrichments_project ON session_enrichments(primary_intent)"
        )
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_session_enrichments_outcome ON session_enrichments(outcome)")
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_session_enrichments_quality ON session_enrichments(session_quality_score)"
        )

        # Session enrichment FTS5
        cursor.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS session_enrichments_fts USING fts5(
                session_summary, what_worked, what_failed, session_id UNINDEXED
            )
        """)

        # Phase commits table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS phase_commits (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                commit_hash TEXT NOT NULL,
                commit_message TEXT,
                phase_name TEXT,
                session_id TEXT,
                project TEXT,
                files_changed TEXT,
                confidence_score REAL,
                outcome TEXT,
                reversibility TEXT,
                created_at TEXT DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now'))
            )
        """)
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_phase_commits_project ON phase_commits(project)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_phase_commits_phase ON phase_commits(phase_name)")

        # source_project_id column
        if "source_project_id" not in existing_cols:
            cursor.execute("ALTER TABLE chunks ADD COLUMN source_project_id TEXT")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_chunks_source_project ON chunks(source_project_id)")

        # ── Knowledge Graph tables ──────────────────────────────────────

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS kg_entities (
                id TEXT PRIMARY KEY,
                entity_type TEXT NOT NULL,
                name TEXT NOT NULL,
                metadata TEXT DEFAULT '{}',
                created_at TEXT DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
                updated_at TEXT DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
                UNIQUE(entity_type, name)
            )
        """)
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_kg_entities_type ON kg_entities(entity_type)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_kg_entities_name ON kg_entities(name)")

        kg_entity_cols = {row[1] for row in cursor.execute("PRAGMA table_info(kg_entities)")}
        if "user_verified" not in kg_entity_cols:
            cursor.execute("ALTER TABLE kg_entities ADD COLUMN user_verified INTEGER DEFAULT 0")

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS kg_relations (
                id TEXT PRIMARY KEY,
                source_id TEXT NOT NULL,
                target_id TEXT NOT NULL,
                relation_type TEXT NOT NULL,
                properties TEXT DEFAULT '{}',
                confidence REAL DEFAULT 1.0,
                created_at TEXT DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
                UNIQUE(source_id, target_id, relation_type)
            )
        """)
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_kg_relations_source ON kg_relations(source_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_kg_relations_target ON kg_relations(target_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_kg_relations_type ON kg_relations(relation_type)")

        kg_rel_cols = {row[1] for row in cursor.execute("PRAGMA table_info(kg_relations)")}
        if "user_verified" not in kg_rel_cols:
            cursor.execute("ALTER TABLE kg_relations ADD COLUMN user_verified INTEGER DEFAULT 0")

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS kg_entity_chunks (
                entity_id TEXT NOT NULL,
                chunk_id TEXT NOT NULL,
                relevance REAL DEFAULT 1.0,
                context TEXT,
                PRIMARY KEY (entity_id, chunk_id)
            )
        """)
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_kg_ec_entity ON kg_entity_chunks(entity_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_kg_ec_chunk ON kg_entity_chunks(chunk_id)")

        cursor.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS kg_vec_entities USING vec0(
                entity_id TEXT PRIMARY KEY,
                embedding FLOAT[1024]
            )
        """)

        cursor.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS kg_entities_fts USING fts5(
                name, metadata, entity_id UNINDEXED
            )
        """)

        # KG FTS5 sync triggers
        cursor.execute("""
            CREATE TRIGGER IF NOT EXISTS kg_entities_fts_insert AFTER INSERT ON kg_entities BEGIN
                INSERT INTO kg_entities_fts(name, metadata, entity_id)
                VALUES (new.name, new.metadata, new.id);
            END
        """)
        cursor.execute("""
            CREATE TRIGGER IF NOT EXISTS kg_entities_fts_delete AFTER DELETE ON kg_entities BEGIN
                DELETE FROM kg_entities_fts WHERE entity_id = old.id;
            END
        """)
        cursor.execute("""
            CREATE TRIGGER IF NOT EXISTS kg_entities_fts_update
            AFTER UPDATE OF name, metadata ON kg_entities BEGIN
                DELETE FROM kg_entities_fts WHERE entity_id = old.id;
                INSERT INTO kg_entities_fts(name, metadata, entity_id)
                VALUES (new.name, new.metadata, new.id);
            END
        """)

        # Entity aliases table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS kg_entity_aliases (
                alias TEXT NOT NULL,
                entity_id TEXT NOT NULL,
                alias_type TEXT DEFAULT 'name',
                created_at TEXT,
                PRIMARY KEY (alias, entity_id)
            )
        """)
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_kg_alias_entity ON kg_entity_aliases(entity_id)")

        # KG standard spec migrations
        for col, default in [
            ("canonical_name", "TEXT"),
            ("description", "TEXT"),
            ("confidence", "REAL DEFAULT 1.0"),
            ("importance", "REAL DEFAULT 0.5"),
            ("valid_from", "TEXT"),
            ("valid_until", "TEXT"),
            ("group_id", "TEXT"),
        ]:
            if col not in kg_entity_cols:
                cursor.execute(f"ALTER TABLE kg_entities ADD COLUMN {col} {default}")

        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_kg_entities_canonical ON kg_entities(canonical_name, entity_type)"
        )
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_kg_entities_valid ON kg_entities(valid_from, valid_until)")

        for col, default in [
            ("fact", "TEXT"),
            ("importance", "REAL DEFAULT 0.5"),
            ("valid_from", "TEXT"),
            ("valid_until", "TEXT"),
            ("expired_at", "TEXT"),
            ("source_chunk_id", "TEXT"),
        ]:
            if col not in kg_rel_cols:
                cursor.execute(f"ALTER TABLE kg_relations ADD COLUMN {col} {default}")

        cursor.execute("CREATE INDEX IF NOT EXISTS idx_kg_relations_validity ON kg_relations(valid_from, valid_until)")

        ec_cols = {row[1] for row in cursor.execute("PRAGMA table_info(kg_entity_chunks)")}
        if "mention_type" not in ec_cols:
            cursor.execute("ALTER TABLE kg_entity_chunks ADD COLUMN mention_type TEXT")

        # kg_current_facts view
        cursor.execute("DROP VIEW IF EXISTS kg_current_facts")
        cursor.execute("""
            CREATE VIEW IF NOT EXISTS kg_current_facts AS
            SELECT * FROM kg_relations
            WHERE (valid_from IS NULL OR valid_from <= strftime('%Y-%m-%dT%H:%M:%fZ','now'))
              AND (valid_until IS NULL OR valid_until >= strftime('%Y-%m-%dT%H:%M:%fZ','now'))
              AND expired_at IS NULL
        """)

        # FTS5 backfill check
        fts_count = list(cursor.execute("SELECT COUNT(*) FROM chunks_fts"))[0][0]
        chunk_count = list(cursor.execute("SELECT COUNT(*) FROM chunks"))[0][0]
        if chunk_count > 0 and fts_count == 0:
            cursor.execute("""
                INSERT INTO chunks_fts(content, chunk_id)
                SELECT content, id FROM chunks
            """)

        # Thread-local storage for per-thread read connections.
        # APSW connections are NOT thread-safe — each thread needs its own.
        # This prevents "Connection is busy in another thread" when parallel
        # MCP tool calls (e.g., brain_search) hit the same VectorStore.
        self._local = threading.local()

    def _get_read_conn(self) -> apsw.Connection:
        """Get or create a per-thread readonly connection."""
        conn = getattr(self._local, "read_conn", None)
        if conn is None:
            conn = apsw.Connection(str(self.db_path), flags=apsw.SQLITE_OPEN_READONLY)
            conn.enableloadextension(True)
            conn.loadextension(sqlite_vec.loadable_path())
            conn.enableloadextension(False)
            conn.setbusytimeout(30_000)
            self._local.read_conn = conn
        return conn

    def _read_cursor(self):
        """Return a cursor for read operations using a per-thread readonly connection."""
        return self._get_read_conn().cursor()

    # ── Chunk CRUD ──────────────────────────────────────────────────────

    def upsert_chunks(self, chunks: List[Dict[str, Any]], embeddings: List[List[float]]) -> int:
        """Upsert chunks with embeddings."""
        if len(chunks) != len(embeddings):
            raise ValueError("Chunks and embeddings must have same length")

        cursor = self.conn.cursor()

        for chunk, embedding in zip(chunks, embeddings):
            chunk_id = chunk["id"]

            cursor.execute(
                """
                INSERT INTO chunks
                (id, content, metadata, source_file, project,
                 content_type, value_type, char_count, source, created_at,
                 conversation_id, position, sender)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    content = excluded.content,
                    metadata = excluded.metadata,
                    source_file = excluded.source_file,
                    project = excluded.project,
                    content_type = excluded.content_type,
                    value_type = excluded.value_type,
                    char_count = excluded.char_count,
                    source = excluded.source,
                    created_at = COALESCE(chunks.created_at, excluded.created_at),
                    conversation_id = COALESCE(excluded.conversation_id, chunks.conversation_id),
                    position = COALESCE(excluded.position, chunks.position),
                    sender = COALESCE(excluded.sender, chunks.sender)
            """,
                (
                    chunk_id,
                    chunk["content"],
                    json.dumps(chunk["metadata"]),
                    chunk["source_file"],
                    chunk.get("project"),
                    chunk.get("content_type"),
                    chunk.get("value_type"),
                    chunk.get("char_count", 0),
                    chunk.get("source", "claude_code"),
                    chunk.get("created_at"),
                    chunk.get("conversation_id"),
                    chunk.get("position"),
                    chunk.get("sender"),
                ),
            )

            # Upsert vector
            cursor.execute("DELETE FROM chunk_vectors WHERE chunk_id = ?", (chunk_id,))
            cursor.execute(
                """
                INSERT INTO chunk_vectors (chunk_id, embedding)
                VALUES (?, ?)
            """,
                (chunk_id, serialize_f32(embedding)),
            )

        return len(chunks)

    def update_chunk(
        self,
        chunk_id: str,
        content: Optional[str] = None,
        tags: Optional[List[str]] = None,
        importance: Optional[float] = None,
        embedding: Optional[List[float]] = None,
    ) -> bool:
        """Update fields on an existing chunk. Returns True if chunk was found."""
        cursor = self.conn.cursor()
        rows = list(cursor.execute("SELECT id FROM chunks WHERE id = ?", (chunk_id,)))
        if not rows:
            return False

        if content is not None:
            cursor.execute(
                "UPDATE chunks SET content = ?, char_count = ?, summary = ? WHERE id = ?",
                (content, len(content), content[:200], chunk_id),
            )
        if tags is not None:
            cursor.execute(
                "UPDATE chunks SET tags = ? WHERE id = ?",
                (json.dumps(tags), chunk_id),
            )
        if importance is not None:
            cursor.execute(
                "UPDATE chunks SET importance = ? WHERE id = ?",
                (float(max(1, min(10, importance))), chunk_id),
            )
        if embedding is not None:
            cursor.execute("DELETE FROM chunk_vectors WHERE chunk_id = ?", (chunk_id,))
            cursor.execute(
                "INSERT INTO chunk_vectors (chunk_id, embedding) VALUES (?, ?)",
                (chunk_id, serialize_f32(embedding)),
            )
        return True

    def archive_chunk(self, chunk_id: str) -> bool:
        """Soft-delete a chunk by setting value_type to ARCHIVED."""
        cursor = self.conn.cursor()
        rows = list(cursor.execute("SELECT id FROM chunks WHERE id = ?", (chunk_id,)))
        if not rows:
            return False
        cursor.execute("UPDATE chunks SET value_type = 'ARCHIVED' WHERE id = ?", (chunk_id,))
        cursor.execute("DELETE FROM chunk_vectors WHERE chunk_id = ?", (chunk_id,))
        return True

    def get_chunk(self, chunk_id: str) -> Optional[Dict[str, Any]]:
        """Get a single chunk by ID."""
        cursor = self.conn.cursor()
        rows = list(
            cursor.execute(
                """SELECT id, content, metadata, source_file, project, content_type,
                      value_type, tags, importance, created_at, summary
               FROM chunks WHERE id = ?""",
                (chunk_id,),
            )
        )
        if not rows:
            return None
        r = rows[0]
        return {
            "id": r[0],
            "content": r[1],
            "metadata": r[2],
            "source_file": r[3],
            "project": r[4],
            "content_type": r[5],
            "value_type": r[6],
            "tags": r[7],
            "importance": r[8],
            "created_at": r[9],
            "summary": r[10],
        }

    # ── Context manager ─────────────────────────────────────────────────

    def close(self) -> None:
        """Close database connections."""
        # Close thread-local read connection if it exists
        if hasattr(self, "_local"):
            read_conn = getattr(self._local, "read_conn", None)
            if read_conn is not None:
                read_conn.close()
                self._local.read_conn = None
        if hasattr(self, "conn"):
            self.conn.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
