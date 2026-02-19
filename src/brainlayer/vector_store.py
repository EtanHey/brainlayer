"""SQLite-vec based vector store for fast search."""

import json
import struct
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import apsw
import apsw.bestpractice
import sqlite_vec

# Apply APSW best practices
apsw.bestpractice.apply(apsw.bestpractice.recommended)


_SOURCE_MIN_CHARS = {
    "whatsapp": 15,
    "telegram": 15,
}
_DEFAULT_MIN_CHARS = 50


def source_aware_min_chars(source: Optional[str]) -> int:
    """Return minimum character count for enrichment based on message source.

    Short-form messaging sources (WhatsApp, Telegram) use a lower threshold
    since meaningful messages are often 15-50 chars.
    """
    if source is None:
        return _DEFAULT_MIN_CHARS
    return _SOURCE_MIN_CHARS.get(source, _DEFAULT_MIN_CHARS)


def _safe_json_loads(value: Any) -> list:
    """Safely parse a JSON string, returning [] on None or invalid JSON."""
    if not value:
        return []
    try:
        return json.loads(value)
    except (json.JSONDecodeError, TypeError):
        return []


def serialize_f32(vector: List[float]) -> bytes:
    """Serialize a float32 vector to bytes for sqlite-vec."""
    return struct.pack(f'{len(vector)}f', *vector)


class VectorStore:
    """SQLite-vec based vector store."""

    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self) -> None:
        """Initialize database with vector extension."""
        self.conn = apsw.Connection(str(self.db_path))
        self.conn.enableloadextension(True)
        self.conn.loadextension(sqlite_vec.loadable_path())
        self.conn.enableloadextension(False)

        cursor = self.conn.cursor()

        # AIDEV-NOTE: busy_timeout is critical for multi-process access (daemon + MCP + enrichment).
        # Without this, concurrent writes get SQLITE_BUSY immediately and crash silently.
        cursor.execute("PRAGMA busy_timeout = 5000")

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

        # Add columns if upgrading existing DB (check existing columns first)
        existing_cols = {row[1] for row in cursor.execute("PRAGMA table_info(chunks)")}
        for col, typ in [
            ("source", "TEXT"), ("sender", "TEXT"), ("language", "TEXT"),
            ("conversation_id", "TEXT"), ("position", "INTEGER"), ("context_summary", "TEXT"),
            ("tags", "TEXT"), ("tag_confidence", "REAL"),
            # Enrichment columns (Phase 5)
            ("summary", "TEXT"),
            ("importance", "REAL"),
            ("intent", "TEXT"),
            ("enriched_at", "TEXT"),
            # Extended enrichment columns (Phase 3 — Gemini backfill)
            ("primary_symbols", "TEXT"),   # JSON array of classes/functions/files
            ("resolved_query", "TEXT"),    # HyDE-style hypothetical question
            ("epistemic_level", "TEXT"),   # hypothesis/substantiated/validated
            ("version_scope", "TEXT"),     # version or system state discussed
            ("debt_impact", "TEXT"),       # introduction/resolution/none
            ("external_deps", "TEXT"),     # JSON array of libraries/APIs
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
        ]:
            cursor.execute(f"CREATE INDEX IF NOT EXISTS {idx} ON chunks({col})")

        # Create vector table with 1024 dimensions for bge-large-en-v1.5
        cursor.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS chunk_vectors USING vec0(
                chunk_id TEXT PRIMARY KEY,
                embedding FLOAT[1024]
            )
        """)

        # FTS5 full-text search table for hybrid search
        cursor.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
                content, chunk_id UNINDEXED
            )
        """)

        # Triggers to keep FTS5 in sync with chunks table
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

        # Phase 8b: Git overlay tables
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
        # Phase 8c: Plan linking columns on session_context
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
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS"
            " idx_file_interactions_path"
            " ON file_interactions(file_path)"
        )
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS"
            " idx_file_interactions_session"
            " ON file_interactions(session_id)"
        )
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS"
            " idx_session_context_project"
            " ON session_context(project)"
        )

        # Phase 8a: Operations table
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
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS"
            " idx_operations_session"
            " ON operations(session_id)"
        )
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS"
            " idx_operations_type"
            " ON operations(operation_type)"
        )

        # Phase 8d: Topic chains table
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
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS"
            " idx_topic_chains_file"
            " ON topic_chains(file_path)"
        )
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS"
            " idx_topic_chains_session"
            " ON topic_chains(session_a)"
        )

        # Check if FTS5 needs backfill (existing DB without FTS5 data)
        fts_count = list(cursor.execute("SELECT COUNT(*) FROM chunks_fts"))[0][0]
        chunk_count = list(cursor.execute("SELECT COUNT(*) FROM chunks"))[0][0]
        if chunk_count > 0 and fts_count == 0:
            cursor.execute("""
                INSERT INTO chunks_fts(content, chunk_id)
                SELECT content, id FROM chunks
            """)

    def upsert_chunks(
        self,
        chunks: List[Dict[str, Any]],
        embeddings: List[List[float]]
    ) -> int:
        """Upsert chunks with embeddings."""
        if len(chunks) != len(embeddings):
            raise ValueError("Chunks and embeddings must have same length")

        cursor = self.conn.cursor()

        for chunk, embedding in zip(chunks, embeddings):
            chunk_id = chunk["id"]

            # Upsert chunk — preserve enrichment columns on re-index
            cursor.execute("""
                INSERT INTO chunks
                (id, content, metadata, source_file, project,
                 content_type, value_type, char_count, source)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    content = excluded.content,
                    metadata = excluded.metadata,
                    source_file = excluded.source_file,
                    project = excluded.project,
                    content_type = excluded.content_type,
                    value_type = excluded.value_type,
                    char_count = excluded.char_count,
                    source = excluded.source
            """, (
                chunk_id,
                chunk["content"],
                json.dumps(chunk["metadata"]),
                chunk["source_file"],
                chunk.get("project"),
                chunk.get("content_type"),
                chunk.get("value_type"),
                chunk.get("char_count", 0),
                chunk.get("source", "claude_code")
            ))

            # Upsert vector - vec0 doesn't support INSERT OR REPLACE, so delete first
            cursor.execute("DELETE FROM chunk_vectors WHERE chunk_id = ?", (chunk_id,))
            cursor.execute("""
                INSERT INTO chunk_vectors (chunk_id, embedding)
                VALUES (?, ?)
            """, (chunk_id, serialize_f32(embedding)))

        return len(chunks)

    def search(
        self,
        query_embedding: Optional[List[float]] = None,
        query_text: Optional[str] = None,
        n_results: int = 10,
        project_filter: Optional[str] = None,
        content_type_filter: Optional[str] = None,
        source_filter: Optional[str] = None,
        sender_filter: Optional[str] = None,
        language_filter: Optional[str] = None,
        tag_filter: Optional[str] = None,
        intent_filter: Optional[str] = None,
        importance_min: Optional[float] = None
    ) -> Dict[str, List]:
        """Search chunks by embedding or text."""

        cursor = self.conn.cursor()

        if query_embedding is not None:
            # Vector similarity search
            query_bytes = serialize_f32(query_embedding)

            where_clauses = []
            params = [query_bytes, n_results]

            if project_filter:
                where_clauses.append("c.project = ?")
                params.insert(-1, project_filter)
            if content_type_filter:
                where_clauses.append("c.content_type = ?")
                params.insert(-1, content_type_filter)
            if source_filter:
                where_clauses.append("c.source = ?")
                params.insert(-1, source_filter)
            if sender_filter:
                where_clauses.append("c.sender = ?")
                params.insert(-1, sender_filter)
            if language_filter:
                where_clauses.append("c.language = ?")
                params.insert(-1, language_filter)
            if tag_filter:
                where_clauses.append("c.tags IS NOT NULL AND json_valid(c.tags) = 1 AND EXISTS (SELECT 1 FROM json_each(c.tags) WHERE value = ?)")
                params.insert(-1, tag_filter)
            if intent_filter:
                where_clauses.append("c.intent = ?")
                params.insert(-1, intent_filter)
            if importance_min is not None:
                where_clauses.append("c.importance >= ?")
                params.insert(-1, importance_min)

            where_sql = ""
            if where_clauses:
                where_sql = "AND " + " AND ".join(where_clauses)

            # sqlite-vec requires k = ? in WHERE clause for KNN queries
            query = f"""
                SELECT c.id, c.content, c.metadata, c.source_file, c.project,
                       c.content_type, c.value_type, c.char_count,
                       v.distance,
                       c.summary, c.tags, c.importance, c.intent
                FROM chunk_vectors v
                JOIN chunks c ON v.chunk_id = c.id
                WHERE v.embedding MATCH ? AND k = ? {where_sql}
                ORDER BY v.distance
            """

            results = list(cursor.execute(query, params))

        elif query_text is not None:
            # Text search using LIKE
            where_clauses = ["content LIKE ?"]
            params = [f"%{query_text}%"]

            if project_filter:
                where_clauses.append("project = ?")
                params.append(project_filter)
            if content_type_filter:
                where_clauses.append("content_type = ?")
                params.append(content_type_filter)
            if source_filter:
                where_clauses.append("source = ?")
                params.append(source_filter)
            if sender_filter:
                where_clauses.append("sender = ?")
                params.append(sender_filter)
            if language_filter:
                where_clauses.append("language = ?")
                params.append(language_filter)
            if tag_filter:
                where_clauses.append("tags IS NOT NULL AND json_valid(tags) = 1 AND EXISTS (SELECT 1 FROM json_each(tags) WHERE value = ?)")
                params.append(tag_filter)
            if intent_filter:
                where_clauses.append("intent = ?")
                params.append(intent_filter)
            if importance_min is not None:
                where_clauses.append("importance >= ?")
                params.append(importance_min)

            params.append(n_results)

            query = f"""
                SELECT id, content, metadata, source_file, project,
                       content_type, value_type, char_count,
                       NULL as distance,
                       summary, tags, importance, intent
                FROM chunks
                WHERE {" AND ".join(where_clauses)}
                ORDER BY char_count DESC
                LIMIT ?
            """

            results = list(cursor.execute(query, params))
        else:
            raise ValueError("Either query_embedding or query_text must be provided")

        # Format results
        ids = []
        documents = []
        metadatas = []
        distances = []

        for row in results:
            ids.append(row[0])  # chunk id
            documents.append(row[1])  # content
            metadata = json.loads(row[2])  # metadata
            metadata.update({
                "source_file": row[3],
                "project": row[4],
                "content_type": row[5],
                "value_type": row[6],
                "char_count": row[7],
            })
            # Enrichment fields (may be None if not yet enriched)
            if row[9]:
                metadata["summary"] = row[9]
            if row[10]:
                try:
                    metadata["tags"] = json.loads(row[10])
                except (json.JSONDecodeError, TypeError):
                    pass
            if row[11] is not None:
                metadata["importance"] = row[11]
            if row[12]:
                metadata["intent"] = row[12]
            metadatas.append(metadata)
            distances.append(row[8])  # distance (None for text search)

        return {
            "ids": [ids],
            "documents": [documents],
            "metadatas": [metadatas],
            "distances": [distances]
        }

    def count(self) -> int:
        """Get total number of chunks."""
        cursor = self.conn.cursor()
        result = list(cursor.execute("SELECT COUNT(*) FROM chunks"))
        return result[0][0] if result else 0

    def get_stats(self) -> Dict[str, Any]:
        """Get collection statistics."""
        count = self.count()

        if count == 0:
            return {"total_chunks": 0, "projects": [], "content_types": []}

        cursor = self.conn.cursor()

        # Get unique projects and content types
        results = list(cursor.execute("""
            SELECT DISTINCT project, content_type
            FROM chunks
            WHERE project IS NOT NULL AND content_type IS NOT NULL
            LIMIT 100
        """))

        projects = set()
        content_types = set()

        for project, content_type in results:
            projects.add(project)
            content_types.add(content_type)

        return {
            "total_chunks": count,
            "projects": list(projects),
            "content_types": list(content_types)
        }

    def get_all_chunks(self, limit: int = 10000) -> List[Dict[str, Any]]:
        """Get all chunks for BM25 fitting (limited for performance)."""
        cursor = self.conn.cursor()
        results = list(cursor.execute("""
            SELECT id, content, metadata, source_file, project, content_type
            FROM chunks
            LIMIT ?
        """, (limit,)))

        return [
            {
                "id": row[0],
                "content": row[1],
                "metadata": json.loads(row[2]) if row[2] else {},
                "source_file": row[3],
                "project": row[4],
                "content_type": row[5]
            }
            for row in results
        ]

    def hybrid_search(
        self,
        query_embedding: List[float],
        query_text: str,
        n_results: int = 10,
        project_filter: Optional[str] = None,
        content_type_filter: Optional[str] = None,
        source_filter: Optional[str] = None,
        sender_filter: Optional[str] = None,
        language_filter: Optional[str] = None,
        tag_filter: Optional[str] = None,
        intent_filter: Optional[str] = None,
        importance_min: Optional[float] = None,
        k: int = 60
    ) -> Dict[str, List]:
        """Hybrid search combining semantic (vector) + keyword (FTS5) via Reciprocal Rank Fusion."""

        # 1. Semantic search — get more results for fusion
        semantic = self.search(
            query_embedding=query_embedding,
            n_results=n_results * 3,
            project_filter=project_filter,
            content_type_filter=content_type_filter,
            source_filter=source_filter,
            sender_filter=sender_filter,
            language_filter=language_filter,
            tag_filter=tag_filter,
            intent_filter=intent_filter,
            importance_min=importance_min,
        )

        # Build semantic rank map: chunk_content -> rank
        semantic_ranks = {}
        for i, (doc, meta) in enumerate(zip(
            semantic["documents"][0], semantic["metadatas"][0]
        )):
            key = meta.get("source_file", "") + "|" + doc[:100]
            semantic_ranks[key] = i

        # 2. FTS5 keyword search
        cursor = self.conn.cursor()
        fts_extra = []
        fts_params: list = [query_text]
        if tag_filter:
            fts_extra.append("AND c.tags IS NOT NULL AND json_valid(c.tags) = 1 AND EXISTS (SELECT 1 FROM json_each(c.tags) WHERE value = ?)")
            fts_params.append(tag_filter)
        if intent_filter:
            fts_extra.append("AND c.intent = ?")
            fts_params.append(intent_filter)
        if importance_min is not None:
            fts_extra.append("AND c.importance >= ?")
            fts_params.append(importance_min)
        fts_params.append(n_results * 3)

        fts_results = list(cursor.execute(f"""
            SELECT f.chunk_id, f.rank,
                   c.content, c.metadata, c.source_file, c.project,
                   c.content_type, c.value_type, c.char_count,
                   c.summary, c.tags, c.importance, c.intent
            FROM chunks_fts f
            JOIN chunks c ON f.chunk_id = c.id
            WHERE chunks_fts MATCH ? {" ".join(fts_extra)}
            ORDER BY f.rank
            LIMIT ?
        """, fts_params))

        # Build FTS rank map
        fts_ranks = {}
        fts_data = {}
        for i, row in enumerate(fts_results):
            chunk_id = row[0]
            fts_ranks[chunk_id] = i
            fts_data[chunk_id] = {
                "content": row[2],
                "metadata": json.loads(row[3]) if row[3] else {},
                "source_file": row[4],
                "project": row[5],
                "content_type": row[6],
                "value_type": row[7],
                "char_count": row[8],
                "summary": row[9],
                "tags": row[10],
                "importance": row[11],
                "intent": row[12],
            }

        # 3. Reciprocal Rank Fusion — deduplicate by chunk_id
        # Build semantic rank map keyed by actual chunk_id
        semantic_by_id = {}
        for i in range(len(semantic["ids"][0])):
            cid = semantic["ids"][0][i]
            if cid and cid not in semantic_by_id:
                semantic_by_id[cid] = {
                    "rank": i,
                    "doc": semantic["documents"][0][i],
                    "meta": semantic["metadatas"][0][i],
                    "dist": semantic["distances"][0][i],
                }

        # Union of all chunk_ids from both sources
        all_chunk_ids = set(semantic_by_id.keys()) | set(fts_ranks.keys())

        scored = []
        for cid in all_chunk_ids:
            score = 0.0
            sem_entry = semantic_by_id.get(cid)
            fts_rank = fts_ranks.get(cid)

            if sem_entry is not None:
                score += 1.0 / (k + sem_entry["rank"])
            if fts_rank is not None:
                score += 1.0 / (k + fts_rank)

            # Get data — prefer semantic (has distance)
            if sem_entry is not None:
                doc = sem_entry["doc"]
                meta = sem_entry["meta"]
                dist = sem_entry["dist"]
            elif cid in fts_data:
                data = fts_data[cid]
                doc = data["content"]
                meta = data["metadata"].copy()
                meta.update({
                    "source_file": data["source_file"],
                    "project": data["project"],
                    "content_type": data["content_type"],
                    "value_type": data["value_type"],
                    "char_count": data["char_count"],
                })
                if data.get("summary"):
                    meta["summary"] = data["summary"]
                if data.get("tags"):
                    try:
                        meta["tags"] = json.loads(data["tags"])
                    except (json.JSONDecodeError, TypeError):
                        pass
                if data.get("importance") is not None:
                    meta["importance"] = data["importance"]
                if data.get("intent"):
                    meta["intent"] = data["intent"]
                dist = None
            else:
                continue

            # Apply filters to FTS-only results
            if fts_rank is not None and sem_entry is None:
                if source_filter and meta.get("source") != source_filter:
                    continue
                if project_filter and meta.get("project") != project_filter:
                    continue

            scored.append((score, cid, doc, meta, dist))

        # Sort by RRF score descending
        scored.sort(key=lambda x: x[0], reverse=True)

        ids = [s[1] for s in scored[:n_results]]
        documents = [s[2] for s in scored[:n_results]]
        metadatas = [s[3] for s in scored[:n_results]]
        distances = [s[4] for s in scored[:n_results]]

        return {
            "ids": [ids],
            "documents": [documents],
            "metadatas": [metadatas],
            "distances": [distances],
        }

    def get_context(
        self,
        chunk_id: str,
        before: int = 3,
        after: int = 3
    ) -> Dict[str, Any]:
        """Get surrounding chunks from the same conversation."""
        cursor = self.conn.cursor()

        # Get the target chunk's conversation_id and position
        target = list(cursor.execute("""
            SELECT conversation_id, position, content, metadata
            FROM chunks WHERE id = ?
        """, (chunk_id,)))

        if not target:
            return {"target": None, "context": [], "error": "Chunk not found"}

        conv_id, position, content, metadata = target[0]

        if not conv_id or position is None:
            return {
                "target": {"id": chunk_id, "content": content, "position": None},
                "context": [],
                "error": "Chunk has no conversation context (conversation_id/position not set)"
            }

        # Get surrounding chunks
        context_rows = list(cursor.execute("""
            SELECT id, content, position, content_type
            FROM chunks
            WHERE conversation_id = ?
              AND position BETWEEN ? AND ?
            ORDER BY position
        """, (conv_id, position - before, position + after)))

        context = []
        for row in context_rows:
            context.append({
                "id": row[0],
                "content": row[1],
                "position": row[2],
                "content_type": row[3],
                "is_target": row[0] == chunk_id,
            })

        return {
            "target": {"id": chunk_id, "content": content, "position": position},
            "context": context,
        }

    def get_unenriched_chunks(
        self,
        batch_size: int = 50,
        content_types: Optional[List[str]] = None,
        min_char_count: Optional[int] = None,
        source: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Get chunks that haven't been enriched yet, for batch processing.

        If min_char_count is not specified, uses source_aware_min_chars()
        to pick an appropriate threshold for the given source.
        """
        cursor = self.conn.cursor()

        effective_min = min_char_count if min_char_count is not None else source_aware_min_chars(source)
        where = ["enriched_at IS NULL", "char_count >= ?"]
        params: list = [effective_min]

        if source:
            where.append("source = ?")
            params.append(source)

        if content_types:
            placeholders = ",".join("?" for _ in content_types)
            where.append(f"content_type IN ({placeholders})")
            params.extend(content_types)

        params.append(batch_size)

        results = list(cursor.execute(f"""
            SELECT id, content, source_file, project, content_type,
                   conversation_id, position, char_count
            FROM chunks
            WHERE {" AND ".join(where)}
            ORDER BY rowid DESC
            LIMIT ?
        """, params))

        return [
            {
                "id": row[0],
                "content": row[1],
                "source_file": row[2],
                "project": row[3],
                "content_type": row[4],
                "conversation_id": row[5],
                "position": row[6],
                "char_count": row[7],
            }
            for row in results
        ]

    def update_enrichment(
        self,
        chunk_id: str,
        summary: Optional[str] = None,
        tags: Optional[List[str]] = None,
        importance: Optional[float] = None,
        intent: Optional[str] = None,
        primary_symbols: Optional[List[str]] = None,
        resolved_query: Optional[str] = None,
        epistemic_level: Optional[str] = None,
        version_scope: Optional[str] = None,
        debt_impact: Optional[str] = None,
        external_deps: Optional[List[str]] = None,
    ) -> None:
        """Update enrichment metadata for a chunk."""
        cursor = self.conn.cursor()
        from datetime import datetime, timezone

        sets = ["enriched_at = ?"]
        params: list = [datetime.now(timezone.utc).isoformat()]

        if summary is not None:
            sets.append("summary = ?")
            params.append(summary)
        if tags is not None:
            sets.append("tags = ?")
            params.append(json.dumps(tags))
        if importance is not None:
            sets.append("importance = ?")
            params.append(importance)
        if intent is not None:
            sets.append("intent = ?")
            params.append(intent)
        if primary_symbols is not None:
            sets.append("primary_symbols = ?")
            params.append(json.dumps(primary_symbols))
        if resolved_query is not None:
            sets.append("resolved_query = ?")
            params.append(resolved_query)
        if epistemic_level is not None:
            sets.append("epistemic_level = ?")
            params.append(epistemic_level)
        if version_scope is not None:
            sets.append("version_scope = ?")
            params.append(version_scope)
        if debt_impact is not None:
            sets.append("debt_impact = ?")
            params.append(debt_impact)
        if external_deps is not None:
            sets.append("external_deps = ?")
            params.append(json.dumps(external_deps))

        params.append(chunk_id)
        # Retry on SQLITE_BUSY — concurrent access from daemon/MCP/enrichment
        import time as _time
        for attempt in range(3):
            try:
                cursor.execute(f"UPDATE chunks SET {', '.join(sets)} WHERE id = ?", params)
                return
            except apsw.BusyError:
                if attempt < 2:
                    _time.sleep(0.5 * (attempt + 1))
                else:
                    raise

    def get_enrichment_stats(self) -> Dict[str, Any]:
        """Get enrichment progress statistics."""
        cursor = self.conn.cursor()
        total = list(cursor.execute("SELECT COUNT(*) FROM chunks"))[0][0]
        enriched = list(cursor.execute("SELECT COUNT(*) FROM chunks WHERE enriched_at IS NOT NULL"))[0][0]
        by_intent = list(cursor.execute("""
            SELECT intent, COUNT(*) FROM chunks
            WHERE intent IS NOT NULL
            GROUP BY intent ORDER BY COUNT(*) DESC
        """))
        return {
            "total_chunks": total,
            "enriched": enriched,
            "remaining": total - enriched,
            "percent": round(enriched / total * 100, 1) if total > 0 else 0,
            "by_intent": {row[0]: row[1] for row in by_intent},
        }

    # ─── Phase 8b: Git Overlay Methods ──────────────────────────────

    def store_session_context(
        self,
        session_id: str,
        project: str,
        branch: Optional[str] = None,
        pr_number: Optional[int] = None,
        commit_shas: Optional[List[str]] = None,
        files_changed: Optional[List[str]] = None,
        started_at: Optional[str] = None,
        ended_at: Optional[str] = None,
        plan_name: Optional[str] = None,
        plan_phase: Optional[str] = None,
        story_id: Optional[str] = None,
    ) -> None:
        """Store git context for a session (upsert).

        Preserves existing plan_name/plan_phase/story_id
        if not provided (avoids wiping plan links on
        git overlay re-runs).
        """
        cursor = self.conn.cursor()
        # Preserve existing plan fields if not provided
        if plan_name is None:
            existing = list(cursor.execute(
                "SELECT plan_name, plan_phase, story_id"
                " FROM session_context"
                " WHERE session_id = ?",
                (session_id,),
            ))
            if existing:
                plan_name = existing[0][0]
                plan_phase = plan_phase or existing[0][1]
                story_id = story_id or existing[0][2]
        cursor.execute("""
            INSERT OR REPLACE INTO session_context
            (session_id, project, branch, pr_number, commit_shas,
             files_changed, started_at, ended_at, created_at,
             plan_name, plan_phase, story_id)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, datetime('now'),
                    ?, ?, ?)
        """, (
            session_id, project, branch, pr_number,
            json.dumps(commit_shas) if commit_shas else None,
            json.dumps(files_changed) if files_changed else None,
            started_at, ended_at,
            plan_name, plan_phase, story_id,
        ))

    def store_file_interactions(
        self, interactions: List[Dict[str, Any]]
    ) -> int:
        """Store file interaction records. Returns count stored."""
        if not interactions:
            return 0
        cursor = self.conn.cursor()
        count = 0
        for i in interactions:
            cursor.execute("""
                INSERT INTO file_interactions
                (file_path, timestamp, session_id, action, chunk_id, project)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                i["file_path"], i.get("timestamp"), i["session_id"],
                i.get("action", "unknown"), i.get("chunk_id"),
                i.get("project"),
            ))
            count += 1
        return count

    def get_file_timeline(
        self,
        file_path: str,
        project: Optional[str] = None,
        limit: int = 50,
    ) -> List[Dict[str, Any]]:
        """Get ordered timeline of interactions with a file."""
        cursor = self.conn.cursor()
        query = """
            SELECT fi.file_path, fi.timestamp, fi.session_id, fi.action,
                   fi.project, sc.branch, sc.pr_number
            FROM file_interactions fi
            LEFT JOIN session_context sc ON fi.session_id = sc.session_id
            WHERE fi.file_path LIKE ?
        """
        params: list = [f"%{file_path}%"]
        if project:
            query += " AND fi.project = ?"
            params.append(project)
        query += " ORDER BY fi.timestamp ASC LIMIT ?"
        params.append(limit)

        results = []
        for row in cursor.execute(query, params):
            results.append({
                "file_path": row[0],
                "timestamp": row[1],
                "session_id": row[2],
                "action": row[3],
                "project": row[4],
                "branch": row[5],
                "pr_number": row[6],
            })
        return results

    def get_session_context(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get git context for a session."""
        cursor = self.conn.cursor()
        rows = list(cursor.execute(
            "SELECT * FROM session_context WHERE session_id = ?",
            (session_id,)
        ))
        if not rows:
            return None
        row = rows[0]
        result = {
            "session_id": row[0],
            "project": row[1],
            "branch": row[2],
            "pr_number": row[3],
            "commit_shas": _safe_json_loads(row[4]),
            "files_changed": _safe_json_loads(row[5]),
            "started_at": row[6],
            "ended_at": row[7],
            "created_at": row[8],
        }
        # Plan linking columns (may not exist in old DBs)
        if len(row) > 9:
            result["plan_name"] = row[9]
            result["plan_phase"] = row[10]
            result["story_id"] = row[11]
        return result

    def update_session_plan(
        self,
        session_id: str,
        plan_name: Optional[str] = None,
        plan_phase: Optional[str] = None,
        story_id: Optional[str] = None,
    ) -> bool:
        """Update plan linking fields for an existing session.

        Returns True if session was found and updated.
        """
        cursor = self.conn.cursor()
        rows = list(cursor.execute(
            "SELECT 1 FROM session_context WHERE session_id = ?",
            (session_id,),
        ))
        if not rows:
            return False
        cursor.execute("""
            UPDATE session_context
            SET plan_name = ?, plan_phase = ?, story_id = ?
            WHERE session_id = ?
        """, (plan_name, plan_phase, story_id, session_id))
        return True

    def get_sessions_by_plan(
        self,
        plan_name: Optional[str] = None,
        project: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Get all sessions linked to a plan (or all linked sessions)."""
        cursor = self.conn.cursor()
        query = (
            "SELECT session_id, project, branch, pr_number,"
            " started_at, ended_at, plan_name, plan_phase, story_id"
            " FROM session_context"
            " WHERE plan_name IS NOT NULL"
        )
        params: list = []
        if plan_name:
            query += " AND plan_name = ?"
            params.append(plan_name)
        if project:
            query += " AND project = ?"
            params.append(project)
        query += " ORDER BY started_at ASC"

        results = []
        for row in cursor.execute(query, params):
            results.append({
                "session_id": row[0],
                "project": row[1],
                "branch": row[2],
                "pr_number": row[3],
                "started_at": row[4],
                "ended_at": row[5],
                "plan_name": row[6],
                "plan_phase": row[7],
                "story_id": row[8],
            })
        return results

    def get_plan_linking_stats(self) -> Dict[str, Any]:
        """Get plan linking statistics."""
        cursor = self.conn.cursor()
        total = list(cursor.execute(
            "SELECT COUNT(*) FROM session_context"
        ))[0][0]
        linked = list(cursor.execute(
            "SELECT COUNT(*) FROM session_context"
            " WHERE plan_name IS NOT NULL"
        ))[0][0]
        plans = list(cursor.execute(
            "SELECT plan_name, COUNT(*) FROM session_context"
            " WHERE plan_name IS NOT NULL"
            " GROUP BY plan_name ORDER BY COUNT(*) DESC"
        ))
        return {
            "total_sessions": total,
            "linked_sessions": linked,
            "unlinked_sessions": total - linked,
            "plans": {row[0]: row[1] for row in plans},
        }

    def clear_plan_links(
        self, project: Optional[str] = None
    ) -> int:
        """Clear plan links. Returns count cleared."""
        cursor = self.conn.cursor()
        if project:
            rows = list(cursor.execute(
                "SELECT COUNT(*) FROM session_context"
                " WHERE plan_name IS NOT NULL AND project = ?",
                (project,),
            ))
            cursor.execute(
                "UPDATE session_context"
                " SET plan_name = NULL, plan_phase = NULL,"
                " story_id = NULL"
                " WHERE project = ?",
                (project,),
            )
        else:
            rows = list(cursor.execute(
                "SELECT COUNT(*) FROM session_context"
                " WHERE plan_name IS NOT NULL"
            ))
            cursor.execute(
                "UPDATE session_context"
                " SET plan_name = NULL, plan_phase = NULL,"
                " story_id = NULL"
            )
        return rows[0][0] if rows else 0

    def get_git_overlay_stats(self) -> Dict[str, Any]:
        """Get git overlay statistics."""
        cursor = self.conn.cursor()
        sessions = list(cursor.execute("SELECT COUNT(*) FROM session_context"))[0][0]
        interactions = list(cursor.execute("SELECT COUNT(*) FROM file_interactions"))[0][0]
        unique_files = list(cursor.execute(
            "SELECT COUNT(DISTINCT file_path) FROM file_interactions"
        ))[0][0]
        return {
            "sessions_with_context": sessions,
            "file_interactions": interactions,
            "unique_files": unique_files,
        }

    def clear_session_git_data(self, session_id: str) -> None:
        """Clear git overlay data for a session (for re-processing)."""
        cursor = self.conn.cursor()
        cursor.execute("DELETE FROM session_context WHERE session_id = ?", (session_id,))
        cursor.execute("DELETE FROM file_interactions WHERE session_id = ?", (session_id,))

    def store_operations(
        self,
        operations: List[Dict[str, Any]],
    ) -> int:
        """Store operation groups.

        Args:
            operations: List of dicts with id, session_id,
                operation_type, chunk_ids, summary, outcome,
                started_at, ended_at, step_count.

        Returns:
            Number of operations stored.
        """
        if not operations:
            return 0
        cursor = self.conn.cursor()
        from datetime import timezone
        now = datetime.now(timezone.utc).isoformat()
        count = 0
        for op in operations:
            chunk_ids_json = json.dumps(
                op.get("chunk_ids", [])
            )
            cursor.execute(
                """INSERT OR REPLACE INTO operations
                (id, session_id, operation_type, chunk_ids,
                 summary, outcome, started_at, ended_at,
                 step_count, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    op["id"],
                    op["session_id"],
                    op.get("operation_type"),
                    chunk_ids_json,
                    op.get("summary"),
                    op.get("outcome"),
                    op.get("started_at"),
                    op.get("ended_at"),
                    op.get("step_count", 0),
                    now,
                ),
            )
            count += 1
        return count

    def get_session_operations(
        self,
        session_id: str,
    ) -> List[Dict[str, Any]]:
        """Get all operations for a session."""
        cursor = self.conn.cursor()
        rows = list(cursor.execute(
            """SELECT id, session_id, operation_type,
                      chunk_ids, summary, outcome,
                      started_at, ended_at, step_count
               FROM operations
               WHERE session_id = ?
               ORDER BY started_at""",
            (session_id,),
        ))
        results = []
        for row in rows:
            chunk_ids = []
            if row[3]:
                try:
                    chunk_ids = json.loads(row[3])
                except (json.JSONDecodeError, TypeError):
                    pass
            results.append({
                "id": row[0],
                "session_id": row[1],
                "operation_type": row[2],
                "chunk_ids": chunk_ids,
                "summary": row[4],
                "outcome": row[5],
                "started_at": row[6],
                "ended_at": row[7],
                "step_count": row[8],
            })
        return results

    def get_operations_stats(self) -> Dict[str, Any]:
        """Get operation grouping statistics."""
        cursor = self.conn.cursor()
        total = list(cursor.execute(
            "SELECT COUNT(*) FROM operations"
        ))[0][0]
        by_type = list(cursor.execute(
            """SELECT operation_type, COUNT(*)
               FROM operations
               GROUP BY operation_type
               ORDER BY COUNT(*) DESC"""
        ))
        sessions = list(cursor.execute(
            """SELECT COUNT(DISTINCT session_id)
               FROM operations"""
        ))[0][0]
        return {
            "total_operations": total,
            "sessions_with_operations": sessions,
            "by_type": {
                (row[0] or "unknown"): row[1]
                for row in by_type
            },
        }

    def clear_session_operations(
        self, session_id: str
    ) -> None:
        """Clear operations for a session."""
        cursor = self.conn.cursor()
        cursor.execute(
            "DELETE FROM operations WHERE session_id = ?",
            (session_id,),
        )

    def store_topic_chains(
        self,
        chains: List[Dict[str, Any]],
    ) -> int:
        """Store topic chain entries."""
        if not chains:
            return 0
        cursor = self.conn.cursor()
        from datetime import timezone
        now = datetime.now(timezone.utc).isoformat()
        count = 0
        for chain in chains:
            cursor.execute(
                """INSERT INTO topic_chains
                (file_path, session_a, session_b,
                 shared_actions, time_delta_hours,
                 project, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (
                    chain["file_path"],
                    chain["session_a"],
                    chain["session_b"],
                    chain.get("shared_actions", 0),
                    chain.get("time_delta_hours"),
                    chain.get("project"),
                    now,
                ),
            )
            count += 1
        return count

    def get_file_chains(
        self,
        file_path: str,
        limit: int = 20,
    ) -> List[Dict[str, Any]]:
        """Get topic chains for a file (sessions linked by file)."""
        cursor = self.conn.cursor()
        rows = list(cursor.execute(
            """SELECT tc.file_path, tc.session_a,
                      tc.session_b, tc.shared_actions,
                      tc.time_delta_hours, tc.project,
                      sa.branch AS branch_a,
                      sb.branch AS branch_b
               FROM topic_chains tc
               LEFT JOIN session_context sa
                   ON tc.session_a = sa.session_id
               LEFT JOIN session_context sb
                   ON tc.session_b = sb.session_id
               WHERE tc.file_path LIKE ?
               ORDER BY tc.time_delta_hours
               LIMIT ?""",
            (f"%{file_path}%", limit),
        ))
        return [
            {
                "file_path": row[0],
                "session_a": row[1],
                "session_b": row[2],
                "shared_actions": row[3],
                "time_delta_hours": row[4],
                "project": row[5],
                "branch_a": row[6],
                "branch_b": row[7],
            }
            for row in rows
        ]

    def get_file_regression(
        self,
        file_path: str,
        project: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Get regression info for a file.

        Finds the last successful operation involving the file,
        then shows all changes after that point.

        Returns:
            Dict with last_success, changes_after, and timeline.
        """
        cursor = self.conn.cursor()

        # Get all interactions for this file, ordered by time
        query = """
            SELECT fi.file_path, fi.timestamp,
                   fi.session_id, fi.action,
                   fi.project,
                   sc.branch, sc.pr_number
            FROM file_interactions fi
            LEFT JOIN session_context sc
                ON fi.session_id = sc.session_id
            WHERE fi.file_path LIKE ?
        """
        params: list = [f"%{file_path}%"]
        if project:
            query += " AND fi.project = ?"
            params.append(project)
        query += " ORDER BY fi.timestamp"

        interactions = list(cursor.execute(query, params))

        if not interactions:
            return {
                "file_path": file_path,
                "timeline": [],
                "last_success": None,
                "changes_after": [],
            }

        # Build timeline
        timeline = []
        for row in interactions:
            timeline.append({
                "file_path": row[0],
                "timestamp": row[1],
                "session_id": row[2],
                "action": row[3],
                "project": row[4],
                "branch": row[5],
                "pr_number": row[6],
            })

        # Find last successful operation for this file
        # Check operations table for success outcomes
        last_success = None
        changes_after = []

        # Get operations that involved this file
        for entry in reversed(timeline):
            sid = entry["session_id"]
            if not sid:
                continue
            ops = list(cursor.execute(
                """SELECT outcome FROM operations
                   WHERE session_id = ?
                   AND outcome = 'success'
                   LIMIT 1""",
                (sid,),
            ))
            if ops:
                last_success = entry
                break

        # Get all entries after last success
        if last_success and last_success.get("timestamp"):
            changes_after = [
                e for e in timeline
                if (e.get("timestamp") or "")
                > last_success["timestamp"]
            ]

        return {
            "file_path": file_path,
            "timeline": timeline,
            "last_success": last_success,
            "changes_after": changes_after,
        }

    def get_topic_chain_stats(self) -> Dict[str, Any]:
        """Get topic chain statistics."""
        cursor = self.conn.cursor()
        total = list(cursor.execute(
            "SELECT COUNT(*) FROM topic_chains"
        ))[0][0]
        files = list(cursor.execute(
            """SELECT COUNT(DISTINCT file_path)
               FROM topic_chains"""
        ))[0][0]
        return {
            "total_chains": total,
            "unique_files": files,
        }

    def clear_topic_chains(
        self, project: Optional[str] = None
    ) -> None:
        """Clear topic chains, optionally for a project."""
        cursor = self.conn.cursor()
        if project:
            cursor.execute(
                "DELETE FROM topic_chains"
                " WHERE project = ?",
                (project,),
            )
        else:
            cursor.execute("DELETE FROM topic_chains")

    def close(self) -> None:
        """Close database connection."""
        if hasattr(self, 'conn'):
            self.conn.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
