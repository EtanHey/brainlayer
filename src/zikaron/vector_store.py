"""SQLite-vec based vector store for fast search."""

import json
import struct
from pathlib import Path
from typing import Any, Dict, List, Optional

import apsw
import apsw.bestpractice
import sqlite_vec

# Apply APSW best practices
apsw.bestpractice.apply(apsw.bestpractice.recommended)


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
        for col, typ in [
            ("source", "TEXT"), ("sender", "TEXT"), ("language", "TEXT"),
            ("conversation_id", "TEXT"), ("position", "INTEGER"), ("context_summary", "TEXT"),
            ("tags", "TEXT"), ("tag_confidence", "REAL"),
        ]:
            try:
                cursor.execute(f"ALTER TABLE chunks ADD COLUMN {col} {typ}")
            except apsw.SQLError:
                pass  # column already exists

        # Indexes for filtering
        for idx, col in [
            ("idx_chunks_source", "source"),
            ("idx_chunks_sender", "sender"),
            ("idx_chunks_conversation", "conversation_id"),
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

            # Upsert chunk
            cursor.execute("""
                INSERT OR REPLACE INTO chunks
                (id, content, metadata, source_file, project,
                 content_type, value_type, char_count, source)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
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
        language_filter: Optional[str] = None
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

            where_sql = ""
            if where_clauses:
                where_sql = "AND " + " AND ".join(where_clauses)

            # sqlite-vec requires k = ? in WHERE clause for KNN queries
            query = f"""
                SELECT c.id, c.content, c.metadata, c.source_file, c.project,
                       c.content_type, c.value_type, c.char_count,
                       v.distance
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

            params.append(n_results)

            query = f"""
                SELECT id, content, metadata, source_file, project,
                       content_type, value_type, char_count,
                       NULL as distance
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
                "char_count": row[7]
            })
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
        fts_results = list(cursor.execute("""
            SELECT f.chunk_id, f.rank,
                   c.content, c.metadata, c.source_file, c.project,
                   c.content_type, c.value_type, c.char_count
            FROM chunks_fts f
            JOIN chunks c ON f.chunk_id = c.id
            WHERE chunks_fts MATCH ?
            ORDER BY f.rank
            LIMIT ?
        """, (query_text, n_results * 3)))

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

    def close(self) -> None:
        """Close database connection."""
        if hasattr(self, 'conn'):
            self.conn.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
