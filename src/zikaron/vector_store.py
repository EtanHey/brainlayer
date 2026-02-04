"""SQLite-vec based vector store for fast search."""

import apsw
import apsw.bestpractice
import sqlite_vec
from pathlib import Path
from typing import Any, Dict, List, Optional
import json
import struct

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
                char_count INTEGER
            )
        """)

        # Create vector table with 1024 dimensions for bge-large-en-v1.5
        cursor.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS chunk_vectors USING vec0(
                chunk_id TEXT PRIMARY KEY,
                embedding FLOAT[1024]
            )
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
                (id, content, metadata, source_file, project, content_type, value_type, char_count)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                chunk_id,
                chunk["content"],
                json.dumps(chunk["metadata"]),
                chunk["source_file"],
                chunk.get("project"),
                chunk.get("content_type"),
                chunk.get("value_type"),
                chunk.get("char_count", 0)
            ))

            # Upsert vector (serialize to bytes)
            cursor.execute("""
                INSERT OR REPLACE INTO chunk_vectors (chunk_id, embedding)
                VALUES (?, ?)
            """, (chunk_id, serialize_f32(embedding)))

        return len(chunks)

    def search(
        self,
        query_embedding: Optional[List[float]] = None,
        query_text: Optional[str] = None,
        n_results: int = 10,
        project_filter: Optional[str] = None,
        content_type_filter: Optional[str] = None
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
        documents = []
        metadatas = []
        distances = []

        for row in results:
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

    def close(self) -> None:
        """Close database connection."""
        if hasattr(self, 'conn'):
            self.conn.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
