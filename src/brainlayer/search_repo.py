"""Search and retrieval methods for VectorStore (mixin)."""

import copy
import json
import math
import os
import time
from collections import OrderedDict
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from ._helpers import _escape_fts5_query, serialize_f32

# ── hybrid_search result cache ───────────────────────────────────────────────
# Caches identical (store, query_text, filters) → results for 60s.
# Warm repeated queries (e.g. hook firing on the same prompt twice) return
# instantly instead of re-running the full 303K-vector brute-force scan.
#
# Correctness constraints:
# - Store-scoped: different DB files must never share cached results.
# - Filter-scoped: all query-affecting filters belong in the cache key.
# - Copy-on-read: callers enrich and mutate result metadata after search.
_HYBRID_CACHE_TTL = 60.0  # seconds
_HYBRID_CACHE_MAX = 128   # max entries (LRU eviction)

# Module-level LRU cache: {cache_key: (result, timestamp)}
_hybrid_cache: "OrderedDict[tuple, tuple[dict, float]]" = OrderedDict()


def _hybrid_cache_key(
    store_key: str,
    query_text: str,
    n_results: int,
    project_filter: Optional[str],
    content_type_filter: Optional[str],
    source_filter: Optional[str],
    sender_filter: Optional[str],
    language_filter: Optional[str],
    tag_filter: Optional[str],
    intent_filter: Optional[str],
    importance_min: Optional[float],
    date_from: Optional[str],
    date_to: Optional[str],
    sentiment_filter: Optional[str],
    entity_id: Optional[str],
    k: int,
) -> tuple:
    return (
        store_key,
        query_text,
        n_results,
        project_filter,
        content_type_filter,
        source_filter,
        sender_filter,
        language_filter,
        tag_filter,
        intent_filter,
        importance_min,
        date_from,
        date_to,
        sentiment_filter,
        entity_id,
        k,
    )


def _clone_hybrid_result(result: Dict[str, List]) -> Dict[str, List]:
    """Return a defensive deep copy of cached hybrid_search results."""
    return copy.deepcopy(result)


class SearchMixin:
    """Search and query methods, mixed into VectorStore."""

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
        importance_min: Optional[float] = None,
        date_from: Optional[str] = None,
        date_to: Optional[str] = None,
        sentiment_filter: Optional[str] = None,
        entity_id: Optional[str] = None,
    ) -> Dict[str, List]:
        """Search chunks by embedding or text.

        Args:
            entity_id: If provided, only return chunks linked to this entity
                       via kg_entity_chunks. Used for per-person memory scoping.
        """

        cursor = self._read_cursor()

        if query_embedding is not None:
            # Vector similarity search
            query_bytes = serialize_f32(query_embedding)

            where_clauses = []
            filter_params: list = []

            if entity_id:
                where_clauses.append("c.id IN (SELECT chunk_id FROM kg_entity_chunks WHERE entity_id = ?)")
                filter_params.append(entity_id)
            if project_filter:
                where_clauses.append("c.project = ?")
                filter_params.append(project_filter)
            if content_type_filter:
                where_clauses.append("c.content_type = ?")
                filter_params.append(content_type_filter)
            if source_filter:
                where_clauses.append("c.source = ?")
                filter_params.append(source_filter)
            if sender_filter:
                where_clauses.append("c.sender = ?")
                filter_params.append(sender_filter)
            if language_filter:
                where_clauses.append("c.language = ?")
                filter_params.append(language_filter)
            if tag_filter:
                where_clauses.append("c.id IN (SELECT chunk_id FROM chunk_tags WHERE tag = ?)")
                filter_params.append(tag_filter)
            if intent_filter:
                where_clauses.append("c.intent = ?")
                filter_params.append(intent_filter)
            if importance_min is not None:
                where_clauses.append("c.importance >= ?")
                filter_params.append(importance_min)
            if date_from:
                where_clauses.append("c.created_at >= ?")
                filter_params.append(date_from)
            if date_to:
                where_clauses.append("c.created_at <= ?")
                filter_params.append(date_to)
            if sentiment_filter:
                where_clauses.append("c.sentiment_label = ?")
                filter_params.append(sentiment_filter)

            where_sql = ""
            if where_clauses:
                where_sql = "AND " + " AND ".join(where_clauses)

            # sqlite-vec KNN: MATCH and k must bind before filter params.
            # Bump k to over-fetch when post-KNN filters may discard most results:
            # - entity_id: entity filter applied post-KNN, most candidates won't match
            # - non-default source: rare sources (youtube, whatsapp) are <0.01% of chunks
            needs_overfetch = entity_id or (source_filter and source_filter != "claude_code")
            effective_k = min(n_results * 10, 1000) if needs_overfetch else n_results
            params = [query_bytes, effective_k] + filter_params
            query = f"""
                SELECT c.id, c.content, c.metadata, c.source_file, c.project,
                       c.content_type, c.value_type, c.char_count,
                       v.distance,
                       c.summary, c.tags, c.importance, c.intent,
                       c.created_at, c.source
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

            if entity_id:
                where_clauses.append("id IN (SELECT chunk_id FROM kg_entity_chunks WHERE entity_id = ?)")
                params.append(entity_id)
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
                where_clauses.append("id IN (SELECT chunk_id FROM chunk_tags WHERE tag = ?)")
                params.append(tag_filter)
            if intent_filter:
                where_clauses.append("intent = ?")
                params.append(intent_filter)
            if importance_min is not None:
                where_clauses.append("importance >= ?")
                params.append(importance_min)
            if date_from:
                where_clauses.append("created_at >= ?")
                params.append(date_from)
            if date_to:
                where_clauses.append("created_at <= ?")
                params.append(date_to)

            params.append(n_results)

            query = f"""
                SELECT id, content, metadata, source_file, project,
                       content_type, value_type, char_count,
                       NULL as distance,
                       summary, tags, importance, intent,
                       created_at, source
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
            try:
                metadata = json.loads(row[2])  # metadata
            except (json.JSONDecodeError, TypeError):
                metadata = {}
            metadata.update(
                {
                    "source_file": row[3],
                    "project": row[4],
                    "content_type": row[5],
                    "value_type": row[6],
                    "char_count": row[7],
                }
            )
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
            # Temporal and source metadata
            if row[13]:
                metadata["created_at"] = row[13]
            if row[14]:
                metadata["source"] = row[14]
            metadatas.append(metadata)
            distances.append(row[8])  # distance (None for text search)

        return {
            "ids": [ids],
            "documents": [documents],
            "metadatas": [metadatas],
            "distances": [distances],
        }

    def enrich_results_with_session_context(self, results: Dict[str, List]) -> Dict[str, List]:
        """Add session enrichment metadata to search results.

        For each result, if its session has been enriched, add session_summary,
        session_outcome, and session_quality_score to the metadata.
        """
        if not results.get("metadatas") or not results["metadatas"][0]:
            return results

        cursor = self._read_cursor()
        # Cache session lookups to avoid repeated queries
        session_cache: Dict[str, Any] = {}

        for meta in results["metadatas"][0]:
            source_file = meta.get("source_file", "")
            if not source_file:
                continue

            # Extract session ID from source_file
            session_id = os.path.splitext(os.path.basename(source_file))[0]
            if not session_id:
                continue

            if session_id not in session_cache:
                rows = list(
                    cursor.execute(
                        """SELECT session_summary, primary_intent, outcome,
                              session_quality_score
                       FROM session_enrichments WHERE session_id = ?""",
                        (session_id,),
                    )
                )
                if rows:
                    session_cache[session_id] = {
                        "session_summary": rows[0][0],
                        "session_intent": rows[0][1],
                        "session_outcome": rows[0][2],
                        "session_quality": rows[0][3],
                    }
                else:
                    session_cache[session_id] = None

            enrichment = session_cache[session_id]
            if enrichment:
                for k, v in enrichment.items():
                    if v is not None:
                        meta[k] = v

        return results

    def count(self) -> int:
        """Get total number of chunks."""
        cursor = self._read_cursor()
        result = list(cursor.execute("SELECT COUNT(*) FROM chunks"))
        return result[0][0] if result else 0

    def get_stats(self) -> Dict[str, Any]:
        """Get collection statistics."""
        count = self.count()

        if count == 0:
            return {"total_chunks": 0, "projects": [], "content_types": []}

        cursor = self._read_cursor()

        # Get unique projects and content types
        results = list(
            cursor.execute("""
            SELECT DISTINCT project, content_type
            FROM chunks
            WHERE project IS NOT NULL AND content_type IS NOT NULL
            LIMIT 100
        """)
        )

        projects = set()
        content_types = set()

        for project, content_type in results:
            projects.add(project)
            content_types.add(content_type)

        return {
            "total_chunks": count,
            "projects": list(projects),
            "content_types": list(content_types),
        }

    def get_all_chunks(self, limit: int = 10000) -> List[Dict[str, Any]]:
        """Get all chunks for BM25 fitting (limited for performance)."""
        cursor = self._read_cursor()
        results = list(
            cursor.execute(
                """
            SELECT id, content, metadata, source_file, project, content_type
            FROM chunks
            LIMIT ?
        """,
                (limit,),
            )
        )

        return [
            {
                "id": row[0],
                "content": row[1],
                "metadata": json.loads(row[2]) if row[2] else {},
                "source_file": row[3],
                "project": row[4],
                "content_type": row[5],
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
        date_from: Optional[str] = None,
        date_to: Optional[str] = None,
        sentiment_filter: Optional[str] = None,
        entity_id: Optional[str] = None,
        k: int = 60,
    ) -> Dict[str, List]:
        """Hybrid search combining semantic (vector) + keyword (FTS5) via Reciprocal Rank Fusion.

        Args:
            entity_id: If provided, only return chunks linked to this entity
                       via kg_entity_chunks. Used for per-person memory scoping.

        Result cache: identical (store + query_text + filters) calls within 60s
        return cached results, avoiding repeated brute-force 303K-vector scans.
        Cache is module-level LRU (128 entries) with defensive copy-on-read.
        """

        # ── Cache lookup ─────────────────────────────────────────────────────
        store_key = os.fspath(getattr(self, "db_path", "<unknown-db>"))
        cache_key = _hybrid_cache_key(
            store_key,
            query_text,
            n_results,
            project_filter,
            content_type_filter,
            source_filter,
            sender_filter,
            language_filter,
            tag_filter,
            intent_filter,
            importance_min,
            date_from,
            date_to,
            sentiment_filter,
            entity_id,
            k,
        )
        now = time.monotonic()
        if cache_key in _hybrid_cache:
            cached_result, cached_at = _hybrid_cache[cache_key]
            if now - cached_at < _HYBRID_CACHE_TTL:
                _hybrid_cache.move_to_end(cache_key)  # LRU touch
                return _clone_hybrid_result(cached_result)
            else:
                del _hybrid_cache[cache_key]

        # 1. Semantic search — get more results for fusion (uses _read_cursor via search())
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
            date_from=date_from,
            date_to=date_to,
            sentiment_filter=sentiment_filter,
            entity_id=entity_id,
        )

        # Build semantic rank map: chunk_content -> rank
        semantic_ranks = {}
        for i, (doc, meta) in enumerate(zip(semantic["documents"][0], semantic["metadatas"][0])):
            key = meta.get("source_file", "") + "|" + doc[:100]
            semantic_ranks[key] = i

        # 2. FTS5 keyword search
        cursor = self._read_cursor()
        fts_extra = []
        # FTS5 MATCH requires escaped query text. Special chars like
        # '.', '*', '"', '(', ')' cause syntax errors if passed raw.
        # Wrap each term in double quotes to treat as literal strings.
        fts_query = _escape_fts5_query(query_text)
        fts_params: list = [fts_query]
        entity_join = ""
        if entity_id:
            entity_join = "JOIN kg_entity_chunks ec ON c.id = ec.chunk_id"
            fts_extra.append("AND ec.entity_id = ?")
            fts_params.append(entity_id)
        if project_filter:
            fts_extra.append("AND c.project = ?")
            fts_params.append(project_filter)
        if source_filter:
            fts_extra.append("AND c.source = ?")
            fts_params.append(source_filter)
        if tag_filter:
            fts_extra.append("AND c.id IN (SELECT chunk_id FROM chunk_tags WHERE tag = ?)")
            fts_params.append(tag_filter)
        if intent_filter:
            fts_extra.append("AND c.intent = ?")
            fts_params.append(intent_filter)
        if importance_min is not None:
            fts_extra.append("AND c.importance >= ?")
            fts_params.append(importance_min)
        if date_from:
            fts_extra.append("AND c.created_at >= ?")
            fts_params.append(date_from)
        if date_to:
            fts_extra.append("AND c.created_at <= ?")
            fts_params.append(date_to)
        if sentiment_filter:
            fts_extra.append("AND c.sentiment_label = ?")
            fts_params.append(sentiment_filter)
        fts_params.append(n_results * 3)

        fts_results = list(
            cursor.execute(
                f"""
            SELECT f.chunk_id, f.rank,
                   c.content, c.metadata, c.source_file, c.project,
                   c.content_type, c.value_type, c.char_count,
                   c.summary, c.tags, c.importance, c.intent,
                   c.created_at, c.source
            FROM chunks_fts f
            JOIN chunks c ON f.chunk_id = c.id
            {entity_join}
            WHERE chunks_fts MATCH ? {" ".join(fts_extra)}
            ORDER BY f.rank
            LIMIT ?
        """,
                fts_params,
            )
        )

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
                "created_at": row[13],
                "source": row[14],
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
                meta.update(
                    {
                        "source_file": data["source_file"],
                        "project": data["project"],
                        "content_type": data["content_type"],
                        "value_type": data["value_type"],
                        "char_count": data["char_count"],
                    }
                )
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
                if data.get("created_at"):
                    meta["created_at"] = data["created_at"]
                if data.get("source"):
                    meta["source"] = data["source"]
                dist = None
            else:
                continue

            # Apply filters to FTS-only results
            if fts_rank is not None and sem_entry is None:
                if source_filter and meta.get("source") != source_filter:
                    continue
                if project_filter and meta.get("project") != project_filter:
                    continue
                if content_type_filter and meta.get("content_type") != content_type_filter:
                    continue

            scored.append((score, cid, doc, meta, dist))

        # Post-RRF boost: importance and recency adjustments
        now = datetime.now(timezone.utc)
        for i, (score, cid, doc, meta, dist) in enumerate(scored):
            boost = 1.0

            # Importance boost: scale 0-10 → 1.0-1.5x multiplier
            imp = meta.get("importance")
            if imp is not None and isinstance(imp, (int, float)):
                boost *= 1.0 + min(max(float(imp), 0), 10) / 20.0  # 10/20 = 0.5 max boost

            # Recency boost: exponential decay with 30-day half-life
            created = meta.get("created_at")
            if created and isinstance(created, str):
                try:
                    dt = datetime.fromisoformat(created.replace("Z", "+00:00"))
                    if dt.tzinfo is None:
                        dt = dt.replace(tzinfo=timezone.utc)
                    age_days = max((now - dt).total_seconds() / 86400, 0)
                    recency = math.exp(-0.023 * age_days)  # ~0.5 at 30 days
                    boost *= 0.7 + 0.3 * recency  # range: 0.7x (old) to 1.0x (fresh)
                except (ValueError, TypeError):
                    pass

            scored[i] = (score * boost, cid, doc, meta, dist)

        # Sort by boosted RRF score descending
        scored.sort(key=lambda x: x[0], reverse=True)

        ids = [s[1] for s in scored[:n_results]]
        documents = [s[2] for s in scored[:n_results]]
        metadatas = [s[3] for s in scored[:n_results]]
        distances = [s[4] for s in scored[:n_results]]

        result = {
            "ids": [ids],
            "documents": [documents],
            "metadatas": [metadatas],
            "distances": [distances],
        }

        # ── Cache store ──────────────────────────────────────────────────────
        _hybrid_cache[cache_key] = (_clone_hybrid_result(result), time.monotonic())
        _hybrid_cache.move_to_end(cache_key)
        if len(_hybrid_cache) > _HYBRID_CACHE_MAX:
            _hybrid_cache.popitem(last=False)  # evict oldest

        return result

    def get_context(self, chunk_id: str, before: int = 3, after: int = 3) -> Dict[str, Any]:
        """Get surrounding chunks from the same conversation."""
        cursor = self._read_cursor()

        # Get the target chunk's conversation_id and position
        target = list(
            cursor.execute(
                """
            SELECT conversation_id, position, content, metadata, content_type
            FROM chunks WHERE id = ?
        """,
                (chunk_id,),
            )
        )

        if not target:
            return {"target": None, "context": [], "error": "Chunk not found"}

        conv_id, position, content, metadata, content_type = target[0]

        if not conv_id or position is None:
            # Standalone chunks (for example, manual-* chunks created via brain_store)
            # have no conversation_id/position. They should still be expandable as a
            # single target chunk instead of being treated as missing.
            return {
                "target": {"id": chunk_id, "content": content, "position": None},
                "context": [
                    {
                        "id": chunk_id,
                        "content": content,
                        "position": None,
                        "content_type": content_type,
                        "is_target": True,
                    }
                ],
            }

        # Get surrounding chunks
        context_rows = list(
            cursor.execute(
                """
            SELECT id, content, position, content_type
            FROM chunks
            WHERE conversation_id = ?
              AND position BETWEEN ? AND ?
            ORDER BY position
        """,
                (conv_id, position - before, position + after),
            )
        )

        context = []
        for row in context_rows:
            context.append(
                {
                    "id": row[0],
                    "content": row[1],
                    "position": row[2],
                    "content_type": row[3],
                    "is_target": row[0] == chunk_id,
                }
            )

        return {
            "target": {"id": chunk_id, "content": content, "position": position},
            "context": context,
        }
