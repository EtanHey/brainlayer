"""Search and retrieval methods for VectorStore (mixin)."""

import copy
import hashlib
import json
import math
import os
import struct
import time
from collections import OrderedDict
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import apsw
import numpy as np

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
_HYBRID_CACHE_MAX = 128  # max entries (LRU eviction)
_MMR_CANDIDATE_LIMIT = 50
_MMR_LAMBDA = 0.65
META_NOISE_PATTERNS = [
    "brain_search(",
    "brain_entity(",
    "brain_recall(",
    "| # | Summary | Rating |",
    "| Summary shown | Tags | Quality |",
    "Grounding Results — Prompt",
]
META_NOISE_PATTERNS_CASEFOLDED = [pattern.casefold() for pattern in META_NOISE_PATTERNS]
AUDIT_RECURSION_TAG_PATTERNS = (
    "{tag_expr} LIKE '%audit%'",
    "{tag_expr} = 'r02'",
    "{tag_expr} GLOB 'r0[0-9]'",
)

# Module-level LRU cache: {cache_key: (result, timestamp)}
_hybrid_cache: "OrderedDict[tuple, tuple[dict, float]]" = OrderedDict()


def clear_hybrid_search_cache(store_key: Any = None) -> None:
    """Clear cached hybrid search results, optionally scoped to a single DB."""
    if store_key is None:
        _hybrid_cache.clear()
        return

    normalized_store_key = os.fspath(store_key)
    stale_keys = [key for key in _hybrid_cache if key and key[0] == normalized_store_key]
    for key in stale_keys:
        _hybrid_cache.pop(key, None)


def _hybrid_embedding_key(query_embedding: Optional[List[float]]) -> bytes:
    """Hash embeddings so cache keys stay stable across equivalent iterables."""
    if query_embedding is None:
        return b""
    embedding_bytes = serialize_f32([float(value) for value in query_embedding])
    return hashlib.sha256(embedding_bytes).digest()


def _hybrid_cache_key(
    store_key: str,
    query_text: str,
    query_embedding: Optional[List[float]],
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
    include_archived: bool = False,
) -> tuple:
    return (
        store_key,
        query_text,
        _hybrid_embedding_key(query_embedding),
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
        include_archived,
    )


def _clone_hybrid_result(result: Dict[str, List]) -> Dict[str, List]:
    """Return a defensive deep copy of cached hybrid_search results."""
    return copy.deepcopy(result)


def _contains_meta_noise(content: Optional[str]) -> bool:
    if not content:
        return False
    content_folded = content.casefold()
    return any(pattern in content_folded for pattern in META_NOISE_PATTERNS_CASEFOLDED)


def _audit_recursion_tag_predicate(tag_expr: str) -> str:
    lowered = f"LOWER(CAST({tag_expr} AS TEXT))"
    return "(" + " OR ".join(pattern.format(tag_expr=lowered) for pattern in AUDIT_RECURSION_TAG_PATTERNS) + ")"


def _audit_recursion_exclusion_sql(chunk_id_expr: str, tags_expr: str, *, use_chunk_tags: bool = True) -> str:
    if use_chunk_tags:
        return (
            "NOT EXISTS ("
            "SELECT 1 FROM chunk_tags audit_tags "
            f"WHERE audit_tags.chunk_id = {chunk_id_expr} "
            f"AND {_audit_recursion_tag_predicate('audit_tags.tag')}"
            ")"
        )

    tags_json = f"CASE WHEN json_valid({tags_expr}) THEN {tags_expr} ELSE '[]' END"
    return (
        "NOT EXISTS ("
        f"SELECT 1 FROM json_each({tags_json}) audit_tags "
        f"WHERE {_audit_recursion_tag_predicate('audit_tags.value')}"
        ")"
    )


def _is_audit_recursion_metadata(meta: dict) -> bool:
    tags = meta.get("tags")
    if not isinstance(tags, list):
        return False
    for tag in tags:
        normalized = str(tag).casefold()
        if "audit" in normalized:
            return True
        if normalized == "r02":
            return True
        if len(normalized) == 3 and normalized[:2] == "r0" and normalized[2].isdigit():
            return True
    return False


class SearchMixin:
    """Search and query methods, mixed into VectorStore."""

    def _audit_recursion_exclusion_sql(self, chunk_id_expr: str, tags_expr: str) -> str:
        return _audit_recursion_exclusion_sql(
            chunk_id_expr,
            tags_expr,
            use_chunk_tags=getattr(self, "_chunk_tags_available", True),
        )

    def _load_chunk_embeddings(self, chunk_ids: List[str]) -> Dict[str, np.ndarray]:
        """Fetch float embeddings for the provided chunk IDs."""
        if not chunk_ids:
            return {}

        cursor = self._read_cursor()
        placeholders = ",".join("?" for _ in chunk_ids)
        rows = list(
            cursor.execute(
                f"SELECT chunk_id, embedding FROM chunk_vectors WHERE chunk_id IN ({placeholders})",
                chunk_ids,
            )
        )
        return {
            chunk_id: np.frombuffer(embedding_blob, dtype=np.float32).copy()
            for chunk_id, embedding_blob in rows
            if embedding_blob
        }

    def _mmr_rerank_scored_results(
        self,
        scored: List[tuple[float, str, str, Dict[str, Any], Any]],
        n_results: int,
    ) -> List[tuple[float, str, str, Dict[str, Any], Any]]:
        """Diversify the top candidate pool with MMR while preserving overall recall."""
        if len(scored) < 2:
            return scored

        candidate_limit = min(len(scored), _MMR_CANDIDATE_LIMIT)
        top_candidates = scored[:candidate_limit]
        tail_candidates = scored[candidate_limit:]

        embeddings_by_id = self._load_chunk_embeddings([candidate[1] for candidate in top_candidates])
        mmr_candidates = [candidate for candidate in top_candidates if candidate[1] in embeddings_by_id]
        if len(mmr_candidates) < 2:
            return scored

        relevance = np.array([candidate[0] for candidate in mmr_candidates], dtype=np.float32)
        rel_max = float(relevance.max())
        if rel_max > 0.0:
            normalized_relevance = relevance / rel_max
        else:
            normalized_relevance = np.ones_like(relevance)

        matrix = np.stack([embeddings_by_id[candidate[1]] for candidate in mmr_candidates]).astype(
            np.float32, copy=False
        )
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-12)
        cosine = np.clip((matrix / norms) @ (matrix / norms).T, -1.0, 1.0)
        np.fill_diagonal(cosine, 0.0)

        selected: list[int] = [int(np.argmax(normalized_relevance))]
        remaining = set(range(len(mmr_candidates))) - set(selected)
        target_count = min(len(mmr_candidates), max(n_results, 1))

        while remaining and len(selected) < target_count:
            remaining_indices = sorted(remaining)
            diversity_penalty = cosine[remaining_indices][:, selected].max(axis=1)
            mmr_scores = (_MMR_LAMBDA * normalized_relevance[remaining_indices]) - (
                (1.0 - _MMR_LAMBDA) * diversity_penalty
            )
            best_idx = remaining_indices[int(np.argmax(mmr_scores))]
            selected.append(best_idx)
            remaining.remove(best_idx)

        reranked_ids = {mmr_candidates[idx][1] for idx in selected}
        reranked = [mmr_candidates[idx] for idx in selected]
        reranked.extend(candidate for candidate in mmr_candidates if candidate[1] not in reranked_ids)

        reranked_iter = iter(reranked)
        recombined = [
            next(reranked_iter) if candidate[1] in embeddings_by_id else candidate for candidate in top_candidates
        ]
        return recombined + tail_candidates

    def _queue_retrieval_strengthening(self, chunk_ids: List[str], now: Optional[float] = None) -> None:
        if getattr(self, "_readonly", False):
            return

        timestamp = time.time() if now is None else now
        with self._retrieval_strengthening_lock:
            for chunk_id in chunk_ids[:10]:
                pending = self._retrieval_strengthening_pending.setdefault(
                    chunk_id,
                    {
                        "retrieval_count_delta": 0,
                        "half_life_days_delta": 0.0,
                        "last_retrieved": timestamp,
                    },
                )
                pending["retrieval_count_delta"] += 1
                pending["half_life_days_delta"] += 3.0
                pending["last_retrieved"] = max(float(pending["last_retrieved"]), timestamp)
            self._retrieval_strengthening_query_count += 1
            should_flush = self._retrieval_strengthening_query_count >= self._retrieval_strengthening_flush_threshold

        if should_flush:
            self.flush_retrieval_strengthening_updates(now=timestamp)

    def _apply_retrieval_strengthening_updates(
        self,
        pending_updates: Dict[str, Dict[str, float]],
        now: Optional[float] = None,
    ) -> None:
        timestamp = time.time() if now is None else now
        cursor = self.conn.cursor()
        for chunk_id, payload in pending_updates.items():
            cursor.execute(
                """
                UPDATE chunks
                SET last_retrieved = ?,
                    retrieval_count = retrieval_count + ?,
                    half_life_days = MIN(COALESCE(half_life_days, 30.0) + ?, 365.0)
                WHERE id = ?
                """,
                (
                    timestamp,
                    int(payload["retrieval_count_delta"]),
                    float(payload["half_life_days_delta"]),
                    chunk_id,
                ),
            )

    def flush_retrieval_strengthening_updates(self, now: Optional[float] = None) -> None:
        if getattr(self, "_readonly", False):
            return

        with self._retrieval_strengthening_lock:
            if not self._retrieval_strengthening_pending:
                self._retrieval_strengthening_query_count = 0
                return
            pending_updates = self._retrieval_strengthening_pending
            self._retrieval_strengthening_pending = {}
            self._retrieval_strengthening_query_count = 0

        try:
            self._apply_retrieval_strengthening_updates(pending_updates, now=now)
        except apsw.BusyError:
            with self._retrieval_strengthening_lock:
                for chunk_id, payload in pending_updates.items():
                    current = self._retrieval_strengthening_pending.setdefault(
                        chunk_id,
                        {
                            "retrieval_count_delta": 0,
                            "half_life_days_delta": 0.0,
                            "last_retrieved": 0.0,
                        },
                    )
                    current["retrieval_count_delta"] += int(payload["retrieval_count_delta"])
                    current["half_life_days_delta"] += float(payload["half_life_days_delta"])
                    current["last_retrieved"] = max(float(current["last_retrieved"]), float(payload["last_retrieved"]))
            return

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
        include_archived: bool = False,
        source_filter_like: Optional[str] = None,
        correction_category: Optional[str] = None,
        include_audit: bool = False,
    ) -> Dict[str, List]:
        """Search chunks by embedding or text.

        Args:
            entity_id: If provided, only return chunks linked to this entity
                       via kg_entity_chunks. Used for per-person memory scoping.
            include_archived: If True, include superseded/aggregated/archived
                              chunks in results (history mode). Default: False.
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
                where_clauses.append("(c.project = ? OR c.project IS NULL)")
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
            if source_filter_like:
                where_clauses.append("c.source LIKE ?")
                filter_params.append(source_filter_like)
            if correction_category:
                where_clauses.append("c.id IN (SELECT chunk_id FROM chunk_tags WHERE tag LIKE ?)")
                filter_params.append(f"correction:{correction_category}%")
            if not include_audit:
                where_clauses.append(self._audit_recursion_exclusion_sql("c.id", "c.tags"))
            if not include_archived:
                where_clauses.append("c.superseded_by IS NULL")
                where_clauses.append("c.aggregated_into IS NULL")
                where_clauses.append("c.archived_at IS NULL")

            where_sql = ""
            if where_clauses:
                where_sql = "AND " + " AND ".join(where_clauses)

            # sqlite-vec KNN: MATCH and k must bind before filter params.
            # Bump k to over-fetch when post-KNN filters may discard most results:
            # - entity_id: entity filter applied post-KNN, most candidates won't match
            # - non-default source: rare sources (youtube, whatsapp) are <0.01% of chunks
            needs_overfetch = (
                entity_id
                or (source_filter and source_filter != "claude_code")
                or source_filter_like
                or correction_category
            )
            effective_k = min(n_results * 10, 1000) if needs_overfetch else n_results
            params = [query_bytes, effective_k] + filter_params
            query = f"""
                SELECT c.id, c.content, c.metadata, c.source_file, c.project,
                       c.content_type, c.value_type, c.char_count,
                       v.distance,
                       c.summary, c.tags, c.importance, c.intent,
                       c.created_at, c.source, c.decay_score
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
                where_clauses.append("(project = ? OR project IS NULL)")
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
            if source_filter_like:
                where_clauses.append("source LIKE ?")
                params.append(source_filter_like)
            if correction_category:
                where_clauses.append("id IN (SELECT chunk_id FROM chunk_tags WHERE tag LIKE ?)")
                params.append(f"correction:{correction_category}%")
            if not include_audit:
                where_clauses.append(self._audit_recursion_exclusion_sql("id", "tags"))
            if not include_archived:
                where_clauses.append("superseded_by IS NULL")
                where_clauses.append("aggregated_into IS NULL")
                where_clauses.append("archived_at IS NULL")

            params.append(n_results)

            query = f"""
                SELECT id, content, metadata, source_file, project,
                       content_type, value_type, char_count,
                       NULL as distance,
                       summary, tags, importance, intent,
                       created_at, source, decay_score
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
            if row[15] is not None:
                metadata["decay_score"] = row[15]
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

    def _binary_search(
        self,
        query_embedding: List[float],
        n_results: int,
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
        include_archived: bool = False,
        source_filter_like: Optional[str] = None,
        correction_category: Optional[str] = None,
        include_audit: bool = False,
    ) -> Dict[str, List]:
        """Run KNN search against binary-quantized vectors."""
        cursor = self._read_cursor()
        query_bytes = serialize_f32(query_embedding)

        where_clauses = []
        filter_params: list = []

        if entity_id:
            where_clauses.append("c.id IN (SELECT chunk_id FROM kg_entity_chunks WHERE entity_id = ?)")
            filter_params.append(entity_id)
        if project_filter:
            where_clauses.append("(c.project = ? OR c.project IS NULL)")
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
        if source_filter_like:
            where_clauses.append("c.source LIKE ?")
            filter_params.append(source_filter_like)
        if correction_category:
            where_clauses.append("c.id IN (SELECT chunk_id FROM chunk_tags WHERE tag LIKE ?)")
            filter_params.append(f"correction:{correction_category}%")
        if not include_audit:
            where_clauses.append(self._audit_recursion_exclusion_sql("c.id", "c.tags"))
        if not include_archived:
            where_clauses.append("c.superseded_by IS NULL")
            where_clauses.append("c.aggregated_into IS NULL")
            where_clauses.append("c.archived_at IS NULL")

        where_sql = ""
        if where_clauses:
            where_sql = "AND " + " AND ".join(where_clauses)

        needs_overfetch = (
            entity_id or (source_filter and source_filter != "claude_code") or source_filter_like or correction_category
        )
        effective_k = min(n_results * 10, 1000) if needs_overfetch else n_results
        params = [query_bytes, effective_k] + filter_params
        results = list(
            cursor.execute(
                f"""
                SELECT c.id, c.content, c.metadata, c.source_file, c.project,
                       c.content_type, c.value_type, c.char_count,
                       v.distance,
                       c.summary, c.tags, c.importance, c.intent,
                       c.created_at, c.source
                FROM chunk_vectors_binary v
                JOIN chunks c ON v.chunk_id = c.id
                WHERE v.embedding MATCH vec_quantize_binary(?) AND k = ? {where_sql}
                ORDER BY v.distance
                """,
                params,
            )
        )

        ids = []
        documents = []
        metadatas = []
        distances = []

        for row in results:
            ids.append(row[0])
            documents.append(row[1])
            try:
                metadata = json.loads(row[2])
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
            if row[13]:
                metadata["created_at"] = row[13]
            if row[14]:
                metadata["source"] = row[14]
            metadatas.append(metadata)
            distances.append(row[8])

        return {
            "ids": [ids],
            "documents": [documents],
            "metadatas": [metadatas],
            "distances": [distances],
        }

    def _rerank_binary_results_with_float(
        self, query_embedding: List[float], semantic_results: Dict[str, List]
    ) -> Dict[str, List]:
        """Restore precision by reranking binary candidates with exact float distance."""
        ids = semantic_results.get("ids", [[]])[0]
        if not ids:
            return semantic_results

        cursor = self._read_cursor()
        placeholders = ",".join("?" for _ in ids)
        rows = list(
            cursor.execute(
                f"SELECT chunk_id, embedding FROM chunk_vectors WHERE chunk_id IN ({placeholders})",
                ids,
            )
        )
        if not rows:
            return semantic_results

        exact_distances = {}
        query = [float(value) for value in query_embedding]
        for chunk_id, embedding_blob in rows:
            vector = struct.unpack(f"{len(embedding_blob) // 4}f", embedding_blob)
            exact_distances[chunk_id] = sum((a - b) ** 2 for a, b in zip(query, vector))

        order = sorted(
            range(len(ids)),
            key=lambda idx: (
                exact_distances.get(
                    ids[idx],
                    semantic_results["distances"][0][idx]
                    if semantic_results["distances"][0][idx] is not None
                    else float("inf"),
                ),
                idx,
            ),
        )

        return {
            "ids": [[semantic_results["ids"][0][idx] for idx in order]],
            "documents": [[semantic_results["documents"][0][idx] for idx in order]],
            "metadatas": [[semantic_results["metadatas"][0][idx] for idx in order]],
            "distances": [
                [
                    exact_distances.get(
                        semantic_results["ids"][0][idx],
                        semantic_results["distances"][0][idx],
                    )
                    for idx in order
                ]
            ],
        }

    def hybrid_search(
        self,
        query_embedding: List[float],
        query_text: str,
        fts_query_override: Optional[str] = None,
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
        include_archived: bool = False,
        kg_boost: bool = False,
        source_filter_like: Optional[str] = None,
        correction_category: Optional[str] = None,
        filter_meta_noise: bool = True,
        include_audit: bool = False,
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
            query_embedding,
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
            include_archived,
        ) + (fts_query_override, kg_boost, source_filter_like, correction_category, filter_meta_noise, include_audit)
        now = time.monotonic()
        if cache_key in _hybrid_cache:
            cached_result, cached_at = _hybrid_cache[cache_key]
            if now - cached_at < _HYBRID_CACHE_TTL:
                _hybrid_cache.move_to_end(cache_key)  # LRU touch
                cached_clone = _clone_hybrid_result(cached_result)
                self._queue_retrieval_strengthening(cached_clone["ids"][0])
                return cached_clone
            else:
                del _hybrid_cache[cache_key]

        # 1. Semantic search leg — prefer binary vectors, fall back to float vectors
        # when the binary index is unavailable (for example readonly live DBs).
        candidate_fetch_count = max(n_results * 3, _MMR_CANDIDATE_LIMIT)
        if getattr(self, "_binary_index_available", False):
            semantic = self._binary_search(
                query_embedding=query_embedding,
                n_results=candidate_fetch_count,
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
                include_archived=include_archived,
                source_filter_like=source_filter_like,
                correction_category=correction_category,
                include_audit=include_audit,
            )
            semantic = self._rerank_binary_results_with_float(query_embedding, semantic)
        else:
            semantic = self.search(
                query_embedding=query_embedding,
                n_results=candidate_fetch_count,
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
                include_archived=include_archived,
                source_filter_like=source_filter_like,
                correction_category=correction_category,
                include_audit=include_audit,
            )

        # Build semantic rank map: chunk_content -> rank
        semantic_ranks = {}
        for i, (doc, meta) in enumerate(zip(semantic["documents"][0], semantic["metadatas"][0])):
            key = meta.get("source_file", "") + "|" + doc[:100]
            semantic_ranks[key] = i

        # 2. FTS5 keyword search
        cursor = self._read_cursor()
        # FTS5 MATCH requires escaped query text. Special chars like
        # '.', '*', '"', '(', ')' cause syntax errors if passed raw.
        # Wrap each term in double quotes to treat as literal strings.
        fts_query = fts_query_override or _escape_fts5_query(query_text)
        fts_results = []
        trigram_fts_results = []
        if fts_query:
            fts_extra = []
            fts_filter_params: list = []
            entity_join = ""
            if entity_id:
                entity_join = "JOIN kg_entity_chunks ec ON c.id = ec.chunk_id"
                fts_extra.append("AND ec.entity_id = ?")
                fts_filter_params.append(entity_id)
            if project_filter:
                fts_extra.append("AND (c.project = ? OR c.project IS NULL)")
                fts_filter_params.append(project_filter)
            if source_filter:
                fts_extra.append("AND c.source = ?")
                fts_filter_params.append(source_filter)
            if sender_filter:
                fts_extra.append("AND c.sender = ?")
                fts_filter_params.append(sender_filter)
            if language_filter:
                fts_extra.append("AND c.language = ?")
                fts_filter_params.append(language_filter)
            if tag_filter:
                fts_extra.append("AND c.id IN (SELECT chunk_id FROM chunk_tags WHERE tag = ?)")
                fts_filter_params.append(tag_filter)
            if intent_filter:
                fts_extra.append("AND c.intent = ?")
                fts_filter_params.append(intent_filter)
            if importance_min is not None:
                fts_extra.append("AND c.importance >= ?")
                fts_filter_params.append(importance_min)
            if date_from:
                fts_extra.append("AND c.created_at >= ?")
                fts_filter_params.append(date_from)
            if date_to:
                fts_extra.append("AND c.created_at <= ?")
                fts_filter_params.append(date_to)
            if sentiment_filter:
                fts_extra.append("AND c.sentiment_label = ?")
                fts_filter_params.append(sentiment_filter)
            if source_filter_like:
                fts_extra.append("AND c.source LIKE ?")
                fts_filter_params.append(source_filter_like)
            if correction_category:
                fts_extra.append("AND c.id IN (SELECT chunk_id FROM chunk_tags WHERE tag LIKE ?)")
                fts_filter_params.append(f"correction:{correction_category}%")
            if not include_audit:
                fts_extra.append(f"AND {self._audit_recursion_exclusion_sql('c.id', 'c.tags')}")
            if filter_meta_noise:
                for pattern in META_NOISE_PATTERNS_CASEFOLDED:
                    fts_extra.append("AND LOWER(c.content) NOT LIKE ?")
                    fts_filter_params.append(f"%{pattern}%")
            if not include_archived:
                fts_extra.append("AND c.superseded_by IS NULL")
                fts_extra.append("AND c.aggregated_into IS NULL")
                fts_extra.append("AND c.archived_at IS NULL")

            def _fetch_fts_rows(table_name: str) -> list[tuple]:
                params = [fts_query, *fts_filter_params, candidate_fetch_count]
                return list(
                    cursor.execute(
                        f"""
                    SELECT f.chunk_id, f.rank,
                           c.content, c.metadata, c.source_file, c.project,
                           c.content_type, c.value_type, c.char_count,
                           c.summary, c.tags, c.importance, c.intent,
                           c.created_at, c.source, c.sender, c.language, c.decay_score
                    FROM {table_name} f
                    JOIN chunks c ON f.chunk_id = c.id
                    {entity_join}
                    WHERE {table_name} MATCH ? {" ".join(fts_extra)}
                    ORDER BY f.rank
                    LIMIT ?
                """,
                        params,
                    )
                )

            fts_results = _fetch_fts_rows("chunks_fts")
            if getattr(self, "_trigram_fts_available", False):
                trigram_fts_results = _fetch_fts_rows("chunks_fts_trigram")

        # Build FTS rank map
        fts_ranks = {}
        trigram_ranks = {}
        keyword_data = {}

        def _ingest_keyword_rows(rows: list[tuple], ranks: dict[str, int]) -> None:
            for i, row in enumerate(rows):
                chunk_id = row[0]
                ranks[chunk_id] = i
                keyword_data.setdefault(
                    chunk_id,
                    {
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
                        "sender": row[15],
                        "language": row[16],
                        "decay_score": row[17],
                    },
                )

        _ingest_keyword_rows(fts_results, fts_ranks)
        _ingest_keyword_rows(trigram_fts_results, trigram_ranks)

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
        all_chunk_ids = set(semantic_by_id.keys()) | set(fts_ranks.keys()) | set(trigram_ranks.keys())

        scored = []
        for cid in all_chunk_ids:
            score = 0.0
            sem_entry = semantic_by_id.get(cid)
            fts_rank = fts_ranks.get(cid)
            trigram_rank = trigram_ranks.get(cid)

            if sem_entry is not None:
                score += 1.0 / (k + sem_entry["rank"])
            if fts_rank is not None:
                score += 1.0 / (k + fts_rank)
            if trigram_rank is not None:
                score += 1.0 / (k + trigram_rank)

            # Get data — prefer semantic (has distance)
            if sem_entry is not None:
                doc = sem_entry["doc"]
                meta = sem_entry["meta"]
                dist = sem_entry["dist"]
            elif cid in keyword_data:
                data = keyword_data[cid]
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
                if data.get("sender"):
                    meta["sender"] = data["sender"]
                if data.get("language"):
                    meta["language"] = data["language"]
                if data.get("decay_score") is not None:
                    meta["decay_score"] = data["decay_score"]
                dist = None
            else:
                continue

            if filter_meta_noise and _contains_meta_noise(doc):
                continue
            if not include_audit and _is_audit_recursion_metadata(meta):
                continue

            # Apply filters to FTS-only results
            if fts_rank is not None and sem_entry is None:
                if source_filter and meta.get("source") != source_filter:
                    continue
                if project_filter and meta.get("project") not in (project_filter, None):
                    continue
                if content_type_filter and meta.get("content_type") != content_type_filter:
                    continue
                if sender_filter and meta.get("sender") != sender_filter:
                    continue
                if language_filter and meta.get("language") != language_filter:
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

            decay_score = meta.get("decay_score")
            if isinstance(decay_score, (int, float)):
                boost *= float(decay_score)

            scored[i] = (score * boost, cid, doc, meta, dist)

        # KG boost: chunks linked to entities detected in the query get a score bump
        if kg_boost and query_text:
            try:
                cursor = self._read_cursor()
                # Find entity IDs that match words in the query
                kg_linked_ids: set = set()
                words = query_text.split()
                for w in words:
                    if len(w) < 3:
                        continue
                    rows = list(
                        cursor.execute(
                            "SELECT id FROM kg_entities WHERE LOWER(name) LIKE ?",
                            (f"%{w.lower()}%",),
                        )
                    )
                    for row in rows:
                        linked = list(
                            cursor.execute(
                                "SELECT chunk_id FROM kg_entity_chunks WHERE entity_id = ?",
                                (row[0],),
                            )
                        )
                        for lrow in linked:
                            kg_linked_ids.add(lrow[0])

                if kg_linked_ids:
                    KG_BOOST_FACTOR = 1.3
                    for i, (score, cid, doc, meta, dist) in enumerate(scored):
                        if cid in kg_linked_ids:
                            scored[i] = (score * KG_BOOST_FACTOR, cid, doc, meta, dist)
            except Exception:
                pass  # KG tables may not exist in all DBs

        # Sort by boosted RRF score descending
        scored.sort(key=lambda x: x[0], reverse=True)
        scored = self._mmr_rerank_scored_results(scored, n_results=n_results)

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

        self._queue_retrieval_strengthening(ids)

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
