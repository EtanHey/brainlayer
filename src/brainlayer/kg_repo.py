"""Knowledge graph CRUD and traversal methods for VectorStore (mixin)."""

import json
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from ._helpers import _escape_fts5_query, serialize_f32
from .phonetic import phonetic_key, phonetic_tokens


class KGMixin:
    """Knowledge graph methods, mixed into VectorStore."""

    @staticmethod
    def _entity_row_to_dict(row: Any) -> Dict[str, Any]:
        return {
            "id": row[0],
            "entity_type": row[1],
            "name": row[2],
            "metadata": json.loads(row[3]) if row[3] else {},
            "created_at": row[4],
            "updated_at": row[5],
            "canonical_name": row[6],
            "description": row[7],
            "confidence": row[8],
            "importance": row[9],
            "valid_from": row[10],
            "valid_until": row[11],
            "group_id": row[12],
            "parent_id": row[13] if len(row) > 13 else None,
        }

    def _fetch_entities_by_lower_name(self, name: str, entity_type: Optional[str] = None) -> List[Dict[str, Any]]:
        cursor = self._read_cursor()
        base_query = """
            SELECT id, entity_type, name, metadata, created_at, updated_at,
                   canonical_name, description, confidence, importance,
                   valid_from, valid_until, group_id, parent_id
            FROM kg_entities
            WHERE LOWER(name) = LOWER(?)
        """
        params: list[Any] = [name]
        if entity_type is not None:
            base_query += " AND entity_type = ?"
            params.append(entity_type)
        rows = list(cursor.execute(base_query, params))
        return [self._entity_row_to_dict(row) for row in rows]

    def _entity_support_score(self, entity_id: str) -> tuple[int, int]:
        cursor = self._read_cursor()
        relation_count = list(
            cursor.execute(
                "SELECT COUNT(*) FROM kg_relations WHERE source_id = ? OR target_id = ?",
                (entity_id, entity_id),
            )
        )[0][0]
        chunk_count = list(cursor.execute("SELECT COUNT(*) FROM kg_entity_chunks WHERE entity_id = ?", (entity_id,)))[
            0
        ][0]
        return relation_count, chunk_count

    def _select_preferred_entity(self, entities: List[Dict[str, Any]]) -> Dict[str, Any]:
        def score(entity: Dict[str, Any]) -> tuple[int, int, int, str]:
            relation_count, chunk_count = self._entity_support_score(entity["id"])
            display_score = 1 if entity["name"] != entity["name"].lower() else 0
            return (relation_count, chunk_count, display_score, entity.get("created_at") or "")

        return max(entities, key=score)

    def normalize_case_variants(self, entity_type: str, name: str) -> Optional[Dict[str, Any]]:
        """Merge case-only duplicates for one entity type and return the kept entity."""
        matches = self._fetch_entities_by_lower_name(name, entity_type=entity_type)
        if not matches:
            return None

        keep = self._select_preferred_entity(matches)
        canonical_name = name.lower().replace(" ", "_")
        now = datetime.now(timezone.utc).isoformat()

        if len(matches) > 1:
            from .pipeline.entity_resolution import merge_entities

            for match in matches:
                if match["id"] == keep["id"]:
                    continue
                merge_entities(self, keep["id"], match["id"])

        self.conn.cursor().execute(
            "UPDATE kg_entities SET canonical_name = ?, updated_at = ? WHERE id = ?",
            (canonical_name, now, keep["id"]),
        )
        return self.get_entity(keep["id"])

    def ensure_named_relation(
        self,
        source_type: str,
        source_name: str,
        target_type: str,
        target_name: str,
        relation_type: str,
        *,
        fact: Optional[str] = None,
    ) -> Optional[str]:
        """Ensure a named relation exists, resolving entities case-insensitively."""
        source = self.get_entity_by_name(source_type, source_name)
        target = self.get_entity_by_name(target_type, target_name)
        if not source or not target:
            return None

        cursor = self._read_cursor()
        existing = list(
            cursor.execute(
                "SELECT id FROM kg_relations WHERE source_id = ? AND target_id = ? AND relation_type = ?",
                (source["id"], target["id"], relation_type),
            )
        )
        if existing:
            return existing[0][0]

        relation_id = f"rel-{relation_type}:{source['id']}:{target['id']}"
        relation_fact = fact or f"{source['name']} {relation_type.lower().replace('_', ' ')} {target['name']}"
        return self.add_relation(
            relation_id=relation_id,
            source_id=source["id"],
            target_id=target["id"],
            relation_type=relation_type,
            fact=relation_fact,
        )

    def _maybe_seed_known_project_relations(self, entity_type: str, name: str) -> None:
        if entity_type != "project" or name.lower() not in {"flowbar", "voicebar"}:
            return
        self.ensure_named_relation(
            source_type="project",
            source_name="FlowBar",
            target_type="project",
            target_name="VoiceBar",
            relation_type="RENAMED_FROM",
            fact="FlowBar was renamed to VoiceBar.",
        )

    def upsert_entity(
        self,
        entity_id: str,
        entity_type: str,
        name: str,
        metadata: Optional[Dict] = None,
        embedding: Optional[List[float]] = None,
        *,
        canonical_name: Optional[str] = None,
        description: Optional[str] = None,
        confidence: Optional[float] = None,
        importance: Optional[float] = None,
        valid_from: Optional[str] = None,
        valid_until: Optional[str] = None,
        group_id: Optional[str] = None,
        parent_id: Optional[str] = None,
    ) -> str:
        """Insert or update a KG entity. Returns the entity ID."""
        cursor = self.conn.cursor()
        meta_json = json.dumps(metadata or {})
        update_meta_json = json.dumps(metadata) if metadata is not None else None
        now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")
        canon = canonical_name or name.lower().replace(" ", "_")
        conf = confidence if confidence is not None else 1.0
        imp = importance if importance is not None else 0.5

        existing = self.normalize_case_variants(entity_type, name)
        if existing is not None:
            cursor.execute(
                """
                UPDATE kg_entities
                SET metadata = COALESCE(?, metadata),
                    canonical_name = COALESCE(?, canonical_name),
                    description = COALESCE(?, description),
                    confidence = COALESCE(?, confidence),
                    importance = COALESCE(?, importance),
                    valid_from = COALESCE(?, valid_from),
                    valid_until = COALESCE(?, valid_until),
                    group_id = COALESCE(?, group_id),
                    parent_id = COALESCE(?, parent_id),
                    updated_at = ?
                WHERE id = ?
                """,
                (
                    update_meta_json,
                    canonical_name,
                    description,
                    confidence,
                    importance,
                    valid_from,
                    valid_until,
                    group_id,
                    parent_id,
                    now,
                    existing["id"],
                ),
            )

            if embedding is not None:
                embedding_bytes = serialize_f32(embedding)
                cursor.execute(
                    "INSERT OR REPLACE INTO kg_vec_entities (entity_id, embedding) VALUES (?, ?)",
                    (existing["id"], embedding_bytes),
                )

            self._maybe_seed_known_project_relations(entity_type, existing["name"])
            return existing["id"]

        cursor.execute(
            """
            INSERT INTO kg_entities (id, entity_type, name, metadata, canonical_name,
                                     description, confidence, importance,
                                     valid_from, valid_until, group_id, parent_id,
                                     created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(entity_type, name) DO UPDATE SET
                metadata = excluded.metadata,
                canonical_name = excluded.canonical_name,
                description = excluded.description,
                confidence = excluded.confidence,
                importance = excluded.importance,
                valid_from = COALESCE(excluded.valid_from, kg_entities.valid_from),
                valid_until = COALESCE(excluded.valid_until, kg_entities.valid_until),
                group_id = COALESCE(excluded.group_id, kg_entities.group_id),
                parent_id = COALESCE(excluded.parent_id, kg_entities.parent_id),
                updated_at = excluded.updated_at
            """,
            (
                entity_id,
                entity_type,
                name,
                meta_json,
                canon,
                description,
                conf,
                imp,
                valid_from,
                valid_until,
                group_id,
                parent_id,
                now,
                now,
            ),
        )

        stored_id = list(
            cursor.execute(
                "SELECT id FROM kg_entities WHERE entity_type = ? AND name = ?",
                (entity_type, name),
            )
        )[0][0]

        if embedding is not None:
            embedding_bytes = serialize_f32(embedding)
            cursor.execute(
                "INSERT OR REPLACE INTO kg_vec_entities (entity_id, embedding) VALUES (?, ?)",
                (stored_id, embedding_bytes),
            )

        self._maybe_seed_known_project_relations(entity_type, name)
        return stored_id

    def add_relation(
        self,
        relation_id: str,
        source_id: str,
        target_id: str,
        relation_type: str,
        properties: Optional[Dict] = None,
        confidence: float = 1.0,
        *,
        fact: Optional[str] = None,
        importance: float = 0.5,
        valid_from: Optional[str] = None,
        valid_until: Optional[str] = None,
        source_chunk_id: Optional[str] = None,
    ) -> str:
        """Add a relationship between two entities. Returns the relation ID."""
        cursor = self.conn.cursor()
        props_json = json.dumps(properties or {})

        cursor.execute(
            """
            INSERT INTO kg_relations (id, source_id, target_id, relation_type, properties, confidence,
                                      fact, importance, valid_from, valid_until, source_chunk_id)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(source_id, target_id, relation_type) DO UPDATE SET
                properties = excluded.properties,
                confidence = excluded.confidence,
                fact = COALESCE(excluded.fact, kg_relations.fact),
                importance = COALESCE(excluded.importance, kg_relations.importance),
                valid_from = COALESCE(excluded.valid_from, kg_relations.valid_from),
                valid_until = COALESCE(excluded.valid_until, kg_relations.valid_until),
                source_chunk_id = COALESCE(excluded.source_chunk_id, kg_relations.source_chunk_id)
            """,
            (
                relation_id,
                source_id,
                target_id,
                relation_type,
                props_json,
                confidence,
                fact,
                importance,
                valid_from,
                valid_until,
                source_chunk_id,
            ),
        )

        stored_id = list(
            cursor.execute(
                "SELECT id FROM kg_relations WHERE source_id = ? AND target_id = ? AND relation_type = ?",
                (source_id, target_id, relation_type),
            )
        )[0][0]
        return stored_id

    def link_entity_chunk(
        self,
        entity_id: str,
        chunk_id: str,
        relevance: float = 1.0,
        context: Optional[str] = None,
        *,
        mention_type: Optional[str] = None,
    ) -> None:
        """Link an entity to an existing chunk."""
        cursor = self.conn.cursor()
        cursor.execute(
            """
            INSERT INTO kg_entity_chunks (entity_id, chunk_id, relevance, context, mention_type)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(entity_id, chunk_id) DO UPDATE SET
                relevance = excluded.relevance,
                context = excluded.context,
                mention_type = CASE
                    WHEN kg_entity_chunks.mention_type = 'explicit' THEN 'explicit'
                    ELSE COALESCE(excluded.mention_type, kg_entity_chunks.mention_type)
                END
            """,
            (entity_id, chunk_id, relevance, context, mention_type),
        )

    def get_entity(self, entity_id: str) -> Optional[Dict[str, Any]]:
        """Get a single entity by ID."""
        cursor = self._read_cursor()
        rows = list(
            cursor.execute(
                """SELECT id, entity_type, name, metadata, created_at, updated_at,
                          canonical_name, description, confidence, importance,
                          valid_from, valid_until, group_id, parent_id
                   FROM kg_entities WHERE id = ?""",
                (entity_id,),
            )
        )
        if not rows:
            return None
        return self._entity_row_to_dict(rows[0])

    def get_entity_by_name(self, entity_type: str, name: str) -> Optional[Dict[str, Any]]:
        """Get an entity by type and name."""
        return self.normalize_case_variants(entity_type, name)

    def get_entity_relations(self, entity_id: str, direction: str = "both") -> List[Dict[str, Any]]:
        """Get all relations for an entity."""
        cursor = self._read_cursor()
        results = []

        if direction in ("outgoing", "both"):
            rows = list(
                cursor.execute(
                    """
                    SELECT r.id, r.source_id, r.target_id, r.relation_type, r.properties, r.confidence,
                           e.name as target_name, e.entity_type as target_type,
                           r.fact, r.importance, r.valid_from, r.valid_until, r.expired_at, r.source_chunk_id
                    FROM kg_relations r
                    JOIN kg_entities e ON r.target_id = e.id
                    WHERE r.source_id = ?
                    """,
                    (entity_id,),
                )
            )
            for row in rows:
                results.append(
                    {
                        "id": row[0],
                        "source_id": row[1],
                        "target_id": row[2],
                        "relation_type": row[3],
                        "properties": json.loads(row[4]) if row[4] else {},
                        "confidence": row[5],
                        "target_name": row[6],
                        "target_type": row[7],
                        "fact": row[8],
                        "importance": row[9],
                        "valid_from": row[10],
                        "valid_until": row[11],
                        "expired_at": row[12],
                        "source_chunk_id": row[13],
                        "direction": "outgoing",
                    }
                )

        if direction in ("incoming", "both"):
            rows = list(
                cursor.execute(
                    """
                    SELECT r.id, r.source_id, r.target_id, r.relation_type, r.properties, r.confidence,
                           e.name as source_name, e.entity_type as source_type,
                           r.fact, r.importance, r.valid_from, r.valid_until, r.expired_at, r.source_chunk_id
                    FROM kg_relations r
                    JOIN kg_entities e ON r.source_id = e.id
                    WHERE r.target_id = ?
                    """,
                    (entity_id,),
                )
            )
            for row in rows:
                results.append(
                    {
                        "id": row[0],
                        "source_id": row[1],
                        "target_id": row[2],
                        "relation_type": row[3],
                        "properties": json.loads(row[4]) if row[4] else {},
                        "confidence": row[5],
                        "source_name": row[6],
                        "source_type": row[7],
                        "fact": row[8],
                        "importance": row[9],
                        "valid_from": row[10],
                        "valid_until": row[11],
                        "expired_at": row[12],
                        "source_chunk_id": row[13],
                        "direction": "incoming",
                    }
                )

        return results

    def get_entity_children(self, entity_id: str, limit: int = 50) -> list:
        """Get child entities that have this entity as parent."""
        cursor = self._read_cursor()
        rows = list(
            cursor.execute(
                """SELECT id, entity_type, name, description, importance
                   FROM kg_entities
                   WHERE parent_id = ? AND (status = 'active' OR status IS NULL)
                   ORDER BY importance DESC, name ASC
                   LIMIT ?""",
                (entity_id, limit),
            )
        )
        return [{"id": r[0], "entity_type": r[1], "name": r[2], "description": r[3], "importance": r[4]} for r in rows]

    def get_entity_parent(self, entity_id: str) -> dict | None:
        """Get the parent entity of a given entity."""
        cursor = self._read_cursor()
        row = list(
            cursor.execute(
                """SELECT p.id, p.entity_type, p.name, p.description, p.importance
                   FROM kg_entities c
                   JOIN kg_entities p ON c.parent_id = p.id
                   WHERE c.id = ?""",
                (entity_id,),
            )
        )
        if not row:
            return None
        r = row[0]
        return {"id": r[0], "entity_type": r[1], "name": r[2], "description": r[3], "importance": r[4]}

    def set_entity_parent(self, entity_id: str, parent_id: str) -> None:
        """Set the parent of an entity."""
        cursor = self.conn.cursor()
        cursor.execute(
            "UPDATE kg_entities SET parent_id = ?, updated_at = strftime('%Y-%m-%dT%H:%M:%fZ','now') WHERE id = ?",
            (parent_id, entity_id),
        )

    def get_entity_chunks(self, entity_id: str, limit: int = 20) -> List[Dict[str, Any]]:
        """Get chunks linked to an entity, ordered by relevance."""
        cursor = self._read_cursor()
        rows = list(
            cursor.execute(
                """
                SELECT ec.chunk_id, ec.relevance, ec.context, ec.mention_type,
                       c.content, c.source_file, c.project, c.content_type, c.created_at
                FROM kg_entity_chunks ec
                JOIN chunks c ON ec.chunk_id = c.id
                WHERE ec.entity_id = ?
                ORDER BY ec.relevance DESC
                LIMIT ?
                """,
                (entity_id, limit),
            )
        )
        return [
            {
                "chunk_id": row[0],
                "relevance": row[1],
                "context": row[2],
                "mention_type": row[3],
                "content": row[4],
                "source_file": row[5],
                "project": row[6],
                "content_type": row[7],
                "created_at": row[8],
            }
            for row in rows
        ]

    def search_entities(
        self,
        query: str,
        entity_type: Optional[str] = None,
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        """Search entities using FTS5 full-text search."""
        cursor = self._read_cursor()
        fts_query = _escape_fts5_query(query, match_mode="or")
        if not fts_query:
            return []

        if entity_type:
            rows = list(
                cursor.execute(
                    """
                    SELECT e.id, e.entity_type, e.name, e.metadata, e.created_at, e.updated_at, f.rank
                    FROM kg_entities_fts f
                    JOIN kg_entities e ON f.entity_id = e.id
                    WHERE kg_entities_fts MATCH ? AND e.entity_type = ?
                    ORDER BY f.rank LIMIT ?
                    """,
                    (fts_query, entity_type, limit),
                )
            )
        else:
            rows = list(
                cursor.execute(
                    """
                    SELECT e.id, e.entity_type, e.name, e.metadata, e.created_at, e.updated_at, f.rank
                    FROM kg_entities_fts f
                    JOIN kg_entities e ON f.entity_id = e.id
                    WHERE kg_entities_fts MATCH ?
                    ORDER BY f.rank LIMIT ?
                    """,
                    (fts_query, limit),
                )
            )

        return [
            {
                "id": row[0],
                "entity_type": row[1],
                "name": row[2],
                "metadata": json.loads(row[3]) if row[3] else {},
                "created_at": row[4],
                "updated_at": row[5],
                "rank": row[6],
            }
            for row in rows
        ]

    def search_entities_semantic(
        self,
        query_embedding: List[float],
        entity_type: Optional[str] = None,
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        """Search entities using vector similarity."""
        cursor = self._read_cursor()
        query_bytes = serialize_f32(query_embedding)

        if entity_type:
            fetch_k = min(max(limit * 3, limit + 50), 500)
            rows = list(
                cursor.execute(
                    """
                    SELECT e.id, e.entity_type, e.name, e.metadata, e.created_at, e.updated_at, v.distance
                    FROM kg_vec_entities v
                    JOIN kg_entities e ON v.entity_id = e.id
                    WHERE v.embedding MATCH ? AND k = ? AND e.entity_type = ?
                    ORDER BY v.distance
                    """,
                    (query_bytes, fetch_k, entity_type),
                )
            )
            rows = rows[:limit]
        else:
            rows = list(
                cursor.execute(
                    """
                    SELECT e.id, e.entity_type, e.name, e.metadata, e.created_at, e.updated_at, v.distance
                    FROM kg_vec_entities v
                    JOIN kg_entities e ON v.entity_id = e.id
                    WHERE v.embedding MATCH ? AND k = ?
                    ORDER BY v.distance
                    """,
                    (query_bytes, limit),
                )
            )

        return [
            {
                "id": row[0],
                "entity_type": row[1],
                "name": row[2],
                "metadata": json.loads(row[3]) if row[3] else {},
                "created_at": row[4],
                "updated_at": row[5],
                "distance": row[6],
            }
            for row in rows
        ]

    def list_entities(
        self,
        entity_type: Optional[str] = None,
        limit: int = 20,
        offset: int = 0,
    ) -> Dict[str, Any]:
        """List entities with optional type filter and pagination."""
        cursor = self._read_cursor()
        where = "WHERE (status = 'active' OR status IS NULL)"
        params: list = []
        if entity_type:
            where += " AND entity_type = ?"
            params.append(entity_type)

        # Get total count
        total = list(cursor.execute(f"SELECT COUNT(*) FROM kg_entities {where}", params))[0][0]

        # Get page
        rows = list(
            cursor.execute(
                f"""SELECT id, entity_type, name, description, importance, created_at
                    FROM kg_entities {where}
                    ORDER BY importance DESC, name ASC
                    LIMIT ? OFFSET ?""",
                params + [limit, offset],
            )
        )

        return {
            "total": total,
            "entities": [
                {
                    "id": r[0],
                    "entity_type": r[1],
                    "name": r[2],
                    "description": r[3],
                    "importance": r[4],
                    "created_at": r[5],
                }
                for r in rows
            ],
        }

    def kg_stats(self) -> Dict[str, Any]:
        """Get KG statistics."""
        cursor = self._read_cursor()
        entity_count = list(cursor.execute("SELECT COUNT(*) FROM kg_entities"))[0][0]
        relation_count = list(cursor.execute("SELECT COUNT(*) FROM kg_relations"))[0][0]
        link_count = list(cursor.execute("SELECT COUNT(*) FROM kg_entity_chunks"))[0][0]
        type_counts = dict(cursor.execute("SELECT entity_type, COUNT(*) FROM kg_entities GROUP BY entity_type"))
        relation_type_counts = dict(
            cursor.execute("SELECT relation_type, COUNT(*) FROM kg_relations GROUP BY relation_type")
        )
        return {
            "entities": entity_count,
            "relations": relation_count,
            "entity_chunk_links": link_count,
            "entity_types": type_counts,
            "relation_types": relation_type_counts,
        }

    def add_entity_alias(self, alias: str, entity_id: str, alias_type: str = "name") -> None:
        """Add an alias for an entity. Idempotent."""
        now = datetime.now(timezone.utc).isoformat()
        cursor = self.conn.cursor()
        cursor.execute(
            """
            INSERT INTO kg_entity_aliases (alias, entity_id, alias_type, created_at)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(alias, entity_id) DO NOTHING
            """,
            (alias, entity_id, alias_type, now),
        )

    def get_entity_by_alias(self, alias: str) -> Optional[Dict[str, Any]]:
        """Look up an entity by alias (case-insensitive)."""
        cursor = self._read_cursor()
        rows = list(
            cursor.execute(
                """
                SELECT e.id, e.entity_type, e.name, e.metadata, e.created_at, e.updated_at
                FROM kg_entity_aliases a
                JOIN kg_entities e ON a.entity_id = e.id
                WHERE LOWER(a.alias) = LOWER(?)
                LIMIT 1
                """,
                (alias,),
            )
        )
        if not rows:
            return None
        row = rows[0]
        return {
            "id": row[0],
            "entity_type": row[1],
            "name": row[2],
            "metadata": json.loads(row[3]) if row[3] else {},
            "created_at": row[4],
            "updated_at": row[5],
        }

    def get_entity_by_phonetic_alias(
        self,
        name: str,
        entity_type: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """Look up an entity by stored BMPM phonetic aliases."""
        query_key = phonetic_key(name)
        query_tokens = phonetic_tokens(name)
        if not query_key or not query_tokens:
            return None

        cursor = self._read_cursor()
        if entity_type:
            rows = list(
                cursor.execute(
                    """
                    SELECT e.id, e.entity_type, e.name, e.metadata, e.created_at, e.updated_at, a.alias
                    FROM kg_entity_aliases a
                    JOIN kg_entities e ON a.entity_id = e.id
                    WHERE a.alias_type = 'phonetic' AND e.entity_type = ?
                    """,
                    (entity_type,),
                )
            )
        else:
            rows = list(
                cursor.execute(
                    """
                    SELECT e.id, e.entity_type, e.name, e.metadata, e.created_at, e.updated_at, a.alias
                    FROM kg_entity_aliases a
                    JOIN kg_entities e ON a.entity_id = e.id
                    WHERE a.alias_type = 'phonetic'
                    """
                )
            )

        best_match = None
        best_score = 0.0
        for row in rows:
            alias_tokens = {token for token in str(row[6]).split() if token}
            if not alias_tokens:
                continue
            overlap = query_tokens & alias_tokens
            if not overlap:
                continue
            score = len(overlap) / len(query_tokens | alias_tokens)
            if row[6] == query_key:
                score = 1.0
            if score <= best_score:
                continue
            best_score = score
            best_match = {
                "id": row[0],
                "entity_type": row[1],
                "name": row[2],
                "metadata": json.loads(row[3]) if row[3] else {},
                "created_at": row[4],
                "updated_at": row[5],
            }

        return best_match

    def get_entity_aliases(self, entity_id: str) -> List[Dict[str, Any]]:
        """Get all aliases for an entity."""
        cursor = self._read_cursor()
        rows = list(
            cursor.execute(
                "SELECT alias, alias_type, created_at FROM kg_entity_aliases WHERE entity_id = ?",
                (entity_id,),
            )
        )
        return [{"alias": row[0], "alias_type": row[1], "created_at": row[2]} for row in rows]

    def soft_close_relation(self, relation_id: str) -> None:
        """Soft-close a relation by setting expired_at to now."""
        cursor = self.conn.cursor()
        now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")
        cursor.execute(
            "UPDATE kg_relations SET expired_at = ? WHERE id = ?",
            (now, relation_id),
        )

    def get_current_facts(
        self,
        entity_id: str,
        relation_type: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Get non-expired relations for an entity (outgoing only)."""
        cursor = self._read_cursor()
        if relation_type:
            rows = list(
                cursor.execute(
                    """
                    SELECT f.id, f.source_id, f.target_id, f.relation_type, f.properties, f.confidence,
                           f.fact, f.importance, f.valid_from, f.valid_until, f.expired_at, f.source_chunk_id,
                           e.name as target_name, e.entity_type as target_type
                    FROM kg_current_facts f
                    JOIN kg_entities e ON f.target_id = e.id
                    WHERE f.source_id = ? AND f.relation_type = ?
                    """,
                    (entity_id, relation_type),
                )
            )
        else:
            rows = list(
                cursor.execute(
                    """
                    SELECT f.id, f.source_id, f.target_id, f.relation_type, f.properties, f.confidence,
                           f.fact, f.importance, f.valid_from, f.valid_until, f.expired_at, f.source_chunk_id,
                           e.name as target_name, e.entity_type as target_type
                    FROM kg_current_facts f
                    JOIN kg_entities e ON f.target_id = e.id
                    WHERE f.source_id = ?
                    """,
                    (entity_id,),
                )
            )
        return [
            {
                "id": row[0],
                "source_id": row[1],
                "target_id": row[2],
                "relation_type": row[3],
                "properties": json.loads(row[4]) if row[4] else {},
                "confidence": row[5],
                "fact": row[6],
                "importance": row[7],
                "valid_from": row[8],
                "valid_until": row[9],
                "expired_at": row[10],
                "source_chunk_id": row[11],
                "target_name": row[12],
                "target_type": row[13],
            }
            for row in rows
        ]

    def traverse(
        self,
        entity_id: str,
        max_depth: int = 2,
        relation_types: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """Multi-hop graph traversal via recursive CTE."""
        cursor = self._read_cursor()

        rel_filter = ""
        params: list = [entity_id, max_depth]
        if relation_types:
            placeholders = ", ".join("?" for _ in relation_types)
            rel_filter = f"AND r.relation_type IN ({placeholders})"
            params = [entity_id] + relation_types + [max_depth]

        if relation_types:
            query = f"""
                WITH RECURSIVE reachable(entity_id, depth, path) AS (
                    SELECT ?, 0, '|' || ? || '|'
                    UNION ALL
                    SELECT r.target_id, re.depth + 1,
                           re.path || r.target_id || '|'
                    FROM reachable re
                    JOIN kg_relations r ON r.source_id = re.entity_id
                    WHERE re.depth < ?
                      AND r.expired_at IS NULL
                      AND re.path NOT LIKE '%|' || r.target_id || '|%'
                      {rel_filter}
                )
                SELECT DISTINCT r.entity_id, r.depth, e.entity_type, e.name
                FROM reachable r
                JOIN kg_entities e ON r.entity_id = e.id
                WHERE r.depth > 0
                ORDER BY r.depth, e.name
            """
            params_list = [entity_id, entity_id] + relation_types + [max_depth]
        else:
            query = """
                WITH RECURSIVE reachable(entity_id, depth, path) AS (
                    SELECT ?, 0, '|' || ? || '|'
                    UNION ALL
                    SELECT r.target_id, re.depth + 1,
                           re.path || r.target_id || '|'
                    FROM reachable re
                    JOIN kg_relations r ON r.source_id = re.entity_id
                    WHERE re.depth < ?
                      AND r.expired_at IS NULL
                      AND re.path NOT LIKE '%|' || r.target_id || '|%'
                )
                SELECT DISTINCT r.entity_id, r.depth, e.entity_type, e.name
                FROM reachable r
                JOIN kg_entities e ON r.entity_id = e.id
                WHERE r.depth > 0
                ORDER BY r.depth, e.name
            """
            params_list = [entity_id, entity_id, max_depth]

        rows = list(cursor.execute(query, params_list))
        return [{"entity_id": row[0], "depth": row[1], "entity_type": row[2], "name": row[3]} for row in rows]

    def resolve_entity(self, name_or_alias: str) -> Optional[Dict[str, Any]]:
        """Resolve a string to a KG entity."""
        # 1. Exact name (case-insensitive)
        cursor = self._read_cursor()
        rows = self._fetch_entities_by_lower_name(name_or_alias)
        if rows:
            normalized: List[Dict[str, Any]] = []
            for match in rows:
                if any(candidate["entity_type"] == match["entity_type"] for candidate in normalized):
                    continue
                normalized_entity = self.normalize_case_variants(match["entity_type"], name_or_alias)
                if normalized_entity is not None:
                    normalized.append(normalized_entity)
            if normalized:
                return self._select_preferred_entity(normalized)

        # 2. Exact alias
        result = self.get_entity_by_alias(name_or_alias)
        if result:
            return result

        # 3. Canonical name match
        rows = list(
            cursor.execute(
                """SELECT id, entity_type, name, metadata, created_at, updated_at,
                          canonical_name, description, confidence, importance,
                          valid_from, valid_until, group_id
                   FROM kg_entities WHERE LOWER(canonical_name) = LOWER(?)""",
                (name_or_alias,),
            )
        )
        if rows:
            row = rows[0]
            return {
                "id": row[0],
                "entity_type": row[1],
                "name": row[2],
                "metadata": json.loads(row[3]) if row[3] else {},
                "created_at": row[4],
                "updated_at": row[5],
                "canonical_name": row[6],
                "description": row[7],
                "confidence": row[8],
                "importance": row[9],
                "valid_from": row[10],
                "valid_until": row[11],
                "group_id": row[12],
            }

        # 4. Stored phonetic alias fallback
        result = self.get_entity_by_phonetic_alias(name_or_alias)
        if result:
            return result

        # 5. FTS5 fuzzy fallback
        results = self.search_entities(name_or_alias, limit=1)
        if results:
            return self.get_entity(results[0]["id"])

        return None

    def kg_search(
        self,
        query: str,
        relation_type: Optional[str] = None,
        limit: int = 20,
    ) -> List[Dict[str, Any]]:
        """Structured KG fact retrieval. Excludes co_occurs_with noise."""
        results: List[Dict[str, Any]] = []

        entity = self.resolve_entity(query)
        if entity:
            cursor = self._read_cursor()

            if relation_type:
                type_filter_src = "AND r.relation_type = ?"
                type_filter_tgt = "AND r.relation_type = ?"
                params = [entity["id"], relation_type, entity["id"], relation_type, limit]
            else:
                type_filter_src = "AND r.relation_type != 'co_occurs_with'"
                type_filter_tgt = "AND r.relation_type != 'co_occurs_with'"
                params = [entity["id"], entity["id"], limit]

            rows = list(
                cursor.execute(
                    f"""
                    SELECT r.id, r.source_id, r.target_id, r.relation_type,
                           r.fact, r.confidence, r.importance,
                           r.source_chunk_id, r.properties,
                           se.name as source_name, se.entity_type as source_type,
                           te.name as target_name, te.entity_type as target_type
                    FROM kg_current_facts r
                    JOIN kg_entities se ON r.source_id = se.id
                    JOIN kg_entities te ON r.target_id = te.id
                    WHERE (r.source_id = ? {type_filter_src})
                       OR (r.target_id = ? {type_filter_tgt})
                    ORDER BY r.importance DESC, r.confidence DESC
                    LIMIT ?
                    """,
                    params,
                )
            )

            for row in rows:
                results.append(
                    {
                        "id": row[0],
                        "source_id": row[1],
                        "target_id": row[2],
                        "relation_type": row[3],
                        "fact": row[4],
                        "confidence": row[5],
                        "importance": row[6],
                        "source_chunk_id": row[7],
                        "properties": json.loads(row[8]) if row[8] else {},
                        "source_entity": {"name": row[9], "entity_type": row[10]},
                        "target_entity": {"name": row[11], "entity_type": row[12]},
                    }
                )

        return results

    def kg_hybrid_search(
        self,
        query_embedding: List[float],
        query_text: str,
        n_results: int = 10,
        entity_name: Optional[str] = None,
        relation_type: Optional[str] = None,
        project_filter: Optional[str] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Combined vector + KG fact retrieval with RRF scoring."""
        RRF_K = 60

        entity_id = None
        if entity_name:
            entity = self.resolve_entity(entity_name)
            if entity:
                entity_id = entity["id"]

        chunk_results = self.hybrid_search(
            query_embedding=query_embedding,
            query_text=query_text,
            n_results=n_results,
            project_filter=project_filter,
            entity_id=entity_id,
            **kwargs,
        )

        search_term = entity_name or query_text
        kg_facts = self.kg_search(
            query=search_term,
            relation_type=relation_type,
            limit=n_results,
        )

        scored_facts = []
        for rank, fact in enumerate(kg_facts):
            rrf_score = 1.0 / (RRF_K + rank)
            importance = fact.get("importance") or 0.5
            confidence = fact.get("confidence") or 1.0
            boosted_score = rrf_score * importance * confidence
            fact["rrf_score"] = round(boosted_score, 6)
            scored_facts.append(fact)

        scored_facts.sort(key=lambda f: f["rrf_score"], reverse=True)

        return {
            "chunks": chunk_results,
            "facts": scored_facts,
        }
