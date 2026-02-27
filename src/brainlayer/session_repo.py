"""Session, operations, enrichment, and topic chain methods for VectorStore (mixin)."""

import json
import logging
import time as _time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import apsw

from ._helpers import _safe_json_loads, source_aware_min_chars

logger = logging.getLogger(__name__)


class SessionMixin:
    """Session management and enrichment methods, mixed into VectorStore."""

    # --- Enrichment CRUD ---

    def get_unenriched_chunks(
        self,
        batch_size: int = 50,
        content_types: Optional[List[str]] = None,
        min_char_count: Optional[int] = None,
        source: Optional[str] = None,
        since_hours: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Get chunks that haven't been enriched yet, for batch processing."""
        cursor = self._read_cursor()

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

        if since_hours is not None:
            where.append(f"created_at > datetime('now', '-{int(since_hours)} hours')")

        params.append(batch_size)

        results = list(
            cursor.execute(
                f"""
            SELECT id, content, source_file, project, content_type,
                   conversation_id, position, char_count
            FROM chunks
            WHERE {" AND ".join(where)}
            ORDER BY rowid DESC
            LIMIT ?
        """,
                params,
            )
        )

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
        sentiment_label: Optional[str] = None,
        sentiment_score: Optional[float] = None,
        sentiment_signals: Optional[List[str]] = None,
    ) -> None:
        """Update enrichment metadata for a chunk."""
        cursor = self.conn.cursor()

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
        if sentiment_label is not None:
            sets.append("sentiment_label = ?")
            params.append(sentiment_label)
        if sentiment_score is not None:
            sets.append("sentiment_score = ?")
            params.append(sentiment_score)
        if sentiment_signals is not None:
            sets.append("sentiment_signals = ?")
            params.append(json.dumps(sentiment_signals))

        params.append(chunk_id)
        for attempt in range(3):
            try:
                cursor.execute(f"UPDATE chunks SET {', '.join(sets)} WHERE id = ?", params)
                return
            except apsw.BusyError:
                if attempt < 2:
                    _time.sleep(0.5 * (attempt + 1))
                else:
                    raise

    def update_sentiment(
        self,
        chunk_id: str,
        label: str,
        score: float,
        signals: Optional[List[str]] = None,
    ) -> None:
        """Update sentiment metadata for a chunk."""
        cursor = self.conn.cursor()

        params: list = [label, score, json.dumps(signals or []), chunk_id]
        for attempt in range(3):
            try:
                cursor.execute(
                    "UPDATE chunks SET sentiment_label = ?, sentiment_score = ?, sentiment_signals = ? WHERE id = ?",
                    params,
                )
                return
            except apsw.BusyError:
                if attempt < 2:
                    _time.sleep(0.5 * (attempt + 1))
                else:
                    raise

    def get_enrichment_stats(self) -> Dict[str, Any]:
        """Get enrichment progress statistics."""
        cursor = self._read_cursor()
        total = list(cursor.execute("SELECT COUNT(*) FROM chunks"))[0][0]
        enriched = list(
            cursor.execute(
                "SELECT COUNT(*) FROM chunks WHERE enriched_at IS NOT NULL AND enriched_at NOT LIKE 'skipped:%'"
            )
        )[0][0]
        skipped = list(cursor.execute("SELECT COUNT(*) FROM chunks WHERE enriched_at LIKE 'skipped:%'"))[0][0]
        remaining = list(cursor.execute("SELECT COUNT(*) FROM chunks WHERE enriched_at IS NULL"))[0][0]
        enrichable = total - skipped
        by_intent = list(
            cursor.execute("""
            SELECT intent, COUNT(*) FROM chunks
            WHERE intent IS NOT NULL
            GROUP BY intent ORDER BY COUNT(*) DESC
        """)
        )
        return {
            "total_chunks": total,
            "enrichable": enrichable,
            "enriched": enriched,
            "skipped": skipped,
            "remaining": remaining,
            "percent": round(enriched / enrichable * 100, 1) if enrichable > 0 else 0,
            "naive_percent": round((enriched + skipped) / total * 100, 1) if total > 0 else 0,
            "by_intent": {row[0]: row[1] for row in by_intent},
        }

    # --- Git Overlay / Session Context ---

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
        """Store git context for a session (upsert)."""
        cursor = self.conn.cursor()
        if plan_name is None:
            existing = list(
                cursor.execute(
                    "SELECT plan_name, plan_phase, story_id FROM session_context WHERE session_id = ?",
                    (session_id,),
                )
            )
            if existing:
                plan_name = existing[0][0]
                plan_phase = plan_phase or existing[0][1]
                story_id = story_id or existing[0][2]
        cursor.execute(
            """
            INSERT OR REPLACE INTO session_context
            (session_id, project, branch, pr_number, commit_shas,
             files_changed, started_at, ended_at, created_at,
             plan_name, plan_phase, story_id)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, datetime('now'),
                    ?, ?, ?)
        """,
            (
                session_id,
                project,
                branch,
                pr_number,
                json.dumps(commit_shas) if commit_shas else None,
                json.dumps(files_changed) if files_changed else None,
                started_at,
                ended_at,
                plan_name,
                plan_phase,
                story_id,
            ),
        )

    def store_file_interactions(self, interactions: List[Dict[str, Any]]) -> int:
        """Store file interaction records. Returns count stored."""
        if not interactions:
            return 0
        cursor = self.conn.cursor()
        count = 0
        for i in interactions:
            cursor.execute(
                """
                INSERT INTO file_interactions
                (file_path, timestamp, session_id, action, chunk_id, project)
                VALUES (?, ?, ?, ?, ?, ?)
            """,
                (
                    i["file_path"],
                    i.get("timestamp"),
                    i["session_id"],
                    i.get("action", "unknown"),
                    i.get("chunk_id"),
                    i.get("project"),
                ),
            )
            count += 1
        return count

    def get_file_timeline(
        self,
        file_path: str,
        project: Optional[str] = None,
        limit: int = 50,
    ) -> List[Dict[str, Any]]:
        """Get ordered timeline of interactions with a file."""
        cursor = self._read_cursor()
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
            results.append(
                {
                    "file_path": row[0],
                    "timestamp": row[1],
                    "session_id": row[2],
                    "action": row[3],
                    "project": row[4],
                    "branch": row[5],
                    "pr_number": row[6],
                }
            )
        return results

    def get_session_context(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get git context for a session."""
        cursor = self._read_cursor()
        rows = list(cursor.execute("SELECT * FROM session_context WHERE session_id = ?", (session_id,)))
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
        """Update plan linking fields for an existing session."""
        cursor = self.conn.cursor()
        rows = list(cursor.execute("SELECT 1 FROM session_context WHERE session_id = ?", (session_id,)))
        if not rows:
            return False
        cursor.execute(
            """
            UPDATE session_context
            SET plan_name = ?, plan_phase = ?, story_id = ?
            WHERE session_id = ?
        """,
            (plan_name, plan_phase, story_id, session_id),
        )
        return True

    def get_sessions_by_plan(
        self,
        plan_name: Optional[str] = None,
        project: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Get all sessions linked to a plan."""
        cursor = self._read_cursor()
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
            results.append(
                {
                    "session_id": row[0],
                    "project": row[1],
                    "branch": row[2],
                    "pr_number": row[3],
                    "started_at": row[4],
                    "ended_at": row[5],
                    "plan_name": row[6],
                    "plan_phase": row[7],
                    "story_id": row[8],
                }
            )
        return results

    def get_plan_linking_stats(self) -> Dict[str, Any]:
        """Get plan linking statistics."""
        cursor = self._read_cursor()
        total = list(cursor.execute("SELECT COUNT(*) FROM session_context"))[0][0]
        linked = list(cursor.execute("SELECT COUNT(*) FROM session_context WHERE plan_name IS NOT NULL"))[0][0]
        plans = list(
            cursor.execute(
                "SELECT plan_name, COUNT(*) FROM session_context"
                " WHERE plan_name IS NOT NULL"
                " GROUP BY plan_name ORDER BY COUNT(*) DESC"
            )
        )
        return {
            "total_sessions": total,
            "linked_sessions": linked,
            "unlinked_sessions": total - linked,
            "plans": {row[0]: row[1] for row in plans},
        }

    def clear_plan_links(self, project: Optional[str] = None) -> int:
        """Clear plan links. Returns count cleared."""
        cursor = self.conn.cursor()
        if project:
            rows = list(
                cursor.execute(
                    "SELECT COUNT(*) FROM session_context WHERE plan_name IS NOT NULL AND project = ?",
                    (project,),
                )
            )
            cursor.execute(
                "UPDATE session_context SET plan_name = NULL, plan_phase = NULL, story_id = NULL WHERE project = ?",
                (project,),
            )
        else:
            rows = list(cursor.execute("SELECT COUNT(*) FROM session_context WHERE plan_name IS NOT NULL"))
            cursor.execute("UPDATE session_context SET plan_name = NULL, plan_phase = NULL, story_id = NULL")
        return rows[0][0] if rows else 0

    def get_git_overlay_stats(self) -> Dict[str, Any]:
        """Get git overlay statistics."""
        cursor = self._read_cursor()
        sessions = list(cursor.execute("SELECT COUNT(*) FROM session_context"))[0][0]
        interactions = list(cursor.execute("SELECT COUNT(*) FROM file_interactions"))[0][0]
        unique_files = list(cursor.execute("SELECT COUNT(DISTINCT file_path) FROM file_interactions"))[0][0]
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

    # --- Operations ---

    def store_operations(self, operations: List[Dict[str, Any]]) -> int:
        """Store operation groups."""
        if not operations:
            return 0
        cursor = self.conn.cursor()
        now = datetime.now(timezone.utc).isoformat()
        count = 0
        for op in operations:
            chunk_ids_json = json.dumps(op.get("chunk_ids", []))
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

    def get_session_operations(self, session_id: str) -> List[Dict[str, Any]]:
        """Get all operations for a session."""
        cursor = self._read_cursor()
        rows = list(
            cursor.execute(
                """SELECT id, session_id, operation_type,
                      chunk_ids, summary, outcome,
                      started_at, ended_at, step_count
               FROM operations
               WHERE session_id = ?
               ORDER BY started_at""",
                (session_id,),
            )
        )
        results = []
        for row in rows:
            chunk_ids = []
            if row[3]:
                try:
                    chunk_ids = json.loads(row[3])
                except (json.JSONDecodeError, TypeError):
                    pass
            results.append(
                {
                    "id": row[0],
                    "session_id": row[1],
                    "operation_type": row[2],
                    "chunk_ids": chunk_ids,
                    "summary": row[4],
                    "outcome": row[5],
                    "started_at": row[6],
                    "ended_at": row[7],
                    "step_count": row[8],
                }
            )
        return results

    def get_operations_stats(self) -> Dict[str, Any]:
        """Get operation grouping statistics."""
        cursor = self._read_cursor()
        total = list(cursor.execute("SELECT COUNT(*) FROM operations"))[0][0]
        by_type = list(
            cursor.execute(
                """SELECT operation_type, COUNT(*)
               FROM operations
               GROUP BY operation_type
               ORDER BY COUNT(*) DESC"""
            )
        )
        sessions = list(cursor.execute("SELECT COUNT(DISTINCT session_id) FROM operations"))[0][0]
        return {
            "total_operations": total,
            "sessions_with_operations": sessions,
            "by_type": {(row[0] or "unknown"): row[1] for row in by_type},
        }

    def clear_session_operations(self, session_id: str) -> None:
        """Clear operations for a session."""
        cursor = self.conn.cursor()
        cursor.execute("DELETE FROM operations WHERE session_id = ?", (session_id,))

    # --- Topic Chains ---

    def store_topic_chains(self, chains: List[Dict[str, Any]]) -> int:
        """Store topic chain entries."""
        if not chains:
            return 0
        cursor = self.conn.cursor()
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

    def get_file_chains(self, file_path: str, limit: int = 20) -> List[Dict[str, Any]]:
        """Get topic chains for a file."""
        cursor = self._read_cursor()
        rows = list(
            cursor.execute(
                """SELECT tc.file_path, tc.session_a,
                      tc.session_b, tc.shared_actions,
                      tc.time_delta_hours, tc.project,
                      sa.branch AS branch_a,
                      sb.branch AS branch_b
               FROM topic_chains tc
               LEFT JOIN session_context sa ON tc.session_a = sa.session_id
               LEFT JOIN session_context sb ON tc.session_b = sb.session_id
               WHERE tc.file_path LIKE ?
               ORDER BY tc.time_delta_hours
               LIMIT ?""",
                (f"%{file_path}%", limit),
            )
        )
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
        """Get regression info for a file."""
        cursor = self._read_cursor()

        query = """
            SELECT fi.file_path, fi.timestamp,
                   fi.session_id, fi.action,
                   fi.project,
                   sc.branch, sc.pr_number
            FROM file_interactions fi
            LEFT JOIN session_context sc ON fi.session_id = sc.session_id
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

        timeline = []
        for row in interactions:
            timeline.append(
                {
                    "file_path": row[0],
                    "timestamp": row[1],
                    "session_id": row[2],
                    "action": row[3],
                    "project": row[4],
                    "branch": row[5],
                    "pr_number": row[6],
                }
            )

        last_success = None
        changes_after = []

        for entry in reversed(timeline):
            sid = entry["session_id"]
            if not sid:
                continue
            ops = list(
                cursor.execute(
                    """SELECT outcome FROM operations
                   WHERE session_id = ?
                   AND outcome = 'success'
                   LIMIT 1""",
                    (sid,),
                )
            )
            if ops:
                last_success = entry
                break

        if last_success and last_success.get("timestamp"):
            changes_after = [e for e in timeline if (e.get("timestamp") or "") > last_success["timestamp"]]

        return {
            "file_path": file_path,
            "timeline": timeline,
            "last_success": last_success,
            "changes_after": changes_after,
        }

    def get_topic_chain_stats(self) -> Dict[str, Any]:
        """Get topic chain statistics."""
        cursor = self._read_cursor()
        total = list(cursor.execute("SELECT COUNT(*) FROM topic_chains"))[0][0]
        files = list(cursor.execute("SELECT COUNT(DISTINCT file_path) FROM topic_chains"))[0][0]
        return {"total_chains": total, "unique_files": files}

    def clear_topic_chains(self, project: Optional[str] = None) -> None:
        """Clear topic chains, optionally for a project."""
        cursor = self.conn.cursor()
        if project:
            cursor.execute("DELETE FROM topic_chains WHERE project = ?", (project,))
        else:
            cursor.execute("DELETE FROM topic_chains")

    # --- Session Enrichment ---

    def upsert_session_enrichment(self, enrichment: Dict[str, Any]) -> None:
        """Insert or update a session enrichment record."""
        cursor = self.conn.cursor()
        enrichment = dict(enrichment)
        session_id = enrichment["session_id"]

        json_fields = [
            "decisions_made",
            "corrections",
            "learnings",
            "mistakes",
            "patterns",
            "topic_tags",
            "tool_usage_stats",
        ]
        for field in json_fields:
            if field in enrichment and not isinstance(enrichment[field], str):
                enrichment[field] = json.dumps(enrichment[field])

        cursor.execute(
            """
            INSERT INTO session_enrichments (
                session_id, file_path, enrichment_version, enrichment_model,
                session_start_time, session_end_time, duration_seconds,
                message_count, user_message_count, assistant_message_count, tool_call_count,
                session_summary, primary_intent, outcome, complexity_score,
                session_quality_score,
                decisions_made, corrections, learnings, mistakes, patterns,
                topic_tags, tool_usage_stats,
                what_worked, what_failed,
                summary_embedding
            ) VALUES (
                ?, ?, ?, ?,
                ?, ?, ?,
                ?, ?, ?, ?,
                ?, ?, ?, ?,
                ?,
                ?, ?, ?, ?, ?,
                ?, ?,
                ?, ?,
                ?
            )
            ON CONFLICT(session_id) DO UPDATE SET
                enrichment_version = excluded.enrichment_version,
                enrichment_model = excluded.enrichment_model,
                enrichment_timestamp = strftime('%Y-%m-%dT%H:%M:%fZ','now'),
                session_start_time = excluded.session_start_time,
                session_end_time = excluded.session_end_time,
                duration_seconds = excluded.duration_seconds,
                message_count = excluded.message_count,
                user_message_count = excluded.user_message_count,
                assistant_message_count = excluded.assistant_message_count,
                tool_call_count = excluded.tool_call_count,
                session_summary = excluded.session_summary,
                primary_intent = excluded.primary_intent,
                outcome = excluded.outcome,
                complexity_score = excluded.complexity_score,
                session_quality_score = excluded.session_quality_score,
                decisions_made = excluded.decisions_made,
                corrections = excluded.corrections,
                learnings = excluded.learnings,
                mistakes = excluded.mistakes,
                patterns = excluded.patterns,
                topic_tags = excluded.topic_tags,
                tool_usage_stats = excluded.tool_usage_stats,
                what_worked = excluded.what_worked,
                what_failed = excluded.what_failed,
                summary_embedding = excluded.summary_embedding
            """,
            (
                session_id,
                enrichment.get("file_path"),
                enrichment.get("enrichment_version", "1.0"),
                enrichment.get("enrichment_model"),
                enrichment.get("session_start_time"),
                enrichment.get("session_end_time"),
                enrichment.get("duration_seconds"),
                enrichment.get("message_count", 0),
                enrichment.get("user_message_count", 0),
                enrichment.get("assistant_message_count", 0),
                enrichment.get("tool_call_count", 0),
                enrichment.get("session_summary"),
                enrichment.get("primary_intent"),
                enrichment.get("outcome"),
                enrichment.get("complexity_score"),
                enrichment.get("session_quality_score"),
                enrichment.get("decisions_made", "[]"),
                enrichment.get("corrections", "[]"),
                enrichment.get("learnings", "[]"),
                enrichment.get("mistakes", "[]"),
                enrichment.get("patterns", "[]"),
                enrichment.get("topic_tags", "[]"),
                enrichment.get("tool_usage_stats", "[]"),
                enrichment.get("what_worked"),
                enrichment.get("what_failed"),
                enrichment.get("summary_embedding"),
            ),
        )

        # Update FTS5
        cursor.execute(
            "DELETE FROM session_enrichments_fts WHERE session_id = ?",
            (session_id,),
        )
        if enrichment.get("session_summary") or enrichment.get("what_worked") or enrichment.get("what_failed"):
            cursor.execute(
                """INSERT INTO session_enrichments_fts
                   (session_summary, what_worked, what_failed, session_id)
                   VALUES (?, ?, ?, ?)""",
                (
                    enrichment.get("session_summary", ""),
                    enrichment.get("what_worked", ""),
                    enrichment.get("what_failed", ""),
                    session_id,
                ),
            )

    _SESSION_ENRICHMENT_COLS = [
        "id",
        "session_id",
        "file_path",
        "enrichment_version",
        "enrichment_model",
        "enrichment_timestamp",
        "session_start_time",
        "session_end_time",
        "duration_seconds",
        "message_count",
        "user_message_count",
        "assistant_message_count",
        "tool_call_count",
        "session_summary",
        "primary_intent",
        "outcome",
        "complexity_score",
        "session_quality_score",
        "decisions_made",
        "corrections",
        "learnings",
        "mistakes",
        "patterns",
        "topic_tags",
        "tool_usage_stats",
        "what_worked",
        "what_failed",
        "summary_embedding",
    ]

    def get_session_enrichment(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get enrichment data for a session."""
        cursor = self._read_cursor()
        rows = list(
            cursor.execute(
                "SELECT * FROM session_enrichments WHERE session_id = ?",
                (session_id,),
            )
        )
        if not rows:
            return None
        row = rows[0]
        result = dict(zip(self._SESSION_ENRICHMENT_COLS, row))
        for field in [
            "decisions_made",
            "corrections",
            "learnings",
            "mistakes",
            "patterns",
            "topic_tags",
            "tool_usage_stats",
        ]:
            result[field] = _safe_json_loads(result.get(field))
        return result

    def list_enriched_sessions(self) -> List[str]:
        """Return session IDs that already have enrichment data."""
        cursor = self._read_cursor()
        return [row[0] for row in cursor.execute("SELECT session_id FROM session_enrichments")]

    def get_session_enrichment_stats(self) -> Dict[str, Any]:
        """Get session enrichment statistics."""
        cursor = self._read_cursor()
        total = list(cursor.execute("SELECT COUNT(*) FROM session_enrichments"))[0][0]
        by_outcome = dict(
            cursor.execute(
                "SELECT outcome, COUNT(*) FROM session_enrichments WHERE outcome IS NOT NULL GROUP BY outcome"
            )
        )
        by_intent = dict(
            cursor.execute(
                "SELECT primary_intent, COUNT(*) FROM session_enrichments WHERE primary_intent IS NOT NULL GROUP BY primary_intent"
            )
        )
        avg_quality = list(
            cursor.execute(
                "SELECT AVG(session_quality_score) FROM session_enrichments WHERE session_quality_score IS NOT NULL"
            )
        )[0][0]
        return {
            "total_enriched_sessions": total,
            "by_outcome": by_outcome,
            "by_intent": by_intent,
            "avg_quality_score": round(avg_quality, 1) if avg_quality else None,
        }
