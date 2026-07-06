from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import apsw
import pytest

from brainlayer.vector_store import VectorStore


def _load_script() -> ModuleType:
    module_path = Path(__file__).resolve().parents[1] / "scripts" / "retro_quarantine_self_pollution.py"
    spec = importlib.util.spec_from_file_location("retro_quarantine_self_pollution", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _insert_chunk(
    store: VectorStore,
    *,
    chunk_id: str,
    source_file: Path | str,
    content: str = "retro quarantine sentinel content",
    content_class: str = "knowledge",
    provenance_class: str | None = "RAW-ETAN-DIRECT",
) -> None:
    store.conn.cursor().execute(
        """INSERT INTO chunks (
            id, content, metadata, source_file, project, content_type,
            char_count, source, importance, created_at, content_class, provenance_class
        ) VALUES (?, ?, '{}', ?, 'retro-test', 'note', ?, 'claude_code', 5,
            '2026-07-03T00:00:00Z', ?, ?)""",
        (chunk_id, content, str(source_file), len(content), content_class, provenance_class),
    )


def test_dry_run_uses_denylist_and_preserves_direct_sessions(tmp_path: Path) -> None:
    script = _load_script()
    db_path = tmp_path / "dry-run.db"
    store = VectorStore(db_path)
    try:
        _insert_chunk(
            store,
            chunk_id="claude-subagent",
            source_file=tmp_path / ".claude" / "projects" / "proj" / "session" / "subagents" / "agent-a1.jsonl",
        )
        _insert_chunk(
            store,
            chunk_id="codex-session",
            source_file=tmp_path / ".codex" / "sessions" / "worker.jsonl",
        )
        _insert_chunk(
            store,
            chunk_id="claude-direct",
            source_file=tmp_path / ".claude" / "projects" / "proj" / "direct-session.jsonl",
        )
        _insert_chunk(store, chunk_id="manual-note", source_file=tmp_path / "notes" / "memory.md")
    finally:
        store.close()

    report = script.run_dry_run(db_path, sample_size=10, random_seed=7)

    assert report["counts"] == {"quarantine_set": 2, "preserved": 2, "total": 4}
    assert report["provider_breakdown"]["quarantine_set"] == {"claude": 1, "codex": 1}
    assert report["audit"]["quarantine_sample_size"] == 2
    assert report["audit"]["direct_session_source_files_checked"] == 1
    assert report["audit"]["direct_session_false_positive_count"] == 0
    assert report["audit"]["stop_gate_passed"] is True


def test_dry_run_stop_gate_fails_when_direct_session_is_classified(tmp_path: Path, monkeypatch) -> None:
    script = _load_script()
    monkeypatch.setenv("BRAINLAYER_INGEST_DENYLIST", "~/.claude/projects/**")
    db_path = tmp_path / "direct-false-positive.db"
    store = VectorStore(db_path)
    try:
        _insert_chunk(
            store,
            chunk_id="claude-direct",
            source_file=tmp_path / ".claude" / "projects" / "proj" / "direct-session.jsonl",
        )
    finally:
        store.close()

    report = script.run_dry_run(db_path, sample_size=10, random_seed=7)

    assert report["counts"] == {"quarantine_set": 1, "preserved": 0, "total": 1}
    assert report["audit"]["direct_session_false_positive_count"] == 1
    assert report["audit"]["stop_gate_passed"] is False


def test_quarantine_unquarantine_round_trip_restores_chunk_and_fts_rows(tmp_path: Path) -> None:
    script = _load_script()
    db_path = tmp_path / "round-trip.db"
    denied_source = tmp_path / ".claude" / "projects" / "proj" / "session" / "subagents" / "agent-a1.jsonl"
    store = VectorStore(db_path)
    try:
        _insert_chunk(
            store,
            chunk_id="claude-subagent",
            source_file=denied_source,
            content="round trip exact fts sentinel",
            provenance_class="RAW-ETAN-DIRECT",
        )
        before = script.capture_restore_state(store.conn.cursor(), ["claude-subagent"])
    finally:
        store.close()

    quarantine_report = script.apply_quarantine_ids(db_path, ["claude-subagent"], batch_size=1, run_id="test-run")
    assert quarantine_report["quarantined"] == 1

    quarantined_store = VectorStore(db_path)
    try:
        cursor = quarantined_store.conn.cursor()
        row = cursor.execute(
            "SELECT content_class, provenance_class FROM chunks WHERE id = 'claude-subagent'"
        ).fetchone()
        knowledge_rows = cursor.execute("SELECT chunk_id FROM chunks_fts WHERE chunk_id = 'claude-subagent'").fetchall()
        operational_rows = cursor.execute(
            "SELECT chunk_id FROM chunks_fts_operational WHERE chunk_id = 'claude-subagent'"
        ).fetchall()
    finally:
        quarantined_store.close()

    assert row == ("operational", "AGENT-INFERENCE")
    assert knowledge_rows == []
    assert operational_rows == [("claude-subagent",)]

    revert_report = script.unquarantine_ids(db_path, ["claude-subagent"], run_id="test-run")
    assert revert_report["restored"] == 1

    restored_store = VectorStore(db_path)
    try:
        after = script.capture_restore_state(restored_store.conn.cursor(), ["claude-subagent"])
    finally:
        restored_store.close()

    assert after == before


def test_quarantine_retrievability_proof_excludes_default_but_preserves_operational_paths(
    tmp_path: Path,
) -> None:
    script = _load_script()
    db_path = tmp_path / "retrievability.db"
    denied_source = tmp_path / ".claude" / "projects" / "proj" / "session" / "subagents" / "agent-a1.jsonl"
    store = VectorStore(db_path)
    try:
        _insert_chunk(
            store,
            chunk_id="claude-subagent",
            source_file=denied_source,
            content="retrievability invariant exactsentinel",
            provenance_class="RAW-ETAN-DIRECT",
        )
    finally:
        store.close()

    quarantine_report = script.apply_quarantine_ids(db_path, ["claude-subagent"], batch_size=1, run_id="proof-run")
    assert quarantine_report["quarantined"] == 1

    proof = script.run_retrievability_proof(db_path, ["claude-subagent"])

    assert proof["passed"] is True
    assert proof["sample_size"] == 1
    assert proof["sample_ids"] == ["claude-subagent"]
    chunk_proof = proof["chunks"]["claude-subagent"]
    assert chunk_proof["default_search_absent"] is True
    assert chunk_proof["operational_search_present"] is True
    assert chunk_proof["expand_fetchable"] is True
    assert chunk_proof["chunk_row_exists"] is True
    assert chunk_proof["content_class"] == "operational"
    assert chunk_proof["operational_fts_rows"] == 1
    assert chunk_proof["default_fts_rows"] == 0


def test_run_apply_refuses_when_retrievability_proof_fails(tmp_path: Path, monkeypatch) -> None:
    script = _load_script()
    db_path = tmp_path / "apply-proof-gate.db"
    backup_path = tmp_path / "apply-proof-gate-backup.db"
    denied_source = tmp_path / ".codex" / "sessions" / "worker.jsonl"
    store = VectorStore(db_path)
    try:
        _insert_chunk(
            store,
            chunk_id="codex-session",
            source_file=denied_source,
            content="apply gate proof sentinel",
        )
    finally:
        store.close()

    def fail_retrievability_proof(db_path_arg, chunk_ids):
        assert db_path_arg == db_path
        assert chunk_ids == ["codex-session"]
        return {"passed": False, "failed_ids": ["codex-session"]}

    monkeypatch.setattr(script, "run_retrievability_proof", fail_retrievability_proof)

    with pytest.raises(RuntimeError, match="retrievability proof failed"):
        script.run_apply(
            db_path,
            backup_path=backup_path,
            confirm_workers_stopped=True,
            confirm_watcher_paused=True,
        )


def test_round_trip_ignores_stale_chunk_fts_rowids(tmp_path: Path) -> None:
    script = _load_script()
    db_path = tmp_path / "stale-rowids.db"
    denied_source = tmp_path / ".claude" / "projects" / "proj" / "session" / "subagents" / "agent-a1.jsonl"
    store = VectorStore(db_path)
    try:
        _insert_chunk(
            store,
            chunk_id="other-knowledge",
            source_file=tmp_path / "notes" / "memory.md",
            content="other row exact fts sentinel",
        )
        _insert_chunk(
            store,
            chunk_id="claude-subagent",
            source_file=denied_source,
            content="stale rowids exact fts sentinel",
        )
        other_rowid = (
            store.conn.cursor().execute("SELECT rowid FROM chunks_fts WHERE chunk_id = 'other-knowledge'").fetchone()[0]
        )
        store.conn.cursor().execute(
            "UPDATE chunk_fts_rowids SET fts_rowid = ? WHERE chunk_id = 'claude-subagent'",
            (other_rowid,),
        )
        before = script.capture_restore_state(store.conn.cursor(), ["claude-subagent"])
    finally:
        store.close()

    quarantine_report = script.apply_quarantine_ids(db_path, ["claude-subagent"], batch_size=1, run_id="stale-run")
    assert quarantine_report["quarantined"] == 1
    revert_report = script.unquarantine_ids(db_path, ["claude-subagent"], run_id="stale-run")
    assert revert_report["restored"] == 1

    restored_store = VectorStore(db_path)
    try:
        after = script.capture_restore_state(restored_store.conn.cursor(), ["claude-subagent"])
    finally:
        restored_store.close()

    assert after == before


def test_round_trip_restores_preexisting_operational_fts_duplicate(tmp_path: Path) -> None:
    script = _load_script()
    db_path = tmp_path / "duplicate-operational-row.db"
    denied_source = tmp_path / ".claude" / "projects" / "proj" / "session" / "subagents" / "agent-a1.jsonl"
    store = VectorStore(db_path)
    try:
        _insert_chunk(
            store,
            chunk_id="claude-subagent",
            source_file=denied_source,
            content="duplicate operational row exact fts sentinel",
            content_class="knowledge",
        )
        store.conn.cursor().execute("""
            INSERT INTO chunks_fts_operational(
                content, summary, tags, resolved_query, key_facts, resolved_queries, chunk_id
            )
            SELECT content, summary, tags, resolved_query, key_facts, resolved_queries, id
            FROM chunks
            WHERE id = 'claude-subagent'
        """)
        store.conn.cursor().execute("""
            UPDATE chunk_fts_rowids
            SET operational_rowid = (
                SELECT rowid FROM chunks_fts_operational WHERE chunk_id = 'claude-subagent'
            )
            WHERE chunk_id = 'claude-subagent'
        """)
        before = script.capture_restore_state(store.conn.cursor(), ["claude-subagent"])
    finally:
        store.close()

    script.apply_quarantine_ids(db_path, ["claude-subagent"], batch_size=1, run_id="duplicate-run")
    script.unquarantine_ids(db_path, ["claude-subagent"], run_id="duplicate-run")

    restored_store = VectorStore(db_path)
    try:
        after = script.capture_restore_state(restored_store.conn.cursor(), ["claude-subagent"])
    finally:
        restored_store.close()

    assert after == before


def test_revert_proof_runs_only_on_snapshot_path(tmp_path: Path) -> None:
    script = _load_script()
    db_path = tmp_path / "source.db"
    snapshot_path = tmp_path / "snapshot.db"
    store = VectorStore(db_path)
    try:
        _insert_chunk(
            store,
            chunk_id="claude-subagent",
            source_file=tmp_path / ".claude" / "projects" / "proj" / "session" / "subagents" / "agent-a1.jsonl",
            content="snapshot revert proof sentinel",
        )
    finally:
        store.close()

    report = script.run_revert_proof(
        db_path,
        snapshot_path=snapshot_path,
        sample_size=1,
        random_seed=11,
        replace_snapshot=True,
    )

    assert report["source_db_path"] == str(db_path)
    assert report["snapshot_db_path"] == str(snapshot_path)
    assert report["sample_size"] == 1
    assert report["byte_identical_restoration"] is True
    assert snapshot_path.exists()


def test_direct_script_execution_bootstraps_operational_fts_for_legacy_snapshot(tmp_path: Path) -> None:
    db_path = tmp_path / "legacy-source.db"
    snapshot_path = tmp_path / "legacy-snapshot.db"
    source_file = tmp_path / ".claude" / "projects" / "proj" / "session" / "subagents" / "agent-a1.jsonl"
    conn = apsw.Connection(str(db_path))
    try:
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE chunks (
                id TEXT PRIMARY KEY,
                content TEXT NOT NULL,
                metadata TEXT NOT NULL,
                source_file TEXT NOT NULL,
                project TEXT,
                content_type TEXT,
                value_type TEXT,
                char_count INTEGER,
                source TEXT,
                created_at TEXT,
                content_class TEXT,
                provenance_class TEXT,
                summary TEXT,
                tags TEXT,
                resolved_query TEXT,
                key_facts TEXT,
                resolved_queries TEXT
            )
        """)
        cursor.execute("""
            CREATE VIRTUAL TABLE chunks_fts USING fts5(
                content, summary, tags, resolved_query, key_facts, resolved_queries, chunk_id UNINDEXED
            )
        """)
        cursor.execute("""
            CREATE VIRTUAL TABLE chunks_fts_trigram USING fts5(
                content, summary, tags, resolved_query, key_facts, resolved_queries, chunk_id UNINDEXED,
                tokenize='trigram'
            )
        """)
        cursor.execute("""
            CREATE TABLE chunk_fts_rowids (
                chunk_id TEXT PRIMARY KEY,
                fts_rowid INTEGER,
                trigram_rowid INTEGER
            )
        """)
        cursor.execute(
            """INSERT INTO chunks (
                id, content, metadata, source_file, project, content_type, char_count, source,
                created_at, content_class, provenance_class
            ) VALUES (?, 'legacy direct execution sentinel', '{}', ?, 'retro-test', 'note', 32,
                'claude_code', '2026-07-03T00:00:00Z', 'knowledge', 'RAW-ETAN-DIRECT')""",
            ("legacy-denylisted", str(source_file)),
        )
        cursor.execute("""
            INSERT INTO chunks_fts(content, summary, tags, resolved_query, key_facts, resolved_queries, chunk_id)
            SELECT content, summary, tags, resolved_query, key_facts, resolved_queries, id FROM chunks
        """)
        cursor.execute("""
            INSERT INTO chunks_fts_trigram(content, summary, tags, resolved_query, key_facts, resolved_queries, chunk_id)
            SELECT content, summary, tags, resolved_query, key_facts, resolved_queries, id FROM chunks
        """)
        cursor.execute("""
            INSERT INTO chunk_fts_rowids(chunk_id, fts_rowid, trigram_rowid)
            SELECT c.id, f.rowid, t.rowid
            FROM chunks c
            JOIN chunks_fts f ON f.chunk_id = c.id
            JOIN chunks_fts_trigram t ON t.chunk_id = c.id
        """)
    finally:
        conn.close()

    script_path = Path(__file__).resolve().parents[1] / "scripts" / "retro_quarantine_self_pollution.py"
    result = subprocess.run(
        [
            sys.executable,
            str(script_path),
            "--db",
            str(db_path),
            "--revert-proof",
            "--snapshot-path",
            str(snapshot_path),
            "--sample-size",
            "1",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert '"byte_identical_restoration": true' in result.stdout
