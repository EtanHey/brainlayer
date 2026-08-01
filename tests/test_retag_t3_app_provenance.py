import json
import sqlite3
from pathlib import Path

import pytest

import scripts.retag_t3_app_provenance as retag_module
from scripts.retag_t3_app_provenance import retag_t3_app_chunks, rollback_t3_app_chunks


def _create_t3_state_db(path: Path, session_id: str) -> None:
    with sqlite3.connect(path) as conn:
        conn.execute(
            """
            CREATE TABLE provider_session_runtime (
                thread_id TEXT PRIMARY KEY,
                provider_name TEXT NOT NULL,
                resume_cursor_json TEXT NOT NULL
            )
            """
        )
        conn.execute(
            "INSERT INTO provider_session_runtime VALUES (?, ?, ?)",
            ("t3-thread", "codex", json.dumps({"threadId": session_id})),
        )


def _create_brain_db(path: Path, app_session_id: str, plain_session_id: str) -> None:
    with sqlite3.connect(path) as conn:
        conn.execute("CREATE TABLE chunks (id TEXT PRIMARY KEY, source_file TEXT, provenance_class TEXT)")
        conn.executemany(
            "INSERT INTO chunks VALUES (?, ?, ?)",
            [
                ("app-1", f"/home/etan/.codex/sessions/rollout-{app_session_id}.jsonl", "codex-session"),
                ("app-2", f"/home/etan/.codex/sessions/rollout-{app_session_id}.jsonl", "RAW-ETAN-DIRECT"),
                ("app-none", f"/home/etan/.codex/sessions/rollout-{app_session_id}.jsonl", None),
                ("app-recon", f"/home/etan/.codex/sessions/rollout-{app_session_id}.jsonl", "recon-agent"),
                ("plain", f"/home/etan/.codex/sessions/rollout-{plain_session_id}.jsonl", "codex-session"),
            ],
        )


def test_retag_only_explicitly_t3_linked_sessions_and_write_rollback_artifact(tmp_path: Path) -> None:
    app_session_id = "11111111-1111-4111-8111-111111111111"
    plain_session_id = "22222222-2222-4222-8222-222222222222"
    state_db = tmp_path / "state.sqlite"
    brain_db = tmp_path / "brainlayer.db"
    rollback_artifact = tmp_path / "rollback.jsonl"
    _create_t3_state_db(state_db, app_session_id)
    _create_brain_db(brain_db, app_session_id, plain_session_id)

    report = retag_t3_app_chunks(
        db_path=brain_db,
        state_db=state_db,
        apply=True,
        rollback_artifact=rollback_artifact,
        batch_size=1,
    )

    assert report == {"linked_sessions": 1, "candidate_chunks": 1, "matched_chunks": 1, "retagged_chunks": 1}
    assert [json.loads(line) for line in rollback_artifact.read_text().splitlines()] == [
        {"id": "app-1", "provenance_class": "codex-session"},
    ]
    with sqlite3.connect(brain_db) as conn:
        assert dict(conn.execute("SELECT id, provenance_class FROM chunks")) == {
            "app-1": "t3-app-session",
            "app-2": "RAW-ETAN-DIRECT",
            "app-none": None,
            "app-recon": "recon-agent",
            "plain": "codex-session",
        }
    assert rollback_t3_app_chunks(db_path=brain_db, rollback_artifact=rollback_artifact, batch_size=1) == {
        "requested_chunks": 1,
        "matched_chunks": 1,
        "restored_chunks": 1,
        "missing_chunks": 0,
        "skipped_non_t3_chunks": 0,
    }
    with sqlite3.connect(brain_db) as conn:
        assert dict(conn.execute("SELECT id, provenance_class FROM chunks")) == {
            "app-1": "codex-session",
            "app-2": "RAW-ETAN-DIRECT",
            "app-none": None,
            "app-recon": "recon-agent",
            "plain": "codex-session",
        }


def test_rollback_can_restore_only_null_prior_values_and_snapshot_current_rows(tmp_path: Path) -> None:
    brain_db = tmp_path / "brainlayer.db"
    rollback_artifact = tmp_path / "rollback.jsonl"
    pre_restore_artifact = tmp_path / "pre-restore.jsonl"
    with sqlite3.connect(brain_db) as conn:
        conn.execute("CREATE TABLE chunks (id TEXT PRIMARY KEY, source_file TEXT, provenance_class TEXT)")
        conn.executemany(
            "INSERT INTO chunks VALUES (?, ?, ?)",
            [
                ("prior-null", "/home/etan/.codex/sessions/a.jsonl", "t3-app-session"),
                ("prior-codex", "/home/etan/.codex/sessions/b.jsonl", "t3-app-session"),
            ],
        )
    rollback_artifact.write_text(
        "\n".join(
            [
                json.dumps({"id": "prior-null", "provenance_class": None}),
                json.dumps({"id": "prior-codex", "provenance_class": "codex-session"}),
            ]
        )
        + "\n"
    )

    assert rollback_t3_app_chunks(
        db_path=brain_db,
        rollback_artifact=rollback_artifact,
        only_null_prior_values=True,
        pre_restore_artifact=pre_restore_artifact,
        batch_size=1,
    ) == {
        "requested_chunks": 1,
        "matched_chunks": 1,
        "restored_chunks": 1,
        "missing_chunks": 0,
        "skipped_non_t3_chunks": 0,
    }
    assert [json.loads(line) for line in pre_restore_artifact.read_text().splitlines()] == [
        {"id": "prior-null", "provenance_class": "t3-app-session"}
    ]
    with sqlite3.connect(brain_db) as conn:
        assert dict(conn.execute("SELECT id, provenance_class FROM chunks")) == {
            "prior-null": None,
            "prior-codex": "t3-app-session",
        }


def test_retag_does_not_overwrite_a_value_changed_after_the_artifact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    app_session_id = "11111111-1111-4111-8111-111111111111"
    state_db = tmp_path / "state.sqlite"
    brain_db = tmp_path / "brainlayer.db"
    rollback_artifact = tmp_path / "rollback.jsonl"
    _create_t3_state_db(state_db, app_session_id)
    _create_brain_db(brain_db, app_session_id, "22222222-2222-4222-8222-222222222222")
    original_write = retag_module._write_rollback_artifact

    def change_value_after_snapshot(path: Path, candidates: list[tuple[str, str | None]]) -> None:
        original_write(path, candidates)
        with sqlite3.connect(brain_db) as conn:
            conn.execute("UPDATE chunks SET provenance_class = 'RAW-ETAN-DIRECT' WHERE id = 'app-1'")

    monkeypatch.setattr(retag_module, "_write_rollback_artifact", change_value_after_snapshot)

    report = retag_t3_app_chunks(
        db_path=brain_db,
        state_db=state_db,
        apply=True,
        rollback_artifact=rollback_artifact,
    )

    assert report["candidate_chunks"] == 1
    assert report["matched_chunks"] == 0
    assert report["retagged_chunks"] == 0
    with sqlite3.connect(brain_db) as conn:
        assert conn.execute("SELECT provenance_class FROM chunks WHERE id = 'app-1'").fetchone()[0] == "RAW-ETAN-DIRECT"


def test_rollback_does_not_overwrite_a_value_changed_after_the_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    brain_db = tmp_path / "brainlayer.db"
    rollback_artifact = tmp_path / "rollback.jsonl"
    pre_restore_artifact = tmp_path / "pre-restore.jsonl"
    with sqlite3.connect(brain_db) as conn:
        conn.execute("CREATE TABLE chunks (id TEXT PRIMARY KEY, source_file TEXT, provenance_class TEXT)")
        conn.execute("INSERT INTO chunks VALUES ('app-1', 'source', 't3-app-session')")
    rollback_artifact.write_text(json.dumps({"id": "app-1", "provenance_class": "codex-session"}) + "\n")
    original_write = retag_module._atomic_write_jsonl

    def change_value_after_snapshot(path: Path, rows: list[tuple[str, str | None]], **kwargs: object) -> None:
        original_write(path, rows, **kwargs)
        with sqlite3.connect(brain_db) as conn:
            conn.execute("UPDATE chunks SET provenance_class = 'RAW-ETAN-DIRECT' WHERE id = 'app-1'")

    monkeypatch.setattr(retag_module, "_atomic_write_jsonl", change_value_after_snapshot)

    report = rollback_t3_app_chunks(
        db_path=brain_db,
        rollback_artifact=rollback_artifact,
        pre_restore_artifact=pre_restore_artifact,
    )

    assert report["matched_chunks"] == 1
    assert report["restored_chunks"] == 0
    with sqlite3.connect(brain_db) as conn:
        assert conn.execute("SELECT provenance_class FROM chunks WHERE id = 'app-1'").fetchone()[0] == "RAW-ETAN-DIRECT"


def test_rollback_skips_missing_ids_without_creating_an_empty_pre_restore_artifact(tmp_path: Path) -> None:
    brain_db = tmp_path / "brainlayer.db"
    rollback_artifact = tmp_path / "rollback.jsonl"
    pre_restore_artifact = tmp_path / "pre-restore.jsonl"
    with sqlite3.connect(brain_db) as conn:
        conn.execute("CREATE TABLE chunks (id TEXT PRIMARY KEY, source_file TEXT, provenance_class TEXT)")
    rollback_artifact.write_text(json.dumps({"id": "deleted", "provenance_class": "codex-session"}) + "\n")

    report = rollback_t3_app_chunks(
        db_path=brain_db,
        rollback_artifact=rollback_artifact,
        pre_restore_artifact=pre_restore_artifact,
    )

    assert report["missing_chunks"] == 1
    assert report["matched_chunks"] == 0
    assert not pre_restore_artifact.exists()


def test_rollback_artifact_replacement_preserves_the_old_file_on_replace_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifact = tmp_path / "rollback.jsonl"
    original = json.dumps({"id": "old", "provenance_class": "codex-session"}) + "\n"
    artifact.write_text(original)

    def fail_replace(source: str, destination: str) -> None:
        raise OSError("simulated replace failure")

    monkeypatch.setattr(retag_module.os, "replace", fail_replace)

    with pytest.raises(OSError, match="simulated replace failure"):
        retag_module._write_rollback_artifact(artifact, [("new", "codex-session")])

    assert artifact.read_text() == original
