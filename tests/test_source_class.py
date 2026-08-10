import json
from pathlib import Path

import pytest

from brainlayer.agent_provenance import classify_provenance
from brainlayer.embeddings import EmbeddedChunk
from brainlayer.index_new import index_chunks_to_sqlite
from brainlayer.ingest_denylist import BRAINLAYER_INGEST_DENYLIST_ENV, is_denylisted
from brainlayer.pipeline.chunk import Chunk
from brainlayer.pipeline.classify import ContentType, ContentValue
from brainlayer.vector_store import VectorStore
from scripts.verify_source_class_visibility import _audit_hidden_class_default_visibility


def _claude_subagent(tmp_path: Path, repo: str, name: str = "agent.jsonl") -> Path:
    return tmp_path / ".claude" / "projects" / f"-Users-test-Gits-{repo}" / "session" / "subagents" / name


def test_five_value_source_class_taxonomy_and_null_for_ambiguous(tmp_path: Path) -> None:
    cases = [
        (tmp_path / ".codex" / "sessions" / "session.jsonl", "cli-agent"),
        (tmp_path / ".gemini" / "sessions" / "session.jsonl", "cli-agent"),
        (_claude_subagent(tmp_path, "domica"), "subagent"),
        (_claude_subagent(tmp_path, "brainlayer"), "fleet-coordination"),
        (_claude_subagent(tmp_path, "brainlayer", "brain-worker"), "brain-worker"),
        (tmp_path / "notes" / "unattributed.jsonl", None),
    ]

    for source_file, expected in cases:
        decision = classify_provenance(str(source_file))
        assert decision.source_class == expected


def test_generic_workflow_subagent_is_not_blanket_denied(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.delenv(BRAINLAYER_INGEST_DENYLIST_ENV, raising=False)
    workflow = _claude_subagent(tmp_path, "brainlayer", "workflows/wf_123/agent.jsonl")
    workflow.parent.mkdir(parents=True)
    workflow.write_text(
        json.dumps({"attributionAgent": "workflow-subagent"}) + "\n",
        encoding="utf-8",
    )

    assert not is_denylisted(workflow)
    assert classify_provenance(str(workflow)).source_class == "fleet-coordination"


def test_indexer_rejects_unknown_supplied_source_class_and_uses_provenance(monkeypatch, tmp_path: Path) -> None:
    source_file = tmp_path / ".codex" / "sessions" / "session.jsonl"
    chunk = Chunk(
        content="A normal Codex memory that must not accept a forged class label.",
        content_type=ContentType.ASSISTANT_TEXT,
        value=ContentValue.MEDIUM,
        metadata={"chunk_id": "classified", "source_class": "brain_worker"},
        char_count=61,
    )
    captured: dict[str, object] = {}

    class FakeStore:
        def upsert_chunks(self, chunks, embeddings, *, deadline_monotonic=None):
            captured["chunks"] = chunks
            return len(chunks)

    monkeypatch.setattr(
        "brainlayer.index_new.embed_chunks",
        lambda chunks, on_progress=None: [EmbeddedChunk(chunk=chunk, embedding=[0.1])],
    )

    assert index_chunks_to_sqlite([chunk], str(source_file), store=FakeStore()) == 1
    assert captured["chunks"][0]["source_class"] == "cli-agent"


def test_per_class_default_visibility_opt_in_and_exact_expansion(tmp_path: Path) -> None:
    store = VectorStore(tmp_path / "source-classes.db")
    rows = [
        ("cli", "cli-agent"),
        ("desktop", "desktop"),
        ("subagent", "subagent"),
        ("brain-worker", "brain-worker"),
        ("fleet", "fleet-coordination"),
        ("legacy", None),
    ]
    for position, (chunk_id, source_class) in enumerate(rows):
        store.conn.cursor().execute(
            """
            INSERT INTO chunks (
                id, content, metadata, source_file, project, content_type,
                char_count, conversation_id, position, source_class
            ) VALUES (?, ?, '{}', ?, 'brainlayer', 'user_message', ?, ?, ?, ?)
            """,
            (
                chunk_id,
                f"source visibility token {chunk_id}",
                f"/{chunk_id}.jsonl",
                len(chunk_id),
                f"conversation-{chunk_id}",
                position,
                source_class,
            ),
        )

    default_ids = set(store.search(query_text="source visibility token", n_results=20)["ids"][0])
    opt_in_ids = set(
        store.search(
            query_text="source visibility token",
            n_results=20,
            include_hidden_source_classes=True,
        )["ids"][0]
    )

    assert default_ids == {"cli", "subagent", "fleet", "legacy"}
    assert opt_in_ids == {"cli", "desktop", "subagent", "fleet", "legacy"}
    for chunk_id in ("cli", "desktop", "subagent", "brain-worker", "fleet", "legacy"):
        assert store.get_context(chunk_id)["target"]["id"] == chunk_id

    store.close()


def test_hidden_class_audit_fails_on_any_sampled_desktop_leak(tmp_path: Path, monkeypatch) -> None:
    store = VectorStore(tmp_path / "source-class-audit.db")
    store.conn.cursor().execute(
        """
        INSERT INTO chunks (
            id, content, metadata, source_file, project, content_type,
            char_count, conversation_id, position, source_class
        ) VALUES ('desktop-leak', 'DesktopAggregateLeakNeedle durable sample', '{}',
                  '/desktop.jsonl', 'brainlayer', 'user_message', 41, 'desktop-conversation', 0, 'desktop')
        """
    )
    real_search = store.search

    def leaking_search(*, query_text, **kwargs):
        result = real_search(query_text=query_text, **kwargs)
        if query_text == "DesktopAggregateLeakNeedle":
            result["ids"][0].append("desktop-leak")
        return result

    monkeypatch.setattr(store, "search", leaking_search)

    with pytest.raises(RuntimeError, match="desktop.*leak"):
        _audit_hidden_class_default_visibility(store, "desktop")
    store.close()
