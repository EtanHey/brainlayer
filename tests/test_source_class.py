import json
from pathlib import Path

from brainlayer.agent_provenance import classify_provenance
from brainlayer.ingest_denylist import BRAINLAYER_INGEST_DENYLIST_ENV, is_denylisted
from brainlayer.vector_store import VectorStore


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
