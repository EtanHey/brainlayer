from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest


def _insert_chunk(
    conn: sqlite3.Connection,
    chunk_id: str,
    content: str,
    *,
    project: str = "brainlayer",
    created_at: str = "2026-06-05T10:00:00Z",
    summary: str = "",
) -> None:
    conn.execute(
        """INSERT INTO chunks
           (id, content, summary, tags, resolved_query, key_facts, resolved_queries, created_at, project)
           VALUES (?, ?, ?, '', '', '', '', ?, ?)""",
        (chunk_id, content, summary, created_at, project),
    )
    conn.execute(
        """INSERT INTO chunks_fts
           (content, summary, tags, resolved_query, key_facts, resolved_queries, chunk_id)
           VALUES (?, ?, '', '', '', '', ?)""",
        (content, summary, chunk_id),
    )


def _create_judge_fixture_db(path: Path) -> None:
    conn = sqlite3.connect(path)
    conn.executescript(
        """
        CREATE TABLE kg_entities (
            id TEXT PRIMARY KEY,
            entity_type TEXT NOT NULL,
            name TEXT NOT NULL,
            status TEXT DEFAULT 'active'
        );
        CREATE TABLE chunks (
            id TEXT PRIMARY KEY,
            content TEXT NOT NULL,
            summary TEXT,
            tags TEXT,
            resolved_query TEXT,
            key_facts TEXT,
            resolved_queries TEXT,
            created_at TEXT,
            project TEXT
        );
        CREATE VIRTUAL TABLE chunks_fts USING fts5(
            content, summary, tags, resolved_query, key_facts, resolved_queries, chunk_id UNINDEXED
        );
        CREATE TABLE kg_entity_chunks (
            entity_id TEXT NOT NULL,
            chunk_id TEXT NOT NULL,
            relevance REAL DEFAULT 1.0,
            context TEXT,
            PRIMARY KEY (entity_id, chunk_id)
        );
        CREATE TABLE kg_relations (
            id TEXT PRIMARY KEY,
            source_id TEXT NOT NULL,
            target_id TEXT NOT NULL,
            relation_type TEXT NOT NULL,
            fact TEXT,
            properties TEXT DEFAULT '{}',
            confidence REAL DEFAULT 1.0,
            expired_at TEXT,
            source_chunk_id TEXT
        );
        """
    )
    entities = [
        ("person-etan", "person", "Etan Heyman"),
        ("easysend-org", "organization", "EasySend"),
        ("easysend-project", "project", "EasySend"),
        ("cantaloupe-org", "organization", "Cantaloupe"),
        ("cantaloupe-person", "person", "Cantaloupe"),
        ("claude-web-tool", "tool", "Claude Web"),
        ("claude-web-project", "project", "Claude-Web"),
    ]
    conn.executemany("INSERT INTO kg_entities (id, entity_type, name) VALUES (?, ?, ?)", entities)

    _insert_chunk(
        conn,
        "chunk-easysend-1",
        "Etan used EasySend as a vendor company/product in delivery work; repo notes are usage records, not ownership.",
        project="easysend-notes",
        summary="EasySend is a company/product Etan used, not his owned project.",
    )
    _insert_chunk(
        conn,
        "chunk-easysend-2",
        "EasySend implementation notes describe customer onboarding and vendor integration constraints.",
        project="client-work",
    )
    _insert_chunk(
        conn,
        "chunk-cantaloupe-1",
        "Cantaloupe was the now-dead/renamed company where Etan worked before later cleanup tasks.",
        project="career",
        summary="Work history says Cantaloupe was an organization, not a person.",
    )
    _insert_chunk(
        conn,
        "chunk-claude-web-1",
        "Claude Web is the claude.ai app/container Etan uses; it contains projects but is not itself his project.",
        project="golems",
    )
    _insert_chunk(
        conn,
        "chunk-claude-web-2",
        "Claude Web research sessions contain project work inside the app container.",
        project="research",
    )
    conn.executemany(
        "INSERT INTO kg_entity_chunks (entity_id, chunk_id, relevance, context) VALUES (?, ?, ?, ?)",
        [
            ("easysend-org", "chunk-easysend-1", 0.98, "company/product usage"),
            ("easysend-project", "chunk-easysend-2", 0.40, "repo presence only"),
            ("cantaloupe-org", "chunk-cantaloupe-1", 0.99, "employment context"),
            ("cantaloupe-person", "chunk-cantaloupe-1", 0.15, "bad person duplicate"),
            ("claude-web-tool", "chunk-claude-web-1", 0.99, "tool container"),
            ("claude-web-project", "chunk-claude-web-2", 0.40, "contains project work"),
        ],
    )
    conn.executemany(
        """INSERT INTO kg_relations
           (id, source_id, target_id, relation_type, fact, confidence, source_chunk_id)
           VALUES (?, ?, ?, ?, ?, ?, ?)""",
        [
            (
                "rel-easysend-uses",
                "person-etan",
                "easysend-org",
                "uses",
                "Etan used EasySend; usage does not imply ownership.",
                0.95,
                "chunk-easysend-1",
            ),
            (
                "rel-cantaloupe-worked",
                "person-etan",
                "cantaloupe-org",
                "worked_at",
                "Etan worked at Cantaloupe during the company period.",
                0.96,
                "chunk-cantaloupe-1",
            ),
            (
                "rel-claude-web-uses",
                "person-etan",
                "claude-web-tool",
                "uses",
                "Etan uses Claude Web as the claude.ai product/app.",
                0.93,
                "chunk-claude-web-1",
            ),
        ],
    )
    conn.commit()
    conn.close()


def _create_repo(root: Path, name: str, readme: str, commit_messages: list[str]) -> None:
    repo = root / name
    repo.mkdir(parents=True)
    (repo / "README.md").write_text(readme, encoding="utf-8")
    for index, message in enumerate(commit_messages):
        (repo / f"note-{index}.txt").write_text(message, encoding="utf-8")


def _cluster(stem: str, members: list[dict]) -> dict:
    return {"stem": stem, "category": "diagnosis-flag", "members": members}


@pytest.fixture
def judge_fixture(tmp_path: Path) -> dict[str, Path]:
    db_path = tmp_path / "brainlayer.db"
    gits_root = tmp_path / "Gits"
    _create_judge_fixture_db(db_path)
    _create_repo(
        gits_root,
        "EasySend",
        "# EasySend\n\nVendor integration notes.\nEtan used this product for client delivery.\n",
        ["vendor import", "capture usage notes"],
    )
    _create_repo(
        gits_root,
        "claude-web-research",
        "# Claude Web Research\n\nResearch sessions inside claude.ai, not an owned app repo.\n",
        ["record claude.ai container behavior"],
    )
    return {"db_path": db_path, "gits_root": gits_root}


def test_gathers_decisive_evidence_for_required_fixture_clusters(judge_fixture: dict[str, Path]):
    from brainlayer.kg_judge import gather_evidence_for_cluster

    clusters = {
        "easysend": _cluster(
            "EasySend",
            [
                {"id": "easysend-org", "name": "EasySend", "type": "organization", "chunks": 1},
                {"id": "easysend-project", "name": "EasySend", "type": "project", "chunks": 1},
            ],
        ),
        "cantaloupe": _cluster(
            "Cantaloupe",
            [
                {"id": "cantaloupe-org", "name": "Cantaloupe", "type": "organization", "chunks": 1},
                {"id": "cantaloupe-person", "name": "Cantaloupe", "type": "person", "chunks": 1},
            ],
        ),
        "claude_web": _cluster(
            "Claude Web",
            [
                {"id": "claude-web-tool", "name": "Claude Web", "type": "tool", "chunks": 1},
                {"id": "claude-web-project", "name": "Claude-Web", "type": "project", "chunks": 1},
            ],
        ),
    }

    packets = {
        key: gather_evidence_for_cluster(
            cluster, db_path=judge_fixture["db_path"], gits_root=judge_fixture["gits_root"]
        )
        for key, cluster in clusters.items()
    }

    easysend_text = packets["easysend"].to_prompt_text()
    assert "Etan used EasySend" in easysend_text
    assert "usage does not imply ownership" in easysend_text
    assert "README.md first lines" in easysend_text

    cantaloupe_text = packets["cantaloupe"].to_prompt_text()
    assert "worked_at" in cantaloupe_text
    assert "Etan worked at Cantaloupe" in cantaloupe_text
    assert "bad person duplicate" in cantaloupe_text

    claude_web_text = packets["claude_web"].to_prompt_text()
    assert "claude.ai app/container" in claude_web_text
    assert "contains projects but is not itself his project" in claude_web_text
    assert "rel-claude-web-uses" in claude_web_text


def test_builds_worker_prompt_with_closed_enum_schema_and_self_gathering_instructions(
    judge_fixture: dict[str, Path],
):
    from brainlayer.kg_judge import build_judge_prompt, gather_evidence_for_cluster

    packet = gather_evidence_for_cluster(
        _cluster(
            "Cantaloupe",
            [
                {"id": "cantaloupe-org", "name": "Cantaloupe", "type": "organization", "chunks": 1},
                {"id": "cantaloupe-person", "name": "Cantaloupe", "type": "person", "chunks": 1},
            ],
        ),
        db_path=judge_fixture["db_path"],
        gits_root=judge_fixture["gits_root"],
    )

    prompt = build_judge_prompt(packet, worker_mode=True)

    assert 'Person ("could I message/meet this individual?", never auto-merge)' in prompt
    assert "D1 Concept→tag" in prompt
    assert "D2 Transient→drop" in prompt
    assert '"proposed_type"' in prompt
    assert '"evidence_cited"' in prompt
    assert '"evidence_degraded"' in prompt
    assert "Run brain_search on the stem" in prompt
    assert "retry once" in prompt
    assert "grep ~/Gits" in prompt
    assert "worked_at" in prompt


def test_mocked_llm_judge_retries_schema_violation_and_requires_cited_evidence(
    judge_fixture: dict[str, Path],
):
    from brainlayer.kg_judge import gather_evidence_for_cluster, judge_cluster_with_llm

    packet = gather_evidence_for_cluster(
        _cluster(
            "EasySend",
            [
                {"id": "easysend-org", "name": "EasySend", "type": "organization", "chunks": 1},
                {"id": "easysend-project", "name": "EasySend", "type": "project", "chunks": 1},
            ],
        ),
        db_path=judge_fixture["db_path"],
        gits_root=judge_fixture["gits_root"],
    )
    first_ref = packet.evidence_refs()[0]
    calls: list[str] = []

    def fake_llm(prompt: str) -> str:
        calls.append(prompt)
        if len(calls) == 1:
            return json.dumps({"stem": "EasySend", "proposed_type": "Project", "evidence_cited": []})
        return json.dumps(
            {
                "stem": "EasySend",
                "proposed_type": "Organization",
                "identity": "EasySend is a company/product Etan used.",
                "merge_disposition": "merge",
                "canonical_suggestion": "EasySend",
                "confidence": "high",
                "evidence_cited": [first_ref],
                "reasoning": "The packet cites vendor usage and no ownership evidence.",
                "evidence_degraded": False,
            }
        )

    verdict = judge_cluster_with_llm(packet, llm=fake_llm, max_attempts=2)

    assert verdict["proposed_type"] == "Organization"
    assert len(calls) == 2
    assert "No verdict without evidence" in calls[0]
    assert "Organization (group acting as one agent" in calls[0]
    assert first_ref in verdict["evidence_cited"]


def test_emit_prompt_files_write_one_self_contained_prompt_per_cluster(
    judge_fixture: dict[str, Path],
    tmp_path: Path,
):
    from brainlayer.kg_judge import emit_prompt_files

    out_dir = tmp_path / "prompts"
    written = emit_prompt_files(
        [
            _cluster(
                "Claude Web",
                [
                    {"id": "claude-web-tool", "name": "Claude Web", "type": "tool", "chunks": 1},
                    {"id": "claude-web-project", "name": "Claude-Web", "type": "project", "chunks": 1},
                ],
            )
        ],
        out_dir,
        db_path=judge_fixture["db_path"],
        gits_root=judge_fixture["gits_root"],
    )

    assert len(written) == 1
    prompt_text = written[0].read_text(encoding="utf-8")
    assert "Claude Web" in prompt_text
    assert "claude.ai product/app" in prompt_text
    assert "Run brain_search on the stem" in prompt_text
    assert "evidence_degraded:true" in prompt_text
    assert "Required verdict JSON schema" in prompt_text


def test_collect_validates_worker_jsons_and_writes_json_plus_markdown(tmp_path: Path):
    from brainlayer.kg_judge import collect_worker_verdicts

    verdict_dir = tmp_path / "worker-verdicts"
    verdict_dir.mkdir()
    (verdict_dir / "001-easysend.json").write_text(
        json.dumps(
            {
                "stem": "EasySend",
                "proposed_type": "Organization",
                "identity": "EasySend is a company/product Etan used.",
                "merge_disposition": "merge",
                "canonical_suggestion": "EasySend",
                "confidence": "high",
                "evidence_cited": ["linked-1"],
                "reasoning": "Usage evidence and no ownership evidence support Organization.",
                "evidence_degraded": False,
            }
        ),
        encoding="utf-8",
    )
    out_json = tmp_path / "merged.json"
    out_md = tmp_path / "merged.md"

    verdicts = collect_worker_verdicts(verdict_dir, out_json=out_json, markdown_path=out_md)

    assert verdicts[0]["stem"] == "EasySend"
    assert json.loads(out_json.read_text(encoding="utf-8"))["verdicts"][0]["proposed_type"] == "Organization"
    markdown = out_md.read_text(encoding="utf-8")
    assert "| stem | proposed type | identity | disposition | confidence | top evidence |" in markdown
    assert (
        "| EasySend | Organization | EasySend is a company/product Etan used. | merge | high | linked-1 |" in markdown
    )


def test_collect_rejects_invalid_worker_json(tmp_path: Path):
    from brainlayer.kg_judge import JudgeSchemaError, collect_worker_verdicts

    verdict_dir = tmp_path / "worker-verdicts"
    verdict_dir.mkdir()
    (verdict_dir / "bad.json").write_text(
        json.dumps(
            {
                "stem": "Cantaloupe",
                "proposed_type": "Person",
                "identity": "Invalid because no cited evidence.",
                "merge_disposition": "merge",
                "canonical_suggestion": "Cantaloupe",
                "confidence": "high",
                "evidence_cited": [],
                "reasoning": "Missing evidence.",
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(JudgeSchemaError, match="evidence_cited"):
        collect_worker_verdicts(verdict_dir)
