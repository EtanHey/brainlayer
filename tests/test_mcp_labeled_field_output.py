"""Regression tests for MCP text output v2 labeled-field markdown."""

import asyncio

from brainlayer.mcp._format import (
    format_entity_simple,
    format_kg_search,
    format_recalled_context,
    format_search_results,
)


def test_brain_search_output_is_labeled_markdown_not_score_table():
    output = format_search_results(
        "auth refactor",
        [
            {
                "chunk_id": "manual-abc123",
                "score": 0.91,
                "source_file": "/tmp/ADR-014.md",
                "date": "2026-04-12T10:00:00Z",
                "summary": "Decision: use refresh tokens with 7-day rotation",
                "snippet": "Refresh-token rotation keeps sessions recoverable while limiting replay risk.",
            }
        ],
        47,
    )

    assert output == (
        '## Search results for "auth refactor" - 1 of 47 shown\n\n'
        "### 1. Decision: use refresh tokens with 7-day rotation\n"
        "- Source: ADR-014.md\n"
        "- Date: 2026-04-12\n"
        "- Preview: Refresh-token rotation keeps sessions recoverable while limiting replay risk."
    )
    assert "score" not in output
    assert "manual-abc123" not in output
    assert not output.strip().startswith("{")


def test_brain_search_source_basename_handles_windows_paths():
    output = format_search_results(
        "path privacy",
        [
            {
                "source_file": r"C:\Users\etan\brainlayer\src\auth.py",
                "date": "2026-04-12T10:00:00Z",
                "snippet": "Windows paths should not leak full absolute paths.",
            }
        ],
        1,
    )

    assert "- Source: auth.py" in output
    assert r"C:\Users" not in output


def test_kg_augmented_brain_search_output_is_labeled_markdown_not_score_table():
    output = format_kg_search(
        "BrainLayer",
        [
            {
                "chunk_id": "kg-abc123",
                "score": 0.93,
                "source_file": "/repo/docs/mcp.md",
                "date": "2026-05-24T11:45:00Z",
                "snippet": "BrainLayer serves readable MCP output.",
            }
        ],
        [{"source": "BrainLayer", "relation": "uses", "target": "MCP"}],
        "brainlayer mcp",
    )

    assert output.startswith('## Search results for "brainlayer mcp" - 1 of 1 shown')
    assert "### KG Facts for BrainLayer" in output
    assert "- BrainLayer uses MCP" in output
    assert "### 1. BrainLayer serves readable MCP output." in output
    assert "- Source: mcp.md" in output
    assert "- Date: 2026-05-24" in output
    assert "score" not in output
    assert "kg-abc123" not in output
    assert "┌" not in output


def test_brain_entity_output_has_kg_facts_and_expired_annotation():
    output = format_entity_simple(
        {
            "name": "JWT middleware",
            "id": "entity-jwt",
            "entity_type": "technology",
            "description": "Authentication middleware used by API services.",
            "relations": [
                {"relation_type": "used_by", "target_name": "api-gateway"},
                {"relation_type": "used_by", "target_name": "admin-service"},
                {
                    "relation_type": "depends_on",
                    "target_name": "legacy-auth-lib",
                    "expired_at": "2026-05-01T00:00:00Z",
                },
            ],
            "chunks": [{"content": "Mentioned in the auth refactor plan as a dependency to remove."}],
        }
    )

    assert "## Entity: JWT middleware" in output
    assert "### KG Facts" in output
    assert "- used_by: api-gateway" in output
    assert "- depends_on: legacy-auth-lib (expired 2026-05-01)" in output
    assert "### Recent context" in output
    assert "### Likely follow-ups" in output
    assert "score" not in output


def test_brain_entity_output_does_not_mark_future_valid_until_expired():
    output = format_entity_simple(
        {
            "name": "JWT middleware",
            "relations": [
                {
                    "relation_type": "depends_on",
                    "target_name": "auth-service",
                    "valid_until": "2026-12-31T00:00:00Z",
                }
            ],
        }
    )

    assert "- depends_on: auth-service" in output
    assert "expired 2026-12-31" not in output


def test_brain_entity_output_includes_provenance_authority_annotations():
    output = format_entity_simple(
        {
            "name": "enrichment",
            "provenance_resolutions": {
                "PRIMARY_BACKEND": {
                    "authoritative": {
                        "value": "Gemini",
                        "provenance_class": "OPERATIONAL-EVIDENCE",
                        "evidence": "220K chunk_origin rows",
                    },
                    "superseded": [
                        {
                            "value": "Groq",
                            "provenance_class": "AGENT-PARAPHRASE",
                            "chunk_id": "c-groq",
                        }
                    ],
                }
            },
        }
    )

    assert "### Provenance Authority" in output
    assert "PRIMARY_BACKEND: Gemini [AUTHORITATIVE · OPERATIONAL-EVIDENCE · 220K chunk_origin rows]" in output
    assert 'superseded: "Groq" (AGENT-PARAPHRASE, c-groq) — reversible' in output


def test_brain_recall_context_output_is_labeled_chunk_markdown():
    output = format_recalled_context(
        "how did we handle session expiry",
        [
            {
                "chunk_id": "chunk-auth-1",
                "source_file": "design-doc/auth-v2.md",
                "content": "We chose sliding-window refresh tokens with a 60-second grace period.",
                "date": "2026-04-12T10:00:00Z",
            }
        ],
    )

    assert output == (
        '## Recalled context for "how did we handle session expiry"\n\n'
        "### Chunk 1 - auth-v2.md\n"
        "We chose sliding-window refresh tokens with a 60-second grace period."
    )
    assert "score" not in output
    assert "chunk-auth-1" not in output
    assert "[{" not in output


def test_brain_recall_context_preserves_position_target_and_type():
    output = format_recalled_context(
        "chunk-auth-1",
        [
            {
                "chunk_id": "chunk-auth-1",
                "source_file": "session.jsonl",
                "content_type": "assistant_text",
                "position": 42,
                "is_target": True,
                "content": "This is the target chunk.",
            }
        ],
    )

    assert "### Chunk 1 - session.jsonl" in output
    assert "- Position: 42" in output
    assert "- Type: assistant_text" in output
    assert "- Target: yes" in output


def test_brain_recall_context_preserves_full_target_content():
    body = "BEGIN " + ("full body detail " * 120) + "END-OF-FULL-CONTENT"

    output = format_recalled_context(
        "chunk-long-body",
        [
            {
                "chunk_id": "chunk-long-body",
                "source_file": "session.jsonl",
                "is_target": True,
                "content": body,
            }
        ],
    )

    assert len(body) > 1500
    assert body in output
    assert "END-OF-FULL-CONTENT" in output


def test_brain_recall_context_compacts_non_target_neighbors():
    body = "NEIGHBOR " + ("surrounding detail " * 120) + "END-OF-NEIGHBOR-CONTENT"

    output = format_recalled_context(
        "chunk-long-body",
        [
            {
                "chunk_id": "chunk-neighbor",
                "source_file": "session.jsonl",
                "is_target": False,
                "content": body,
            },
            {
                "chunk_id": "chunk-long-body",
                "source_file": "session.jsonl",
                "is_target": True,
                "content": "TARGET " + ("full target detail " * 120) + "END-OF-FULL-CONTENT",
            },
        ],
    )

    assert len(body) > 1500
    assert "END-OF-NEIGHBOR-CONTENT" not in output
    assert "END-OF-FULL-CONTENT" in output


def test_brain_recall_context_uses_project_when_source_file_missing():
    output = format_recalled_context(
        "session context",
        [
            {
                "chunk_id": "chunk-session-1",
                "project": "brainlayer",
                "content": "Session recall row did not carry source_file.",
            }
        ],
    )

    assert "### Chunk 1 - brainlayer" in output
    assert "unknown" not in output


def test_brain_recall_tool_declares_anthropic_max_result_size():
    from brainlayer.mcp import list_tools

    tools = asyncio.run(list_tools())
    recall = next(tool for tool in tools if tool.name == "brain_recall")
    annotation_dump = recall.annotations.model_dump(by_alias=True)

    assert annotation_dump["anthropic/maxResultSizeChars"] >= 100_000
