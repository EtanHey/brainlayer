"""Agent/reference docs must retire production while retaining read contracts."""

import json
import re
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_agent_instructions_do_not_run_or_resume_enrichment():
    text = (ROOT / "AGENTS.md").read_text()
    assert not re.search(r"brainlayer\s+enrich\b", text)
    assert "## Enrichment (retired)" in text
    assert "enrichment_controller.enrich_realtime uses Gemini" not in text
    for retained in ("GOOGLE_API_KEY", "BRAINLAYER_REQUIRE_GOOGLE_API_KEY", "exit-78", "re-renders"):
        assert retained in text
    assert "docs/enrichment.md" in text


def test_architecture_retains_historical_rows_without_live_model_nodes():
    text = (ROOT / "docs/architecture.md").read_text()
    assert "Chunk Enrichment<br/>" not in text
    assert "Session Enrichment<br/>" not in text
    assert "session_enrichments" in text
    assert "historical" in text.lower()
    assert "Brain Graph" in text
    assert "enrichment.md" in text


def test_writer_reference_has_no_enrichment_producer_advertisement():
    text = (ROOT / "docs/arbitration.md").read_text()
    assert "templates set this for watch and enrichment" not in text
    assert "retired" in text.lower()
    assert "brainlayer flush" in text
    assert "brainlayer repair-fts" in text


def test_metadata_and_documented_tool_tables_match_native_inventory():
    source = (ROOT / "brain-bar/Sources/BrainBar/MCPRouter.swift").read_text()
    definitions = source.split("static let toolDefinitions:", 1)[1]
    names = set(re.findall(r'^\s*"name": "(brain_[^"]+)"', definitions, re.M))
    assert "brain_enrich" not in names
    assert {"brain_search", "brain_store", "brain_digest", "brain_entity"} <= names
    for relative in ("README.md", "docs/mcp-tools.md"):
        text = (ROOT / relative).read_text()
        documented = set(re.findall(r"^\| `(brain_\w+)` \|", text, re.M))
        assert documented == names, (relative, documented ^ names)
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]
    server = json.loads((ROOT / "server.json").read_text())
    for description in (project["description"], server["description"]):
        assert f"{len(names)} BrainBar MCP tools" in description
