"""Synthetic behavior checks for the offline diagnostic, never a live search test."""

import importlib.util
import sqlite3
from pathlib import Path

import pytest


def harness():
    path = Path(__file__).parents[1] / "scripts/diagnose_literal_recall.py"
    spec = importlib.util.spec_from_file_location("recall_diagnostic", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_uses_real_python_query_builder_without_importing_model():
    mod = harness()
    escape = mod.python_builder(Path(__file__).parents[1])
    assert escape("cedar parser") == '"cedar" "parser"'
    assert escape("why did cedar parser stall") == '"why" OR "did" OR "cedar" OR "parser" OR "stall"'


def test_fts_ablation_separates_candidate_loss_from_top_five_loss():
    mod = harness()
    db = mod.fts_index([{"id": "target", "content": "cedar parser stale"}])
    assert mod.ranked(db, '"cedar" AND "stall"') == []
    assert mod.ranked(db, '"cedar" OR "stall"')[0] == "target"
    db.close()


def test_synonym_is_not_a_literal_match():
    mod = harness()
    db = mod.fts_index([{"id": "target", "content": "cedar stale"}])
    assert mod.ranked(db, '"stall"') == []
    assert mod.ranked(db, '"stal"*') == ["target"]
    db.close()


def test_readonly_connection_refuses_write(tmp_path):
    mod = harness()
    path = tmp_path / "fixture.db"
    with sqlite3.connect(path) as db:
        db.execute("CREATE TABLE fixture (n)")
    db = mod.readonly(path)
    with pytest.raises(sqlite3.OperationalError, match="readonly"):
        db.execute("INSERT INTO fixture VALUES (1)")
    db.close()


def test_long_query_quotes_syntax_and_fails_closed_on_unknown_builder(tmp_path):
    mod = harness()
    escape = mod.python_builder(Path(__file__).parents[1])
    db = mod.fts_index([{"id": "target", "content": "cedar parser"}])
    assert mod.ranked(db, escape('cedar OR parser "quoted"')) == ["target"]
    with pytest.raises(FileNotFoundError):
        mod.python_builder(tmp_path)
    db.close()


def test_alias_expansion_can_reintroduce_conjunction():
    """A conditional override is distinct from the default OR builder."""
    import ast

    mod = harness()
    repo = Path(__file__).parents[1]
    path = repo / "src/brainlayer/mcp/search_handler.py"
    node = next(
        n
        for n in ast.parse(path.read_text()).body
        if isinstance(n, ast.FunctionDef) and n.name == "_expanded_fts_query"
    )
    namespace = {
        "Any": object,
        "_escape_fts5_query": mod.python_builder(repo),
        "_lexical_defense_variants": lambda token: [token, "cedar"] if token == "birch" else [token],
        "_kg_alias_variants": lambda token, store: [],
    }
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), namespace)
    query = "why did birch parser stall"
    override = namespace["_expanded_fts_query"](query, None)
    db = mod.fts_index([{"id": "target", "content": "cedar parser stale"}])
    assert mod.ranked(db, mod.python_builder(repo)(query)) == ["target"]
    assert " AND " in override and mod.ranked(db, override) == []
    db.close()


def test_offline_arms_preserve_candidate_ids(tmp_path):
    import json

    mod = harness()
    (tmp_path / "corpus").mkdir()
    (tmp_path / "corpus/fixture.md").write_text("cedar parser stale")
    (tmp_path / "corpus-manifest.json").write_text(json.dumps([{"id": "G01", "corpus_file": "fixture.md"}]))
    (tmp_path / "questions.json").write_text(json.dumps([{"id": "Q1", "question": "why did cedar parser stall"}]))
    result = mod.analyze(tmp_path, Path(__file__).parents[1], None)
    arms = result["cohorts"]["curated_documents"]["queries"][0]["arms"]
    assert arms["python_actual"]["top5"] == ["G01"]
    assert arms["forced_and_counterfactual"]["candidates"] == []
    assert all(arm["error"] is None for arm in arms.values())
