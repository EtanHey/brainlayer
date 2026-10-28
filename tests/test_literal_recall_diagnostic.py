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
        for n in ast.parse(path.read_text(encoding="utf-8")).body
        if isinstance(n, ast.FunctionDef) and n.name == "_expanded_fts_query"
    )
    namespace = {
        "Any": object,
        "_escape_fts5_query": mod.python_builder(repo),
        "_lexical_defense_variants": lambda token: [token],
        "_kg_alias_variants": lambda token, store: [],
    }
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), namespace)
    query = "why did birch parser stall"
    assert namespace["_expanded_fts_query"](query, None) is None
    namespace["_kg_alias_variants"] = lambda token, store: ["cedar"] if token == "birch" else []
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


def audit_fixture(tmp_path):
    import hashlib
    import json

    audit = tmp_path / "audit"
    (audit / "corpus").mkdir(parents=True)
    doc = audit / "corpus/one.md"
    doc.write_bytes("cedar שלום".encode("utf-8"))
    q = {"id": "Q1", "question": "cedar", "relevant": ["G01"]}
    values = {
        "questions.json": [q],
        "groundtruth-freeze.json": {"questions": [q]},
        "corpus-manifest.json": [
            {
                "id": "G01",
                "corpus_file": "one.md",
                "sha256": hashlib.sha256(doc.read_bytes()).hexdigest(),
                "source": str(doc),
            }
        ],
        "brain-benchmark.json": [
            {"id": "Q1", "question": "cedar", "result": {"content": [{"type": "text", "text": "- ID: target"}]}}
            for _ in range(5)
        ],
        "brain-source-scoring.json": {
            "scores": [{"id": "Q1", "arm": "literal", "units": ["G01"], "hits": ["G01"], "recall_at_5": 1}]
        },
        "brain-guided.json": [],
        "brain-alias-verification.json": [],
        "structural-linked-benchmark.json": [{"id": "Q1", "arm": "rg", "recall_at_5": 1}],
    }
    for name, value in values.items():
        (audit / name).write_bytes(json.dumps(value, ensure_ascii=False).encode("utf-8"))
    (audit / "REPORT.md").write_bytes(b"Synthetic audit")
    return audit


def test_snapshot_refuses_metadata_changed_after_it_was_read(tmp_path, monkeypatch):
    mod = harness()
    audit = audit_fixture(tmp_path)
    target = audit / "questions.json"
    original_bytes, original_text = Path.read_bytes, Path.read_text
    mutated = False

    def change_after_read(path, value):
        nonlocal mutated
        if path == target and not mutated:
            mutated = True
            target.write_bytes(b'[{"id":"Q1","question":"changed","relevant":["G01"]}]')
        return value

    monkeypatch.setattr(Path, "read_bytes", lambda path: change_after_read(path, original_bytes(path)))
    monkeypatch.setattr(Path, "read_text", lambda path, *a, **k: change_after_read(path, original_text(path, *a, **k)))
    with pytest.raises(ValueError, match="audit inputs changed"):
        mod.snapshot(audit)


@pytest.mark.parametrize("stage", ["json", "builder", "corpus"])
def test_input_text_is_utf8_even_with_a_legacy_locale(tmp_path, monkeypatch, stage):
    mod = harness()
    audit = audit_fixture(tmp_path)
    original = Path.read_text
    monkeypatch.setattr(
        Path, "read_text", lambda p, encoding=None, **k: original(p, encoding=encoding or "cp1252", **k)
    )
    if stage == "json":
        path = tmp_path / "input.json"
        path.write_bytes('{"value":"שלום"}'.encode("utf-8"))
        assert mod.read(path) == {"value": "שלום"}
    elif stage == "builder":
        source = tmp_path / "src/brainlayer/_helpers.py"
        source.parent.mkdir(parents=True)
        source.write_bytes('def _escape_fts5_query(q): return q.replace("שלום", "hello")'.encode("utf-8"))
        assert mod.python_builder(tmp_path)("שלום") == "hello"
    else:
        index = mod.fts_index

        def inspect_rows(rows):
            assert rows[0]["content"] == "cedar שלום"
            return index(rows)

        monkeypatch.setattr(mod, "fts_index", inspect_rows)
        mod.analyze(audit, Path(__file__).parents[1], None)


def test_swift_mirror_keeps_the_native_token_joining_contract():
    mirror = harness().swift_mirror
    assert mirror("cedar parser stale") == '"cedar"* "parser"* "stale"*'
    assert mirror("a b c d") == '"a"* OR "b"* OR "c"* OR "d"*'
    assert mirror('say "hi" a*') == '"say"* "hi"* "a"*'
    assert mirror("   ") == '""'


@pytest.mark.parametrize("existing", [True, False])
def test_cli_preserves_prior_or_concurrently_created_receipt(tmp_path, monkeypatch, existing):
    import sys

    mod = harness()
    output = tmp_path / "receipt.json"
    if existing:
        output.write_bytes(b"prior evidence\n")

    def snapshot(_audit):
        assert not existing, "existing output must refuse before gathering inputs"
        return {"state": "new"}

    original_open = Path.open

    def concurrent_open(path, mode="r", *args, **kwargs):
        if path == output and mode in {"w", "x"} and not existing:
            with original_open(path, "x") as winner:
                winner.write("prior evidence\n")
        return original_open(path, mode, *args, **kwargs)

    monkeypatch.setattr(Path, "open", concurrent_open)
    monkeypatch.setattr(mod, "snapshot", snapshot)
    monkeypatch.setattr(sys, "argv", ["diagnose", "snapshot", "--audit", str(tmp_path), "--output", str(output)])
    with pytest.raises(SystemExit) as exc:
        mod.main()
    assert exc.value.code == 2
    assert output.read_bytes() == b"prior evidence\n"


@pytest.mark.parametrize("serializable", [True, False])
def test_cli_serializes_before_creating_a_new_receipt(tmp_path, monkeypatch, capsys, serializable):
    import hashlib
    import json
    import sys

    mod = harness()
    output = tmp_path / "receipt.json"
    monkeypatch.setattr(mod, "snapshot", lambda _: {"value": "synthetic" if serializable else object()})
    monkeypatch.setattr(sys, "argv", ["diagnose", "snapshot", "--audit", str(tmp_path), "--output", str(output)])
    if serializable:
        mod.main()
        assert json.loads(output.read_text())["value"] == "synthetic"
        assert json.loads(capsys.readouterr().out)["sha256"] == hashlib.sha256(output.read_bytes()).hexdigest()
    else:
        with pytest.raises(TypeError):
            mod.main()
        assert not output.exists()
