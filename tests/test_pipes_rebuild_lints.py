"""Source ratchets for the pipes rebuild's audited invalidation and oracle boundaries."""

import ast
import builtins
import io
import os
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
# Replacement and lifecycle deletion are intentional; content invalidation is guarded below.
VECTOR_SITES = {
    ("vector_store.py", "_upsert_chunk_vector"): 2,
    ("vector_store.py", "_delete_chunk_vector"): 2,
    ("vector_store.py", "upsert_chunks"): 1,
    ("vector_store.py", "archive_chunk"): 1,
    ("vector_store.py", "supersede_chunk"): 1,
    ("drain.py", "_insert_precomputed_embedding"): 2,
    ("dedupe.py", "_invalidate_merged_vectors"): 1,
    ("dedupe.py", "merge_duplicate_chunk"): 2,
    ("dedupe.py", "merge_existing_chunk_content"): 1,
    ("source_class_migration.py", "_remove_brain_worker_fts_rows"): 1,
    ("source_class_migration.py", "_remove_brain_worker_vector_rows"): 1,
    ("doctor.py", "_cleanup_probe"): 1,
    ("index_completeness.py", "_delete_index_row"): 1,
    ("index_completeness.py", "_undo_one_run"): 2,
}


def functions(source):
    return {n.name: n for n in ast.walk(ast.parse(source)) if isinstance(n, ast.FunctionDef)}


def vector_sites(source):
    tree = ast.parse(source)
    parents = {child: parent for parent in ast.walk(tree) for child in ast.iter_child_nodes(parent)}
    sites = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        name = getattr(node.func, "attr", getattr(node.func, "id", ""))
        sql = ast.unparse(node.args[0]).upper() if node.args else ""
        if name not in ("_delete_chunk_vector", "_invalidate_merged_vectors") and not (
            name == "execute" and "DELETE FROM" in sql and ("CHUNK_VECTORS" in sql or "{TABLE}" in sql)
        ):
            continue
        parent = node
        while parent in parents and not isinstance(parent, ast.FunctionDef):
            parent = parents[parent]
        sites.append((parent.name if isinstance(parent, ast.FunctionDef) else "<module>", node))
    return sites


def check_vector_sources(sources):
    inventory = {}
    for path, source in sources.items():
        for function, _ in vector_sites(source):
            key = (path, function)
            inventory[key] = inventory.get(key, 0) + 1
    assert inventory == VECTOR_SITES, "new vector deletion site requires an audited guard"
    for path, function in [("dedupe.py", "merge_duplicate_chunk"), ("dedupe.py", "merge_existing_chunk_content")]:
        fn = functions(sources[path])[function]
        expected = "embedding_changed = content_changed and embedding_input(str(existing_content)) != embedding_input(merged_content)"
        assert any(isinstance(n, ast.Assign) and ast.unparse(n) == expected for n in ast.walk(fn))
        calls = [
            n
            for n in ast.walk(fn)
            if isinstance(n, ast.Call) and getattr(n.func, "id", "") == "_invalidate_merged_vectors"
        ]
        guarded = [
            n
            for guard in ast.walk(fn)
            if isinstance(guard, ast.If) and ast.unparse(guard.test) == "embedding_changed"
            for stmt in guard.body
            for n in ast.walk(stmt)
        ]
        assert all(call in guarded for call in calls), "merge invalidation must compare embedding inputs"
    fn = functions(sources["vector_store.py"])["upsert_chunks"]
    assert any(
        isinstance(n, ast.Assign)
        and ast.unparse(n)
        == (
            "same_embedding_input = previous is not None and embedding_input(previous[0]) == embedding_input(prepared['content'])"
        )
        for n in ast.walk(fn)
    )
    guarded = [
        n
        for guard in ast.walk(fn)
        if isinstance(guard, ast.If) and ast.unparse(guard.test) == "same_embedding_input"
        for stmt in guard.orelse
        for n in ast.walk(stmt)
    ]
    calls = [
        n for n in ast.walk(fn) if isinstance(n, ast.Call) and getattr(n.func, "attr", "") == "_delete_chunk_vector"
    ]
    assert calls and all(call in guarded for call in calls)


def check_identity(source):
    # Exact-hash dedupe and same-id repost admission retain their existing normalization.
    # Original-record membership and merge assembly must never use that lossy normalization.
    allowed = {"normalize_for_exact_hash", "normalized_exact_hash", "merge_existing_chunk_seen"}
    for name, fn in functions(source).items():
        if name not in allowed:
            assert not any(isinstance(n, ast.Name) and n.id == "normalize_for_exact_hash" for n in ast.walk(fn)), name
    assert ast.unparse(functions(source)["_whitespace_normalized"].body[-1]) == "return ' '.join(content.split())"


def check_oracle(source):
    # A stdlib-only import allowlist also blocks indirect production re-exports.
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            modules = [a.name for a in node.names] if isinstance(node, ast.Import) else [node.module]
            assert all(m in {"datetime", "json", "sys", "pathlib", "collections"} for m in modules), modules
        if isinstance(node, ast.Call):
            assert getattr(node.func, "id", "") not in {"__import__", "eval", "exec"}


def test_vector_deletions_are_allowlisted_and_content_guarded():
    check_vector_sources(
        {
            p.relative_to(ROOT / "src/brainlayer").as_posix(): p.read_text()
            for p in (ROOT / "src/brainlayer").rglob("*.py")
        }
    )


def test_original_identity_never_uses_lossy_exact_hash():
    check_identity((ROOT / "src/brainlayer/dedupe.py").read_text())


@pytest.mark.parametrize("path", ["tests/verdict_replay_oracle.py", "tests/fixtures/verdict_replay/build_oracle.py"])
def test_verdict_oracle_imports_are_independent(path):
    check_oracle((ROOT / path).read_text())


@pytest.mark.parametrize("rule", ["vector", "guard", "identity", "oracle", "reexport"])
def test_planted_violations_are_rejected(rule):
    sources = {
        p.relative_to(ROOT / "src/brainlayer").as_posix(): p.read_text()
        for p in (ROOT / "src/brainlayer").rglob("*.py")
    }
    with pytest.raises(AssertionError):
        if rule == "vector":
            sources["planted.py"] = 'def bad(cursor):\n cursor.execute("DELETE FROM chunk_vectors")'
            check_vector_sources(sources)
        elif rule == "guard":
            sources["dedupe.py"] = sources["dedupe.py"].replace("if embedding_changed:", "if True:")
            check_vector_sources(sources)
        elif rule == "identity":
            check_identity(sources["dedupe.py"] + "\ndef bad_identity(text):\n return normalize_for_exact_hash(text)\n")
        else:
            check_oracle("from brainlayer import " + ("health_verdict" if rule == "oracle" else "health_check"))


# Capture before the isolation fixture rewrites HOME and module-level Paths.
PROTECTED = str(Path.home() / ".local/share/brainlayer")


@pytest.mark.parametrize("opener", ["builtin", "io", "path", "os"])
@pytest.mark.parametrize("relative", ["job-alerts.json", "logs/maintenance.log"])
def test_production_runtime_write_is_refused(opener, relative):
    path = Path(PROTECTED) / relative
    with pytest.raises(RuntimeError, match="production BrainLayer runtime write refused"):
        if opener == "os":
            os.open(path, os.O_WRONLY | os.O_CREAT)
        else:
            {"builtin": builtins.open, "io": io.open, "path": lambda p, mode: p.open(mode)}[opener](path, "w")


def test_job_alert_report_hits_write_guard():
    from brainlayer.job_alerts import report

    with pytest.raises(RuntimeError, match="production BrainLayer runtime write refused"):
        report("planted", "must not write", path=Path(PROTECTED) / "job-alerts.json")


def test_maintenance_main_hits_write_guard(monkeypatch):
    from brainlayer import maintenance

    monkeypatch.setenv("BRAINLAYER_MAINTENANCE_LOG_PATH", str(Path(PROTECTED) / "logs/maintenance.log"))

    def abort(*args, **kwargs):
        raise maintenance.MaintenanceAbort("planted refusal", code=1)

    monkeypatch.setattr(maintenance, "run_maintenance", abort)
    with pytest.raises(RuntimeError, match="production BrainLayer runtime write refused"):
        maintenance.main(["--light"])


def test_health_check_hits_write_guard():
    from brainlayer.health_check import HealthCheckConfig, run_health_check

    with pytest.raises(RuntimeError, match="production BrainLayer runtime write refused"):
        run_health_check(HealthCheckConfig(state_path=Path(PROTECTED) / "logs/health-check-state.json"))


def test_symlinked_protected_root_stays_protected(tmp_path, monkeypatch):
    """CodeRabbit #1603: a root that is itself a symlink must not let resolved writes through."""
    import tests.conftest as guard

    real_root = tmp_path / "real-runtime"
    real_root.mkdir()
    link_root = tmp_path / "home" / ".local" / "share" / "brainlayer"
    link_root.parent.mkdir(parents=True)
    link_root.symlink_to(real_root, target_is_directory=True)
    monkeypatch.setattr(guard, "_PROTECTED_BRAINLAYER_ROOTS", (link_root,))

    assert guard._is_protected_runtime_path(real_root / "job-alerts.json")  # resolved target of the root
    assert guard._is_protected_runtime_path(link_root / "job-alerts.json")  # lexical path through the root
    escaped = tmp_path / "elsewhere.json"
    (real_root / "out").symlink_to(escaped)
    assert guard._is_protected_runtime_path(link_root / "out")  # inside the root, even if it points out
    assert not guard._is_protected_runtime_path(tmp_path / "unrelated.json")
