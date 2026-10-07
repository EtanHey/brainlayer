"""Dormant script removal witnesses and isolated retirement exit probes."""

import ast
import os
import subprocess
import sys
from pathlib import Path

import pytest

from brainlayer.import_sweep import _child_env

ROOT = Path(os.environ.get("BRAINLAYER_IMPORT_SWEEP_SOURCE_ROOT", Path(__file__).resolve().parents[1])).resolve()
REMOVED_SCRIPTS = (
    "cloud_stream.py",
    "enrichment_pilot.py",
    "enrich_recent.py",
    "enrichment_backfill.py",
    "batch_submit_paced.py",
    "vertex_poll_import.py",
    "monitor_batch_reenrichment.py",
    "run_abcde_enrich.py",
    "enrichment_llm_judge.py",
)
TRIMMED = {
    "cloud_stream.py": {"run_stream", "main"},
    "enrichment_pilot.py": {"call_gemini", "main", "analyze_results", "write_report"},
}
BLOCKED_SCRIPTS = (
    "cloud_stream.py",
    "enrichment_pilot.py",
    "enrich_recent.py",
    "enrichment_backfill.py",
    "batch_submit_paced.py",
)


def test_retired_script_functions_are_removed():
    for script, removed in TRIMMED.items():
        path = ROOT / "scripts" / script
        if script in REMOVED_SCRIPTS:
            assert not path.exists(), script
            continue
        tree = ast.parse(path.read_text())
        definitions = {
            node.name
            for node in ast.walk(tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
        }
        assert not removed & definitions, (script, removed & definitions)


@pytest.mark.parametrize("script", BLOCKED_SCRIPTS)
def test_remaining_blocked_scripts_exit_before_dependencies_or_io(script, tmp_path):
    path = ROOT / "scripts" / script
    if script in REMOVED_SCRIPTS:
        assert not path.exists(), script
        return
    tree = ast.parse(path.read_text())
    call = tree.body[2]
    assert isinstance(call, ast.Expr) and isinstance(call.value, ast.Call)
    assert ast.unparse(call.value.func) == "_gate_sys.exit"
    code = r"""
import builtins, pathlib, socket, sys
path = pathlib.Path(sys.argv[1])
source = path.read_text()
real_import = builtins.__import__
def guarded_import(name, *args, **kwargs):
    if name.split('.')[0] in {'brainlayer', 'src', 'google', 'requests', 'apsw', 'sqlite3', 'subprocess'}:
        raise AssertionError('dependency reached before retirement exit: ' + name)
    return real_import(name, *args, **kwargs)
def deny(*args, **kwargs):
    raise AssertionError('network attempted during retirement probe')
def audit(event, args):
    if event == 'open' or event.startswith(('socket.', 'subprocess.')):
        raise AssertionError('I/O attempted after script load: ' + event)
builtins.__import__ = guarded_import
socket.socket.connect = deny
socket.create_connection = deny
sys.addaudithook(audit)
sys.argv = [str(path), '--run', '--dry-run', '--workers', '100']
exec(compile(source, str(path), 'exec'), {'__name__': '__main__', '__file__': str(path)})
"""
    env = _child_env(tmp_path)
    env.update(GOOGLE_API_KEY="synthetic-not-a-key", GROQ_API_KEY="synthetic-not-a-key")
    result = subprocess.run(
        [sys.executable, "-I", "-c", code, str(path)], env=env, cwd=tmp_path, capture_output=True, text=True, timeout=30
    )
    assert result.returncode == 1, result.stdout + result.stderr
    assert "GATED OFF" in result.stderr or "RETIRED" in result.stderr
    assert "AssertionError" not in result.stderr


@pytest.mark.parametrize("script", REMOVED_SCRIPTS)
def test_removed_script_is_absent_and_old_invocation_cannot_run(script, tmp_path):
    path = ROOT / "scripts" / script
    assert not path.exists(), script
    history = tmp_path / "history.db"
    checkpoint = tmp_path / "checkpoint.json"
    history.write_bytes(b"synthetic retained history")
    checkpoint.write_bytes(b'{"state":"retained"}\n')
    env = _child_env(tmp_path)
    env.update(GOOGLE_API_KEY="synthetic-not-a-key", GROQ_API_KEY="synthetic-not-a-key")
    result = subprocess.run(
        [sys.executable, "-I", str(path), "--help"], env=env, cwd=tmp_path, capture_output=True, text=True, timeout=30
    )
    assert result.returncode == 2, result.stdout + result.stderr
    assert "can't open file" in result.stderr
    assert history.read_bytes() == b"synthetic retained history"
    assert checkpoint.read_bytes() == b'{"state":"retained"}\n'


def test_removed_scripts_have_no_source_or_install_references():
    paths = [ROOT / "pyproject.toml", ROOT / "README.md"]
    for directory in ("src", "scripts", "hooks", ".github", "docs"):
        paths.extend(
            path
            for path in (ROOT / directory).rglob("*")
            if path.is_file() and path.suffix in {".py", ".sh", ".yml", ".yaml", ".toml", ".md", ".json", ".plist"}
        )
    retired_modules = {"scripts." + Path(script).stem for script in REMOVED_SCRIPTS}
    findings = []
    for path in paths:
        text = path.read_text()
        for script in REMOVED_SCRIPTS:
            if script in text:
                findings.append((str(path.relative_to(ROOT)), script))
        if path.suffix == ".py":
            for node in ast.walk(ast.parse(text)):
                if isinstance(node, ast.ImportFrom) and node.module in retired_modules:
                    findings.append((str(path.relative_to(ROOT)), node.module))
                if isinstance(node, ast.Constant) and isinstance(node.value, str) and node.value in retired_modules:
                    findings.append((str(path.relative_to(ROOT)), node.value))
                if isinstance(node, ast.ImportFrom) and node.module == "scripts":
                    for alias in node.names:
                        if alias.name + ".py" in REMOVED_SCRIPTS:
                            findings.append((str(path.relative_to(ROOT)), alias.name))
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        if (
                            alias.name.startswith("scripts.")
                            and alias.name.rsplit(".", 1)[-1] + ".py" in REMOVED_SCRIPTS
                        ):
                            findings.append((str(path.relative_to(ROOT)), alias.name))
    assert not findings, findings
