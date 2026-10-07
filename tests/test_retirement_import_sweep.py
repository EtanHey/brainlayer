"""Source import contracts, without executing maintenance jobs."""

import ast
import json
import os
import subprocess
import sys
from pathlib import Path

from brainlayer.import_sweep import _child_env

ROOT = Path(os.environ.get("BRAINLAYER_IMPORT_SWEEP_SOURCE_ROOT", Path(__file__).resolve().parents[1])).resolve()


def _targets():
    targets = []
    for path in sorted((ROOT / "src" / "brainlayer").rglob("*.py")):
        parts = list(path.relative_to(ROOT / "src").with_suffix("").parts)
        if parts[-1] == "__init__":
            parts.pop()
        targets.append(("module", ".".join(parts)))
    for path in sorted((ROOT / "scripts").glob("*.py")):
        imports = []
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.ImportFrom) and (node.module or "").split(".")[0] == "brainlayer":
                imports.append(ast.unparse(node))
            elif isinstance(node, ast.Import):
                aliases = [alias for alias in node.names if alias.name.split(".")[0] == "brainlayer"]
                if aliases:
                    imports.append(ast.unparse(ast.Import(names=aliases)))
        if imports:
            targets.append(("script", (path.name, imports)))
    return targets


TARGETS = _targets()


def test_all_source_import_contracts(tmp_path):
    # Includes function-local imports. Scripts have unguarded exits/signal
    # handlers, so check imports without executing a maintenance job.
    code = """
import importlib, json, pathlib, socket, sys, traceback
sys.path.insert(0, sys.argv[1])
def deny(*args, **kwargs):
    raise RuntimeError('network disabled during source import sweep')
socket.socket.connect = deny
socket.create_connection = deny
failures = []
for kind, target in json.loads(sys.argv[2]):
    try:
        if kind == 'module':
            module = importlib.import_module(target)
            assert pathlib.Path(module.__file__).resolve().is_relative_to(pathlib.Path(sys.argv[1]).resolve())
        else:
            for statement in target[1]:
                exec(statement, {})
    except BaseException:
        failures.append(f'{kind}: {target}\\n{traceback.format_exc()}')
print(f'Import inventory: {len(json.loads(sys.argv[2]))} targets; {len(failures)} failures')
if failures:
    print('\\n'.join(failures))
    sys.exit(1)
"""
    env = _child_env(tmp_path)
    env["BRAINLAYER_FORBID_BRAINBAR_SOCKET"] = "1"
    result = subprocess.run(
        [sys.executable, "-I", "-c", code, str(ROOT / "src"), json.dumps(TARGETS)],
        env=env,
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=180,
    )
    print(result.stdout)
    assert result.returncode == 0, result.stdout + result.stderr


def test_import_inventory_has_every_package_file_and_session_finish_script():
    assert sum(kind == "module" for kind, _ in TARGETS) == len(list((ROOT / "src" / "brainlayer").rglob("*.py")))
    assert any(kind == "script" and target[0] == "kg_session_finish.py" for kind, target in TARGETS)
