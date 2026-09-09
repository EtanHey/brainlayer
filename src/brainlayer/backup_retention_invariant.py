"""Static CI guard for the JSONL backup retention invariant introduced in PR #815.

Behavior tests prove today's examples. This guard also pins the production call graph so a
future refactor cannot keep the fixtures green while bypassing the surviving-copy evidence.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path


def _function(tree: ast.AST, name: str) -> ast.FunctionDef | None:
    return next(
        (node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == name),
        None,
    )


def _call_name(call: ast.Call) -> str | None:
    if isinstance(call.func, ast.Name):
        return call.func.id
    if isinstance(call.func, ast.Attribute):
        return call.func.attr
    return None


def _calls(function: ast.FunctionDef, name: str) -> list[ast.Call]:
    return [node for node in ast.walk(function) if isinstance(node, ast.Call) and _call_name(node) == name]


def _has_compare(function: ast.FunctionDef, *, operator: type[ast.cmpop], terms: tuple[str, ...]) -> bool:
    for node in ast.walk(function):
        if not isinstance(node, ast.Compare) or not any(isinstance(op, operator) for op in node.ops):
            continue
        rendered = ast.unparse(node)
        if all(term in rendered for term in terms):
            return True
    return False


def _passes_live_inventory(call: ast.Call) -> bool:
    if any(
        keyword.arg == "surviving_archives"
        and isinstance(keyword.value, ast.Name)
        and keyword.value.id == "surviving_archives"
        for keyword in call.keywords
    ):
        return True
    return len(call.args) >= 3 and isinstance(call.args[2], ast.Name) and call.args[2].id == "surviving_archives"


def _verified_upload_delete_lines(function: ast.FunctionDef) -> set[int]:
    lines: set[int] = set()
    for node in ast.walk(function):
        if not isinstance(node, ast.If):
            continue
        condition = ast.unparse(node.test)
        if 'result["verified"]' not in condition and "result['verified']" not in condition:
            continue
        if "upload" not in condition:
            continue
        lines.update(child.lineno for child in ast.walk(node) if isinstance(child, ast.Call))
    return lines


def inspect_jsonl_retention_invariant(source: str) -> list[str]:
    """Return deterministic violations of the PR #815 surviving-copy contract."""
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        return [f"jsonl_backup.py is not valid Python: {exc}"]

    errors: list[str] = []
    state_matches = _function(tree, "_state_matches")
    select_candidates = _function(tree, "_select_backup_candidates")
    update_state = _function(tree, "_update_state_for_uploaded")
    run_backup = _function(tree, "run_backup")
    required = {
        "_state_matches": state_matches,
        "_select_backup_candidates": select_candidates,
        "_update_state_for_uploaded": update_state,
        "run_backup": run_backup,
    }
    for name, function in required.items():
        if function is None:
            errors.append(f"required retention function is missing: {name}")
    if errors:
        return errors

    assert state_matches is not None
    assert select_candidates is not None
    assert update_state is not None
    assert run_backup is not None

    if not _has_compare(
        state_matches,
        operator=ast.NotIn,
        terms=("archive_id", "surviving_archives"),
    ):
        errors.append("coverage must reject archive IDs absent from the live Drive inventory")
    if not _has_compare(
        state_matches,
        operator=ast.NotEq,
        terms=("live_md5", "recorded_md5"),
    ):
        errors.append("coverage must reject a surviving Drive object whose archived bytes changed")
    if not _has_compare(
        state_matches,
        operator=ast.Eq,
        terms=("recorded_hash", "_sha256_file(candidate.path)"),
    ):
        errors.append("coverage must compare the live source bytes with the archived source digest")

    select_calls = _calls(select_candidates, "_state_matches")
    if not select_calls or not any(_passes_live_inventory(call) for call in select_calls):
        errors.append("candidate selection must pass the live Drive inventory into the coverage predicate")

    list_calls = _calls(run_backup, "_list_surviving_archives")
    selection_calls = _calls(run_backup, "_select_backup_candidates")
    if (
        not list_calls
        or not selection_calls
        or min(call.lineno for call in list_calls) >= min(call.lineno for call in selection_calls)
    ):
        errors.append("run_backup must list surviving Drive objects before selecting covered files")
    elif not any(_passes_live_inventory(call) for call in selection_calls):
        errors.append("run_backup must hand its live Drive inventory to candidate selection")

    state_write_calls = _calls(run_backup, "_update_state_for_uploaded")
    required_state_keywords = {"archive_id", "archive_md5", "digests"}
    if not state_write_calls or not any(
        required_state_keywords <= {keyword.arg for keyword in call.keywords if keyword.arg}
        for call in state_write_calls
    ):
        errors.append("uploaded state must persist archive identity, archive bytes, and source-byte digests")

    prune_calls = _calls(run_backup, "prune_drive_backups")
    archive_unlinks = [
        call
        for call in _calls(run_backup, "unlink")
        if isinstance(call.func, ast.Attribute)
        and isinstance(call.func.value, ast.Name)
        and call.func.value.id == "archive_path"
    ]
    delete_calls = [*prune_calls, *archive_unlinks]
    verified_lines = _verified_upload_delete_lines(run_backup)
    if not prune_calls:
        errors.append("the Drive retention deletion call disappeared instead of retaining its safety contract")
    if not archive_unlinks:
        errors.append("the local staging deletion call disappeared instead of retaining its safety contract")
    if any(call.lineno not in verified_lines for call in delete_calls):
        errors.append("backup deletion calls must remain inside verified-upload control flow")
    if (
        state_write_calls
        and delete_calls
        and max(call.lineno for call in state_write_calls) >= min(call.lineno for call in delete_calls)
    ):
        errors.append("surviving-copy provenance must be persisted before any backup deletion call")

    return errors


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    path = Path(args[0]) if args else Path("src/brainlayer/jsonl_backup.py")
    errors = inspect_jsonl_retention_invariant(path.read_text(encoding="utf-8"))
    if errors:
        for error in errors:
            print(f"FAIL: {error}")
        return 1
    print(f"PASS: PR #815 JSONL retention invariant is structurally intact in {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
