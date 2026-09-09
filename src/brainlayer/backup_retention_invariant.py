"""Static CI guard for the JSONL backup retention invariant introduced in PR #815.

Behavior tests prove today's examples. This guard also pins the production call graph so a
future refactor cannot keep the fixtures green while bypassing the surviving-copy evidence.
It deliberately proves call-graph shape and the specific md5 producer/consumer seam from #815;
it is not general data-flow analysis. Behavioral tests own proof that runtime values are populated.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

REFACTOR_GUIDANCE = (
    "PR #815 shipped an integrity check that could never execute while its tests passed. "
    "If you refactored deliberately, UPDATE this guard; do not delete it."
)


def _with_refactor_guidance(errors: list[str]) -> list[str]:
    return [*errors, REFACTOR_GUIDANCE] if errors else errors


def _function(tree: ast.AST, name: str) -> ast.FunctionDef | None:
    definitions = [
        node for node in getattr(tree, "body", []) if isinstance(node, ast.FunctionDef) and node.name == name
    ]
    return definitions[0] if len(definitions) == 1 else None


def _call_name(call: ast.Call) -> str | None:
    if isinstance(call.func, ast.Name):
        return call.func.id
    if isinstance(call.func, ast.Attribute):
        return call.func.attr
    return None


def _calls(function: ast.FunctionDef, name: str) -> list[ast.Call]:
    return [node for node in ast.walk(function) if isinstance(node, ast.Call) and _call_name(node) == name]


def _parent_map(tree: ast.AST) -> dict[ast.AST, ast.AST]:
    return {child: parent for parent in ast.walk(tree) for child in ast.iter_child_nodes(parent)}


def _inside_statically_dead_branch(node: ast.AST, parents: dict[ast.AST, ast.AST]) -> bool:
    child = node
    while parent := parents.get(child):
        if isinstance(parent, ast.If) and isinstance(parent.test, ast.Constant):
            if parent.test.value is False and child in parent.body:
                return True
            if parent.test.value is True and child in parent.orelse:
                return True
        child = parent
    return False


def _has_reachable_compare(
    function: ast.FunctionDef,
    *,
    operator: type[ast.cmpop],
    terms: tuple[str, ...],
    parents: dict[ast.AST, ast.AST],
) -> bool:
    for node in ast.walk(function):
        if not isinstance(node, ast.Compare) or not any(isinstance(op, operator) for op in node.ops):
            continue
        if _inside_statically_dead_branch(node, parents):
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


def _is_exact_verified_upload_gate(node: ast.If) -> bool:
    expected = ast.parse('result["verified"] and upload', mode="eval").body
    return ast.dump(node.test, include_attributes=False) == ast.dump(expected, include_attributes=False)


def _calls_in_statements(statements: list[ast.stmt]) -> list[ast.Call]:
    return [child for statement in statements for child in ast.walk(statement) if isinstance(child, ast.Call)]


def _direct_expression_call(statement: ast.stmt, name: str) -> ast.Call | None:
    if not isinstance(statement, ast.Expr) or not isinstance(statement.value, ast.Call):
        return None
    return statement.value if _call_name(statement.value) == name else None


def _allowed_coverage_return(node: ast.Return, parents: dict[ast.AST, ast.AST]) -> bool:
    if isinstance(node.value, ast.Constant) and node.value.value is False:
        return True
    if isinstance(node.value, ast.Compare):
        rendered = ast.unparse(node.value)
        return (
            any(isinstance(operator, ast.Eq) for operator in node.value.ops)
            and "recorded_hash" in rendered
            and "_sha256_file(candidate.path)" in rendered
        )
    if not (isinstance(node.value, ast.Constant) and node.value.value is True):
        return False
    parent = parents.get(node)
    if not isinstance(parent, ast.If) or node not in parent.body:
        return False
    expected = ast.parse("surviving_archives is None", mode="eval").body
    return ast.dump(parent.test, include_attributes=False) == ast.dump(expected, include_attributes=False)


def _has_exact_single_assignment(
    function: ast.FunctionDef,
    *,
    target: str,
    expression: str,
    parents: dict[ast.AST, ast.AST],
) -> bool:
    stores = [
        node
        for node in ast.walk(function)
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store) and node.id == target
    ]
    if len(stores) != 1:
        return False
    assignment = parents.get(stores[0])
    if not isinstance(assignment, ast.Assign) or assignment.targets != [stores[0]]:
        return False
    expected = ast.parse(expression, mode="eval").body
    return ast.dump(assignment.value, include_attributes=False) == ast.dump(expected, include_attributes=False)


def _keyword_matches(call: ast.Call, *, name: str, expression: str) -> bool:
    expected = ast.parse(expression, mode="eval").body
    return any(
        keyword.arg == name
        and ast.dump(keyword.value, include_attributes=False) == ast.dump(expected, include_attributes=False)
        for keyword in call.keywords
    )


def _upload_requests_md5(function: ast.FunctionDef) -> bool:
    for call in _calls(function, "post"):
        if not call.args or not isinstance(call.args[0], ast.Constant) or not isinstance(call.args[0].value, str):
            continue
        fields = parse_qs(urlsplit(call.args[0].value).query).get("fields", [])
        if any("md5Checksum" in value.split(",") for value in fields):
            return True
    return False


def inspect_jsonl_retention_invariant(source: str, *, backup_daily_source: str) -> list[str]:
    """Return deterministic violations of the PR #815 surviving-copy contract."""
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        return [f"jsonl_backup.py is not valid Python: {exc}"]

    try:
        backup_daily_tree = ast.parse(backup_daily_source)
    except SyntaxError as exc:
        return [f"backup_daily.py is not valid Python: {exc}"]

    errors: list[str] = []
    state_matches = _function(tree, "_state_matches")
    select_candidates = _function(tree, "_select_backup_candidates")
    update_state = _function(tree, "_update_state_for_uploaded")
    run_backup = _function(tree, "run_backup")
    upload_file = _function(backup_daily_tree, "upload_file_to_drive_raw")
    required = {
        "_state_matches": state_matches,
        "_select_backup_candidates": select_candidates,
        "_update_state_for_uploaded": update_state,
        "run_backup": run_backup,
        "upload_file_to_drive_raw": upload_file,
    }
    for name, function in required.items():
        if function is None:
            errors.append(f"required retention function is missing: {name}")
    if errors:
        return _with_refactor_guidance(errors)

    assert state_matches is not None
    assert select_candidates is not None
    assert update_state is not None
    assert run_backup is not None
    assert upload_file is not None
    parents = _parent_map(tree)

    if not _has_exact_single_assignment(
        state_matches,
        target="recorded_md5",
        expression='entry.get("archive_md5")',
        parents=parents,
    ):
        errors.append("coverage must read the recorded archive md5 from persisted state")

    if not _has_reachable_compare(
        state_matches,
        operator=ast.NotIn,
        terms=("archive_id", "surviving_archives"),
        parents=parents,
    ):
        errors.append("coverage must reject archive IDs absent from the live Drive inventory")
    if not _has_reachable_compare(
        state_matches,
        operator=ast.NotEq,
        terms=("live_md5", "recorded_md5"),
        parents=parents,
    ):
        errors.append("coverage must reject a surviving Drive object whose archived bytes changed")
    if not _has_reachable_compare(
        state_matches,
        operator=ast.Eq,
        terms=("recorded_hash", "_sha256_file(candidate.path)"),
        parents=parents,
    ):
        errors.append("coverage must compare the live source bytes with the archived source digest")
    if any(
        not _allowed_coverage_return(node, parents)
        for node in ast.walk(state_matches)
        if isinstance(node, ast.Return) and not _inside_statically_dead_branch(node, parents)
    ):
        errors.append("every successful coverage path must require surviving-copy evidence")

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

    verified_gates = [
        node for node in ast.walk(run_backup) if isinstance(node, ast.If) and _is_exact_verified_upload_gate(node)
    ]
    verified_gate = verified_gates[0] if len(verified_gates) == 1 else None
    safe_calls = _calls_in_statements(verified_gate.body) if verified_gate is not None else []

    persistence: list[tuple[int, ast.Call, ast.Call]] = []
    if verified_gate is not None:
        for index, statement in enumerate(verified_gate.body):
            atomic_write = _direct_expression_call(statement, "_atomic_write_json")
            if atomic_write is None:
                continue
            updates = [
                child
                for child in ast.walk(atomic_write)
                if isinstance(child, ast.Call) and _call_name(child) == "_update_state_for_uploaded"
            ]
            if len(updates) == 1:
                persistence.append((index, atomic_write, updates[0]))

    required_state_keywords = {"archive_id", "archive_md5", "digests"}
    if not persistence or not any(
        required_state_keywords <= {keyword.arg for keyword in call.keywords if keyword.arg}
        for _, _, call in persistence
    ):
        errors.append("uploaded state must persist archive identity, archive bytes, and source-byte digests")
    if not any(
        _keyword_matches(call, name="archive_md5", expression='uploaded.get("md5Checksum")')
        for _, _, call in persistence
    ):
        errors.append("uploaded state must persist md5Checksum from the upload response")

    if not _upload_requests_md5(upload_file):
        errors.append("Drive upload must request md5Checksum from the API")

    prune_calls = _calls(run_backup, "prune_drive_backups")
    archive_unlinks = [
        call
        for call in _calls(run_backup, "unlink")
        if isinstance(call.func, ast.Attribute)
        and isinstance(call.func.value, ast.Name)
        and call.func.value.id == "archive_path"
    ]
    delete_calls = [*prune_calls, *archive_unlinks]
    if not prune_calls:
        errors.append("the Drive retention deletion call disappeared instead of retaining its safety contract")
    if not archive_unlinks:
        errors.append("the local staging deletion call disappeared instead of retaining its safety contract")
    if verified_gate is None or any(call not in safe_calls for call in delete_calls):
        errors.append("backup deletion calls must remain inside verified-upload control flow")
    deletion_statement_indexes = (
        [
            index
            for index, statement in enumerate(verified_gate.body)
            if any(call in delete_calls for call in ast.walk(statement) if isinstance(call, ast.Call))
        ]
        if verified_gate is not None
        else []
    )
    if (
        not persistence
        or not deletion_statement_indexes
        or min(index for index, _, _ in persistence) >= min(deletion_statement_indexes)
    ):
        errors.append("surviving-copy provenance must be durably persisted before any backup deletion call")

    return _with_refactor_guidance(errors)


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    path = Path(args[0]) if args else Path("src/brainlayer/jsonl_backup.py")
    backup_daily_path = path.with_name("backup_daily.py")
    if not backup_daily_path.exists():
        print(f"FAIL: required sibling source is missing: {backup_daily_path}")
        print(f"FAIL: {REFACTOR_GUIDANCE}")
        return 1
    errors = inspect_jsonl_retention_invariant(
        path.read_text(encoding="utf-8"),
        backup_daily_source=backup_daily_path.read_text(encoding="utf-8"),
    )
    if errors:
        for error in errors:
            print(f"FAIL: {error}")
        return 1
    print(f"PASS: PR #815 JSONL retention invariant is structurally intact in {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
