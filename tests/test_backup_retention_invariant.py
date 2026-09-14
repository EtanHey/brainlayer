import re
from pathlib import Path

DRIVE_HARD_DELETE_ALLOWLIST: frozenset[str] = frozenset()


def _repo_drive_hard_delete_references(root: Path = Path("src/brainlayer")) -> list[str]:
    pattern = re.compile(r"\.files\(\)\s*\.delete\s*\(")
    return [
        f"{path}:{line_number}"
        for path in sorted(root.rglob("*.py"))
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1)
        if pattern.search(line) and path.as_posix() not in DRIVE_HARD_DELETE_ALLOWLIST
    ]


def test_brainlayer_source_has_empty_drive_hard_delete_allowlist() -> None:
    assert DRIVE_HARD_DELETE_ALLOWLIST == frozenset()
    assert _repo_drive_hard_delete_references() == []


def test_repo_scan_rejects_a_multiline_drive_files_delete(tmp_path: Path) -> None:
    module = tmp_path / "multiline_delete.py"
    module.write_text(
        "def purge(service, file_id):\n"
        "    (\n"
        "        service.files()\n"
        "        .delete(fileId=file_id, supportsAllDrives=True)\n"
        "        .execute()\n"
        "    )\n",
        encoding="utf-8",
    )

    assert _repo_drive_hard_delete_references(tmp_path) != []


def _backup_daily_with_pruner_statement(statement: str) -> str:
    source = Path("src/brainlayer/backup_daily.py").read_text(encoding="utf-8")
    marker = "        service.files().update(\n"
    mutated = source.replace(marker, f"{statement}{marker}", 1)
    assert mutated != source
    return mutated


def _assert_backup_daily_hard_delete_rejected(source: str) -> None:
    from brainlayer.backup_retention_invariant import inspect_backup_daily_retention_invariant

    assert "backup_daily retention must not hard-delete Drive objects" in inspect_backup_daily_retention_invariant(
        source
    )


def test_backup_daily_invariant_rejects_an_injected_drive_hard_delete() -> None:
    from brainlayer.backup_retention_invariant import inspect_backup_daily_retention_invariant

    source = Path("src/brainlayer/backup_daily.py").read_text(encoding="utf-8")
    mutated = source.replace(
        '        service.files().update(\n            fileId=item["id"],\n            body={"trashed": True},\n',
        '        service.files().delete(\n            fileId=item["id"],\n',
        1,
    )

    assert mutated != source
    assert "backup_daily retention must not hard-delete Drive objects" in inspect_backup_daily_retention_invariant(
        mutated
    )


def test_backup_daily_invariant_rejects_a_files_resource_alias_delete() -> None:
    from brainlayer.backup_retention_invariant import inspect_backup_daily_retention_invariant

    source = Path("src/brainlayer/backup_daily.py").read_text(encoding="utf-8")
    mutated = source.replace(
        '        service.files().update(\n            fileId=item["id"],\n            body={"trashed": True},\n',
        '        files = service.files()\n        files.delete(\n            fileId=item["id"],\n',
        1,
    )

    assert mutated != source
    assert "backup_daily retention must not hard-delete Drive objects" in inspect_backup_daily_retention_invariant(
        mutated
    )


def test_backup_daily_invariant_rejects_an_annotated_files_alias_delete() -> None:
    _assert_backup_daily_hard_delete_rejected(
        _backup_daily_with_pruner_statement(
            "        files: Any = service.files()\n        files.delete(fileId=item['id'])\n"
        )
    )


def test_backup_daily_invariant_rejects_getattr_delete() -> None:
    _assert_backup_daily_hard_delete_rejected(
        _backup_daily_with_pruner_statement(
            '        getattr(service.files(), "delete")(fileId=item["id"], supportsAllDrives=True)\n'
        )
    )


def test_backup_daily_invariant_rejects_a_bound_delete_method() -> None:
    _assert_backup_daily_hard_delete_rejected(
        _backup_daily_with_pruner_statement(
            '        purge = service.files().delete\n        purge(fileId=item["id"], supportsAllDrives=True)\n'
        )
    )


def test_backup_daily_invariant_rejects_a_walrus_files_alias_delete() -> None:
    _assert_backup_daily_hard_delete_rejected(
        _backup_daily_with_pruner_statement(
            '        (files := service.files()).delete(fileId=item["id"], supportsAllDrives=True)\n'
        )
    )


def test_backup_daily_invariant_rejects_raw_drive_rest_delete() -> None:
    _assert_backup_daily_hard_delete_rejected(
        _backup_daily_with_pruner_statement(
            "        requests.delete(f\"https://www.googleapis.com/drive/v3/files/{item['id']}\")\n"
        )
    )


def test_backup_daily_invariant_rejects_raw_drive_request_delete() -> None:
    _assert_backup_daily_hard_delete_rejected(
        _backup_daily_with_pruner_statement(
            '        requests.request("DELETE", f"https://www.googleapis.com/drive/v3/files/{item[\'id\']}")\n'
        )
    )


def test_jsonl_retention_guard_failure_tells_refactors_to_update_not_delete(tmp_path: Path, capsys) -> None:
    from brainlayer.backup_retention_invariant import main
    from tests import test_jsonl_backup

    test_jsonl_backup.test_jsonl_retention_invariant_is_a_ci_guard_not_only_a_behavior_fixture()

    unsafe_path = tmp_path / "jsonl_backup.py"
    unsafe_path.write_text("def run_backup():\n    pass\n", encoding="utf-8")

    assert main([str(unsafe_path)]) == 1
    output = capsys.readouterr().out
    assert "UPDATE this guard; do not delete it" in output
    assert "PR #815" in output
