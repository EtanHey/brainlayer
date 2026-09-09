from pathlib import Path


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
