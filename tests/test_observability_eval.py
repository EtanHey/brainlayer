from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path
from types import SimpleNamespace

import jsonschema
import pytest

from scripts import build_observability_fixture as builder
from scripts import observability_eval as evaluator

REQUIRED_FAILURES = {
    "missing_source_class",
    "missing_log",
    "malformed_log",
    "missing_launchd",
    "empty_db",
    "clock_skew",
}


def _tree_digest(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(candidate for candidate in root.rglob("*") if candidate.is_file()):
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def test_builder_is_byte_deterministic(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"

    builder.build_fixture_bundle(first, seed=20260913)
    builder.build_fixture_bundle(second, seed=20260913)

    assert _tree_digest(first) == _tree_digest(second)


def test_builder_uses_vector_store_schema_without_handwritten_ddl(tmp_path: Path) -> None:
    root = tmp_path / "bundle"
    builder.build_fixture_bundle(root, seed=20260913)

    regular_case = next(case for case in builder.case_definitions() if case.failure != "missing_source_class")
    with sqlite3.connect(root / regular_case.db_path) as connection:
        columns = {row[1] for row in connection.execute("PRAGMA table_info(chunks)")}

    assert {"provenance_class", "source_class", "archived_at", "superseded_by"} <= columns
    assert "CREATE TABLE" not in Path(builder.__file__).read_text(encoding="utf-8").upper()


def test_cases_cover_every_fail_closed_shape_in_both_splits() -> None:
    cases = builder.case_definitions()

    for split in ("dev", "heldout"):
        assert {case.failure for case in cases if case.split == split} >= REQUIRED_FAILURES
    for case in cases:
        expected = "heldout" if hashlib.sha256(case.case_id.encode()).digest()[0] < 0x60 else "dev"
        assert case.split == expected


def test_frozen_schema_validates_every_dev_golden() -> None:
    fixture_root = Path("tests/fixtures/observability")
    schema = json.loads((fixture_root / "observability-schema.v1.json").read_text(encoding="utf-8"))
    cases = json.loads((fixture_root / "cases.json").read_text(encoding="utf-8"))["cases"]

    for case in cases:
        if case["split"] != "dev":
            continue
        golden = json.loads((fixture_root / case["golden"]).read_text(encoding="utf-8"))
        jsonschema.validate(golden, schema)


def test_heldout_goldens_are_not_committed() -> None:
    fixture_root = Path("tests/fixtures/observability")
    cases = json.loads((fixture_root / "cases.json").read_text(encoding="utf-8"))["cases"]

    assert all(not (fixture_root / case["golden"]).exists() for case in cases if case["split"] == "heldout")


def test_recon_log_and_launchd_shapes_are_frozen() -> None:
    root = Path("tests/fixtures/observability")
    healthy_jsonl = (root / "logs/healthy-dev/jsonl-backup.log").read_text(encoding="utf-8").splitlines()
    daily = (root / "logs/healthy-dev/backup-daily.log").read_text(encoding="utf-8").splitlines()
    errors = (root / "logs/backup-errors-dev/backup-daily.log").read_text(encoding="utf-8").splitlines()

    receipts = [json.loads(line) for line in healthy_jsonl]
    assert "archive_id" not in receipts[0] and "forever_files" not in receipts[0]
    assert "forever_files" in receipts[1] and "archive_id" not in receipts[1]
    assert {"archive_id", "md5Checksum"} <= receipts[2].keys()
    assert sum(not line.startswith("{") for line in daily) == 2
    assert all(json.loads(line)["error_type"] == "FileNotFoundError" for line in errors)
    assert json.loads((root / "logs/no-op-dev/jsonl-backup.log").read_text())["status"] == "no-op"
    assert "Could not find service" in (root / "launchd/no-op-dev.txt").read_text()
    assert (root / "launchd/missing-launchd-dev.txt").read_text() == ""


def test_db_census_emitter_and_unknown_author_shapes_are_frozen() -> None:
    root = Path("tests/fixtures/observability")
    with sqlite3.connect(root / "db/healthy-dev.sqlite") as connection:
        rows = connection.execute(
            "SELECT metadata, source, sender, source_file, provenance_class, source_class FROM chunks"
        ).fetchall()
    assert all("attributionAgent" not in json.loads(row[0]) for row in rows)
    assert {"realtime_watcher", "claude_code", "codex_cli", "mcp"} <= {row[1] for row in rows}
    assert any(row[1] is None and row[2] == "assistant" for row in rows)
    assert any(row[1] is None and row[2] is None and row[3] == "realtime-hook" for row in rows)
    assert any(row[4] is None for row in rows)
    assert any(row[4] == "unknown" and row[5] is not None for row in rows)

    golden = evaluator.load_golden("healthy-dev")
    derivations = {item["derived_from"] for item in golden["emitters"]["by_emitter"]}
    assert derivations == {"source", "sender", "source_file"}
    assert golden["author_unknown"]["never_classified"]["count"] > 0
    assert golden["author_unknown"]["classified_unknown"]["count"] > 0


def test_mutated_golden_control_fails_field_by_field() -> None:
    expected = evaluator.load_golden("healthy-dev")
    actual = json.loads(json.dumps(expected))
    actual["stores"]["total_chunks"] += 1

    result = evaluator.grade_payload(
        case=evaluator.load_case("healthy-dev"),
        actual=actual,
        expected=expected,
        opened_inputs=evaluator.declared_inputs(evaluator.load_case("healthy-dev")),
    )

    assert result.passed is False
    assert result.field_mismatches == ["$.stores.total_chunks: expected 25, actual 26"]


def test_malformed_producer_json_is_a_grade_not_a_traceback(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_run(*args: object, env: dict[str, str], **kwargs: object) -> SimpleNamespace:
        Path(env["BRAINLAYER_OBSERVABILITY_PATH"]).write_text("{malformed", encoding="utf-8")
        return SimpleNamespace(returncode=0, stderr="", stdout="")

    monkeypatch.setattr(evaluator.subprocess, "run", fake_run)
    case = evaluator.load_case("healthy-dev")
    result = evaluator._run_case(case, Path("tests/fixtures/observability"), Path.cwd(), None)

    assert result.passed is False
    assert result.field_mismatches[0].startswith("$: malformed producer JSON:")


def test_mock_green_detects_numbers_when_a_required_input_is_missing() -> None:
    case = evaluator.load_case("missing-log-dev")
    actual = evaluator.load_golden("healthy-dev")

    result = evaluator.grade_payload(
        case=case,
        actual=actual,
        expected=actual,
        opened_inputs=evaluator.declared_inputs(case),
    )

    assert any(item.startswith("MOCK_GREEN $.backups") for item in result.mock_green)


def test_traceability_requires_exact_declared_opened_input_set() -> None:
    case = evaluator.load_case("healthy-dev")
    expected = evaluator.load_golden("healthy-dev")
    declared = evaluator.declared_inputs(case)

    result = evaluator.grade_payload(
        case=case,
        actual=expected,
        expected=expected,
        opened_inputs=declared[:-1],
    )

    assert result.traceability == [f"missing opened input: {declared[-1]}"]
