from __future__ import annotations

import hashlib
import json
import os
import shutil
import sqlite3
import sys
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import jsonschema
import pytest

from scripts import build_observability_fixture as builder
from scripts import check_observability_golden_inputs as golden_inputs
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


def test_dev_golden_input_receipts_match_fixture_artifacts() -> None:
    assert (
        golden_inputs.check(
            fixture_root=evaluator.FIXTURES,
            heldout_root=None,
            split="dev",
            write=False,
        )
        == []
    )


def test_golden_input_checker_reports_malformed_database_as_drift(tmp_path: Path) -> None:
    fixture = tmp_path / "fixtures"
    shutil.copytree(evaluator.FIXTURES, fixture)
    case = next(item for item in json.loads((fixture / "cases.json").read_text())["cases"] if item["split"] == "dev")
    (fixture / case["inputs"]["db"]).write_bytes(b"not sqlite")

    findings = golden_inputs.check(fixture_root=fixture, heldout_root=None, split="dev", write=False)

    assert any("status: malformed" in finding for finding in findings)


def test_golden_input_checker_write_mode_repairs_real_drift_and_exits_nonzero(tmp_path: Path, monkeypatch) -> None:
    fixture = tmp_path / "fixtures"
    shutil.copytree(evaluator.FIXTURES, fixture)
    case = next(item for item in json.loads((fixture / "cases.json").read_text())["cases"] if item["split"] == "dev")
    log_path = fixture / case["inputs"]["jsonl_backup_log"]
    log_path.write_text(log_path.read_text() + "drift\n")
    monkeypatch.setattr(
        sys,
        "argv",
        ["check_observability_golden_inputs.py", "--fixture-root", str(fixture), "--split", "dev", "--write"],
    )

    assert golden_inputs.main() == 1
    assert golden_inputs.check(fixture_root=fixture, heldout_root=None, split="dev", write=False) == []


def test_builder_uses_vector_store_schema_without_handwritten_ddl(tmp_path: Path) -> None:
    root = tmp_path / "bundle"
    builder.build_fixture_bundle(root, seed=20260913)

    regular_case = next(case for case in builder.case_definitions() if case.failure != "missing_source_class")
    with sqlite3.connect(root / regular_case.db_path) as connection:
        columns = {row[1] for row in connection.execute("PRAGMA table_info(chunks)")}

    assert {"provenance_class", "source_class", "archived_at", "superseded_by"} <= columns
    assert "CREATE TABLE" not in Path(builder.__file__).read_text(encoding="utf-8").upper()


def test_runner_stages_pinned_mtimes_without_mutating_fixture(tmp_path: Path) -> None:
    source_root = tmp_path / "fixture"
    source_root.mkdir()
    source = source_root / "db/healthy-dev.sqlite"
    source.parent.mkdir()
    source.write_bytes(b"fixture")
    os.utime(source, (1, 1))
    case = json.loads(Path("tests/fixtures/observability/cases.json").read_text())["cases"][0]
    case = {
        **case,
        "declared_inputs": ["db/healthy-dev.sqlite"],
        "input_mtimes": {"db/healthy-dev.sqlite": "2026-09-13T12:00:00Z"},
    }

    staged_root = tmp_path / "staged"
    evaluator._stage_case_inputs(case, source_root, staged_root)

    assert source.stat().st_mtime == 1
    assert (
        staged_root.joinpath("db/healthy-dev.sqlite").stat().st_mtime
        == datetime.fromisoformat("2026-09-13T12:00:00+00:00").timestamp()
    )


def test_runner_stages_future_clock_skew_log_mtime_without_mutating_fixture(tmp_path: Path) -> None:
    source_root = tmp_path / "fixture"
    source_root.mkdir()
    for relative in ("db/clock-skew-dev.sqlite", "logs/clock-skew-dev/jsonl-backup.log"):
        source = source_root / relative
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_bytes(b"fixture")
        os.utime(source, (1, 1))
    original = evaluator.load_case("clock-skew-dev")
    declared = ["db/clock-skew-dev.sqlite", "logs/clock-skew-dev/jsonl-backup.log"]
    case = {
        **original,
        "declared_inputs": declared,
        "input_mtimes": {path: original["input_mtimes"][path] for path in declared},
    }

    staged_root = tmp_path / "staged"
    evaluator._stage_case_inputs(case, source_root, staged_root)

    assert (source_root / "logs/clock-skew-dev/jsonl-backup.log").stat().st_mtime == 1
    assert (
        staged_root.joinpath("logs/clock-skew-dev/jsonl-backup.log").stat().st_mtime
        == datetime.fromisoformat("2026-09-13T16:00:00+00:00").timestamp()
    )


def test_runner_fails_closed_when_input_mtime_is_missing(tmp_path: Path) -> None:
    case = evaluator.load_case("healthy-dev")
    case = {**case, "input_mtimes": {}}
    result = evaluator._run_case(case, Path("tests/fixtures/observability"), Path.cwd(), None)
    assert result.field_mismatches == [
        "$: input staging failed: missing input_mtimes for declared inputs: "
        + ", ".join(sorted(case["declared_inputs"]))
    ]


def test_faithful_stub_requires_runner_mtime_staging(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fixture = tmp_path / "fixture"
    shutil.copytree("tests/fixtures/observability", fixture)
    case = evaluator.load_case("healthy-dev", fixture)
    os.utime(fixture / case["inputs"]["db"], (1, 1))

    def stub(*args: object, env: dict[str, str], **kwargs: object) -> SimpleNamespace:
        payload = evaluator.load_golden("healthy-dev", fixture)
        actual = json.loads(json.dumps(payload))
        expected_mtime = case["input_mtimes"][case["inputs"]["db"]]
        observed = (
            __import__("datetime")
            .datetime.fromtimestamp(Path(env["BRAINLAYER_DB"]).stat().st_mtime, __import__("datetime").UTC)
            .isoformat()
            .replace("+00:00", "Z")
        )
        if observed != expected_mtime:
            for section in actual.values():
                if isinstance(section, dict):
                    for item in section.get("inputs", []):
                        if isinstance(item, dict) and item.get("path") == case["inputs"]["db"]:
                            item["mtime"] = observed
        Path(env["BRAINLAYER_OBSERVABILITY_PATH"]).write_text(json.dumps(actual), encoding="utf-8")
        Path(env["BRAINLAYER_OBSERVABILITY_TRACE_PATH"]).write_text(
            json.dumps(case["declared_inputs"]), encoding="utf-8"
        )
        return SimpleNamespace(returncode=0, stderr="", stdout="")

    monkeypatch.setattr(evaluator.subprocess, "run", stub)
    assert evaluator._run_case(case, fixture, Path.cwd(), None).passed
    assert not evaluator._run_case(case, fixture, Path.cwd(), None, stage_inputs=False).passed


def test_cases_cover_every_fail_closed_shape_in_both_splits() -> None:
    cases = builder.case_definitions()
    manifest = json.loads(Path(builder.MANIFEST).read_text(encoding="utf-8"))["cases"]

    for split in ("dev", "heldout"):
        assert {case.failure for case in cases if case.split == split} >= REQUIRED_FAILURES
    for entry, case in zip(manifest, cases, strict=True):
        expected = "heldout" if hashlib.sha256(case.case_id.encode()).digest()[0] < 0x60 else "dev"
        assert entry["split"] == case.split == expected


def test_builder_rejects_manifest_split_drift(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    manifest = json.loads(Path(builder.MANIFEST).read_text(encoding="utf-8"))
    manifest["cases"][0]["split"] = "heldout" if manifest["cases"][0]["split"] == "dev" else "dev"
    path = tmp_path / "cases.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    monkeypatch.setattr(builder, "MANIFEST", path)

    with pytest.raises(ValueError, match="manifest split does not match derived split"):
        builder.case_definitions()


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
    cases = json.loads((root / "cases.json").read_text(encoding="utf-8"))["cases"]
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
    assert all(case["inputs"]["disabled_dir"] in case["declared_inputs"] for case in cases)


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
        opened_inputs=evaluator.load_case("healthy-dev")["declared_inputs"],
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


@pytest.mark.parametrize("trace", [[None, "db/healthy-dev.sqlite"], [{}]])
def test_malformed_input_trace_entries_are_a_grade_not_a_traceback(
    monkeypatch: pytest.MonkeyPatch, trace: list[object]
) -> None:
    def fake_run(*args: object, env: dict[str, str], **kwargs: object) -> SimpleNamespace:
        Path(env["BRAINLAYER_OBSERVABILITY_PATH"]).write_text(
            json.dumps(evaluator.load_golden("healthy-dev")), encoding="utf-8"
        )
        Path(env["BRAINLAYER_OBSERVABILITY_TRACE_PATH"]).write_text(json.dumps(trace), encoding="utf-8")
        return SimpleNamespace(returncode=0, stderr="", stdout="")

    monkeypatch.setattr(evaluator.subprocess, "run", fake_run)
    case = evaluator.load_case("healthy-dev")
    result = evaluator._run_case(case, Path("tests/fixtures/observability"), Path.cwd(), None)

    assert result.field_mismatches == ["$: malformed input trace: expected a JSON list of strings"]


def test_producer_timeout_is_a_grade_not_a_traceback(monkeypatch: pytest.MonkeyPatch) -> None:
    def timeout(*args: object, **kwargs: object) -> None:
        raise evaluator.subprocess.TimeoutExpired("producer", 30)

    monkeypatch.setattr(evaluator.subprocess, "run", timeout)
    case = evaluator.load_case("healthy-dev")
    result = evaluator._run_case(case, Path("tests/fixtures/observability"), Path.cwd(), None)
    assert result.field_mismatches == ["$: producer timed out after 30 seconds"]


def test_producer_spawn_error_is_a_grade_not_a_traceback(monkeypatch: pytest.MonkeyPatch) -> None:
    def spawn_error(*args: object, **kwargs: object) -> None:
        raise OSError("synthetic spawn failure")

    monkeypatch.setattr(evaluator.subprocess, "run", spawn_error)
    case = evaluator.load_case("healthy-dev")
    result = evaluator._run_case(case, Path("tests/fixtures/observability"), Path.cwd(), None)
    assert result.field_mismatches == ["$: producer could not start: synthetic spawn failure"]


def test_run_case_resolves_relative_roots_and_scrubs_ambient_brainlayer_env(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: dict[str, object] = {}

    def fake_run(*args: object, cwd: Path, env: dict[str, str], **kwargs: object) -> SimpleNamespace:
        captured.update(cwd=cwd, env=env)
        return SimpleNamespace(returncode=1, stderr="synthetic", stdout="")

    producer = tmp_path / "producer"
    fixture = Path("tests/fixtures/observability")
    producer.mkdir()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("BRAINLAYER_ENRICH_BACKEND", "ollama")
    monkeypatch.setattr(evaluator.subprocess, "run", fake_run)
    monkeypatch.setattr(evaluator, "load_golden", lambda *args, **kwargs: {})

    evaluator._run_case(evaluator.load_case("healthy-dev", evaluator.FIXTURES), fixture, Path("producer"), None)

    env = captured["env"]
    assert captured["cwd"] == producer.resolve()
    assert env["PYTHONPATH"] == str(producer.resolve() / "src")
    assert "BRAINLAYER_ENRICH_BACKEND" not in env


def test_main_rejects_producer_root_without_brainlayer_package(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        evaluator.sys,
        "argv",
        [
            "observability_eval.py",
            "--split",
            "dev",
            "--producer-root",
            str(tmp_path),
            "--baseline-sha",
            "test",
        ],
    )

    with pytest.raises(SystemExit, match="2"):
        evaluator.main()


def test_heldout_digest_changes_when_a_golden_changes(tmp_path: Path) -> None:
    cases = [{"split": "heldout", "golden": "golden/a.json"}]
    golden = tmp_path / "golden/a.json"
    golden.parent.mkdir()
    golden.write_text('{"answer": 1}\n', encoding="utf-8")
    sealed = evaluator._heldout_digest(tmp_path, cases)

    golden.write_text('{"answer": 2}\n', encoding="utf-8")
    assert evaluator._heldout_digest(tmp_path, cases) != sealed


def test_main_rejects_changed_heldout_before_grading(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cases = [{"case_id": "sealed", "split": "heldout", "golden": "golden/sealed.json"}]
    golden = tmp_path / "golden/sealed.json"
    golden.parent.mkdir()
    golden.write_text("{}\n", encoding="utf-8")
    (tmp_path / "cases.json").write_text(
        json.dumps({"heldout_goldens_sha256": "0" * 64, "cases": cases}), encoding="utf-8"
    )
    monkeypatch.setattr(
        evaluator.sys,
        "argv",
        [
            "observability_eval.py",
            "--split",
            "heldout",
            "--fixture-root",
            str(tmp_path),
            "--heldout-golden-root",
            str(tmp_path),
            "--baseline-sha",
            "test",
        ],
    )
    monkeypatch.setattr(evaluator, "_run_case", lambda *args: pytest.fail("grading began before seal validation"))

    with pytest.raises(SystemExit, match="2"):
        evaluator.main()


def test_main_requires_heldout_root_before_grading(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (tmp_path / "cases.json").write_text(
        json.dumps({"heldout_goldens_sha256": "0" * 64, "cases": [{"case_id": "sealed", "split": "heldout"}]}),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        evaluator.sys,
        "argv",
        ["observability_eval.py", "--split", "heldout", "--fixture-root", str(tmp_path), "--baseline-sha", "test"],
    )
    monkeypatch.setattr(evaluator, "_run_case", lambda *args: pytest.fail("grading began without a seal"))

    with pytest.raises(SystemExit, match="2"):
        evaluator.main()


def test_mock_green_detects_numbers_when_a_required_input_is_missing() -> None:
    case = evaluator.load_case("missing-log-dev")
    actual = evaluator.load_golden("healthy-dev")

    result = evaluator.grade_payload(
        case=case,
        actual=actual,
        expected=actual,
        opened_inputs=case["declared_inputs"],
    )

    assert any(item.startswith("MOCK_GREEN $.backups") for item in result.mock_green)


def test_traceability_requires_exact_declared_opened_input_set() -> None:
    case = evaluator.load_case("healthy-dev")
    expected = evaluator.load_golden("healthy-dev")
    declared = case["declared_inputs"]

    result = evaluator.grade_payload(
        case=case,
        actual=expected,
        expected=expected,
        opened_inputs=declared[:-1],
    )

    assert result.traceability == [f"missing opened input: {declared[-1]}"]
