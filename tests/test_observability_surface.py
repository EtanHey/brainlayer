from __future__ import annotations

import json
import os
import shutil
import sqlite3
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import pytest
from typer.testing import CliRunner

REPO = Path(__file__).resolve().parents[1]
FIXTURES = REPO / "tests/fixtures/observability"
OWNED_SECTIONS = ("stores", "emitters", "author_unknown")
def _dev_cases() -> list[dict[str, object]]:
    manifest = json.loads((FIXTURES / "cases.json").read_text(encoding="utf-8"))
    return [case for case in manifest["cases"] if case["split"] == "dev"]
def _stage_db(case: dict[str, object], tmp_path: Path) -> Path:
    inputs = case["inputs"]
    assert isinstance(inputs, dict)
    source = FIXTURES / str(inputs["db"])
    relative = Path(str(inputs["db"])) if not Path(str(inputs["db"])).is_absolute() else Path("db") / source.name
    staged = tmp_path / "inputs" / relative
    staged.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, staged)
    fixed = int(datetime.fromisoformat(str(case["generated_at"]).replace("Z", "+00:00")).timestamp())
    os.utime(staged, (fixed, fixed))
    return staged
def _run_case(case: dict[str, object], tmp_path: Path) -> tuple[dict[str, object], list[str]]:
    inputs = case["inputs"]
    assert isinstance(inputs, dict)
    staged_db = _stage_db(case, tmp_path)
    output = tmp_path / "observability.json"
    trace = tmp_path / "trace.json"
    env = {key: os.environ[key] for key in ("HOME", "PATH", "BRAINLAYER_FORBID_EMBEDDING_MODEL") if key in os.environ}
    input_env = {"jsonl_backup_log": "BRAINLAYER_OBSERVABILITY_JSONL_BACKUP_LOG", "backup_daily_log": "BRAINLAYER_OBSERVABILITY_BACKUP_DAILY_LOG", "launchd_output": "BRAINLAYER_OBSERVABILITY_LAUNCHD_OUTPUT", "disabled_dir": "BRAINLAYER_OBSERVABILITY_DISABLED_DIR"}  # fmt: skip
    env.update({target: str(FIXTURES / str(inputs[source])) for source, target in input_env.items()})
    env.update(
        {
            "PYTHONPATH": str(REPO / "src"),
            "BRAINLAYER_DB": str(staged_db),
            "BRAINLAYER_OBSERVABILITY_PATH": str(output),
            "BRAINLAYER_OBSERVABILITY_TRACE_PATH": str(trace),
            "BRAINLAYER_OBSERVABILITY_INPUT_ROOT": str(tmp_path / "inputs"),
            "BRAINLAYER_OBSERVABILITY_NOW": str(case["generated_at"]),
        }
    )
    run = subprocess.run(
        [sys.executable, "-m", "brainlayer.observability_surface"],
        cwd=REPO,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert run.returncode == 0, run.stderr
    return json.loads(output.read_text(encoding="utf-8")), json.loads(trace.read_text(encoding="utf-8"))
@pytest.mark.parametrize("case", _dev_cases(), ids=lambda case: str(case["case_id"]))
def test_dev_goldens_for_owned_sections(case: dict[str, object], tmp_path: Path) -> None:
    actual, trace = _run_case(case, tmp_path)
    golden_path = FIXTURES / "golden" / f"{case['case_id']}.json"
    expected = json.loads(golden_path.read_text(encoding="utf-8"))
    for section in OWNED_SECTIONS:
        assert actual[section] == expected[section]
    assert actual["db_path"] == expected["db_path"]
    assert trace.count(str(case["inputs"]["db"])) == 1
@pytest.mark.parametrize(("secret", "prefix"), [("sk-ant-" + "A" * 30, "[REDACTED:anthropic]"), ("Q7mV2pL9xR4cT8nW3kY6dF1sH5jB", "[REDACTED:quarantined]")])  # fmt: skip
def test_preview_is_secret_scrubbed_and_limited_to_80_chars(tmp_path: Path, secret: str, prefix: str) -> None:
    case = next(case for case in _dev_cases() if case["case_id"] == "healthy-dev")
    db = tmp_path / "preview.sqlite"
    shutil.copy2(FIXTURES / str(case["inputs"]["db"]), db)
    connection = sqlite3.connect(db)
    connection.execute("PRAGMA journal_mode=DELETE")
    connection.execute("UPDATE chunks SET content = ? WHERE id = 'synthetic-00'", (secret + " " + "x" * 100,))
    connection.commit()
    connection.close()
    frozen = int(datetime.fromisoformat(str(case["generated_at"]).replace("Z", "+00:00")).timestamp())
    os.utime(db, (frozen, frozen))
    case = {**case, "inputs": {**case["inputs"], "db": str(db)}}

    actual, _ = _run_case(case, tmp_path)
    preview = actual["stores"]["latest"][0]["preview"]
    assert secret not in preview
    assert preview.startswith(prefix)
    assert len(preview) <= 80
def test_offset_timestamps_are_compared_as_utc_instants(tmp_path: Path) -> None:
    case = next(case for case in _dev_cases() if case["case_id"] == "healthy-dev")
    db = tmp_path / "offset.sqlite"
    shutil.copy2(FIXTURES / str(case["inputs"]["db"]), db)
    with sqlite3.connect(db) as connection:
        connection.execute("PRAGMA journal_mode=DELETE")
        connection.executemany("UPDATE chunks SET created_at = ? WHERE id = ?", [("2026-09-13T13:00:00+02:00", "synthetic-00"), ("2026-09-13T11:30:00Z", "synthetic-01"), ("2026-09-12T13:00:00+14:00", "synthetic-02")])  # fmt: skip
    case = {**case, "inputs": {**case["inputs"], "db": str(db)}}
    actual, _ = _run_case(case, tmp_path)
    assert actual["stores"]["latest"][0]["chunk_id"] == "synthetic-01"
    assert actual["stores"]["in_window"]["count"] == 11
    assert sum(item["count_in_window"] for item in actual["emitters"]["by_emitter"]) == 11
def test_malformed_timestamp_degrades_sections_and_names_row(tmp_path: Path) -> None:
    case = next(case for case in _dev_cases() if case["case_id"] == "healthy-dev")
    db = tmp_path / "malformed-time.sqlite"
    shutil.copy2(FIXTURES / str(case["inputs"]["db"]), db)
    with sqlite3.connect(db) as connection:
        connection.execute("PRAGMA journal_mode=DELETE")
        connection.execute("UPDATE chunks SET created_at = 'not-a-time' WHERE id = 'synthetic-00'")
    actual, _ = _run_case({**case, "inputs": {**case["inputs"], "db": str(db)}}, tmp_path)
    for section in OWNED_SECTIONS:
        assert actual[section]["state"] == "unmeasurable"
        assert "synthetic-00" in actual[section]["reason"]
def test_trace_is_written_when_build_fails(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    import brainlayer.observability_surface as surface

    case = next(case for case in _dev_cases() if case["case_id"] == "healthy-dev")
    db, trace = _stage_db(case, tmp_path), tmp_path / "trace.json"
    monkeypatch.setattr(surface, "_stores", lambda *_: (_ for _ in ()).throw(RuntimeError("boom")))
    with pytest.raises(RuntimeError, match="boom"):
        surface.build_document(env={"BRAINLAYER_DB": str(db), "BRAINLAYER_OBSERVABILITY_INPUT_ROOT": str(tmp_path / "inputs"), "BRAINLAYER_OBSERVABILITY_TRACE_PATH": str(trace), "BRAINLAYER_OBSERVABILITY_NOW": str(case["generated_at"])})  # fmt: skip
    assert json.loads(trace.read_text()) == [str(case["inputs"]["db"])]
def test_trace_only_input_is_excluded_from_recorder_section_inputs(tmp_path: Path) -> None:
    from brainlayer.observability_surface import InputRecorder

    recorder = InputRecorder(root=tmp_path, trace_path=None, now=datetime.now().astimezone())
    included = recorder(tmp_path / "missing")
    recorder(tmp_path, rows_or_bytes=0, in_section_inputs=False)
    assert recorder.section_inputs == [included]
@pytest.mark.parametrize(("source_file", "expected"), [("/Users/x/.claude/projects/-Users-x-Gits-brainlayer/session.jsonl", "brainlayer"), ("/Users/x/.codex/sessions/2026/09/13/rollout.jsonl", "codex"), ("brainbar-store", "brainbar-store"), ("realtime-hook", "realtime-hook"), ("unknown", "unknown"), ("", "unknown")])  # fmt: skip
def test_source_file_emitter_derivation(source_file: str, expected: str) -> None:
    from brainlayer.observability_surface import derive_emitter

    assert derive_emitter(None, None, source_file) == (expected, "source_file")
def test_cli_observability_stdout(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from brainlayer.cli import app

    case = next(case for case in _dev_cases() if case["case_id"] == "empty-db-dev")
    monkeypatch.setenv("BRAINLAYER_DB", str(_stage_db(case, tmp_path)))
    monkeypatch.setenv("BRAINLAYER_OBSERVABILITY_NOW", str(case["generated_at"]))
    result = CliRunner().invoke(app, ["observability", "--stdout"])
    assert result.exit_code == 0
    assert json.loads(result.stdout)["stores"]["total_chunks"] == 0
    monkeypatch.setenv("BRAINLAYER_OBSERVABILITY_PRODUCER_ROOT", str(REPO.parent))
    assert CliRunner().invoke(app, ["observability", "--stdout"]).exit_code == 1
