from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


REPO = Path(__file__).resolve().parents[1]
FIXTURES = REPO / "tests/fixtures/observability"
OWNED_SECTIONS = ("stores", "emitters", "author_unknown")


def _dev_cases() -> list[dict[str, object]]:
    manifest = json.loads((FIXTURES / "cases.json").read_text(encoding="utf-8"))
    return [case for case in manifest["cases"] if case["split"] == "dev"]


def _run_case(case: dict[str, object], tmp_path: Path) -> tuple[dict[str, object], list[str]]:
    inputs = case["inputs"]
    assert isinstance(inputs, dict)
    output = tmp_path / "observability.json"
    trace = tmp_path / "trace.json"
    env = {
        key: os.environ[key]
        for key in ("HOME", "PATH", "BRAINLAYER_FORBID_EMBEDDING_MODEL")
        if key in os.environ
    }
    env.update(
        {
            "PYTHONPATH": str(REPO / "src"),
            "BRAINLAYER_DB": str(FIXTURES / str(inputs["db"])),
            "BRAINLAYER_OBSERVABILITY_PATH": str(output),
            "BRAINLAYER_OBSERVABILITY_TRACE_PATH": str(trace),
            "BRAINLAYER_OBSERVABILITY_INPUT_ROOT": str(FIXTURES),
            "BRAINLAYER_OBSERVABILITY_JSONL_BACKUP_LOG": str(FIXTURES / str(inputs["jsonl_backup_log"])),
            "BRAINLAYER_OBSERVABILITY_BACKUP_DAILY_LOG": str(FIXTURES / str(inputs["backup_daily_log"])),
            "BRAINLAYER_OBSERVABILITY_LAUNCHD_OUTPUT": str(FIXTURES / str(inputs["launchd_output"])),
            "BRAINLAYER_OBSERVABILITY_DISABLED_DIR": str(FIXTURES / str(inputs["disabled_dir"])),
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


def test_preview_is_secret_scrubbed_and_limited_to_80_chars(tmp_path: Path) -> None:
    import shutil
    import sqlite3

    case = next(case for case in _dev_cases() if case["case_id"] == "healthy-dev")
    db = tmp_path / "preview.sqlite"
    shutil.copy2(FIXTURES / str(case["inputs"]["db"]), db)
    secret = "sk-ant-" + "A" * 30
    connection = sqlite3.connect(db)
    connection.execute("UPDATE chunks SET content = ? WHERE id = 'synthetic-00'", (secret + "x" * 100,))
    connection.commit()
    connection.close()
    case = {**case, "inputs": {**case["inputs"], "db": str(db)}}

    actual, _ = _run_case(case, tmp_path)
    preview = actual["stores"]["latest"][0]["preview"]
    assert secret not in preview
    assert preview.startswith("[REDACTED:anthropic]")
    assert len(preview) <= 80


@pytest.mark.parametrize(
    ("source_file", "expected"),
    [
        ("/Users/x/.claude/projects/-Users-x-Gits-brainlayer/session.jsonl", "brainlayer"),
        ("/Users/x/.codex/sessions/2026/09/13/rollout.jsonl", "codex"),
        ("brainbar-store", "brainbar-store"),
        ("realtime-hook", "realtime-hook"),
        ("unknown", "unknown"),
        ("", "unknown"),
    ],
)
def test_source_file_emitter_derivation(source_file: str, expected: str) -> None:
    from brainlayer.observability_surface import derive_emitter

    assert derive_emitter(None, None, source_file) == (expected, "source_file")
