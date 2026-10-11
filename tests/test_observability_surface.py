from __future__ import annotations

import builtins
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


def test_invalid_timestamps_are_counted_but_excluded_from_windowed_metrics(tmp_path: Path) -> None:
    case = next(case for case in _dev_cases() if case["case_id"] == "healthy-dev")
    db = tmp_path / "malformed-time.sqlite"
    shutil.copy2(FIXTURES / str(case["inputs"]["db"]), db)
    with sqlite3.connect(db) as connection:
        connection.execute("PRAGMA journal_mode=DELETE")
        connection.execute("UPDATE chunks SET created_at = NULL WHERE id = 'synthetic-00'")
        connection.execute("UPDATE chunks SET created_at = '2026-05-28T~12:35:00Z' WHERE id = 'synthetic-01'")
    actual, _ = _run_case({**case, "inputs": {**case["inputs"], "db": str(db)}}, tmp_path)
    for section in OWNED_SECTIONS:
        assert actual[section]["state"] == "measured"
        assert actual[section]["inputs"][0]["skipped_lines"] == 4
    assert actual["stores"]["total_chunks"] == 27
    assert actual["stores"]["in_window"]["count"] == 10
    assert all(item["chunk_id"] not in {"synthetic-00", "synthetic-01"} for item in actual["stores"]["latest"])


def test_missing_created_at_column_stays_fail_closed_with_document_and_trace(tmp_path: Path) -> None:
    case = next(case for case in _dev_cases() if case["case_id"] == "healthy-dev")
    db = tmp_path / "missing-created-at.sqlite"
    shutil.copy2(FIXTURES / str(case["inputs"]["db"]), db)
    with sqlite3.connect(db) as connection:
        connection.execute("PRAGMA journal_mode=DELETE")
        connection.execute("DROP INDEX idx_chunks_created")
        connection.execute("DROP INDEX idx_chunks_current_active")
        connection.execute("ALTER TABLE chunks DROP COLUMN created_at")
    actual, trace = _run_case({**case, "inputs": {**case["inputs"], "db": str(db)}}, tmp_path)
    assert any(item.endswith("missing-created-at.sqlite") for item in trace)
    for section in OWNED_SECTIONS:
        assert actual[section] == {
            "state": "unmeasurable",
            "reason": "required column missing: chunks.created_at",
            "inputs": actual[section]["inputs"],
        }


def test_latest_excludes_invalid_rows_even_when_they_would_fill_the_limit(tmp_path: Path) -> None:
    case = next(case for case in _dev_cases() if case["case_id"] == "healthy-dev")
    db = tmp_path / "latest-invalid.sqlite"
    shutil.copy2(FIXTURES / str(case["inputs"]["db"]), db)
    with sqlite3.connect(db) as connection:
        connection.execute("PRAGMA journal_mode=DELETE")
        connection.execute("UPDATE chunks SET created_at = NULL WHERE id NOT IN ('synthetic-00', 'synthetic-01')")
    actual, _ = _run_case({**case, "inputs": {**case["inputs"], "db": str(db)}}, tmp_path)
    assert [item["chunk_id"] for item in actual["stores"]["latest"]] == ["synthetic-00", "synthetic-01"]


@pytest.mark.parametrize("row_count", [0, 3, 600])
def test_latest_fetches_content_only_for_selected_rows(row_count: int) -> None:
    from datetime import UTC, timedelta

    from brainlayer.observability_surface import _stores

    now = datetime(2026, 10, 9, 1, 0, tzinfo=UTC)
    content_reads: list[str] = []
    with sqlite3.connect(":memory:") as connection:
        connection.execute(
            "CREATE TABLE items (id TEXT PRIMARY KEY, created_at TEXT, content TEXT, "
            "source_class TEXT, source TEXT, sender TEXT, source_file TEXT, content_class TEXT)"
        )
        connection.executemany(
            "INSERT INTO items VALUES (?, ?, ?, 'cli', 'codex', NULL, 'synthetic', 'user_message')",
            [
                (f"chunk-{index:04}", (now + timedelta(seconds=index)).isoformat(), f"payload {index}")
                for index in range(row_count)
            ],
        )

        def read_content(value: str) -> str:
            content_reads.append(value)
            return value

        # A view makes SQLite's payload evaluation observable without a timing threshold.
        # Ascending insertion puts a new row into the top-five sorter on every step.
        connection.create_function("read_content", 1, read_content)
        connection.execute(
            "CREATE VIEW chunks AS SELECT rowid, id, created_at, read_content(content) AS content, "
            "source_class, source, sender, source_file, content_class FROM items"
        )
        columns = {row[1] for row in connection.execute("PRAGMA table_info(chunks)")}
        stores = _stores(connection, columns, {}, now)

    selected = list(range(max(0, row_count - 5), row_count))[::-1]
    assert [row["chunk_id"] for row in stores["latest"]] == [f"chunk-{index:04}" for index in selected]
    assert [row["preview"] for row in stores["latest"]] == [f"payload {index}" for index in selected]
    assert len(content_reads) == len(selected)


def test_latest_keeps_timestamp_ties_and_nullable_ids() -> None:
    from datetime import UTC

    from brainlayer.observability_surface import _stores

    with sqlite3.connect(":memory:") as connection:
        connection.execute(
            "CREATE TABLE chunks (id TEXT PRIMARY KEY, created_at TEXT, content TEXT, "
            "source_class TEXT, source TEXT, sender TEXT, source_file TEXT, content_class TEXT)"
        )
        connection.executemany(
            "INSERT INTO chunks VALUES (?, ?, ?, 'cli', 'codex', NULL, 'synthetic', 'user_message')",
            [
                (None, "2026-10-09T03:00:00+02:00", "first nullable id"),
                (None, "2026-10-09T01:00:00Z", "second nullable id"),
                ("a", "2026-10-09T01:00:00Z", "tied a"),
                ("b", "2026-10-09T03:00:00+02:00", "tied b"),
                ("c", "2026-10-09T01:00:00Z", "tied c"),
                ("older", "2026-10-08T23:59:59Z", "excluded older"),
                ("invalid", "invalid", "excluded invalid"),
            ],
        )
        columns = {row[1] for row in connection.execute("PRAGMA table_info(chunks)")}
        stores = _stores(connection, columns, {}, datetime(2026, 10, 9, 2, 0, tzinfo=UTC))

    assert [row["chunk_id"] for row in stores["latest"]] == ["None", "None", "a", "b", "c"]
    assert {row["preview"] for row in stores["latest"]} == {
        "first nullable id",
        "second nullable id",
        "tied a",
        "tied b",
        "tied c",
    }
    assert {row["stored_at"] for row in stores["latest"]} == {"2026-10-09T01:00:00Z"}


def test_trace_is_written_when_build_fails(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    import brainlayer.observability_surface as surface

    case = next(case for case in _dev_cases() if case["case_id"] == "healthy-dev")
    db, trace = _stage_db(case, tmp_path), tmp_path / "trace.json"
    monkeypatch.setattr(surface, "_stores", lambda *_: (_ for _ in ()).throw(RuntimeError("boom")))
    document, _ = surface.build_document(env={"BRAINLAYER_DB": str(db), "BRAINLAYER_OBSERVABILITY_INPUT_ROOT": str(tmp_path / "inputs"), "BRAINLAYER_OBSERVABILITY_TRACE_PATH": str(trace), "BRAINLAYER_OBSERVABILITY_NOW": str(case["generated_at"])})  # fmt: skip
    assert document["stores"] == {
        "state": "unmeasurable",
        "reason": "stores raised RuntimeError: boom",
        "inputs": document["stores"]["inputs"],
    }
    trace_items = json.loads(trace.read_text())
    assert trace_items[0] == str(case["inputs"]["db"])
    assert len(trace_items) == 5


def test_transitive_backup_import_failure_names_the_missing_dependency(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import brainlayer.observability_surface as surface

    case = next(case for case in _dev_cases() if case["case_id"] == "healthy-dev")
    db = _stage_db(case, tmp_path)
    original_import = builtins.__import__

    def fail_backup_import(name, globals=None, locals=None, fromlist=(), level=0):
        if level == 1 and name == "observability_backup":
            raise ModuleNotFoundError("No module named 'idna'", name="idna")
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fail_backup_import)
    document, _ = surface.build_document(
        env={
            "BRAINLAYER_DB": str(db),
            "BRAINLAYER_OBSERVABILITY_INPUT_ROOT": str(tmp_path / "inputs"),
            "BRAINLAYER_OBSERVABILITY_NOW": str(case["generated_at"]),
        }
    )

    assert document["backups"] == {
        "state": "unmeasurable",
        "reason": "backups import failed: ModuleNotFoundError: No module named 'idna'",
        "inputs": [],
    }


def test_trace_only_input_is_excluded_from_recorder_section_inputs(tmp_path: Path) -> None:
    from brainlayer.observability_surface import InputRecorder

    recorder = InputRecorder(root=tmp_path, trace_path=None, now=datetime.now().astimezone())
    included = recorder(tmp_path / "missing")
    recorder(tmp_path, rows_or_bytes=0, in_section_inputs=False)
    assert recorder.section_inputs == [included]


def test_input_recorder_fails_closed_on_unreadable_file(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from brainlayer.observability_surface import InputRecorder

    path = tmp_path / "unreadable.log"
    path.write_text("receipt\n", encoding="utf-8")
    original_open = Path.open

    def refuse_open(self: Path, *args, **kwargs):
        if self == path:
            raise PermissionError("permission denied")
        return original_open(self, *args, **kwargs)

    monkeypatch.setattr(Path, "open", refuse_open)
    recorder = InputRecorder(root=tmp_path, trace_path=None, now=datetime.now().astimezone())
    item = recorder(path)
    assert item["status"] == "malformed"
    assert item["sha256_first_64kb"] is None
    assert item["rows_or_bytes"] is None
    assert recorder.section_inputs == [item]


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
