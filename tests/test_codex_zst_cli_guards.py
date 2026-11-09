import pytest
from typer.testing import CliRunner

from brainlayer.cli import app
from brainlayer.watcher import OffsetRegistry
from tests.test_codex_zstd_stream import compress as _compress
from tests.test_codex_zstd_stream import rollout

compress = _compress


@pytest.mark.parametrize("held", [False, True])
def test_backfill_reports_every_unfinished_archive(compress, tmp_path, monkeypatch, held):
    archives = [
        compress(f".codex/sessions/a{i}.jsonl.zst", rollout(f"archive {i} prompt long enough " * 3)) for i in range(3)
    ]
    monkeypatch.setenv("BRAINLAYER_DB", str(tmp_path / "x.db"))
    monkeypatch.setenv("BRAINLAYER_QUEUE_DIR", str(tmp_path / "queue"))
    monkeypatch.setenv("BRAINLAYER_CODEX_ZST_ADMISSION", "bounded")
    monkeypatch.setenv("BRAINLAYER_CODEX_ZST_FILES_PER_CYCLE", "1")
    registry = tmp_path / "offsets.json"
    if held:
        saved = OffsetRegistry(registry)
        for path in archives:
            saved.set(str(path.with_suffix("")), 10, path.stat().st_ino)
        saved.flush()
    result = CliRunner().invoke(
        app, ["watch-backfill", "--home", str(tmp_path), "--registry", str(registry), "--max-cycles", "1"]
    )
    assert result.exit_code == (0 if held else 2), result.output
    assert f"held_archives={3 if held else 2}" in result.output
    assert f"represented_cold={3 if held else 0}" in result.output
    assert f"unadmitted={0 if held else 2}" in result.output


def test_missing_manifest_is_clean_cli_error(tmp_path):
    result = CliRunner().invoke(app, ["ingest-codex", "--from-manifest", str(tmp_path / "missing.jsonl")])
    assert result.exit_code == 2
    assert "manifest file not found" in result.output
    assert not isinstance(result.exception, FileNotFoundError)


def test_manifest_cli_emits_other_machine_and_invalid_receipts(tmp_path, monkeypatch):
    import json

    selected = tmp_path / "manifest.jsonl"
    selected.write_text('{"machine":"M1"}\n{"machine":"m4"}\n{}\n')
    monkeypatch.setenv("BRAINLAYER_DB", str(tmp_path / "fixture.db"))
    result = CliRunner().invoke(
        app, ["ingest-codex", "--from-manifest", str(selected), "--manifest-machine", "M4", "--dry-run"]
    )
    assert result.exit_code == 2, result.output
    rows = [json.loads(line) for line in result.output.splitlines()]
    assert len(rows) == 3
    assert rows[0]["state"] == "skipped_other_machine"
    assert all(row["reason"] == "invalid_machine" for row in rows[1:])
    assert all(row["live_chunks_at_start"] == row["duplicate_content"] == row["filtered"] == 0 for row in rows)


@pytest.mark.parametrize(
    "args,reason", [(["--dry-run", "--manifest-machine", "m4"], "must be M1 or M4"), ([], "--receipts-out is required")]
)
def test_manifest_options_fail_before_processing(tmp_path, args, reason):
    manifest = tmp_path / "empty.jsonl"
    manifest.write_text("")
    result = CliRunner().invoke(app, ["ingest-codex", "--from-manifest", str(manifest), *args])
    assert result.exit_code == 2 and reason in result.output


def test_receipts_cli_refuses_existing_file(tmp_path):
    manifest, ledger = tmp_path / "empty.jsonl", tmp_path / "ledger.jsonl"
    manifest.write_text("")
    ledger.write_text("retain me")
    result = CliRunner().invoke(app, ["ingest-codex", "--from-manifest", str(manifest), "--receipts-out", str(ledger)])
    assert result.exit_code == 2 and "File exists" in result.output
    assert ledger.read_text() == "retain me"


def test_backfill_completed_restart_exits_zero_without_decode(compress, tmp_path, monkeypatch):
    compress(".codex/sessions/done.jsonl.zst", rollout())
    monkeypatch.setenv("BRAINLAYER_DB", str(tmp_path / "fixture.db"))
    monkeypatch.setenv("BRAINLAYER_QUEUE_DIR", str(tmp_path / "queue"))
    monkeypatch.setenv("BRAINLAYER_CODEX_ZST_ADMISSION", "bounded")
    args = [
        "watch-backfill",
        "--home",
        str(tmp_path),
        "--registry",
        str(tmp_path / "offsets.json"),
        "--max-cycles",
        "10",
    ]
    assert CliRunner().invoke(app, args).exit_code == 0
    monkeypatch.setattr("brainlayer.codex_zst.ZstdDecodedFile", lambda *_: pytest.fail("completed archive decoded"))
    result = CliRunner().invoke(app, args[:-1] + ["1"])
    assert result.exit_code == 0, result.output
    assert "processed_entries=0" in result.output
