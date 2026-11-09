import json
import uuid

import pytest

from brainlayer.ingest.codex_manifest import ingest_codex_manifest
from brainlayer.watcher import OffsetRegistry
from tests.test_codex_zst_manifest import manifest
from tests.test_codex_zst_watcher import watcher
from tests.test_codex_zstd_stream import compress as _compress
from tests.test_codex_zstd_stream import rollout

compress = _compress


def test_manifest_budgets_and_malformed_archive(compress, tmp_path, monkeypatch):
    monkeypatch.setenv("BRAINLAYER_QUEUE_DIR", str(tmp_path / "queue"))
    a = compress("a.jsonl.zst", rollout())
    b = compress("b.jsonl.zst", rollout())
    selected = manifest(tmp_path, [a, b])
    receipts = list(ingest_codex_manifest(selected, tmp_path / "db.sqlite", dry_run=True, max_files=1))
    assert receipts[0]["decodable"] is True
    assert receipts[1]["skip_reason"] == "admission-budget"
    receipt = list(ingest_codex_manifest(selected, tmp_path / "db.sqlite", dry_run=True, max_decoded_bytes=1))[0]
    assert receipt["decoded_bytes"] == 1 and receipt["skip_reason"] == "decoded-byte-budget"
    assert not (tmp_path / "queue").exists()
    a.write_bytes(b"bad archive")
    selected = manifest(tmp_path, [a])
    receipt = list(
        ingest_codex_manifest(selected, tmp_path / "db.sqlite", receipts_out=tmp_path / f"{uuid.uuid4().hex}.jsonl")
    )[0]
    assert "decode failed" in receipt["skip_reason"] and receipt["queued"] == 0


def test_manifest_cli_dry_run_is_read_only(compress, tmp_path, monkeypatch):
    from typer.testing import CliRunner

    from brainlayer.cli import app

    monkeypatch.setenv("BRAINLAYER_DB", str(tmp_path / "db.sqlite"))
    monkeypatch.setenv("BRAINLAYER_QUEUE_DIR", str(tmp_path / "queue"))
    selected = manifest(tmp_path, [compress("cli.jsonl.zst", rollout())])
    result = CliRunner().invoke(app, ["ingest-codex", "--from-manifest", str(selected), "--dry-run"])
    assert result.exit_code == 0, result.output
    receipt = json.loads(result.output)
    assert receipt["would_insert"] == 1 and receipt["queued"] == 0
    assert not (tmp_path / "offsets.json").exists()
    assert not (tmp_path / "queue").exists() and not (tmp_path / "db.sqlite").exists()


def test_manifest_missing_inode_and_byte_budget_never_reset_offsets(compress, tmp_path, monkeypatch):
    monkeypatch.setenv("BRAINLAYER_QUEUE_DIR", str(tmp_path / "queue"))
    archive = compress("checkpoint.jsonl.zst", rollout())
    selected = manifest(tmp_path, [archive])
    original = OffsetRegistry(tmp_path / "offsets.json")
    original.set(str(archive.with_suffix("")), 999999, 123)
    original.flush()
    before = (tmp_path / "offsets.json").read_bytes()
    receipt = list(
        ingest_codex_manifest(
            selected, tmp_path / "db.sqlite", max_decoded_bytes=1, receipts_out=tmp_path / "budget.jsonl"
        )
    )[0]
    assert receipt["skip_reason"] == "decoded-byte-budget"
    assert (tmp_path / "offsets.json").read_bytes() == before
    row = json.loads(selected.read_text())
    row["inode"] = None
    selected.write_text(json.dumps(row) + "\n")
    receipt = list(ingest_codex_manifest(selected, tmp_path / "db.sqlite", dry_run=True))[0]
    assert receipt["skip_reason"] == "missing-expected-inode" and receipt["decoded_bytes"] == 0
    assert (tmp_path / "offsets.json").read_bytes() == before


@pytest.mark.parametrize("damage", ["malformed", "truncated"])
def test_bad_archive_retains_checkpoint_and_surfaces_failure(compress, tmp_path, monkeypatch, damage):
    path = compress("bad.jsonl.zst", rollout())
    data = path.read_bytes()
    path.write_bytes(b"bad archive" if damage == "malformed" else data[:-4])
    instance, entries = watcher(tmp_path, monkeypatch)
    instance.poll_once()
    logical = str(path.with_suffix(""))
    assert logical in instance._file_ingestion_failures
    assert instance.registry.get(logical)[0] == 0
    assert not entries
    assert not instance._tailers[logical].decoded_eof
    instance.stop()


def test_one_shot_backfill_is_off_by_default_then_replays_once(compress, tmp_path, monkeypatch):
    from typer.testing import CliRunner

    from brainlayer.cli import app

    archive = compress(".codex/sessions/one.jsonl.zst", rollout())
    monkeypatch.setenv("BRAINLAYER_DB", str(tmp_path / "db.sqlite"))
    monkeypatch.setenv("BRAINLAYER_QUEUE_DIR", str(tmp_path / "queue"))
    monkeypatch.setenv("BRAINLAYER_CODEX_ZST_ADMISSION", "off")
    args = ["watch-backfill", "--home", str(tmp_path), "--registry", str(tmp_path / "offsets.json")]
    runner = CliRunner()
    off = runner.invoke(app, args + ["--dry-run"])
    assert off.exit_code == 0 and "candidate_files=0" in off.output
    monkeypatch.setenv("BRAINLAYER_CODEX_ZST_ADMISSION", "bounded")
    first = runner.invoke(app, args)
    assert first.exit_code == 0, first.output
    assert len(list((tmp_path / "queue").glob("*.jsonl"))) == 1
    second = runner.invoke(app, args)
    assert second.exit_code == 0, second.output
    assert len(list((tmp_path / "queue").glob("*.jsonl"))) == 1
    assert archive.exists()


def test_active_decoder_finishes_before_a_newer_archive_is_admitted(compress, tmp_path, monkeypatch):
    import os

    old = compress("old.jsonl.zst", rollout())
    os.utime(old, (1, 1))
    instance, _ = watcher(tmp_path, monkeypatch, cap=1, budget=1)
    instance.poll_once()
    newer = compress("new.jsonl.zst", rollout())
    instance.poll_once()
    assert str(newer.with_suffix("")) not in instance._tailers
    assert instance._tailers[str(old.with_suffix(""))].decoder.position == 2
    instance.stop()


def test_queue_backpressure_holds_archives(compress, tmp_path, monkeypatch):
    compress("held.jsonl.zst", rollout())
    instance, entries = watcher(tmp_path, monkeypatch)
    queue = tmp_path / "queue"
    queue.mkdir()
    for i in range(25):
        (queue / f"{i}.jsonl").write_text("{}\n")
    assert instance.poll_once() == 0
    assert not entries and str(tmp_path / "held.jsonl") not in instance._tailers


def test_backfill_budget_exit_reaps_retained_decoder(compress, tmp_path, monkeypatch):
    from typer.testing import CliRunner

    from brainlayer import codex_zst
    from brainlayer.cli import app

    compress(".codex/sessions/pending.jsonl.zst", rollout())
    monkeypatch.setenv("BRAINLAYER_DB", str(tmp_path / "db.sqlite"))
    monkeypatch.setenv("BRAINLAYER_QUEUE_DIR", str(tmp_path / "queue"))
    monkeypatch.setenv("BRAINLAYER_CODEX_ZST_ADMISSION", "bounded")
    monkeypatch.setenv("BRAINLAYER_CODEX_ZST_BYTES_PER_CYCLE", "1")
    original = codex_zst.ZstdDecodedFile
    decoders = []

    def capture(path):
        result = original(path)
        decoders.append(result)
        return result

    monkeypatch.setattr(codex_zst, "ZstdDecodedFile", capture)
    result = CliRunner().invoke(app, ["watch-backfill", "--home", str(tmp_path), "--max-cycles", "1"])
    assert result.exit_code == 2 and "held_archives=1" in result.output
    assert decoders and all(decoder.process.poll() is not None for decoder in decoders)


def test_manifest_real_cli_process_uses_only_fixture_queue(compress, tmp_path, monkeypatch):
    import os
    import subprocess
    import sys
    from pathlib import Path

    archive = compress("process.jsonl.zst", rollout())
    selected = manifest(tmp_path, [archive])
    monkeypatch.setenv("BRAINLAYER_DB", str(tmp_path / "process.db"))
    monkeypatch.setenv("BRAINLAYER_QUEUE_DIR", str(tmp_path / "queue"))
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join((str(Path(__file__).resolve().parents[1] / "src"), env.get("PYTHONPATH", "")))
    command = [sys.executable, "-m", "brainlayer", "ingest-codex", "--from-manifest", str(selected)]
    before = {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    dry = subprocess.run(command + ["--dry-run"], env=env, capture_output=True, text=True, timeout=30)
    assert dry.returncode == 0, dry.stderr
    assert json.loads(dry.stdout)["would_insert"] == 1
    assert {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()} == before
    real = subprocess.run(
        command + ["--receipts-out", str(tmp_path / "receipts.jsonl")],
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert real.returncode == 0, real.stderr
    assert json.loads(real.stdout)["queued"] == 1
    assert len(list((tmp_path / "queue").glob("*.jsonl"))) == 1
    assert not (tmp_path / "process.db").exists() and not (tmp_path / "offsets.json").exists()
    assert archive.read_bytes() == before[archive]


def test_malformed_header_is_reported_without_stopping_the_manifest(compress, tmp_path, monkeypatch):
    monkeypatch.setenv("BRAINLAYER_QUEUE_DIR", str(tmp_path / "queue"))
    bad = compress("bad-header.jsonl.zst", b'{"type":"session_meta","payload":"invalid"}\n')
    good = compress("valid-header.jsonl.zst", rollout())
    receipts = list(
        ingest_codex_manifest(
            manifest(tmp_path, [bad, good]), tmp_path / "db.sqlite", receipts_out=tmp_path / "header-receipts.jsonl"
        )
    )
    assert receipts[0]["skip_reason"] == "session_meta payload is not an object"
    assert receipts[0]["queued"] == 0
    assert receipts[1]["decodable"] is True and receipts[1]["queued"] == 1


def test_all_zst_mapping_owners_exist_at_final_stack_tip():
    import re
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    script = (root / "scripts/run_tests.sh").read_text()
    rows = "\n".join(
        line
        for line in script.splitlines()
        if "zst_tests=(" in line
        or "for rel in test_codex_zst_acceptance.py" in line
        or "for rel in test_t3_app_provenance.py" in line
    )
    owners = set(re.findall(r"\btest_[a-z0-9_]+\.py\b", rows))
    assert len(owners) >= 25
    assert not [name for name in sorted(owners) if not (root / "tests" / name).is_file()]
