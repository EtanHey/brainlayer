import json
import sqlite3
import uuid

from brainlayer.ingest.codex_manifest import ingest_codex_manifest
from tests.test_codex_zst_manifest import manifest
from tests.test_codex_zstd_stream import compress as _compress
from tests.test_codex_zstd_stream import rollout

compress = _compress


def test_real_codex_v2_archive_dry_run_keeps_alarm_state(compress, tmp_path, monkeypatch):
    from brainlayer import t3_linkage_alarm

    native = "aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa"
    filename = "bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb"
    state = tmp_path / "statev2.sqlite"
    with sqlite3.connect(state) as db:
        db.execute(
            "CREATE TABLE orchestration_v2_projection_provider_threads (provider_thread_id TEXT, provider TEXT, payload_json TEXT, driver TEXT, provider_instance_id TEXT)"
        )
        db.execute(
            "INSERT INTO orchestration_v2_projection_provider_threads VALUES (?, ?, ?, ?, ?)",
            (
                "fixture",
                "codex",
                json.dumps({"nativeThreadRef": {"driver": "codex", "nativeId": native}}),
                "codex",
                "codex",
            ),
        )
    monkeypatch.setenv("BRAINLAYER_DB", str(tmp_path / "db.sqlite"))
    monkeypatch.setenv("BRAINLAYER_QUEUE_DIR", str(tmp_path / "queue"))
    monkeypatch.setenv("BRAINLAYER_T3_STATE_DB", str(state))
    monkeypatch.setenv("BRAINLAYER_T3_PROJECTION_VERSION", "2")
    records = [json.loads(line) for line in rollout().splitlines()]
    records[0]["payload"].update(id=native, originator="T3 Code")
    data = b"".join((json.dumps(row) + "\n").encode() for row in records)
    archive = compress(f".codex/sessions/rollout-{filename}.jsonl.zst", data)
    selected = manifest(tmp_path, [archive])
    alarm = t3_linkage_alarm._status_path(f"t3-runtime:2:{state.resolve()}")
    alarm.parent.mkdir(parents=True)
    alarm.write_text('{"state":"existing-alarm"}\n')
    before = {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    receipt = list(ingest_codex_manifest(selected, tmp_path / "db.sqlite", dry_run=True))[0]
    assert receipt["skip_reason"] == "" and receipt["would_insert"] == 1
    assert {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()} == before
    assert (
        list(
            ingest_codex_manifest(
                selected, tmp_path / "db.sqlite", receipts_out=tmp_path / f"{uuid.uuid4().hex}.receipts.jsonl"
            )
        )[0]["queued"]
        == 1
    )
    event = json.loads(next((tmp_path / "queue").glob("*.jsonl")).read_text())
    assert event["provenance_class"] == "t3-app-session"
    assert event["source_file"] == str(archive.with_suffix(""))
