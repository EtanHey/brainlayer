import json

import pytest

from brainlayer.alarm import BrainLayerAlarm
from brainlayer.t3_provenance import codex_header_identity
from tests.test_codex_zst_watcher import compress as _compress
from tests.test_codex_zst_watcher import watcher
from tests.test_codex_zstd_stream import rollout

compress = _compress


def test_header_retention_is_compact_and_pruned(compress, tmp_path, monkeypatch):
    native = "0193aaaa-bbbb-7ccc-8ddd-eeeeeeeeeeee"
    records = [json.loads(line) for line in rollout().splitlines()]
    records[0]["payload"].update(id=native, instructions="A" * 100000, originator="T3 Code")
    data = b"".join((json.dumps(r) + "\n").encode() for r in records)
    path = compress("header.jsonl.zst", data)
    instance, entries = watcher(tmp_path, monkeypatch)
    instance.poll_once()
    key = str(path.with_suffix(""))
    header = instance._codex_headers[key]
    assert len(repr(header)) < 300
    assert codex_header_identity(key, header=header) == (native, True)
    assert header.size == len((json.dumps(records[0]) + "\n").encode())
    assert entries[0]["_codex_header"] is header
    path.unlink()
    instance.poll_once()
    assert key not in instance._codex_headers
    instance.stop()


def test_compact_header_keeps_invalid_fork_decision(compress, tmp_path, monkeypatch):
    from brainlayer.t3_linkage_alarm import read_only

    records = [json.loads(line) for line in rollout().splitlines()]
    records[0]["payload"].update(
        id="0193aaaa-bbbb-7ccc-8ddd-eeeeeeeeeeee", session_id="0193aaaa-bbbb-7ccc-8ddd-ffffffffffff"
    )
    path = compress("bad-header.jsonl.zst", b"".join((json.dumps(r) + "\n").encode() for r in records))
    instance, entries = watcher(tmp_path, monkeypatch)
    instance.poll_once()
    with read_only(), pytest.raises(BrainLayerAlarm, match="session identity"):
        codex_header_identity(path.with_suffix(""), header=entries[0]["_codex_header"])
    instance.stop()


def test_restart_transition_replays_header_and_project(compress, tmp_path, monkeypatch):
    native = "0193aaaa-bbbb-7ccc-8ddd-eeeeeeeeeeee"
    records = [json.loads(line) for line in rollout().splitlines()]
    records[0]["payload"].update(id=native, originator="T3 Code")
    prefix = b"".join((json.dumps(r) + "\n").encode() for r in records)
    state = tmp_path / "state.sqlite"
    import sqlite3

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
    monkeypatch.setenv("BRAINLAYER_T3_STATE_DB", str(state))
    monkeypatch.setenv("BRAINLAYER_T3_PROJECTION_VERSION", "2")
    plain = tmp_path / ".codex/sessions/rollout.jsonl"
    plain.parent.mkdir(parents=True)
    plain.write_bytes(prefix)
    instance, _ = watcher(tmp_path, monkeypatch)
    instance.poll_once()
    instance.registry.flush()
    instance.stop()
    # Cold plain checkpoint is first restored, then Codex publishes its archive.
    resumed, entries = watcher(tmp_path, monkeypatch, budget=11)
    resumed.poll_once()
    tail = rollout("A distinct late user prompt with project metadata.").splitlines(keepends=True)[1]
    compress(".codex/sessions/rollout.jsonl.zst", prefix + tail)
    plain.unlink()
    for _ in range(200):
        resumed.poll_once()
    assert len(entries) == 1
    assert entries[0]["cwd"] == "/fixture/repo"
    assert codex_header_identity(plain, header=entries[0]["_codex_header"]) == (native, True)
    from brainlayer.t3_linkage_alarm import read_only
    from brainlayer.watcher_bridge import create_flush_callback

    rows = []
    monkeypatch.setattr("brainlayer.watcher_bridge.enqueue_watcher_chunk", lambda **row: rows.append(row))
    with read_only():
        result = create_flush_callback(tmp_path / "fixture.db", arbitrated=True)(entries)
    assert not result.deferred_entries
    assert len(rows) == 1 and rows[0]["project"] == "repo"
    assert rows[0]["provenance_class"] == "t3-app-session"
    resumed.stop()
