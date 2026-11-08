"""Archive header retention and decoder ownership."""

import json
import threading

import pytest

from tests.test_codex_zst_watcher import compress as _compress
from tests.test_codex_zst_watcher import watcher
from tests.test_codex_zstd_stream import rollout

compress = _compress


def test_off_plain_files_retain_no_headers(tmp_path, monkeypatch):
    header = {"type": "session_meta", "payload": {"id": "x", "cwd": "/r", "instructions": "A" * 20000}}
    for i in range(5):
        (tmp_path / f"s{i}.jsonl").write_bytes(
            (json.dumps(header) + "\n").encode() + rollout().splitlines(keepends=True)[1]
        )
    instance, _ = watcher(tmp_path, monkeypatch, enabled=False)
    instance.poll_once()
    assert not instance._codex_sessions
    assert not getattr(instance, "_codex_headers", {})
    assert all(getattr(t, "first_record", None) is None for t in instance._tailers.values())


def test_archive_retains_only_id_cwd_and_drops_with_tailer(compress, tmp_path, monkeypatch):
    data = rollout().replace(
        b'"cwd": "/fixture/repo"', b'"cwd": "/fixture/repo", "instructions": "' + b"A" * 20000 + b'"'
    )
    path = compress("archive.jsonl.zst", data)
    instance, entries = watcher(tmp_path, monkeypatch)
    instance.poll_once()
    key = str(path.with_suffix(""))
    assert instance._codex_sessions[key] == ("native-id", "/fixture/repo")
    assert "sessionId" not in entries[0]
    assert entries[0]["codex_native_session_id"] == "native-id"
    path.unlink()
    instance.poll_once()
    assert key not in instance._tailers
    assert key not in instance._codex_sessions
    assert key not in getattr(instance, "_codex_headers", {})
    instance.stop()


@pytest.mark.parametrize("drop", ["vanish", "denylist"])
def test_drop_closes_live_decoder(compress, tmp_path, monkeypatch, drop):
    path = compress("active.jsonl.zst", rollout())
    instance, _ = watcher(tmp_path, monkeypatch, budget=1)
    instance.poll_once()
    decoder = instance._tailers[str(path.with_suffix(""))].decoder
    if drop == "vanish":
        path.unlink()
    else:
        monkeypatch.setenv("BRAINLAYER_INGEST_DENYLIST", str(path))
    instance.poll_once()
    assert decoder.process.stdout.closed
    assert decoder.process.poll() is not None
    instance.stop()


def test_stop_waits_for_poll_thread_before_closing(compress, tmp_path, monkeypatch):
    compress("active.jsonl.zst", rollout())
    instance, _ = watcher(tmp_path, monkeypatch, budget=1)
    instance.poll_once()
    tailer = next(iter(instance._tailers.values()))
    original_close = tailer.close
    calls, errors = [], []
    entered, release = threading.Event(), threading.Event()

    def close():
        calls.append(threading.get_ident())
        original_close()

    def poll():
        entered.set()
        assert release.wait(5)

    def run():
        try:
            instance.start()
        except BaseException as error:
            errors.append(error)

    monkeypatch.setattr(tailer, "close", close)
    monkeypatch.setattr(instance, "poll_once", poll)
    thread = threading.Thread(target=run)
    thread.start()
    try:
        assert entered.wait(5)
        instance.stop()
        assert not calls
    finally:
        release.set()
        thread.join(5)
    assert not errors and not thread.is_alive()
    assert calls == [thread.ident]
