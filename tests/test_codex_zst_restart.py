"""Only archives confirmed by bounded admission can resume after restart."""

from tests.test_codex_zst_watcher import compress as _compress
from tests.test_codex_zst_watcher import rollout, watcher

compress = _compress


def test_restart_continues_fingerprinted_archive(compress, tmp_path, monkeypatch):
    records = rollout("first prompt " * 8)
    data = records + rollout("second prompt " * 80).splitlines(keepends=True)[1]
    physical = compress("resume.jsonl.zst", data)
    logical = str(physical.with_suffix(""))
    w, before = watcher(tmp_path, monkeypatch, budget=len(records))
    w.poll_once()
    w.registry.flush()
    proof = w.registry.fingerprint(logical)
    assert proof is not None
    marker = getattr(w.registry, "archive", lambda _: None)(logical)
    w.stop()
    peer, after = watcher(tmp_path, monkeypatch, budget=17)
    peer.poll_once()
    peer.registry.flush()
    peer.stop()
    peer, after = watcher(tmp_path, monkeypatch, budget=17)
    for _ in range(200):
        peer.poll_once()
    assert len(before) == len(after) == 1
    assert marker == dict(
        size=physical.stat().st_size, mtime_ns=physical.stat().st_mtime_ns, ino=physical.stat().st_ino
    )
    assert after[0]["message"] != before[0]["message"]
    assert peer.registry.get(logical)[0] == len(data)
    assert peer.registry.fingerprint(logical) != proof
    assert peer.archive_status()["represented_cold"] == 0
    peer.stop()


def test_represented_cold_visible_in_health(compress, tmp_path, monkeypatch):
    path = compress("held.jsonl.zst", rollout())
    w, entries = watcher(tmp_path, monkeypatch)
    key = str(path.with_suffix(""))
    w.registry.set(key, 20, 123)
    w.poll_once()
    assert not entries and key not in w._tailers
    assert w.archive_status()["represented_cold"] == 1
    w.health_path = tmp_path / "health.json"
    w._write_health_snapshot(w._discover_jsonl_files())
    import json

    assert json.loads(w.health_path.read_text())["held_archives"] == 1
    w.stop()
