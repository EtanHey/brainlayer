"""Deterministic confirmation interleavings and durable decoded-EOF guards (r7)."""

import json
import os

import pytest

from brainlayer import codex_zst
from tests.test_codex_zst_watcher import compress as _compress
from tests.test_codex_zst_watcher import rollout, watcher

compress = _compress


@pytest.mark.parametrize("budget,confirm_first", [(40, True), (40, False), (100000, False)])
def test_stale_confirmation_keeps_proven_checkpoint(compress, tmp_path, monkeypatch, budget, confirm_first):
    head = rollout().splitlines(keepends=True)[0]
    data = head + b"".join(rollout(f"record {n} long enough " * 10).splitlines(keepends=True)[1] for n in range(8))
    arch = compress("strand.jsonl.zst", data)
    key = str(arch.with_suffix(""))
    w, before = watcher(tmp_path, monkeypatch, max_lines=3)
    w.indexer.batch_size = 1000
    w.poll_once()
    if confirm_first:
        w.indexer.flush()
    confirmed = w.registry.get(key)[0]
    w.poll_once()  # unflushed records from the old tailer
    stale = list(w.indexer._buffer)
    hidden = arch.with_suffix(".hidden")
    arch.rename(hidden)
    w.poll_once()
    hidden.rename(arch)
    w._zst_limits = (2, budget)
    w.poll_once()  # recreated tailer cannot prove the old batch's end
    w._tailers[key].read_new_lines(max_lines=100, max_bytes=100000)
    w._advance_confirmed_batch({key: stale[-1]["_line_end_offset"]}, stale)
    assert w.registry.get(key)[0] == confirmed  # even after the new tailer catches up
    w.indexer.flush()
    assert w.registry.get(key)[0] == 0 or (
        w.registry.fingerprint(key) is not None and w.registry.archive(key) is not None
    )
    assert w.registry.get(key)[0] >= confirmed
    w.registry.flush()
    w.stop()
    peer, after = watcher(tmp_path, monkeypatch)
    for _ in range(12):
        peer.poll_once()
    assert peer.registry.get(key)[0] == len(data)
    assert len({e["_line_end_offset"] for e in before + after}) == 8
    assert not any(peer.archive_status()[k] for k in ("unadmitted", "incomplete", "failed"))
    peer.stop()


def test_completed_restart_never_decodes_and_stat_change_rechecks(compress, tmp_path, monkeypatch):
    paths = [compress(f"done{i}.jsonl.zst", rollout(f"done {i} " * 40)) for i in range(4)]
    w, entries = watcher(tmp_path, monkeypatch, cap=1)
    st = paths[-1].stat()
    marker = dict(size=st.st_size, mtime_ns=st.st_mtime_ns, ino=st.st_ino)
    w.registry.set(str(paths[-1].with_suffix("")), len(rollout("done 3 " * 40)), 0, archive=marker)
    accept = w.indexer.on_flush

    def unavailable(_):
        raise OSError("forced acknowledgement failure")

    w.indexer.on_flush = unavailable
    w.poll_once()
    assert w.archive_status()["incomplete"] == 1
    w.indexer.on_flush = accept
    w.indexer.flush()
    for _ in range(8):
        w.poll_once()
    assert len(entries) == 4
    w.registry.flush()
    saved = json.loads((tmp_path / "offsets.json").read_text())
    if hasattr(w.registry, "fingerprint"):
        fields = {"offset", "inode", "mtime", "generation", "fingerprint", "archive"}
        assert all(set(v) <= fields for k, v in saved.items() if not k.startswith("__"))
    w.stop()
    real, opened = codex_zst.ZstdDecodedFile, []
    monkeypatch.setattr(codex_zst, "ZstdDecodedFile", lambda path: opened.append(path) or real(path))
    peer, again = watcher(tmp_path, monkeypatch, cap=1)
    for _ in range(4):
        peer.poll_once()
    assert opened == [] and again == []
    assert not any(peer.archive_status()[k] for k in ("unadmitted", "incomplete", "failed"))
    path = paths[0]
    os.utime(path, ns=(path.stat().st_atime_ns, 1700000000000000000))
    peer.poll_once()
    assert opened == [path]
    opened.clear()
    # This changes mtime_ns while its floating-point mtime stays identical.
    os.utime(path, ns=(path.stat().st_atime_ns, 1700000000000000001))
    peer.poll_once()
    assert opened == [path]
    peer.stop()


def test_prefixless_marked_checkpoint_does_not_starve(compress, tmp_path, monkeypatch):
    paths = [compress(f"good{i}.jsonl.zst", rollout(f"good {i}")) for i in range(3)]
    bad = compress("legacy.jsonl.zst", rollout("legacy prefix"))
    os.utime(bad, (5000, 5000))
    for p in paths:
        os.utime(p, (1000, 1000))
    w, entries = watcher(tmp_path, monkeypatch, cap=1)
    st = bad.stat()
    w.registry.set(
        str(bad.with_suffix("")), 20, 0, archive=dict(size=st.st_size, mtime_ns=st.st_mtime_ns, ino=st.st_ino)
    )
    for _ in range(10):
        w.poll_once()
    assert {e["_source_file"] for e in entries} == {str(p.with_suffix("")) for p in [*paths, bad]}
    assert not any(w.archive_status()[k] for k in ("unadmitted", "incomplete", "failed"))
    w.stop()


def test_terminal_archive_rechecks_nanosecond_stat_change(compress, tmp_path, monkeypatch):
    path = compress("mismatch.jsonl.zst", rollout())
    path.write_bytes(b"not a zstd archive")
    os.utime(path, ns=(1700000000000000000, 1700000000000000000))
    w, _ = watcher(tmp_path, monkeypatch)
    st = path.stat()
    w.poll_once()
    real, opened = codex_zst.ZstdDecodedFile, []
    monkeypatch.setattr(codex_zst, "ZstdDecodedFile", lambda p: opened.append(p) or real(p))
    w.poll_once()
    assert opened == []
    os.utime(path, ns=(st.st_atime_ns, st.st_mtime_ns + 1))
    w.poll_once()
    assert opened == [path]
    w.stop()
