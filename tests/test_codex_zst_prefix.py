"""Decoded-prefix proofs use the same consumed-byte contract as plain tailers."""

import hashlib
import inspect
import json
import subprocess

import pytest

from brainlayer import watcher as watcher_module
from brainlayer.codex_zst import ZstdTailer
from brainlayer.watcher import JSONLTailer


def archive(tmp_path, data):
    path = tmp_path / "rollout.jsonl.zst"
    path.write_bytes(subprocess.check_output(["zstd", "-q", "-c"], input=data))
    return path


def fingerprint(data, offset):
    return watcher_module._source_fingerprint(
        offset, hashlib.blake2b(data[: min(4096, offset)], digest_size=16).digest(), data[max(0, offset - 256) : offset]
    )


def resumed(path, offset, data):
    if "resume_fingerprint" in inspect.signature(ZstdTailer).parameters:
        return ZstdTailer(str(path), offset=offset, resume_fingerprint=fingerprint(data, offset))
    tailer = ZstdTailer(str(path), offset=offset)
    tailer._resume_prefix = (hashlib.sha256(data[:offset]).hexdigest(), None)
    return tailer


def drain(tailer, budget=13):
    lines = []
    try:
        for _ in range(100):
            lines.extend(tailer.read_new_lines(max_bytes=budget))
            if tailer.last_error or tailer.decoded_eof:
                break
        return lines
    finally:
        tailer.close()


def test_decoded_resume_and_snapshot(tmp_path):
    data = b'{"id":1}\n{"id":2}\n'
    path = archive(tmp_path, data)
    tailer = resumed(path, 9, data)
    snapshot = tailer.snapshot()
    assert [r["id"] for r in drain(tailer)] == [2]
    assert tailer.fingerprint_at(len(data)) == fingerprint(data, len(data))
    tailer.restore_snapshot(snapshot)
    assert [r["id"] for r in drain(tailer)] == [2]


def test_mismatched_fingerprint_reconciles_from_zero(tmp_path):
    data = b'{"id":1}\n{"id":2}\n'
    tailer = resumed(archive(tmp_path, data), 9, b'{"id":0}\n')
    assert [r["id"] for r in drain(tailer)] == [1, 2]
    assert tailer.offset == len(data) and tailer.last_error is None
    assert tailer._archive_reconcile_through == 9


def test_legacy_checkpoint_replays_without_trusting_stat(tmp_path):
    data = b'{"id":1}\n{"id":2}\n'
    path = archive(tmp_path, data)
    tailer = ZstdTailer(str(path), offset=9)
    assert [r["id"] for r in drain(tailer)] == [1, 2]
    assert tailer.offset == 18 and tailer.last_error is None
    assert tailer._archive_reconcile_through == 9
    assert tailer.fingerprint_at(18) == fingerprint(data, 18)


@pytest.mark.parametrize("suffix", [b'{"id":2}', b"broken-tail"])
def test_final_bytes_enter_prefix_proof(tmp_path, suffix):
    data = b'{"id":1}\n' + suffix
    tailer = ZstdTailer(str(archive(tmp_path, data)))
    drain(tailer)
    if tailer.failed_record:
        tailer.discard_failed_record()
    assert tailer.offset == len(data)
    assert tailer.fingerprint_at(len(data)) == fingerprint(data, len(data))
    # Plain quarantine uses exactly the same digest feed, including the rejected bytes.
    plain = tmp_path / "plain.jsonl"
    plain.write_bytes(data)
    peer = JSONLTailer(str(plain))
    peer.read_new_lines()
    peer._consume_anchor(peer._buffer)
    assert peer._head_hash.digest() == tailer._head_hash.digest() and peer.anchor == tailer.anchor


def test_resume_verifies_head_sample_and_reconciles_mismatch(tmp_path):
    records = [json.dumps({"id": n, "text": "x" * 500}).encode() + b"\n" for n in range(15)]
    data = b"".join(records)
    changed = data.replace(b'"id": 0', b'"id": 9')
    tailer = resumed(archive(tmp_path, changed), len(data) - len(records[-1]), data)
    assert len(drain(tailer, budget=512)) == 15
    assert tailer.last_error is None
