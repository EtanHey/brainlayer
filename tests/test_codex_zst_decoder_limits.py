"""Decoder lifetime and resolver precedence over real compressed fixtures."""

import tracemalloc

from brainlayer.zstd_stream import ZstdDecodedFile, resolve_zstd
from tests.test_codex_zstd_stream import compress as _compress
from tests.test_codex_zstd_stream import rollout

compress = _compress


def test_decoder_memory_is_bounded_and_early_close_reaps_child(compress):
    path = compress("large.jsonl.zst", (b'{"message":"' + b"x" * 1000 + b'"}\n') * 25000)
    tracemalloc.start()
    with ZstdDecodedFile(path) as stream:
        total = sum(len(line) for line in stream)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    assert total > 24 * 1024 * 1024
    assert peak < 2 * 1024 * 1024
    with ZstdDecodedFile(path) as stream:
        assert stream.read(1) == b"{"
    assert stream.process.poll() is not None


def test_ingest_override_precedes_backup_override(compress, monkeypatch):
    path = compress("override.jsonl.zst", rollout())
    binary = resolve_zstd()
    monkeypatch.setenv("BRAINLAYER_BACKUP_ZSTD_PATH", "/missing/backup/zstd")
    monkeypatch.setenv("BRAINLAYER_INGEST_ZSTD_PATH", binary)
    with ZstdDecodedFile(path) as stream:
        assert b"native-id" in b"".join(stream)
