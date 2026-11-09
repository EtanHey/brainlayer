"""Terminal archive records are quarantinable without re-decoding."""

from brainlayer.codex_zst import ZstdTailer
from tests.test_codex_zstd_stream import compress as _compress
from tests.test_codex_zstd_stream import rollout

compress = _compress


def test_truncated_final_record_is_terminal_and_quarantinable(compress):
    path = compress("bad.jsonl.zst", rollout() + b'{"partial":')
    tailer = ZstdTailer(str(path))
    tailer.read_new_lines()
    assert tailer.decoded_eof
    assert tailer.terminal_failure == "truncated_final_record"
    assert tailer.failed_record == b'{"partial":'
    assert str(tailer.last_error).startswith("truncated_final_record")
    assert tailer.decoder is None
