from brainlayer.ingest.codex_manifest import _read_entries
from tests.test_codex_zstd_stream import compress as _compress
from tests.test_codex_zstd_stream import rollout

compress = _compress


def test_quarantined_final_record_keeps_complete_window(compress):
    data = rollout() + b'{"partial":'
    path = compress("tail.jsonl.zst", data)
    receipt = dict(records=0, decoded_bytes=0)
    entries = list(_read_entries(path, len(data) + 1, receipt))
    assert len(entries) == 1 and len(entries[0]) == 1
    assert receipt["quarantined_tail"] and receipt["decoded_bytes_total"] == len(data)
