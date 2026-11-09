import sqlite3

import pytest

from brainlayer.ingest.codex_manifest import _committed, _read_entries
from tests.test_codex_zstd_stream import compress as _compress
from tests.test_codex_zstd_stream import rollout

compress = _compress


@pytest.mark.parametrize("suffix,reason", [(b"invalid\n", "malformed_record@")])
def test_json_record_errors_are_distinct_from_decoder_errors(compress, suffix, reason):
    path = compress("record.jsonl.zst", rollout() + suffix)
    receipt = dict(records=0, decoded_bytes=0)
    with pytest.raises(ValueError, match=reason):
        list(_read_entries(path, 1048576, receipt))


def test_reader_reports_decoded_size_and_native_metadata(compress):
    data = rollout()
    path = compress("total.jsonl.zst", data)
    receipt = dict(records=0, decoded_bytes=0)
    batches = list(_read_entries(path, len(data) + 1, receipt))
    assert receipt["decoded_bytes_total"] == len(data)
    assert batches[0][0]["codex_native_session_id"] == "native-id"
    assert "sessionId" not in batches[0][0]


def test_one_committed_connection_can_check_multiple_rows(tmp_path, monkeypatch):
    path = tmp_path / "fixture.db"
    with sqlite3.connect(path) as db:
        db.execute("CREATE TABLE chunks(id TEXT, content TEXT)")
        db.execute("INSERT INTO chunks VALUES ('one','content')")
    original = sqlite3.connect
    opened = []

    def connect(*args, **kwargs):
        opened.append(args[0])
        return original(*args, **kwargs)

    monkeypatch.setattr(sqlite3, "connect", connect)
    with _committed(path) as conn:
        assert conn.execute("SELECT id FROM chunks WHERE id = ?", ("one",)).fetchone()
        assert not conn.execute("SELECT id FROM chunks WHERE id = ?", ("two",)).fetchone()
    assert len(opened) == 1 and "mode=ro" in opened[0]
    with pytest.raises(sqlite3.ProgrammingError):
        conn.execute("SELECT 1")


def test_invalid_session_payload_is_a_record_error(compress):
    path = compress("bad-header.jsonl.zst", b'{"type":"session_meta","payload":"invalid"}\n')
    with pytest.raises(ValueError, match="session_meta payload is not an object"):
        list(_read_entries(path, 1048576, dict(records=0, decoded_bytes=0)))
