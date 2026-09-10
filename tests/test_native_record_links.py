import apsw
import pytest

from brainlayer.native_record_links import (
    MAX_PRESERVED_SEGMENT_BYTES,
    NativeRecordLinkConflict,
    UnsupportedNativeRecordProvider,
    catalog_native_record,
    ensure_native_record_links_schema,
    find_native_uuid_records,
)


def _conn():
    return apsw.Connection(":memory:")


def _record(**overrides):
    value = {
        "provider": "claude",
        "session_id": "session-a",
        "source_file": "/fixture/a.jsonl",
        "record_start_offset": 4,
        "record_end_offset": 44,
        "raw_record_hash": "a" * 64,
        "declared_headers": {
            "/uuid": "native-1",
            "/parentUuid": None,
            "/logicalParentUuid": {"unexpected": True},
            "/compactMetadata/preservedSegment": {"kind": "opaque"},
        },
    }
    value.update(overrides)
    return value


def test_filtered_shaped_record_has_no_chunk_dependency_and_explicit_parent_states():
    conn = _conn()
    ensure_native_record_links_schema(conn)
    result = catalog_native_record(conn, **_record())
    assert result.inserted is True
    row = conn.execute(
        "SELECT native_uuid, parent_uuid_state, parent_uuid_value, logical_parent_uuid_state, "
        "preserved_segment_json, preserved_segment_omitted FROM native_record_links"
    ).fetchone()
    assert row == ("native-1", "null", None, "invalid", '{"kind":"opaque"}', 0)


def test_exact_reinsert_is_idempotent_and_conflicting_metadata_refuses_overwrite():
    conn = _conn()
    ensure_native_record_links_schema(conn)
    first = catalog_native_record(conn, **_record())
    second = catalog_native_record(conn, **_record())
    assert second == first.__class__(record_key=first.record_key, inserted=False)
    changed = _record(declared_headers={"/uuid": "native-2"})
    with pytest.raises(NativeRecordLinkConflict):
        catalog_native_record(conn, **changed)
    assert conn.execute("SELECT native_uuid FROM native_record_links").fetchall() == [("native-1",)]


@pytest.mark.parametrize(
    ("headers", "expected"),
    [
        ({}, ("missing", None, "missing", None)),
        ({"/parentUuid": None, "/logicalParentUuid": "lp"}, ("null", None, "string", "lp")),
        ({"/parentUuid": "p", "/logicalParentUuid": ["bad"]}, ("string", "p", "invalid", None)),
    ],
)
def test_parent_missing_null_string_and_invalid_are_distinct(headers, expected):
    conn = _conn()
    ensure_native_record_links_schema(conn)
    data = _record(declared_headers=headers)
    catalog_native_record(conn, **data)
    assert (
        conn.execute(
            "SELECT parent_uuid_state, parent_uuid_value, logical_parent_uuid_state, logical_parent_uuid_value "
            "FROM native_record_links"
        ).fetchone()
        == expected
    )


def test_same_uuid_across_sources_remains_separate_and_lookup_is_namespaced():
    conn = _conn()
    ensure_native_record_links_schema(conn)
    catalog_native_record(conn, **_record())
    catalog_native_record(
        conn,
        **_record(
            source_file="/fixture/b.jsonl", record_start_offset=8, record_end_offset=48, raw_record_hash="b" * 64
        ),
    )
    rows = find_native_uuid_records(conn, provider="claude", session_id="session-a", native_uuid="native-1")
    assert [row["source_file"] for row in rows] == ["/fixture/a.jsonl", "/fixture/b.jsonl"]


def test_outer_rollback_removes_catalog_write():
    conn = _conn()
    ensure_native_record_links_schema(conn)
    conn.execute("BEGIN")
    catalog_native_record(conn, **_record())
    conn.execute("ROLLBACK")
    assert conn.execute("SELECT COUNT(*) FROM native_record_links").fetchone()[0] == 0


def test_multibyte_preserved_segment_is_omitted_at_byte_cap_and_provider_is_explicit():
    conn = _conn()
    ensure_native_record_links_schema(conn)
    headers = {"/compactMetadata/preservedSegment": "א" * (MAX_PRESERVED_SEGMENT_BYTES + 1)}
    catalog_native_record(conn, **_record(declared_headers=headers))
    assert conn.execute(
        "SELECT preserved_segment_json, preserved_segment_omitted FROM native_record_links"
    ).fetchone() == (None, 1)
    with pytest.raises(UnsupportedNativeRecordProvider):
        catalog_native_record(conn, **_record(provider="codex"))
