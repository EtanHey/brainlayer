import apsw
import pytest

from brainlayer.native_record_links import (
    NativeRecordLinkConflict,
    UnsupportedNativeRecordProvider,
    catalog_native_record,
    ensure_native_record_links_schema,
    find_native_uuid_records,
)


def conn():
    return apsw.Connection(":memory:")


def report(target="task-1"):
    return {
        "version": 1,
        "scope": "supported_tool_calls_v1",
        "targets": [
            {
                "tool_name": "mcp__codex_app__send_message_to_thread",
                "target": target,
                "target_kind": "task_handle",
                "declared_unverified": True,
                "call_id": "call-1",
                "raw_container_pointer": "/payload/arguments",
                "decoded_pointer": "/threadId",
                "pointer_kind": "decoded_json_string",
            }
        ],
        "diagnostics": ["declared only"],
    }


def record(**overrides):
    value = {
        "provider": "claude",
        "session_id": "s",
        "source_file": "/a.jsonl",
        "record_start_offset": 0,
        "record_end_offset": 10,
        "raw_record_hash": "a" * 64,
        "declared_headers": {"/uuid": "u", "/parentUuid": None},
    }
    value.update(overrides)
    return value


def test_prior_schema_migrates_and_v0_to_v1_preserves_physical_identity():
    db = conn()
    db.execute("""CREATE TABLE native_record_links (
        record_key TEXT PRIMARY KEY,provider TEXT NOT NULL,session_id TEXT NOT NULL,source_file TEXT NOT NULL,
        record_start_offset INTEGER NOT NULL,record_end_offset INTEGER NOT NULL,raw_record_hash TEXT NOT NULL,
        native_uuid TEXT,parent_uuid_state TEXT NOT NULL,parent_uuid_value TEXT,logical_parent_uuid_state TEXT NOT NULL,
        logical_parent_uuid_value TEXT,declared_headers_json TEXT NOT NULL,preserved_segment_json TEXT,
        preserved_segment_sha256 TEXT,preserved_segment_omitted INTEGER NOT NULL DEFAULT 0)""")
    ensure_native_record_links_schema(db)
    assert {"target_extraction_version", "declared_targets_json", "target_diagnostics_json"} <= {
        x[1] for x in db.execute("PRAGMA table_info(native_record_links)")
    }
    first = catalog_native_record(db, **record())
    upgraded = catalog_native_record(db, **record(), target_report=report())
    assert upgraded == first.__class__(first.record_key, False)
    assert db.execute("SELECT target_extraction_version FROM native_record_links").fetchone() == (1,)
    catalog_native_record(db, **record())  # Old writer does not erase v1 evidence.
    assert db.execute("SELECT declared_targets_json FROM native_record_links").fetchone()[0].find("task-1") >= 0


def test_replay_conflict_and_unknown_newer_version_refuse_overwrite():
    db = conn()
    ensure_native_record_links_schema(db)
    first = catalog_native_record(db, **record(), target_report=report())
    assert catalog_native_record(db, **record(), target_report=report()) == first.__class__(first.record_key, False)
    with pytest.raises(NativeRecordLinkConflict):
        catalog_native_record(db, **record(), target_report=report("task-2"))
    db.execute("UPDATE native_record_links SET target_extraction_version=2 WHERE record_key=?", (first.record_key,))
    with pytest.raises(NativeRecordLinkConflict):
        catalog_native_record(db, **record(), target_report=report())
    catalog_native_record(db, **record())  # Legacy caller remains non-destructive.


def test_only_helper_contract_fields_are_persisted_and_bounded():
    db = conn()
    ensure_native_record_links_schema(db)
    bad = report()
    bad["arguments"] = "private body"
    with pytest.raises(ValueError):
        catalog_native_record(db, **record(), target_report=bad)
    bad = report()
    bad["targets"][0]["declared_unverified"] = False
    with pytest.raises(ValueError):
        catalog_native_record(db, **record(), target_report=bad)


def test_provider_namespaces_and_claude_only_chain_lookup():
    db = conn()
    ensure_native_record_links_schema(db)
    claude = catalog_native_record(db, **record())
    codex = catalog_native_record(db, **record(provider="codex"))
    assert claude.record_key != codex.record_key
    assert (
        find_native_uuid_records(db, provider="claude", session_id="s", native_uuid="u")[0]["record_key"]
        == claude.record_key
    )
    with pytest.raises(UnsupportedNativeRecordProvider):
        find_native_uuid_records(db, provider="codex", session_id="s", native_uuid="u")


@pytest.mark.parametrize("overrides", [{"session_id": ""}, {"source_file": ""}, {"record_end_offset": 0}])
def test_explicit_missing_namespace_or_locator_refuses_record(overrides):
    db = conn()
    ensure_native_record_links_schema(db)
    with pytest.raises(ValueError):
        catalog_native_record(db, **record(**overrides))


def test_outer_rollback_leaves_no_physical_or_derived_evidence():
    db = conn()
    ensure_native_record_links_schema(db)
    db.execute("BEGIN")
    catalog_native_record(db, **record(), target_report=report())
    db.execute("ROLLBACK")
    assert db.execute("SELECT COUNT(*) FROM native_record_links").fetchone() == (0,)


@pytest.mark.parametrize("version", [True, 1.0, "1"])
def test_target_report_requires_exact_integer_v1(version):
    db = conn()
    ensure_native_record_links_schema(db)
    candidate = report()
    candidate["version"] = version
    with pytest.raises(ValueError):
        catalog_native_record(db, **record(), target_report=candidate)


def test_target_report_rejects_malformed_maps_and_helper_list_overflow():
    db = conn()
    ensure_native_record_links_schema(db)
    for key, value in (("target", None), ("tool_name", ""), ("target_kind", "unknown"), ("pointer_kind", "bad")):
        candidate = report()
        candidate["targets"][0][key] = value
        with pytest.raises(ValueError):
            catalog_native_record(db, **record(), target_report=candidate)
    candidate = report()
    candidate["targets"] *= 26
    with pytest.raises(ValueError):
        catalog_native_record(db, **record(), target_report=candidate)
    candidate = report()
    candidate["diagnostics"] *= 33
    with pytest.raises(ValueError):
        catalog_native_record(db, **record(), target_report=candidate)


def test_target_report_accepts_both_real_helper_pointer_shapes():
    db = conn()
    ensure_native_record_links_schema(db)
    candidate = report()
    candidate["targets"][0].update(
        target_kind="surface",
        raw_container_pointer=None,
        decoded_pointer="/message/content/0/input/surface",
        pointer_kind="direct_raw_json",
    )
    result = catalog_native_record(db, **record(), target_report=candidate)
    assert result.inserted
