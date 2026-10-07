import json
from copy import deepcopy

import pytest

from scripts.ci_ratchet_table import GREEN, NA, RED, row_watcher_heartbeat
from scripts.watcher_heartbeat_ratchet import ROOT, SOURCES, digest, process_declaration, store_content

HEAD = "a" * 40


@pytest.fixture
def evidence(tmp_path, monkeypatch):
    # Isolate this receipt unit fixture's Git identity; native proof never uses this patch.
    monkeypatch.setattr("scripts.brainbar_source_identity.source_identity", lambda _: {"head": HEAD, "dirty": False})
    artifacts = tmp_path / "heartbeat-evidence"
    artifacts.mkdir()
    (artifacts / "heartbeat-consumer").write_bytes(b"synthetic retained collector fixture")
    enum_file = artifacts / "ProcessEvidence.swift"
    enum_file.write_text(process_declaration())
    compile_sources = {
        p: digest(ROOT / p) for p in SOURCES if p.endswith(".swift") and not p.endswith("PipelineState.swift")
    }
    compile_sources["ProcessEvidence.swift"] = digest(enum_file)
    expected = dict.fromkeys(
        ["fresh", "refresh_recovery", "boundary_300", "future", "timezone", "scan_recovery"], "running"
    )
    expected.update(dict.fromkeys(["stale", "unrelated_store", "boundary_over_300", "scan_in_progress"], "degraded"))
    expected.update(missing="unknown", malformed="unknown", stopped="stopped")
    return {
        "schema": 2,
        "status": "measured",
        "head": HEAD,
        "sources": {p: digest(ROOT / p) for p in SOURCES},
        "binary_sha256": digest(artifacts / "heartbeat-consumer"),
        "compile_sources": compile_sources,
        "stores": [
            {
                "path": f"mcp-1-{i:032x}.jsonl",
                "event": {
                    "kind": "store_memory",
                    "source": "mcp",
                    "memory_type": "note",
                    "chunk_id": f"manual-{i:016x}",
                    "content": store_content(i),
                },
            }
            for i in range(4)
        ],
        "completed_health": {
            "poll_count": 3,
            "actual_poll_count": 3,
            "previous_poll_count": 2,
            "updated_at": "2026-10-07T10:00:02+00:00",
            "previous_updated_at": "2026-10-07T10:00:00+00:00",
            "completion_started_at": "2026-10-07T10:00:01+00:00",
            "completion_finished_at": "2026-10-07T10:00:03+00:00",
        },
        "cases": [{"name": k, "expected": v, "observed": v} for k, v in expected.items()],
        "producer_poll_counts": [1, 2],
        "watcher_queued_chunks": 1,
        "scan_progress": {"stat_calls": 1, "independent_stores": 3, "completed_poll": 3, "delay_seconds": 1.1},
    }


def verdict(tmp_path, data):
    path = tmp_path / "report.json"
    path.write_text(json.dumps(data))
    return row_watcher_heartbeat(path, None, HEAD).status


def test_complete_report_and_explicit_trigger_skip(tmp_path, evidence):
    assert verdict(tmp_path, evidence) == GREEN
    assert row_watcher_heartbeat(None, "unaffected paths", HEAD).status == NA
    assert row_watcher_heartbeat(None, None, HEAD).status == RED
    assert row_watcher_heartbeat(tmp_path / "absent", "irrelevant", HEAD).status == RED


@pytest.mark.parametrize(
    "fault",
    [
        "case_missing",
        "case_duplicate",
        "wrong_state",
        "wrong_expected",
        "head",
        "source",
        "unmeasured",
        "nonadvancing",
        "bool_count",
        "no_ingest",
        "binary_missing",
        "malformed",
        "scan_missing",
    ],
)
def test_incomplete_or_false_evidence_fails_closed(tmp_path, evidence, fault):
    data = deepcopy(evidence)
    if fault == "case_missing":
        data["cases"].pop()
    elif fault == "case_duplicate":
        data["cases"][-1] = data["cases"][0]
    elif fault in {"wrong_state", "wrong_expected"}:
        data["cases"][0]["observed" if fault == "wrong_state" else "expected"] = "degraded"
    elif fault == "head":
        data["head"] = "c" * 40
    elif fault == "source":
        data["sources"][SOURCES[0]] = "d" * 64
    elif fault == "unmeasured":
        data["status"] = "failed"
    elif fault == "nonadvancing":
        data["producer_poll_counts"] = [1, 1]
    elif fault == "bool_count":
        data["producer_poll_counts"] = [True, 2]
    elif fault == "no_ingest":
        data["watcher_queued_chunks"] = 0
    elif fault == "binary_missing":
        data.pop("binary_sha256")
    elif fault == "scan_missing":
        data.pop("scan_progress")
    else:
        data = []
    assert verdict(tmp_path, data) == RED


@pytest.mark.parametrize(
    "fault",
    [
        "discarded_stores",
        "duplicate_path",
        "duplicate_identity",
        "malformed_event",
        "wrong_content",
        "missing_event",
        "published_regression",
        "published_missing_count",
        "published_bool_count",
        "missing_timestamp",
        "regressed_timestamp",
        "future_timestamp",
        "naive_timestamp",
        "memory_disagreement",
        "fake_digest",
        "missing_artifact",
        "changed_artifact",
        "escaped_artifact",
        "changed_compile_source",
        "changed_enum",
        "claimed_store_count",
    ],
)
def test_r1_false_green_witnesses_fail_closed(tmp_path, evidence, fault):
    data = deepcopy(evidence)
    stores, completed = data["stores"], data["completed_health"]
    artifact = tmp_path / "heartbeat-evidence/heartbeat-consumer"
    if fault == "discarded_stores":
        data["stores"] = []
    elif fault == "duplicate_path":
        stores[2]["path"] = stores[1]["path"]
    elif fault == "duplicate_identity":
        stores[2]["event"]["chunk_id"] = stores[1]["event"]["chunk_id"]
    elif fault == "malformed_event":
        stores[1]["event"] = []
    elif fault == "wrong_content":
        stores[1]["event"]["content"] = "discarded"
    elif fault == "missing_event":
        stores.pop()
    elif fault == "published_regression":
        completed["poll_count"] = 2
    elif fault == "published_missing_count":
        completed.pop("poll_count")
    elif fault == "published_bool_count":
        completed["poll_count"] = True
    elif fault == "missing_timestamp":
        completed.pop("updated_at")
    elif fault == "regressed_timestamp":
        completed["updated_at"] = completed["previous_updated_at"]
    elif fault == "future_timestamp":
        completed["updated_at"] = "2026-10-08T10:00:02+00:00"
    elif fault == "naive_timestamp":
        completed["updated_at"] = "2026-10-07T10:00:02"
    elif fault == "memory_disagreement":
        completed["actual_poll_count"] = 4
    elif fault == "fake_digest":
        data["binary_sha256"] = "0" * 64
    elif fault == "missing_artifact":
        artifact.unlink()
    elif fault == "changed_artifact":
        artifact.write_bytes(b"substituted")
    elif fault == "escaped_artifact":
        outside = tmp_path.parent / (tmp_path.name + "-outside")
        outside.write_bytes(artifact.read_bytes())
        artifact.unlink()
        artifact.symlink_to(outside)
    elif fault == "changed_compile_source":
        data["compile_sources"]["ProcessEvidence.swift"] = "0" * 64
    elif fault == "changed_enum":
        (tmp_path / "heartbeat-evidence/ProcessEvidence.swift").write_text("wrong enum")
    elif fault == "claimed_store_count":
        data["scan_progress"]["independent_stores"] = 4
    assert verdict(tmp_path, data) == RED


@pytest.mark.parametrize("field,value", [("head", "b" * 40), ("dirty", True)])
def test_collector_checks_actual_checkout_identity(tmp_path, evidence, monkeypatch, field, value):
    identity = dict(head=HEAD, dirty=False)
    identity[field] = value
    monkeypatch.setattr("scripts.brainbar_source_identity.source_identity", lambda _: identity)
    assert verdict(tmp_path, evidence) == RED


def test_unreadable_git_identity_fails_closed(tmp_path, evidence, monkeypatch):
    def unavailable(_):
        raise RuntimeError("Git identity unavailable")

    monkeypatch.setattr("scripts.brainbar_source_identity.source_identity", unavailable)
    assert verdict(tmp_path, evidence) == RED
