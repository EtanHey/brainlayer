"""Tests for kg_review_session — voice/visual flag-batch review session driver.

Pure-file driver: reads the flag-batch JSON, maintains a decisions JSON
(stub contract, merge-safe for two writers), no DB access.
"""

import json

import pytest

from brainlayer.kg_review_session import (
    apply_rule,
    cluster_id,
    load_flag_batch,
    next_undecided,
    record_decision,
    speak_text,
    stats,
)

FLAG_BATCH = {
    "diagnosis-flag": [
        {
            "stem": "agent c",
            "size": 3,
            "members": [
                {"id": "p1", "name": "Agent C", "type": "person", "chunks": 4},
                {"id": "t1", "name": "Agent C", "type": "tool", "chunks": 3},
                {"id": "t2", "name": "Agent-C", "type": "tool", "chunks": 2},
            ],
        },
    ],
    "case-only": [
        {
            "stem": "agent",
            "size": 2,
            "members": [
                {"id": "a1", "name": "agent", "type": "topic", "chunks": 643},
                {"id": "a2", "name": "Agent", "type": "concept", "chunks": 30},
            ],
        },
        {
            "stem": "docker",
            "size": 2,
            "members": [
                {"id": "d1", "name": "docker", "type": "tool", "chunks": 50},
                {"id": "d2", "name": "Docker", "type": "tool", "chunks": 70},
            ],
        },
    ],
}


@pytest.fixture
def batch_file(tmp_path):
    p = tmp_path / "flag-batch.json"
    p.write_text(json.dumps(FLAG_BATCH))
    return p


@pytest.fixture
def decisions_file(tmp_path):
    return tmp_path / "decisions.json"


def test_cluster_id_is_category_and_stem():
    assert cluster_id("case-only", "agent") == "case-only:agent"


def test_load_flag_batch_flattens_with_ids(batch_file):
    clusters = load_flag_batch(batch_file)
    ids = [c["cluster_id"] for c in clusters]
    assert "diagnosis-flag:agent c" in ids
    assert "case-only:docker" in ids
    assert all("category" in c and "members" in c for c in clusters)


def test_next_undecided_walks_category_in_order(batch_file, decisions_file):
    c = next_undecided(batch_file, decisions_file, category="case-only")
    assert c["cluster_id"] == "case-only:agent"


def test_record_decision_then_next_advances(batch_file, decisions_file):
    record_decision(
        decisions_file,
        cluster_id="case-only:agent",
        decision={
            "action": "keep_all",
            "note": "topic tag vs concept, different things",
            "source": "voice",
        },
    )
    c = next_undecided(batch_file, decisions_file, category="case-only")
    assert c["cluster_id"] == "case-only:docker"


def test_record_decision_is_merge_safe_and_stamped(decisions_file):
    record_decision(
        decisions_file,
        cluster_id="case-only:docker",
        decision={"action": "merge_all", "canonical_id": "d2", "source": "voice"},
    )
    data = json.loads(decisions_file.read_text())
    d = data["decisions"]["case-only:docker"]
    assert d["action"] == "merge_all"
    assert d["canonical_id"] == "d2"
    assert d["source"] == "voice"
    assert "decided_at" in d
    # second writer updates a different cluster without clobbering
    record_decision(
        decisions_file,
        cluster_id="case-only:agent",
        decision={"action": "keep_all", "source": "visual"},
    )
    data = json.loads(decisions_file.read_text())
    assert set(data["decisions"]) == {"case-only:docker", "case-only:agent"}


def test_record_decision_validates_action(decisions_file):
    with pytest.raises(ValueError):
        record_decision(
            decisions_file,
            cluster_id="x:y",
            decision={"action": "explode", "source": "voice"},
        )


def test_mixed_decision_requires_member_map(decisions_file):
    with pytest.raises(ValueError):
        record_decision(
            decisions_file,
            cluster_id="x:y",
            decision={"action": "mixed", "source": "voice"},
        )


def test_speak_text_mentions_names_types_and_counts(batch_file):
    clusters = load_flag_batch(batch_file)
    cluster = [c for c in clusters if c["cluster_id"] == "diagnosis-flag:agent c"][0]
    text = speak_text(cluster)
    assert "Agent C" in text
    assert "person" in text and "tool" in text
    assert "3" in text  # member count or chunks


def test_apply_rule_bulk_decides_matching_undecided(batch_file, decisions_file):
    n = apply_rule(
        batch_file,
        decisions_file,
        rule={
            "match": {"category": "case-only"},
            "action": "merge_all",
            "canonical": "most_chunks",
            "note": "case-only variants merge, most chunks wins",
            "source": "voice",
        },
    )
    assert n == 2
    data = json.loads(decisions_file.read_text())
    docker = data["decisions"]["case-only:docker"]
    assert docker["action"] == "merge_all"
    assert docker["canonical_id"] == "d2"  # 70 chunks > 50
    assert docker["source"] == "rule"
    assert data["rules"], "rule itself must be recorded"
    # rule must NOT overwrite an existing manual decision
    record_decision(
        decisions_file,
        cluster_id="diagnosis-flag:agent c",
        decision={"action": "keep_all", "source": "voice"},
    )
    n2 = apply_rule(
        batch_file,
        decisions_file,
        rule={
            "match": {"category": "diagnosis-flag"},
            "action": "merge_all",
            "canonical": "most_chunks",
            "source": "voice",
        },
    )
    assert n2 == 0


def test_apply_rule_rejects_mixed_and_invalid_actions(batch_file, decisions_file):
    # mixed cannot be bulk-applied: it requires a per-member map per cluster
    with pytest.raises(ValueError):
        apply_rule(
            batch_file,
            decisions_file,
            rule={"match": {"category": "case-only"}, "action": "mixed", "source": "voice"},
        )
    with pytest.raises(ValueError):
        apply_rule(
            batch_file,
            decisions_file,
            rule={"match": {"category": "case-only"}, "action": "explode", "source": "voice"},
        )
    # nothing persisted by rejected rules
    assert not decisions_file.exists() or json.loads(decisions_file.read_text())["decisions"] == {}


def test_cli_runs_from_plain_checkout_without_pythonpath(batch_file, decisions_file):
    import os
    import subprocess
    import sys as _sys
    from pathlib import Path

    script = Path(__file__).resolve().parents[1] / "scripts" / "kg_review_session.py"
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    result = subprocess.run(
        [
            _sys.executable,
            str(script),
            "stats",
            "--batch",
            str(batch_file),
            "--decisions",
            str(decisions_file),
        ],
        capture_output=True,
        text=True,
        env=env,
    )
    assert result.returncode == 0, result.stderr
    assert "case-only" in result.stdout


def test_stats_reports_progress(batch_file, decisions_file):
    record_decision(
        decisions_file,
        cluster_id="case-only:agent",
        decision={"action": "keep_all", "source": "voice"},
    )
    s = stats(batch_file, decisions_file)
    assert s["case-only"]["total"] == 2
    assert s["case-only"]["decided"] == 1
    assert s["diagnosis-flag"]["decided"] == 0
