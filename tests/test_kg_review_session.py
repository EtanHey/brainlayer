"""Tests for kg_review_session — voice/visual flag-batch review session driver.

Pure-file driver: reads the flag-batch JSON, maintains the dashboard-compatible
kg-flag-decisions-v1 file, and has no DB access.
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
                {"id": "d2", "name": "Docker", "type": "tool", "chunks": 70},
                {"id": "d1", "name": "docker", "type": "tool", "chunks": 50},
            ],
        },
    ],
}

DECISIONS_SCHEMA = "kg-flag-decisions-v1"
V1_SOURCES = {"explicit", "rule", "voice", "voice-rule"}
RULE_SOURCES = {"rule", "voice-rule"}


def assert_v1(data):
    assert data["schema"] == DECISIONS_SCHEMA
    assert "version" not in data
    assert "decisions" not in data
    assert set(data) >= {"schema", "source", "rules", "per_category", "counts", "merge", "keep"}

    exported = data["merge"] + data["keep"]
    assert all(item["source"] in V1_SOURCES for item in exported)
    assert data["counts"]["merge_clusters"] == len(data["merge"])
    assert data["counts"]["rows_merged_away"] == sum(len(item["members"]) for item in data["merge"])
    assert data["counts"]["keep"] == len(data["keep"])
    assert data["counts"]["explicit"] == sum(1 for item in exported if item["source"] not in RULE_SOURCES)
    assert data["counts"]["by_rule"] == sum(1 for item in exported if item["source"] in RULE_SOURCES)
    for cat, row in data["per_category"].items():
        assert set(row) == {"total", "explicit", "by_rule", "undecided", "rule"}, cat
        assert row["explicit"] + row["by_rule"] + row["undecided"] == row["total"], cat


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
    assert {c["cluster_id"]: c["item_kind"] for c in clusters} == {
        "diagnosis-flag:agent c": "cluster",
        "case-only:agent": "cluster",
        "case-only:docker": "cluster",
    }


def test_next_undecided_is_idempotent_until_decision_or_skip(batch_file, decisions_file):
    first = next_undecided(batch_file, decisions_file, category="case-only")
    second = next_undecided(batch_file, decisions_file, category="case-only")
    assert first["cluster_id"] == "case-only:agent"
    assert second["cluster_id"] == "case-only:agent"

    record_decision(
        batch_file,
        decisions_file,
        cluster_id="case-only:agent",
        decision={"action": "skip", "note": "ask later", "source": "voice"},
    )

    c = next_undecided(batch_file, decisions_file, category="case-only")
    assert c["cluster_id"] == "case-only:docker"
    data = json.loads(decisions_file.read_text())
    assert data["merge"] == []
    assert data["keep"] == []
    assert data["skipped"][0]["stem"] == "agent"
    assert data["skipped"][0]["note"] == "ask later"
    assert data["per_category"]["case-only"]["undecided"] == 2
    assert_v1(data)


def test_record_decision_then_next_advances(batch_file, decisions_file):
    record_decision(
        batch_file,
        decisions_file,
        cluster_id="case-only:agent",
        decision={
            "action": "keep",
            "note": "topic tag vs concept, different things",
            "source": "voice",
        },
    )
    c = next_undecided(batch_file, decisions_file, category="case-only")
    assert c["cluster_id"] == "case-only:docker"
    data = json.loads(decisions_file.read_text())
    assert data["keep"][0]["source"] == "voice"
    assert data["keep"][0]["note"] == "topic tag vs concept, different things"
    assert "decided_at" in data["keep"][0]
    assert data["counts"]["keep"] == 1
    assert data["counts"]["explicit"] == 1
    assert_v1(data)


def test_record_decision_is_merge_safe_and_stamped(batch_file, decisions_file):
    record_decision(
        batch_file,
        decisions_file,
        cluster_id="case-only:docker",
        decision={
            "action": "merge",
            "canonical_id": "d1",
            "note": "lower-case is the alias I meant",
            "source": "voice",
        },
    )
    data = json.loads(decisions_file.read_text())
    d = data["merge"][0]
    assert d["canonical"]["id"] == "d1"  # user-named canonical overrides members[0]
    assert [m["id"] for m in d["members"]] == ["d2"]
    assert d["source"] == "voice"
    assert d["note"] == "lower-case is the alias I meant"
    assert "decided_at" in d
    # second writer updates a different cluster without clobbering
    record_decision(
        batch_file,
        decisions_file,
        cluster_id="case-only:agent",
        decision={"action": "keep", "source": "explicit"},
    )
    data = json.loads(decisions_file.read_text())
    assert {(d["category"], d["stem"]) for d in data["merge"]} == {("case-only", "docker")}
    assert {(d["category"], d["stem"]) for d in data["keep"]} == {("case-only", "agent")}
    assert_v1(data)


def test_record_merge_excludes_context_members_from_canonical_and_losers(tmp_path, decisions_file):
    batch = {
        "contested": [
            {
                "stem": "android eas",
                "size": 4,
                "members": [
                    {
                        "id": "ctx-android eas",
                        "name": "CONTESTED - judge said Technology",
                        "type": "context",
                        "chunks": 0,
                    },
                    {"id": "project-android", "name": "Android EAS", "type": "project", "chunks": 1},
                    {"id": "tool-android", "name": "Android EAS", "type": "tool", "chunks": 7},
                    {"id": "tech-android", "name": "Android EAS", "type": "technology", "chunks": 3},
                ],
            }
        ]
    }
    batch_file = tmp_path / "context-batch.json"
    batch_file.write_text(json.dumps(batch))

    record_decision(
        batch_file,
        decisions_file,
        cluster_id="contested:android eas",
        decision={"action": "merge", "source": "voice"},
    )

    data = json.loads(decisions_file.read_text())
    merge = data["merge"][0]
    assert merge["canonical"] == {"id": "tool-android", "name": "Android EAS", "type": "tool"}
    assert [member["id"] for member in merge["members"]] == ["tech-android", "project-android"]
    assert "ctx-android eas" not in json.dumps(merge)
    assert_v1(data)


def test_record_merge_rejects_context_canonical_override(tmp_path, decisions_file):
    batch = {
        "contested": [
            {
                "stem": "android eas",
                "size": 3,
                "members": [
                    {
                        "id": "ctx-android eas",
                        "name": "CONTESTED - judge said Technology",
                        "type": "context",
                        "chunks": 0,
                    },
                    {"id": "tool-android", "name": "Android EAS", "type": "tool", "chunks": 7},
                    {"id": "tech-android", "name": "Android EAS", "type": "technology", "chunks": 3},
                ],
            }
        ]
    }
    batch_file = tmp_path / "context-batch.json"
    batch_file.write_text(json.dumps(batch))

    with pytest.raises(ValueError, match="canonical override .* is not a real merge member"):
        record_decision(
            batch_file,
            decisions_file,
            cluster_id="contested:android eas",
            decision={"action": "merge", "source": "voice", "canonical_id": "ctx-android eas"},
        )


def test_record_merge_rejects_fewer_than_two_real_members(tmp_path, decisions_file):
    batch = {
        "contested": [
            {
                "stem": "android eas",
                "size": 2,
                "members": [
                    {
                        "id": "ctx-android eas",
                        "name": "CONTESTED - judge said Technology",
                        "type": "context",
                        "chunks": 0,
                    },
                    {"id": "tool-android", "name": "Android EAS", "type": "tool", "chunks": 7},
                ],
            }
        ]
    }
    batch_file = tmp_path / "context-batch.json"
    batch_file.write_text(json.dumps(batch))

    with pytest.raises(ValueError, match="needs at least two real merge members"):
        record_decision(
            batch_file,
            decisions_file,
            cluster_id="contested:android eas",
            decision={"action": "merge", "source": "voice"},
        )

    assert not decisions_file.exists()


def test_record_decision_validates_action(batch_file, decisions_file):
    with pytest.raises(ValueError):
        record_decision(
            batch_file,
            decisions_file,
            cluster_id="x:y",
            decision={"action": "explode", "source": "voice"},
        )


def test_mixed_decision_requires_member_map_and_exports_v1_gap(batch_file, decisions_file):
    with pytest.raises(ValueError):
        record_decision(
            batch_file,
            decisions_file,
            cluster_id="x:y",
            decision={"action": "mixed", "source": "voice"},
        )

    member_map = {"p1": "merge", "t1": "keep", "t2": "keep"}
    record_decision(
        batch_file,
        decisions_file,
        cluster_id="diagnosis-flag:agent c",
        decision={"action": "mixed", "members": member_map, "note": "person vs tools", "source": "voice"},
    )
    data = json.loads(decisions_file.read_text())
    assert data["merge"] == []
    assert data["keep"][0]["stem"] == "agent c"
    assert data["keep"][0]["note"].startswith('MIXED: {"p1": "merge", "t1": "keep", "t2": "keep"}')
    assert data["needs_v1_1"][0]["members"] == member_map
    assert data["needs_v1_1"][0]["note"] == "person vs tools"
    assert data["counts"]["keep"] == 1
    assert data["counts"]["explicit"] == 1
    assert_v1(data)


def test_speak_text_mentions_names_types_and_counts(batch_file):
    clusters = load_flag_batch(batch_file)
    cluster = [c for c in clusters if c["cluster_id"] == "diagnosis-flag:agent c"][0]
    text = speak_text(cluster)
    assert "Agent C" in text
    assert "person" in text and "tool" in text
    assert "3" in text  # member count or chunks


def test_apply_rule_bulk_decides_matching_undecided(batch_file, decisions_file):
    record_decision(
        batch_file,
        decisions_file,
        cluster_id="case-only:agent",
        decision={"action": "keep", "source": "voice"},
    )
    n = apply_rule(
        batch_file,
        decisions_file,
        rule={
            "match": {"category": "case-only"},
            "action": "merge",
            "note": "case-only variants merge, most chunks wins",
            "source": "voice",
        },
    )
    assert n == 1
    data = json.loads(decisions_file.read_text())
    docker = data["merge"][0]
    assert docker["canonical"]["id"] == "d2"  # members[0] from the flag batch
    assert docker["source"] == "voice-rule"
    assert docker["note"] == "case-only variants merge, most chunks wins"
    assert data["rules"] == {"case-only": "merge"}
    assert data["keep"][0]["stem"] == "agent"  # rule must NOT overwrite manual decisions
    assert data["per_category"]["case-only"] == {
        "total": 2,
        "explicit": 1,
        "by_rule": 1,
        "undecided": 0,
        "rule": "merge",
    }
    assert_v1(data)

    n2 = apply_rule(
        batch_file,
        decisions_file,
        rule={
            "match": {"category": "case-only"},
            "action": "keep",
            "source": "voice",
        },
    )
    assert n2 == 0
    data = json.loads(decisions_file.read_text())
    assert data["keep"][0]["stem"] == "agent"


def test_apply_rule_rejects_mixed_and_invalid_actions(batch_file, decisions_file):
    for action in ("mixed", "skip", "explode"):
        with pytest.raises(ValueError):
            apply_rule(
                batch_file,
                decisions_file,
                rule={"match": {"category": "case-only"}, "action": action, "source": "voice"},
            )
    # nothing persisted by rejected rules
    assert not decisions_file.exists()


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
        batch_file,
        decisions_file,
        cluster_id="case-only:agent",
        decision={"action": "keep", "source": "voice"},
    )
    s = stats(batch_file, decisions_file)
    assert s["per_category"]["case-only"]["total"] == 2
    assert s["per_category"]["case-only"]["explicit"] == 1
    assert s["per_category"]["case-only"]["undecided"] == 1
    assert s["per_category"]["diagnosis-flag"]["undecided"] == 1
    assert s["counts"]["keep"] == 1


def test_dashboard_export_round_trips_voice_decision_as_v1(batch_file, decisions_file):
    dashboard_export = {
        "schema": DECISIONS_SCHEMA,
        "source": "kg-phase1-flag-batch-2026-06-05",
        "rules": {},
        "per_category": {
            "diagnosis-flag": {"total": 1, "explicit": 0, "by_rule": 0, "undecided": 1, "rule": None},
            "case-only": {"total": 2, "explicit": 1, "by_rule": 0, "undecided": 1, "rule": None},
        },
        "counts": {"merge_clusters": 1, "rows_merged_away": 1, "keep": 0, "explicit": 1, "by_rule": 0},
        "dashboard_state": {"sample": True},
        "merge": [
            {
                "stem": "agent",
                "category": "case-only",
                "source": "explicit",
                "canonical": {"id": "a1", "name": "agent", "type": "topic"},
                "members": [{"id": "a2", "name": "Agent", "type": "concept"}],
                "dashboard_flag": "untouched",
            }
        ],
        "keep": [],
    }
    decisions_file.write_text(json.dumps(dashboard_export))

    record_decision(
        batch_file,
        decisions_file,
        cluster_id="diagnosis-flag:agent c",
        decision={"action": "keep", "note": "real person vs tools", "source": "voice"},
    )

    data = json.loads(decisions_file.read_text())
    assert data["schema"] == dashboard_export["schema"]
    assert data["source"] == dashboard_export["source"]
    assert data["rules"] == dashboard_export["rules"]
    assert data["dashboard_state"] == dashboard_export["dashboard_state"]
    assert data["merge"] == dashboard_export["merge"]
    assert data["keep"][0]["stem"] == "agent c"
    assert data["counts"] == {"merge_clusters": 1, "rows_merged_away": 1, "keep": 1, "explicit": 2, "by_rule": 0}
    assert_v1(data)


def test_voice_rewrite_preserves_unknown_dashboard_fields(batch_file, decisions_file):
    decisions_file.write_text(
        json.dumps(
            {
                "schema": DECISIONS_SCHEMA,
                "source": "kg-phase1-flag-batch-2026-06-05",
                "rules": {},
                "per_category": {},
                "counts": {},
                "merge": [],
                "keep": [
                    {
                        "stem": "agent",
                        "category": "case-only",
                        "source": "explicit",
                        "dashboard_row_id": "row-17",
                    }
                ],
            }
        )
    )

    record_decision(
        batch_file,
        decisions_file,
        cluster_id="case-only:agent",
        decision={"action": "keep", "note": "non-deterministic voice answer", "source": "voice"},
    )

    data = json.loads(decisions_file.read_text())
    assert data["keep"][0]["source"] == "voice"
    assert data["keep"][0]["note"] == "non-deterministic voice answer"
    assert data["keep"][0]["dashboard_row_id"] == "row-17"
    assert_v1(data)


def test_record_decision_stamps_schema_on_legacy_decisions_file(batch_file, decisions_file):
    decisions_file.write_text(
        json.dumps(
            {
                "source": "kg-phase1-flag-batch-2026-06-05",
                "rules": {},
                "per_category": {},
                "counts": {},
                "merge": [],
                "keep": [],
            }
        )
    )

    record_decision(
        batch_file,
        decisions_file,
        cluster_id="case-only:agent",
        decision={"action": "keep", "note": "legacy file upgrade", "source": "voice"},
    )

    data = json.loads(decisions_file.read_text())
    assert data["schema"] == DECISIONS_SCHEMA
    assert data["keep"][0]["note"] == "legacy file upgrade"
    assert_v1(data)


# --- speak_text v2: question items + TTS noise fixes (Etan session blockers) ---


def _q_cluster():
    return {
        "stem": "Q1 of 6 — invocable substrates",
        "category": "etan-queue",
        "members": [{"id": "q1", "name": "DICTIONARY QUESTION: are languages Tools?", "type": "question", "chunks": 0}],
    }


def _tool_cluster():
    return {
        "stem": "github",
        "category": "etan-queue",
        "members": [
            {"id": "a", "name": "github", "type": "tool", "chunks": 4},
            {"id": "b", "name": "GitHub", "type": "technology", "chunks": 2},
        ],
    }


def test_speak_question_item_uses_capture_verbatim_prompt():
    from brainlayer.kg_review_session import speak_text

    s = speak_text(_q_cluster())
    assert "Merge, keep separate" not in s
    assert "DICTIONARY QUESTION: are languages Tools?" in s
    assert "capture" in s.lower() and "verbatim" in s.lower()


def test_speak_does_not_respeak_identical_single_name():
    from brainlayer.kg_review_session import speak_text

    s = speak_text(_q_cluster())
    assert s.count("DICTIONARY QUESTION: are languages Tools?") == 1


def test_speak_drops_chunk_counts():
    from brainlayer.kg_review_session import speak_text

    s = speak_text(_tool_cluster())
    assert "chunks" not in s
    assert "Merge, keep separate, mixed, or skip?" in s
