import json
import os
import subprocess
import sys
from copy import deepcopy
from pathlib import Path

import pytest

from brainlayer.kg_session_harvest import harvest_session

SESSION_BATCH = {
    "etan-queue": [
        {
            "stem": "Q1 of 6 - invocable substrates",
            "size": 1,
            "members": [
                {
                    "id": "q1",
                    "name": "DICTIONARY QUESTION: Tools or widen Technology?",
                    "type": "question",
                    "chunks": 0,
                }
            ],
        },
        {
            "stem": "Q2 of 6 - sprints and QA rounds",
            "size": 1,
            "members": [
                {
                    "id": "q2",
                    "name": "DICTIONARY QUESTION: Sprint, QA pass, or process?",
                    "type": "question",
                    "chunks": 0,
                }
            ],
        },
        {
            "stem": "Q3 of 6 - external media",
            "size": 1,
            "members": [{"id": "q3", "name": "DICTIONARY QUESTION: Media source?", "type": "question", "chunks": 0}],
        },
        {
            "stem": "Q4 of 6 - agents as subtype",
            "size": 1,
            "members": [{"id": "q4", "name": "DICTIONARY QUESTION: Agent subtype?", "type": "question", "chunks": 0}],
        },
        {
            "stem": "Q5 of 6 - orphaned persons",
            "size": 1,
            "members": [{"id": "q5", "name": "DICTIONARY QUESTION: Person fallback?", "type": "question", "chunks": 0}],
        },
        {
            "stem": "Q6 of 6 - concept demotion method",
            "size": 1,
            "members": [{"id": "q6", "name": "DICTIONARY QUESTION: Demote how?", "type": "question", "chunks": 0}],
        },
        {
            "stem": "android eas",
            "size": 4,
            "members": [
                {
                    "id": "ctx-android eas",
                    "name": "CONTESTED - judge said Technology; red-team refuted.",
                    "type": "context",
                    "chunks": 0,
                },
                {"id": "project-android", "name": "Android EAS", "type": "project", "chunks": 1},
                {"id": "tool-android", "name": "Android EAS", "type": "tool", "chunks": 7},
                {"id": "tech-android", "name": "Android EAS", "type": "technology", "chunks": 3},
            ],
        },
        {
            "stem": "agent mix",
            "size": 3,
            "members": [
                {"id": "ctx-agent mix", "name": "CONTESTED - mixed case.", "type": "context", "chunks": 0},
                {"id": "agent-tool", "name": "Agent Mix", "type": "tool", "chunks": 4},
                {"id": "agent-concept", "name": "Agent Mix", "type": "concept", "chunks": 2},
            ],
        },
    ]
}


def _write_json(path: Path, payload: dict) -> Path:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


@pytest.fixture
def batch_file(tmp_path: Path) -> Path:
    return _write_json(tmp_path / "etan-session.json", SESSION_BATCH)


@pytest.fixture
def decisions_file(tmp_path: Path) -> Path:
    answers = [
        {"stem": "Q1 of 6 - invocable substrates", "category": "etan-queue", "source": "voice", "note": "widen Technology", "decided_at": "2026-06-06T01:00:00Z"},
        {"stem": "Q2 of 6 - sprints and QA rounds", "category": "etan-queue", "source": "voice", "note": "process vocabulary", "decided_at": "2026-06-06T01:01:00Z"},
        {"stem": "Q3 of 6 - external media", "category": "etan-queue", "source": "voice", "note": "external-source", "decided_at": "2026-06-06T01:02:00Z"},
        {"stem": "Q4 of 6 - agents as subtype", "category": "etan-queue", "source": "voice", "note": "agent subtype", "decided_at": "2026-06-06T01:03:00Z"},
        {"stem": "Q5 of 6 - orphaned persons", "category": "etan-queue", "source": "voice", "note": "do not invent people", "decided_at": "2026-06-06T01:04:00Z"},
        {"stem": "Q6 of 6 - concept demotion method", "category": "etan-queue", "source": "voice", "note": "demote by evidence", "decided_at": "2026-06-06T01:05:00Z"},
    ]
    payload = {
        "schema": "kg-flag-decisions-v1",
        "source": "etan-session-2026-06-06",
        "rules": {},
        "per_category": {},
        "counts": {"merge_clusters": 3, "rows_merged_away": 6, "keep": 2, "explicit": 5, "by_rule": 0},
        "merge": [
            answers[1],
            {
                "stem": "android eas",
                "category": "etan-queue",
                "source": "voice",
                "canonical": {"id": "ctx-android eas", "name": "CONTESTED - judge said Technology", "type": "context"},
                "members": [
                    {"id": "project-android", "name": "Android EAS", "type": "project"},
                    {"id": "tool-android", "name": "Android EAS", "type": "tool"},
                    {"id": "tech-android", "name": "Android EAS", "type": "technology"},
                    {"id": "ctx-android eas", "name": "CONTESTED - judge said Technology", "type": "context"},
                ],
                "note": "merge the real Android EAS entries",
                "decided_at": "2026-06-06T01:06:00Z",
            },
        ],
        "keep": [
            answers[0],
            answers[2],
            answers[3],
            answers[4],
            answers[5],
            {
                "stem": "agent mix",
                "category": "etan-queue",
                "source": "voice",
                "note": 'MIXED: {"agent-concept": "keep", "agent-tool": "merge", "ctx-agent mix": "keep"}; split tool from concept',
                "decided_at": "2026-06-06T01:07:00Z",
            },
        ],
        "needs_v1_1": [
            {
                "stem": "agent mix",
                "category": "etan-queue",
                "source": "voice",
                "members": {"ctx-agent mix": "keep", "agent-tool": "merge", "agent-concept": "keep"},
                "note": "split tool from concept",
                "decided_at": "2026-06-06T01:07:00Z",
            }
        ],
    }
    return _write_json(tmp_path / "session-decisions.json", payload)


def _contains_ctx(value: object) -> bool:
    if isinstance(value, dict):
        return any(_contains_ctx(k) or _contains_ctx(v) for k, v in value.items())
    if isinstance(value, list):
        return any(_contains_ctx(item) for item in value)
    return isinstance(value, str) and value.startswith("ctx-")


def test_harvest_splits_question_answers_by_stem_not_action(batch_file: Path, decisions_file: Path, tmp_path: Path):
    answers_path = tmp_path / "answers.md"
    clean_path = tmp_path / "clean-decisions.json"

    result = harvest_session(batch_file, decisions_file, answers_path=answers_path, decisions_path=clean_path)

    answers_json = json.loads((tmp_path / "answers.json").read_text(encoding="utf-8"))
    assert [row["stem"] for row in answers_json] == [
        "Q1 of 6 - invocable substrates",
        "Q2 of 6 - sprints and QA rounds",
        "Q3 of 6 - external media",
        "Q4 of 6 - agents as subtype",
        "Q5 of 6 - orphaned persons",
        "Q6 of 6 - concept demotion method",
    ]
    assert answers_json[1]["question"] == "DICTIONARY QUESTION: Sprint, QA pass, or process?"
    assert answers_json[1]["answer"] == "process vocabulary"
    assert result["answers_count"] == 6

    clean = json.loads(clean_path.read_text(encoding="utf-8"))
    clean_stems = {item["stem"] for item in clean["merge"] + clean["keep"]}
    assert "Q1 of 6 - invocable substrates" not in clean_stems
    assert "Q2 of 6 - sprints and QA rounds" not in clean_stems
    assert "android eas" in clean_stems


def test_harvest_strips_ctx_members_and_rederives_canonical_from_chunk_count(
    batch_file: Path, decisions_file: Path, tmp_path: Path
):
    clean_path = tmp_path / "clean-decisions.json"

    harvest_session(batch_file, decisions_file, answers_path=tmp_path / "answers.md", decisions_path=clean_path)

    clean = json.loads(clean_path.read_text(encoding="utf-8"))
    android = clean["merge"][0]
    assert android["canonical"] == {"id": "tool-android", "name": "Android EAS", "type": "tool"}
    assert [member["id"] for member in android["members"]] == ["project-android", "tech-android"]
    assert not _contains_ctx(clean)


def test_harvest_filters_mixed_member_maps(batch_file: Path, decisions_file: Path, tmp_path: Path):
    clean_path = tmp_path / "clean-decisions.json"

    harvest_session(batch_file, decisions_file, answers_path=tmp_path / "answers.md", decisions_path=clean_path)

    clean = json.loads(clean_path.read_text(encoding="utf-8"))
    assert clean["needs_v1_1"][0]["members"] == {"agent-tool": "merge", "agent-concept": "keep"}
    assert not _contains_ctx(clean["needs_v1_1"])


def test_harvest_allows_non_id_note_text_containing_ctx_token(
    batch_file: Path, decisions_file: Path, tmp_path: Path
):
    data = json.loads(decisions_file.read_text(encoding="utf-8"))
    for item in data["keep"]:
        if item["stem"] == "agent mix":
            item["note"] = "MIXED: {\"agent-concept\": \"keep\", \"agent-tool\": \"merge\"}; compare ctx-local wording"
    decisions_file.write_text(json.dumps(data), encoding="utf-8")

    harvest_session(
        batch_file,
        decisions_file,
        answers_path=tmp_path / "answers.md",
        decisions_path=tmp_path / "clean.json",
    )


def test_harvest_clean_decisions_round_trip_through_apply_validator(
    batch_file: Path, decisions_file: Path, tmp_path: Path
):
    from scripts.kg_cleanup_apply import load_decisions

    clean_path = tmp_path / "clean-decisions.json"

    harvest_session(batch_file, decisions_file, answers_path=tmp_path / "answers.md", decisions_path=clean_path)

    clean = load_decisions(clean_path)
    assert clean["counts"] == {
        "merge_clusters": 1,
        "rows_merged_away": 2,
        "keep": 1,
        "explicit": 2,
        "by_rule": 0,
    }


def test_harvest_fails_loud_on_unknown_decision_stem(batch_file: Path, decisions_file: Path, tmp_path: Path):
    data = json.loads(decisions_file.read_text(encoding="utf-8"))
    data["keep"].append({"stem": "missing stem", "category": "etan-queue", "source": "voice"})
    decisions_file.write_text(json.dumps(data), encoding="utf-8")

    with pytest.raises(ValueError, match="unknown decision stem"):
        harvest_session(batch_file, decisions_file, answers_path=tmp_path / "answers.md", decisions_path=tmp_path / "clean.json")


def test_harvest_fails_loud_on_non_object_decision_item(
    batch_file: Path, decisions_file: Path, tmp_path: Path
):
    data = json.loads(decisions_file.read_text(encoding="utf-8"))
    data["merge"].append("not an object")
    decisions_file.write_text(json.dumps(data), encoding="utf-8")

    with pytest.raises(ValueError, match="decision item in 'merge'"):
        harvest_session(
            batch_file,
            decisions_file,
            answers_path=tmp_path / "answers.md",
            decisions_path=tmp_path / "clean.json",
        )


def test_harvest_fails_loud_on_non_object_decisions_file(
    batch_file: Path, tmp_path: Path
):
    decisions_path = tmp_path / "bad-decisions.json"
    decisions_path.write_text(json.dumps(["not", "an", "object"]), encoding="utf-8")

    with pytest.raises(ValueError, match="session decisions must be a JSON object"):
        harvest_session(
            batch_file,
            decisions_path,
            answers_path=tmp_path / "answers.md",
            decisions_path=tmp_path / "clean.json",
        )


def test_harvest_fails_loud_on_malformed_mixed_note_with_ctx_member(
    batch_file: Path, decisions_file: Path, tmp_path: Path
):
    data = json.loads(decisions_file.read_text(encoding="utf-8"))
    for item in data["keep"]:
        if item["stem"] == "agent mix":
            item["note"] = 'MIXED: {"ctx-agent mix": "keep"'
    decisions_file.write_text(json.dumps(data), encoding="utf-8")

    with pytest.raises(ValueError, match="MIXED note payload must be valid JSON"):
        harvest_session(
            batch_file,
            decisions_file,
            answers_path=tmp_path / "answers.md",
            decisions_path=tmp_path / "clean.json",
        )


def test_harvest_fails_loud_when_merge_selects_no_real_losers(
    batch_file: Path, decisions_file: Path, tmp_path: Path
):
    data = json.loads(decisions_file.read_text(encoding="utf-8"))
    for item in data["merge"]:
        if item["stem"] == "android eas":
            item["members"] = [
                {"id": "ctx-android eas", "name": "CONTESTED - judge said Technology", "type": "context"}
            ]
    decisions_file.write_text(json.dumps(data), encoding="utf-8")

    with pytest.raises(ValueError, match="selects no real members"):
        harvest_session(
            batch_file,
            decisions_file,
            answers_path=tmp_path / "answers.md",
            decisions_path=tmp_path / "clean.json",
        )


def test_harvest_keys_clusters_by_category_and_stem(
    decisions_file: Path, tmp_path: Path
):
    batch = deepcopy(SESSION_BATCH)
    batch["other-queue"] = [
        {
            "stem": "android eas",
            "size": 2,
            "members": [
                {"id": "ctx-other android eas", "name": "Other contested context", "type": "context", "chunks": 0},
                {"id": "other-android", "name": "Android EAS", "type": "project", "chunks": 9},
            ],
        }
    ]
    batch_path = _write_json(tmp_path / "duplicate-stems.json", batch)
    clean_path = tmp_path / "clean.json"

    harvest_session(
        batch_path,
        decisions_file,
        answers_path=tmp_path / "answers.md",
        decisions_path=clean_path,
    )

    clean = json.loads(clean_path.read_text(encoding="utf-8"))
    assert clean["merge"][0]["category"] == "etan-queue"
    assert clean["merge"][0]["canonical"]["id"] == "tool-android"
    assert all(member["id"] != "other-android" for member in clean["merge"][0]["members"])


def test_harvest_per_category_counts_skipped_and_undecided_non_question_clusters(
    decisions_file: Path, tmp_path: Path
):
    batch = deepcopy(SESSION_BATCH)
    batch["etan-queue"].extend(
        [
            {
                "stem": "manual skip",
                "size": 2,
                "members": [
                    {"id": "ctx-manual skip", "name": "Skip context", "type": "context", "chunks": 0},
                    {"id": "skip-real", "name": "Manual Skip", "type": "concept", "chunks": 1},
                ],
            },
            {
                "stem": "still undecided",
                "size": 2,
                "members": [
                    {"id": "ctx-still undecided", "name": "Undecided context", "type": "context", "chunks": 0},
                    {"id": "undecided-real", "name": "Still Undecided", "type": "concept", "chunks": 1},
                ],
            },
        ]
    )
    data = json.loads(decisions_file.read_text(encoding="utf-8"))
    data["skipped"] = [
        {"stem": "manual skip", "category": "etan-queue", "source": "voice", "decided_at": "2026-06-06T01:08:00Z"}
    ]
    batch_path = _write_json(tmp_path / "partial-session.json", batch)
    decisions_path = _write_json(tmp_path / "partial-decisions.json", data)
    clean_path = tmp_path / "clean.json"

    harvest_session(
        batch_path,
        decisions_path,
        answers_path=tmp_path / "answers.md",
        decisions_path=clean_path,
    )

    clean = json.loads(clean_path.read_text(encoding="utf-8"))
    assert clean["per_category"]["etan-queue"] == {
        "total": 4,
        "explicit": 2,
        "by_rule": 0,
        "undecided": 2,
        "rule": None,
    }
    assert clean["counts"] == {
        "merge_clusters": 1,
        "rows_merged_away": 2,
        "keep": 1,
        "explicit": 2,
        "by_rule": 0,
    }


def test_harvest_fails_loud_on_malformed_batch_member(decisions_file: Path, tmp_path: Path):
    batch = deepcopy(SESSION_BATCH)
    del batch["etan-queue"][6]["members"][2]["type"]
    batch_path = _write_json(tmp_path / "bad-batch.json", batch)

    with pytest.raises(ValueError, match="invalid member.*android eas"):
        harvest_session(
            batch_path,
            decisions_file,
            answers_path=tmp_path / "answers.md",
            decisions_path=tmp_path / "clean.json",
        )


def test_cli_runs_from_plain_checkout_without_pythonpath(batch_file: Path, decisions_file: Path, tmp_path: Path):
    script = Path(__file__).resolve().parents[1] / "scripts" / "kg_session_harvest.py"
    answers_path = tmp_path / "answers.md"
    clean_path = tmp_path / "clean-decisions.json"
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}

    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--batch",
            str(batch_file),
            "--session-decisions",
            str(decisions_file),
            "--answers",
            str(answers_path),
            "--decisions",
            str(clean_path),
        ],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert json.loads(clean_path.read_text(encoding="utf-8"))["counts"]["merge_clusters"] == 1
