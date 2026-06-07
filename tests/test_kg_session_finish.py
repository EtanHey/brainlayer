import json
from copy import deepcopy
from pathlib import Path

from brainlayer.kg_session_finish import finish_session
from tests.test_kg_session_harvest import SESSION_BATCH


def _write_json(path: Path, payload: dict) -> Path:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def _session_decisions() -> dict:
    answers = [
        {
            "stem": f"Q{index} of 6 - {label}",
            "category": "etan-queue",
            "source": "voice",
            "note": f"answer {index}",
            "decided_at": f"2026-06-06T01:0{index}:00Z",
        }
        for index, label in enumerate(
            [
                "invocable substrates",
                "sprints and QA rounds",
                "external media",
                "agents as subtype",
                "orphaned persons",
                "concept demotion method",
            ],
            start=1,
        )
    ]
    return {
        "schema": "kg-flag-decisions-v1",
        "source": "etan-session-2026-06-06",
        "rules": {},
        "per_category": {},
        "counts": {"merge_clusters": 1, "rows_merged_away": 2, "keep": 1, "explicit": 2, "by_rule": 0},
        "merge": [
            {
                "stem": "android eas",
                "category": "etan-queue",
                "source": "voice",
                "canonical": {"id": "tool-android", "name": "Android EAS", "type": "tool"},
                "members": [
                    {"id": "project-android", "name": "Android EAS", "type": "project"},
                    {"id": "tech-android", "name": "Android EAS", "type": "technology"},
                ],
                "note": "merge real Android EAS entries",
                "decided_at": "2026-06-06T01:06:00Z",
            }
        ],
        "keep": [
            *answers,
            {
                "stem": "agent mix",
                "category": "etan-queue",
                "source": "voice",
                "note": "keep tool and concept separate",
                "decided_at": "2026-06-06T01:07:00Z",
            },
        ],
    }


def _pass_verdict(stem: str, disposition: str) -> dict:
    return {
        "stem": stem,
        "proposed_type": "Tool",
        "identity": f"{stem} is supported by the supplied evidence.",
        "merge_disposition": disposition,
        "canonical_suggestion": "Android EAS" if disposition == "merge" else "Agent Mix",
        "confidence": "high",
        "evidence_cited": ["fixture"],
        "reasoning": "Fixture verdict.",
        "evidence_degraded": False,
    }


def test_finish_summary_shape_and_questions_never_member_mapped(tmp_path: Path):
    batch_path = _write_json(tmp_path / "batch.json", SESSION_BATCH)
    decisions_path = _write_json(tmp_path / "decisions.json", _session_decisions())
    applied_paths: list[Path] = []

    def judge(item: dict, _cluster: dict) -> dict:
        disposition = "merge" if item["stem"] == "android eas" else "keep"
        return _pass_verdict(item["stem"], disposition)

    def apply(path: Path, _run_id: str, *, execute: bool) -> dict:
        assert execute is True
        applied_paths.append(path)
        return {"ok": True}

    summary = finish_session(batch_path, decisions_path, run_id="test-run", judge_func=judge, apply_func=apply)

    assert summary == {
        "harvested": 1,
        "questions": 6,
        "clean_decisions": 2,
        "judge_pass": 2,
        "judge_fail": 0,
        "applied": 2,
        "queued_for_review": 0,
        "run_id": "test-run",
        "reversible": True,
    }
    clean = json.loads((tmp_path / "decisions.clean.json").read_text(encoding="utf-8"))
    assert {item["stem"] for item in clean["merge"] + clean["keep"]} == {"android eas", "agent mix"}
    passes = json.loads(applied_paths[0].read_text(encoding="utf-8"))
    assert {item["stem"] for item in passes["merge"] + passes["keep"]} == {"android eas", "agent mix"}


def test_finish_failed_judge_decision_is_queued_not_applied(tmp_path: Path):
    batch_path = _write_json(tmp_path / "batch.json", SESSION_BATCH)
    decisions_path = _write_json(tmp_path / "decisions.json", _session_decisions())
    applied_docs: list[dict] = []

    def judge(item: dict, _cluster: dict) -> dict:
        if item["stem"] == "android eas":
            return _pass_verdict(item["stem"], "merge")
        verdict = _pass_verdict(item["stem"], "split")
        verdict["canonical_suggestion"] = None
        return verdict

    def apply(path: Path, _run_id: str, *, execute: bool) -> dict:
        applied_docs.append(json.loads(path.read_text(encoding="utf-8")))
        return {"ok": True}

    summary = finish_session(batch_path, decisions_path, run_id="test-run", judge_func=judge, apply_func=apply)

    assert summary["judge_pass"] == 1
    assert summary["judge_fail"] == 1
    assert summary["applied"] == 1
    assert summary["queued_for_review"] == 1
    assert [item["stem"] for item in applied_docs[0]["merge"]] == ["android eas"]
    assert applied_docs[0]["keep"] == []
    queue = json.loads((tmp_path / "decisions.review-queue.json").read_text(encoding="utf-8"))
    assert list(queue) == ["etan-queue"]
    assert queue["etan-queue"][0]["stem"] == "agent mix"
    assert queue["etan-queue"][0]["item_kind"] == "cluster"


def test_finish_no_judge_applies_all_clean_decisions(tmp_path: Path):
    batch_path = _write_json(tmp_path / "batch.json", SESSION_BATCH)
    decisions_path = _write_json(tmp_path / "decisions.json", _session_decisions())
    applied_docs: list[dict] = []

    def apply(path: Path, _run_id: str, *, execute: bool) -> dict:
        applied_docs.append(json.loads(path.read_text(encoding="utf-8")))
        return {"ok": True}

    summary = finish_session(batch_path, decisions_path, run_id="test-run", no_judge=True, apply_func=apply)

    assert summary["judge_pass"] == 2
    assert summary["judge_fail"] == 0
    assert summary["applied"] == 2
    assert {item["stem"] for item in applied_docs[0]["merge"] + applied_docs[0]["keep"]} == {"android eas", "agent mix"}


def test_finish_dry_run_mutates_nothing(tmp_path: Path):
    batch_path = _write_json(tmp_path / "batch.json", SESSION_BATCH)
    decisions_path = _write_json(tmp_path / "decisions.json", _session_decisions())
    before = {path.name for path in tmp_path.iterdir()}
    apply_called = False

    def judge(item: dict, _cluster: dict) -> dict:
        disposition = "merge" if item["stem"] == "android eas" else "keep"
        return _pass_verdict(item["stem"], disposition)

    def apply(_path: Path, _run_id: str, *, execute: bool) -> dict:
        nonlocal apply_called
        apply_called = True
        return {"ok": True}

    summary = finish_session(batch_path, decisions_path, dry_run=True, judge_func=judge, apply_func=apply)

    assert summary["applied"] == 0
    assert summary["queued_for_review"] == 0
    assert apply_called is False
    assert {path.name for path in tmp_path.iterdir()} == before


def test_finish_cli_prints_one_summary_json_line(tmp_path: Path, capsys):
    from brainlayer.kg_session_finish import main

    batch_path = _write_json(tmp_path / "batch.json", SESSION_BATCH)
    decisions_path = _write_json(tmp_path / "decisions.json", _session_decisions())

    exit_code = main(
        [
            "--batch",
            str(batch_path),
            "--decisions",
            str(decisions_path),
            "--run-id",
            "test-run",
            "--no-judge",
            "--dry-run",
        ]
    )

    assert exit_code == 0
    lines = capsys.readouterr().out.splitlines()
    assert len(lines) == 1
    summary = json.loads(lines[0])
    assert set(summary) == {
        "harvested",
        "questions",
        "clean_decisions",
        "judge_pass",
        "judge_fail",
        "applied",
        "queued_for_review",
        "run_id",
        "reversible",
    }


def test_finish_writes_review_queue_item_kind_for_failed_merge(tmp_path: Path):
    batch = deepcopy(SESSION_BATCH)
    batch_path = _write_json(tmp_path / "batch.json", batch)
    decisions_path = _write_json(tmp_path / "decisions.json", _session_decisions())
    applied_docs: list[dict] = []

    def judge(item: dict, _cluster: dict) -> dict:
        verdict = _pass_verdict(item["stem"], "keep")
        verdict["canonical_suggestion"] = "Android EAS"
        return verdict

    def apply(path: Path, _run_id: str, *, execute: bool) -> dict:
        assert execute is True
        applied_docs.append(json.loads(path.read_text(encoding="utf-8")))
        return {"ok": True}

    summary = finish_session(batch_path, decisions_path, run_id="test-run", judge_func=judge, apply_func=apply)

    assert summary["judge_fail"] == 1
    assert [item["stem"] for item in applied_docs[0]["keep"]] == ["agent mix"]
    queue = json.loads((tmp_path / "decisions.review-queue.json").read_text(encoding="utf-8"))
    assert queue["etan-queue"][0]["stem"] == "android eas"
    assert queue["etan-queue"][0]["item_kind"] == "cluster"
