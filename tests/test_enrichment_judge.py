from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest


def _create_snapshot(path: Path) -> None:
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE chunks (id TEXT PRIMARY KEY, content TEXT NOT NULL)")
    conn.execute("INSERT INTO chunks (id, content) VALUES (?, ?)", ("chunk-1", "BrainLayer uses SQLite."))
    conn.commit()
    conn.close()


def _enrichment(**overrides):
    payload = {
        "summary": "BrainLayer PR-JUDGE adds an offline JSONL judge harness with calibration gates.",
        "key_facts": ["PR-JUDGE", "offline JSONL judge harness", "calibration gates"],
        "tags": ["brainlayer", "eval", "judge"],
        "importance": 8,
        "intent": "implementing",
        "primary_symbols": ["enrichment_judge.py"],
        "resolved_query": "How does BrainLayer judge enrichment quality?",
        "resolved_queries": [
            "How does BrainLayer judge enrichment quality?",
            "BrainLayer PR-JUDGE offline JSONL judge harness",
            "BrainLayer calibration gates for enrichment judge",
        ],
        "epistemic_level": "validated",
        "version_scope": "PR-JUDGE",
        "debt_impact": "none",
        "external_deps": [],
        "entities": [
            {"name": "BrainLayer", "type": "project", "relation": "memory layer"},
            {"name": "SQLite", "type": "technology", "relation": "experiment store"},
        ],
        "sentiment_label": "neutral",
        "sentiment_score": 0.0,
        "sentiment_signals": [],
    }
    payload.update(overrides)
    return payload


def _judge_response(_request):
    return {
        "reason": {
            "faithfulness": "The enrichment only uses facts present in the chunk.",
            "usefulness": "The summary and queries help future retrieval.",
            "entity_coverage": "The main supported entities are present.",
        },
        "score": {
            "faithfulness": 5,
            "usefulness": 4,
            "entity_coverage": 5,
        },
        "rationale": "Faithful and useful with the key entities covered.",
    }


def test_inline_batch_scores_jsonl_and_records_reproducibility_metadata(tmp_path):
    from brainlayer.eval.enrichment_judge import DEFAULT_TEMPERATURE, JUDGE_PROMPT_HASH, score_jsonl_inline

    input_path = tmp_path / "judge-input.jsonl"
    output_path = tmp_path / "judge-output.jsonl"
    row = {
        "chunk_id": "chunk-1",
        "variant_id": "A",
        "chunk_text": "BrainLayer PR-JUDGE adds an offline JSONL judge harness using SQLite experiment storage.",
        "enrichment": _enrichment(),
        "model": "gemini-test",
        "prompt_hash": "variant-prompt-hash",
        "gold": {
            "must_capture_facts": ["PR-JUDGE", "offline JSONL judge harness"],
            "gold_entities": [
                {"name": "BrainLayer", "type": "project"},
                {"name": "SQLite", "type": "technology"},
            ],
            "gold_importance": 8,
        },
    }
    input_path.write_text(json.dumps(row) + "\n", encoding="utf-8")

    written = score_jsonl_inline(
        input_path,
        output_path,
        judge=_judge_response,
        judge_model="codex-subscription-gpt-5.5",
        judge_cli_version="codex-cli 1.2.3",
    )

    assert written == 1
    scored = json.loads(output_path.read_text(encoding="utf-8"))
    assert scored["chunk_id"] == "chunk-1"
    assert scored["variant_id"] == "A"
    assert scored["scores"] == {
        "faithfulness": 5,
        "usefulness": 4,
        "entity_coverage": 5,
        "composite": pytest.approx(4.7),
    }
    assert scored["judge_metadata"]["temperature"] == DEFAULT_TEMPERATURE == 0
    assert scored["judge_metadata"]["judge_prompt_hash"] == JUDGE_PROMPT_HASH
    assert scored["judge_metadata"]["judge_model"] == "codex-subscription-gpt-5.5"
    assert scored["judge_metadata"]["judge_cli_version"] == "codex-cli 1.2.3"
    assert scored["judge_metadata"]["response_schema"] == "two_step_reason_then_score_v1"
    assert scored["deterministic_pre_signals"]["schema"]["passed"] is True
    assert scored["deterministic_pre_signals"]["banned_pattern_hit"] is False
    assert scored["request"]["judge_prompt_hash"] == JUDGE_PROMPT_HASH


def test_score_jsonl_inline_persists_llm_judgments_to_experiment_store(tmp_path):
    from brainlayer.eval.enrichment_judge import score_jsonl_inline
    from brainlayer.eval.experiment_store import ExperimentStore

    snapshot_db = tmp_path / "experiments" / "abcde-snapshot.db"
    experiment_db = tmp_path / "experiments" / "abcde-experiment.db"
    snapshot_db.parent.mkdir()
    _create_snapshot(snapshot_db)
    input_path = tmp_path / "judge-input.jsonl"
    output_path = tmp_path / "judge-output.jsonl"

    with ExperimentStore(experiment_db_path=experiment_db, snapshot_db_path=snapshot_db) as store:
        chunk_id = store.upsert_chunk(
            source_chunk_id="chunk-1",
            raw_text="BrainLayer PR-JUDGE adds an offline JSONL judge harness using SQLite experiment storage.",
            content_type="code",
            content_class="test",
            strata={},
        )
        store.upsert_variant(
            chunk_id=chunk_id,
            variant_id="A",
            enrichment=_enrichment(),
            model="gemini-test",
            prompt_hash="variant-prompt-hash",
        )
        input_path.write_text(
            json.dumps(
                {
                    "chunk_id": chunk_id,
                    "variant_id": "A",
                    "chunk_text": "BrainLayer PR-JUDGE adds an offline JSONL judge harness using SQLite.",
                    "enrichment": _enrichment(),
                }
            )
            + "\n",
            encoding="utf-8",
        )

        score_jsonl_inline(input_path, output_path, judge=_judge_response, experiment_store=store)

    conn = sqlite3.connect(experiment_db)
    row = conn.execute("SELECT source, scores_json, rationale FROM exp_judgments").fetchone()
    conn.close()

    assert row[0] == "llm"
    assert json.loads(row[1])["judge_metadata"]["temperature"] == 0
    assert "Faithful and useful" in row[2]


def test_prepare_batch_jsonl_writes_prompt_packages_without_calling_metered_api(tmp_path):
    from brainlayer.eval.enrichment_judge import JUDGE_PROMPT_HASH, prepare_batch_jsonl

    input_path = tmp_path / "judge-input.jsonl"
    prepared_path = tmp_path / "judge-prepared.jsonl"
    input_path.write_text(
        json.dumps(
            {
                "chunk_id": "chunk-1",
                "variant_id": "B",
                "chunk_text": "BrainLayer uses an experiment namespace.",
                "enrichment": _enrichment(),
            }
        )
        + "\n",
        encoding="utf-8",
    )

    assert prepare_batch_jsonl(input_path, prepared_path, judge_model="codex-subscription") == 1
    prepared = json.loads(prepared_path.read_text(encoding="utf-8"))
    assert prepared["mode"] == "inline_or_subscription_batch"
    assert prepared["temperature"] == 0
    assert prepared["judge_prompt_hash"] == JUDGE_PROMPT_HASH
    assert prepared["strict_json_contract"]["required_top_level_keys"] == ["reason", "score", "rationale"]
    assert "Do not call a metered API" in prepared["system_instruction"]


def test_calibration_passes_when_kappa_and_spearman_meet_floors():
    from brainlayer.eval.enrichment_judge import calibrate_judge

    result = calibrate_judge(
        judge_scores=[{"composite": 5}, {"composite": 4}, {"composite": 2}, {"composite": 1}],
        human_scores=[{"composite": 5}, {"composite": 4}, {"composite": 2}, {"composite": 1}],
    )

    assert result.quarantined is False
    assert result.kappa == 1.0
    assert result.spearman_rho == 1.0


def test_calibration_quarantines_below_hard_floor():
    from brainlayer.eval.enrichment_judge import JudgeQuarantinedError, calibrate_judge

    with pytest.raises(JudgeQuarantinedError) as exc:
        calibrate_judge(
            judge_scores=[{"composite": 5}, {"composite": 5}, {"composite": 1}, {"composite": 1}],
            human_scores=[{"composite": 1}, {"composite": 1}, {"composite": 5}, {"composite": 5}],
            kappa_floor=0.6,
            spearman_floor=0.7,
        )

    result = exc.value.result
    assert result.quarantined is True
    assert result.kappa < 0.6
    assert result.spearman_rho < 0.7
