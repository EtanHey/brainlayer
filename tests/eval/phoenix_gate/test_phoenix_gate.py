from __future__ import annotations

import json

import pytest

from brainlayer.eval.phoenix_gate.baseline_store import (
    DEFAULT_BASELINE_STORE_PATH,
    BaselineRecord,
    JsonBaselineStore,
)
from brainlayer.eval.phoenix_gate.models import BaselineKey, ExperimentScore, HarnessFault
from brainlayer.eval.phoenix_gate.phoenix_client import (
    DEFAULT_BASE_URL,
    PHOENIX_TAILNET_BASE_URL,
    aggregate_evaluator_means,
    extract_dataset_examples,
    validate_evaluators_for_run,
)
from brainlayer.eval.phoenix_gate.regression_gate import RegressionGate
from brainlayer.eval.phoenix_gate.triggers import diff_rerun_triggers


def _key(
    *,
    condition: str = "control",
    model_version: str = "claude-opus-4-8-1m",
    catalog_context: str = "minimal",
) -> BaselineKey:
    return BaselineKey(
        surface="cmux-mcp",
        mode="usage",
        condition=condition,
        model_version=model_version,
        catalog_context=catalog_context,
        suite_version="v0-proof",
    )


def _score(
    *,
    key: BaselineKey | None = None,
    means: dict[str, float] | None = None,
    created_at: str = "2026-06-03T18:52:10+00:00",
    experiment_id: str = "RXhwZXJpbWVudDoxMQ==",
) -> ExperimentScore:
    return ExperimentScore(
        experiment_id=experiment_id,
        dataset_id="RGF0YXNldDo0",
        created_at=created_at,
        key=key or _key(),
        evaluator_means=means or {"executed_not_relayed": 1.0, "success": 1.0},
    )


def test_baseline_key_requires_all_six_canonical_metadata_fields() -> None:
    metadata = {
        "surface": "cmux-mcp",
        "mode": "usage",
        "condition": "baseline_live",
        "model_version": "claude-opus-4-8-1m",
        "catalog_context": "full-fleet-live",
        "suite_version": "v0-proof",
    }

    key = BaselineKey.from_metadata(metadata)

    assert key == BaselineKey(
        surface="cmux-mcp",
        mode="usage",
        condition="baseline_live",
        model_version="claude-opus-4-8-1m",
        catalog_context="full-fleet-live",
        suite_version="v0-proof",
    )

    missing_model_version = dict(metadata)
    missing_model_version.pop("model_version")
    missing_model_version["model"] = "Opus-1M"

    with pytest.raises(HarnessFault, match="model_version"):
        BaselineKey.from_metadata(missing_model_version)


def test_phoenix_base_url_is_tailnet_default_not_localhost() -> None:
    assert PHOENIX_TAILNET_BASE_URL == "http://100.114.179.86:6006"
    assert DEFAULT_BASE_URL == PHOENIX_TAILNET_BASE_URL


def test_aggregate_evaluator_means_allows_true_zero_but_fails_silent_zero() -> None:
    rows = [
        {
            "annotations": [
                {"name": "executed_not_relayed", "annotator_kind": "CODE", "score": 1.0},
                {"name": "focus_before_send", "annotator_kind": "CODE", "score": 1.0},
                {"name": "abs_le2_columns", "annotator_kind": "CODE", "score": 0.0},
                {"name": "success", "annotator_kind": "CODE", "score": 1.0},
            ]
        }
    ]

    means = aggregate_evaluator_means(
        rows,
        expected_evaluators={"executed_not_relayed", "focus_before_send", "abs_le2_columns", "success"},
    )

    assert means == {
        "abs_le2_columns": 0.0,
        "executed_not_relayed": 1.0,
        "focus_before_send": 1.0,
        "success": 1.0,
    }

    silent_zero_rows = [
        {
            "annotations": [
                {"name": "executed_not_relayed", "annotator_kind": "CODE", "score": 0.0},
                {"name": "focus_before_send", "annotator_kind": "CODE", "score": 0.0},
                {"name": "abs_le2_columns", "annotator_kind": "CODE", "score": 0.0},
                {"name": "success", "annotator_kind": "CODE", "score": 0.0},
            ]
        }
    ]

    with pytest.raises(HarnessFault, match="all-zero"):
        aggregate_evaluator_means(
            silent_zero_rows,
            expected_evaluators={"executed_not_relayed", "focus_before_send", "abs_le2_columns", "success"},
        )


def test_aggregate_evaluator_means_fails_loud_on_absent_expected_evaluator() -> None:
    rows = [{"annotations": [{"name": "success", "annotator_kind": "CODE", "score": 1.0}]}]

    with pytest.raises(HarnessFault, match="executed_not_relayed"):
        aggregate_evaluator_means(rows, expected_evaluators={"success", "executed_not_relayed"})


def test_phoenix_evaluators_must_be_non_empty_dict_not_list() -> None:
    evaluators = {"success": object()}

    assert validate_evaluators_for_run(evaluators) is evaluators

    with pytest.raises(HarnessFault, match="dict"):
        validate_evaluators_for_run([object()])

    with pytest.raises(HarnessFault, match="non-empty"):
        validate_evaluators_for_run({})


def test_get_dataset_read_shape_extracts_examples_without_empty_inputs() -> None:
    payload = {
        "data": {
            "dataset_id": "RGF0YXNldDoz",
            "version_id": "RGF0YXNldFZlcnNpb246NA==",
            "examples": [
                {
                    "id": "RGF0YXNldEV4YW1wbGU6OTQ1",
                    "input": {"case_id": "F3", "mode": "usage", "intent": "Use cmux, not a relay."},
                    "output": {"gold_primary": "send_to"},
                    "metadata": {"failure_id": "F3"},
                }
            ],
        }
    }

    examples = extract_dataset_examples(payload)

    assert examples[0].input["case_id"] == "F3"
    assert examples[0].output["gold_primary"] == "send_to"
    assert examples[0].metadata["failure_id"] == "F3"

    bad_payload = {"data": {"examples": [{"id": "bad", "input": {}, "output": {}, "metadata": {}}]}}
    with pytest.raises(HarnessFault, match="empty input"):
        extract_dataset_examples(bad_payload)


def test_baseline_store_identity_uses_tuple_and_created_at_not_experiment_global_id(tmp_path) -> None:
    path = tmp_path / "phoenix-baselines.json"
    store = JsonBaselineStore(path)
    key = _key()

    store.add_green(
        BaselineRecord(
            key=key,
            created_at="2026-06-03T18:51:49+00:00",
            evaluator_means={"success": 1.0},
            source_experiment_id="RXhwZXJpbWVudDo3",
        )
    )
    store.add_green(
        BaselineRecord(
            key=key,
            created_at="2026-06-03T18:51:50+00:00",
            evaluator_means={"success": 0.9},
            source_experiment_id="RXhwZXJpbWVudDo3",
        )
    )

    raw = json.loads(path.read_text())
    identities = [record["identity"] for record in raw["baselines"]]

    assert [json.loads(identity) for identity in identities] == [
        [
            "cmux-mcp",
            "usage",
            "control",
            "claude-opus-4-8-1m",
            "minimal",
            "v0-proof",
            "2026-06-03T18:51:49+00:00",
        ],
        [
            "cmux-mcp",
            "usage",
            "control",
            "claude-opus-4-8-1m",
            "minimal",
            "v0-proof",
            "2026-06-03T18:51:50+00:00",
        ],
    ]
    assert all("RXhwZXJpbWVudDo3" not in identity for identity in identities)
    assert store.latest_green_exact(key).evaluator_means["success"] == 0.9


def test_baseline_store_wraps_malformed_json_as_harness_fault(tmp_path) -> None:
    path = tmp_path / "phoenix-baselines.json"
    path.write_text("{not-json")

    with pytest.raises(HarnessFault, match="Failed to read baseline store"):
        JsonBaselineStore(path).all_records()


def test_default_baseline_store_path_is_not_brainlayer_db() -> None:
    assert DEFAULT_BASELINE_STORE_PATH.name != "brainlayer.db"
    assert "brainlayer.db" not in str(DEFAULT_BASELINE_STORE_PATH)


def test_regression_gate_flags_drop_and_attributes_description_axis(tmp_path) -> None:
    store = JsonBaselineStore(tmp_path / "baselines.json")
    store.add_green(
        BaselineRecord(
            key=_key(condition="control"),
            created_at="2026-06-03T18:51:49+00:00",
            evaluator_means={"tool_accuracy": 1.0, "no_wrong_calls": 1.0},
            source_experiment_id="RXhwZXJpbWVudDo3",
        )
    )
    candidate = _score(
        key=_key(condition="variant1_pr121"),
        means={"tool_accuracy": 0.75, "no_wrong_calls": 1.0},
        experiment_id="RXhwZXJpbWVudDo4",
    )

    verdict = RegressionGate(store, threshold=0.0).evaluate(candidate)

    assert verdict.alarm is True
    assert verdict.status == "ITERATE"
    assert verdict.regression_attributed_to == "description"
    assert verdict.regressions[0].evaluator == "tool_accuracy"
    assert verdict.regressions[0].delta == pytest.approx(-0.25)


def test_regression_gate_rejects_non_finite_threshold(tmp_path) -> None:
    with pytest.raises(ValueError, match="finite"):
        RegressionGate(JsonBaselineStore(tmp_path / "baselines.json"), threshold=float("nan"))


def test_regression_gate_rekeys_model_upgrade_without_alarm_when_scores_hold(tmp_path) -> None:
    store = JsonBaselineStore(tmp_path / "baselines.json")
    store.add_green(
        BaselineRecord(
            key=_key(model_version="claude-opus-4-8-1m"),
            created_at="2026-06-03T18:51:50+00:00",
            evaluator_means={"success": 1.0, "executed_not_relayed": 1.0},
            source_experiment_id="RXhwZXJpbWVudDoxMA==",
        )
    )
    candidate = _score(
        key=_key(model_version="claude-opus-4-9-1m"),
        means={"success": 1.0, "executed_not_relayed": 1.0},
        experiment_id="RXhwZXJpbWVudDoxMQ==",
    )

    verdict = RegressionGate(store, threshold=0.0).evaluate(candidate)

    assert verdict.alarm is False
    assert verdict.status == "SHIP"
    assert verdict.regression_attributed_to == "model"
    assert verdict.requires_new_baseline is True


def test_regression_gate_same_tuple_drop_has_unknown_attribution(tmp_path) -> None:
    store = JsonBaselineStore(tmp_path / "baselines.json")
    store.add_green(
        BaselineRecord(
            key=_key(),
            created_at="2026-06-03T18:51:50+00:00",
            evaluator_means={"success": 1.0},
            source_experiment_id="RXhwZXJpbWVudDoxMA==",
        )
    )
    candidate = _score(key=_key(), means={"success": 0.0}, experiment_id="RXhwZXJpbWVudDoxMA==")

    verdict = RegressionGate(store, threshold=0.0).evaluate(candidate)

    assert verdict.alarm is True
    assert verdict.regression_attributed_to == "unknown"


def test_rerun_triggers_include_model_version_as_mandatory_axis() -> None:
    previous = {
        "description_hash": "desc-a",
        "skill_hash": "skill-a",
        "model_version": "claude-opus-4-8-1m",
        "catalog_context": "minimal",
        "suite_version": "v0-proof",
    }

    current = {
        "description_hash": "desc-a",
        "skill_hash": "skill-a",
        "model_version": "claude-opus-4-9-1m",
        "catalog_context": "minimal",
        "suite_version": "v0-proof",
    }

    diff = diff_rerun_triggers(previous, current)

    assert diff.rerun_required is True
    assert diff.reasons == ("model_version_changed",)


def test_rerun_triggers_fail_loud_on_missing_manifest_axis() -> None:
    previous = {
        "description_hash": "desc-a",
        "skill_hash": "skill-a",
        "model_version": "claude-opus-4-8-1m",
        "catalog_context": "minimal",
        "suite_version": "v0-proof",
    }
    current = dict(previous)
    current.pop("model_version")

    with pytest.raises(HarnessFault, match="model_version"):
        diff_rerun_triggers(previous, current)


def test_empty_phoenix_experiment_payload_fails_as_no_evaluator_output() -> None:
    with pytest.raises(HarnessFault, match="no evaluator output"):
        aggregate_evaluator_means({"data": []}, expected_evaluators={"success"})
