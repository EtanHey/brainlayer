"""REST-only Phoenix client helpers for regression-gate inputs."""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.parse
import urllib.request
from collections import defaultdict
from collections.abc import Callable, Mapping
from typing import Any

from brainlayer.eval.phoenix_gate.models import BaselineKey, DatasetExample, ExperimentScore, HarnessFault

PHOENIX_TAILNET_BASE_URL = "http://100.114.179.86:6006"
DEFAULT_BASE_URL = os.environ.get("PHOENIX_BASE_URL", PHOENIX_TAILNET_BASE_URL)


def validate_evaluators_for_run(evaluators: object) -> Mapping[str, Any]:
    """Guard the Phoenix gotcha where a list silently zeroes evaluator output."""
    if not isinstance(evaluators, Mapping):
        raise HarnessFault("Phoenix evaluators must be passed as a non-empty dict, not a list")
    if not evaluators:
        raise HarnessFault("Phoenix evaluators dict must be non-empty")
    if not all(isinstance(name, str) and name for name in evaluators):
        raise HarnessFault("Phoenix evaluator names must be non-empty strings")
    return evaluators


def _coerce_rows(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        rows = payload
    elif isinstance(payload, dict):
        for candidate_key in ("data", "rows", "examples"):
            if candidate_key in payload:
                rows = payload[candidate_key]
                break
        else:
            rows = None
    else:
        rows = None
    if not isinstance(rows, list):
        raise HarnessFault("Phoenix experiment JSON must be a list of rows")
    if not all(isinstance(row, dict) for row in rows):
        raise HarnessFault("Phoenix experiment JSON rows must be objects")
    return rows


def _annotation_score(annotation: Mapping[str, Any]) -> float:
    if annotation.get("error"):
        raise HarnessFault(f"Phoenix evaluator {annotation.get('name')!r} errored: {annotation['error']}")
    value = annotation.get("score")
    if value is None:
        value = annotation.get("label")
    if isinstance(value, bool):
        return float(int(value))
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        stripped = value.strip().lower()
        if stripped in {"true", "pass", "passed", "yes"}:
            return 1.0
        if stripped in {"false", "fail", "failed", "no"}:
            return 0.0
        try:
            return float(stripped)
        except ValueError as exc:
            raise HarnessFault(f"Phoenix evaluator {annotation.get('name')!r} has non-numeric label {value!r}") from exc
    raise HarnessFault(f"Phoenix evaluator {annotation.get('name')!r} is missing a numeric score")


def aggregate_evaluator_means(payload: Any, *, expected_evaluators: set[str] | frozenset[str]) -> dict[str, float]:
    rows = _coerce_rows(payload)
    scores: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        annotations = row.get("annotations")
        if not isinstance(annotations, list):
            raise HarnessFault("Phoenix experiment row missing annotations list")
        for annotation in annotations:
            if not isinstance(annotation, Mapping):
                raise HarnessFault("Phoenix annotation must be an object")
            name = annotation.get("name")
            if not isinstance(name, str) or not name:
                raise HarnessFault("Phoenix annotation missing evaluator name")
            scores[name].append(_annotation_score(annotation))

    means = {name: sum(values) / len(values) for name, values in sorted(scores.items()) if values}
    if not means:
        raise HarnessFault("Phoenix experiment has no evaluator output")

    missing = sorted(expected_evaluators - set(means))
    if missing:
        raise HarnessFault("Phoenix experiment missing expected evaluator(s): " + ", ".join(missing))

    scored_expected = {name: means[name] for name in expected_evaluators}
    # Fail-loud on all-zero to catch Phoenix evaluator setup bugs (e.g. list instead of dict)
    # where harness silently produces zero scores. While this prevents detecting legitimate
    # all-zero regressions, the Phoenix list→dict coercion gotcha is caught earlier by
    # validate_evaluators_for_run, so in practice this guard catches missing annotations
    # or broken evaluator runs that would otherwise appear as silent SHIP verdicts.
    if scored_expected and all(value == 0.0 for value in scored_expected.values()):
        raise HarnessFault("Phoenix experiment evaluator output is all-zero; likely harness fault")
    return means


def extract_dataset_examples(payload: Any) -> list[DatasetExample]:
    if not isinstance(payload, dict):
        raise HarnessFault("Phoenix dataset payload must be an object")
    data = payload.get("data", payload)
    if not isinstance(data, dict):
        raise HarnessFault("Phoenix dataset data must be an object")
    rows = data.get("examples")
    if not isinstance(rows, list):
        raise HarnessFault("Phoenix get_dataset read payload missing data.examples")

    examples: list[DatasetExample] = []
    for row in rows:
        if not isinstance(row, dict):
            raise HarnessFault("Phoenix dataset example must be an object")
        input_value = row.get("input")
        output_value = row.get("output", {})
        metadata_value = row.get("metadata", {})
        if not isinstance(input_value, dict) or not input_value:
            raise HarnessFault(f"Phoenix dataset example {row.get('id')!r} has empty input")
        if not isinstance(output_value, dict):
            raise HarnessFault(f"Phoenix dataset example {row.get('id')!r} has non-object output")
        if not isinstance(metadata_value, dict):
            raise HarnessFault(f"Phoenix dataset example {row.get('id')!r} has non-object metadata")
        examples.append(
            DatasetExample(
                id=None if row.get("id") is None else str(row.get("id")),
                input=dict(input_value),
                output=dict(output_value),
                metadata=dict(metadata_value),
            )
        )
    if not examples:
        raise HarnessFault("Phoenix get_dataset read payload contains no examples")
    return examples


class PhoenixRestClient:
    """Small REST client for Phoenix v1 endpoints used by the gate."""

    def __init__(
        self,
        *,
        base_url: str = DEFAULT_BASE_URL,
        request_json: Callable[[str], Any] | None = None,
        timeout_seconds: float = 10.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self._request_json = request_json
        self.timeout_seconds = timeout_seconds

    def _url(self, path: str) -> str:
        return f"{self.base_url}/{path.lstrip('/')}"

    def _get_json(self, path: str) -> Any:
        url = self._url(path)
        if self._request_json is not None:
            return self._request_json(url)
        request = urllib.request.Request(url, headers={"Accept": "application/json"})
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                return json.loads(response.read().decode("utf-8"))
        except (urllib.error.URLError, json.JSONDecodeError) as exc:
            raise HarnessFault(f"Phoenix REST request failed for {url}: {exc}") from exc

    @staticmethod
    def _quote_id(value: str) -> str:
        return urllib.parse.quote(value, safe="")

    def list_dataset_experiments(self, dataset_id: str) -> list[dict[str, Any]]:
        payload = self._get_json(f"/v1/datasets/{self._quote_id(dataset_id)}/experiments")
        if not isinstance(payload, dict) or not isinstance(payload.get("data"), list):
            raise HarnessFault("Phoenix dataset experiments payload missing data list")
        return payload["data"]

    def get_dataset_examples(self, dataset_id: str) -> list[DatasetExample]:
        payload = self._get_json(f"/v1/datasets/{self._quote_id(dataset_id)}/examples")
        return extract_dataset_examples(payload)

    def get_experiment(self, experiment_id: str) -> dict[str, Any]:
        payload = self._get_json(f"/v1/experiments/{self._quote_id(experiment_id)}")
        if not isinstance(payload, dict):
            raise HarnessFault("Phoenix experiment payload must be an object")
        if isinstance(payload.get("data"), dict):
            return payload["data"]
        return payload

    def get_experiment_json(self, experiment_id: str) -> list[dict[str, Any]]:
        payload = self._get_json(f"/v1/experiments/{self._quote_id(experiment_id)}/json")
        return _coerce_rows(payload)

    def load_experiment_score(
        self,
        experiment_id: str,
        *,
        expected_evaluators: set[str] | frozenset[str],
    ) -> ExperimentScore:
        experiment = self.get_experiment(experiment_id)
        metadata = experiment.get("metadata")
        if not isinstance(metadata, Mapping):
            raise HarnessFault("Phoenix experiment metadata must be an object")
        key = BaselineKey.from_metadata(metadata)
        rows = self.get_experiment_json(experiment_id)
        return ExperimentScore(
            experiment_id=str(experiment.get("id", experiment_id)),
            dataset_id=None if experiment.get("dataset_id") is None else str(experiment.get("dataset_id")),
            created_at=str(experiment.get("created_at", "")),
            key=key,
            evaluator_means=aggregate_evaluator_means(rows, expected_evaluators=expected_evaluators),
        )
