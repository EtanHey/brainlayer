"""Shared models for the Phoenix regression gate."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Mapping

BASELINE_KEY_FIELDS = (
    "surface",
    "mode",
    "condition",
    "model_version",
    "catalog_context",
    "suite_version",
)


class HarnessFault(RuntimeError):
    """Raised when Phoenix/gate data is structurally unsafe to score."""


@dataclass(frozen=True)
class BaselineKey:
    """Canonical Phoenix baseline identity.

    The tuple deliberately excludes Phoenix experiment global IDs because the
    live store can reuse them after deletion. ``created_at`` is added by
    ``BaselineRecord`` for immutable baseline identity.
    """

    surface: str
    mode: str
    condition: str
    model_version: str
    catalog_context: str
    suite_version: str

    @classmethod
    def from_metadata(cls, metadata: Mapping[str, Any]) -> "BaselineKey":
        missing = [
            field
            for field in BASELINE_KEY_FIELDS
            if not isinstance(metadata.get(field), str) or not str(metadata.get(field)).strip()
        ]
        if missing:
            raise HarnessFault(
                "Experiment metadata missing canonical baseline tuple fields: " + ", ".join(missing)
            )
        return cls(**{field: str(metadata[field]).strip() for field in BASELINE_KEY_FIELDS})

    def as_tuple(self) -> tuple[str, str, str, str, str, str]:
        return (
            self.surface,
            self.mode,
            self.condition,
            self.model_version,
            self.catalog_context,
            self.suite_version,
        )

    def to_dict(self) -> dict[str, str]:
        return {field: value for field, value in zip(BASELINE_KEY_FIELDS, self.as_tuple(), strict=True)}

    def identity_without_created_at(self) -> str:
        return json.dumps(list(self.as_tuple()), separators=(",", ":"), ensure_ascii=True)

    def scope_tuple(self) -> tuple[str, str, str]:
        """Comparable suite scope: surface + mode + suite version."""
        return (self.surface, self.mode, self.suite_version)


@dataclass(frozen=True)
class DatasetExample:
    id: str | None
    input: dict[str, Any]
    output: dict[str, Any]
    metadata: dict[str, Any]


@dataclass(frozen=True)
class ExperimentScore:
    experiment_id: str
    dataset_id: str | None
    created_at: str
    key: BaselineKey
    evaluator_means: dict[str, float]


@dataclass(frozen=True)
class RegressionFinding:
    evaluator: str
    baseline: float
    observed: float
    delta: float


@dataclass(frozen=True)
class RegressionVerdict:
    status: str
    alarm: bool
    regression_attributed_to: str
    regressions: tuple[RegressionFinding, ...]
    candidate_key: BaselineKey
    baseline_key: BaselineKey
    candidate_created_at: str
    baseline_created_at: str
    requires_new_baseline: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "alarm": self.alarm,
            "regression_attributed_to": self.regression_attributed_to,
            "requires_new_baseline": self.requires_new_baseline,
            "candidate_key": self.candidate_key.to_dict(),
            "baseline_key": self.baseline_key.to_dict(),
            "candidate_created_at": self.candidate_created_at,
            "baseline_created_at": self.baseline_created_at,
            "regressions": [
                {
                    "evaluator": finding.evaluator,
                    "baseline": finding.baseline,
                    "observed": finding.observed,
                    "delta": finding.delta,
                }
                for finding in self.regressions
            ],
        }
