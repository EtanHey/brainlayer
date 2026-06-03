"""Regression verdict logic for Phoenix evaluator vectors."""

from __future__ import annotations

import math

from brainlayer.eval.phoenix_gate.baseline_store import BaselineRecord, JsonBaselineStore
from brainlayer.eval.phoenix_gate.models import (
    BaselineKey,
    ExperimentScore,
    HarnessFault,
    RegressionFinding,
    RegressionVerdict,
)

OTEL_TIER2_ENABLED = False


def _attribute_regression(baseline_key: BaselineKey, candidate_key: BaselineKey) -> str:
    changed: set[str] = set()
    if baseline_key.condition != candidate_key.condition:
        changed.add("description")
    if baseline_key.model_version != candidate_key.model_version:
        changed.add("model")
    if baseline_key.catalog_context != candidate_key.catalog_context:
        changed.add("catalog")

    if not changed:
        return "unknown"
    if len(changed) == 1:
        return next(iter(changed))
    return "both"


class RegressionGate:
    """Compare a Phoenix experiment vector against the latest GREEN baseline."""

    def __init__(self, baseline_store: JsonBaselineStore, *, threshold: float = 0.0) -> None:
        if not math.isfinite(threshold) or threshold < 0:
            raise ValueError("threshold must be a finite value >= 0")
        self.baseline_store = baseline_store
        self.threshold = threshold

    def _resolve_baseline(self, score: ExperimentScore) -> BaselineRecord:
        baseline = self.baseline_store.latest_green_for_comparison(score.key)
        if baseline is None:
            raise HarnessFault(
                "No GREEN Phoenix baseline found for "
                f"surface={score.key.surface!r}, mode={score.key.mode!r}, suite_version={score.key.suite_version!r}"
            )
        return baseline

    def evaluate(self, score: ExperimentScore) -> RegressionVerdict:
        if not score.evaluator_means:
            raise HarnessFault("Candidate experiment evaluator_means must be non-empty")
        baseline = self._resolve_baseline(score)
        missing = sorted(set(baseline.evaluator_means) - set(score.evaluator_means))
        if missing:
            raise HarnessFault("Candidate experiment missing baseline evaluator(s): " + ", ".join(missing))

        regressions: list[RegressionFinding] = []
        for evaluator, baseline_value in sorted(baseline.evaluator_means.items()):
            observed = float(score.evaluator_means[evaluator])
            delta = observed - float(baseline_value)
            if -delta > self.threshold:
                regressions.append(
                    RegressionFinding(
                        evaluator=evaluator,
                        baseline=float(baseline_value),
                        observed=observed,
                        delta=delta,
                    )
                )

        attribution = _attribute_regression(baseline.key, score.key)
        alarm = bool(regressions)
        status = "ITERATE" if alarm else ("SHIP_SAFE" if attribution == "unknown" else "SHIP")
        requires_new_baseline = not alarm and baseline.key != score.key
        return RegressionVerdict(
            status=status,
            alarm=alarm,
            regression_attributed_to=attribution,
            regressions=tuple(regressions),
            candidate_key=score.key,
            baseline_key=baseline.key,
            candidate_created_at=score.created_at,
            baseline_created_at=baseline.created_at,
            requires_new_baseline=requires_new_baseline,
        )
