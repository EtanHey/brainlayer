"""Re-run trigger manifest checks for Phoenix eval suites."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from brainlayer.eval.phoenix_gate.models import HarnessFault

TRIGGER_FIELDS = (
    "description_hash",
    "skill_hash",
    "model_version",
    "catalog_context",
    "suite_version",
)

FIELD_REASONS = {
    "description_hash": "description_changed",
    "skill_hash": "skill_changed",
    "model_version": "model_version_changed",
    "catalog_context": "catalog_context_changed",
    "suite_version": "suite_version_changed",
}


@dataclass(frozen=True)
class TriggerDiff:
    rerun_required: bool
    reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {"rerun_required": self.rerun_required, "reasons": list(self.reasons)}


def _require_manifest_fields(name: str, manifest: Mapping[str, Any]) -> None:
    missing = [
        field
        for field in TRIGGER_FIELDS
        if not isinstance(manifest.get(field), str) or not str(manifest.get(field)).strip()
    ]
    if missing:
        raise HarnessFault(f"{name} trigger manifest missing required field(s): " + ", ".join(missing))


def diff_rerun_triggers(previous: Mapping[str, Any], current: Mapping[str, Any]) -> TriggerDiff:
    """Return the exact trigger reasons that require a fresh Phoenix eval run."""
    _require_manifest_fields("previous", previous)
    _require_manifest_fields("current", current)

    reasons = tuple(
        FIELD_REASONS[field] for field in TRIGGER_FIELDS if str(previous[field]).strip() != str(current[field]).strip()
    )
    return TriggerDiff(rerun_required=bool(reasons), reasons=reasons)
