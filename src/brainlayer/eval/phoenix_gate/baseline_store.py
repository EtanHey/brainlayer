"""Versioned JSON baseline store for Phoenix regression gates."""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from brainlayer.eval.phoenix_gate.models import BaselineKey, HarnessFault

DEFAULT_BASELINE_STORE_PATH = Path(__file__).with_name("phoenix_baselines.json")
GREEN_STATUS = "GREEN"


@dataclass(frozen=True)
class BaselineRecord:
    key: BaselineKey
    created_at: str
    evaluator_means: dict[str, float]
    source_experiment_id: str | None = None
    status: str = GREEN_STATUS

    @property
    def identity(self) -> str:
        return json.dumps([*self.key.as_tuple(), self.created_at], separators=(",", ":"), ensure_ascii=True)

    def to_dict(self) -> dict[str, Any]:
        return {
            "identity": self.identity,
            "key": self.key.to_dict(),
            "created_at": self.created_at,
            "evaluator_means": dict(sorted(self.evaluator_means.items())),
            "source_experiment_id": self.source_experiment_id,
            "status": self.status,
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "BaselineRecord":
        return cls(
            key=BaselineKey.from_metadata(raw["key"]),
            created_at=str(raw["created_at"]),
            evaluator_means={str(name): float(value) for name, value in raw["evaluator_means"].items()},
            source_experiment_id=raw.get("source_experiment_id"),
            status=str(raw.get("status", GREEN_STATUS)),
        )


class JsonBaselineStore:
    """Read/write immutable GREEN baseline records from a JSON file."""

    def __init__(self, path: str | Path = DEFAULT_BASELINE_STORE_PATH) -> None:
        self.path = Path(path)

    def _read_records(self) -> list[BaselineRecord]:
        if not self.path.exists():
            return []
        try:
            payload = json.loads(self.path.read_text())
        except (json.JSONDecodeError, OSError) as exc:
            raise HarnessFault(f"Failed to read baseline store {self.path}: {exc}") from exc
        if not isinstance(payload, dict) or not isinstance(payload.get("baselines", []), list):
            raise HarnessFault(f"Baseline store {self.path} must contain a baselines list")
        try:
            return [BaselineRecord.from_dict(record) for record in payload.get("baselines", [])]
        except (KeyError, TypeError, ValueError) as exc:
            raise HarnessFault(f"Baseline store {self.path} contains malformed baseline records: {exc}") from exc

    def _write_records(self, records: list[BaselineRecord]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        ordered = sorted(records, key=lambda record: record.identity)
        payload = {"baselines": [record.to_dict() for record in ordered]}
        try:
            content = json.dumps(payload, indent=2, sort_keys=True) + "\n"
            fd, temp_path = tempfile.mkstemp(dir=self.path.parent, prefix=f".{self.path.name}.", text=True)
            try:
                os.write(fd, content.encode("utf-8"))
                os.fsync(fd)
                os.close(fd)
                os.replace(temp_path, self.path)
            except Exception:
                os.close(fd) if fd >= 0 else None
                Path(temp_path).unlink(missing_ok=True)
                raise
        except OSError as exc:
            raise HarnessFault(f"Failed to write baseline store {self.path}: {exc}") from exc

    def add_green(self, record: BaselineRecord) -> None:
        if record.status != GREEN_STATUS:
            raise HarnessFault(f"Only GREEN baselines can be stored, got status={record.status!r}")
        if not record.created_at:
            raise HarnessFault("Baseline created_at is required")
        if not record.evaluator_means:
            raise HarnessFault("Baseline evaluator_means must be non-empty")
        records = self._read_records()
        identities = {existing.identity for existing in records}
        if record.identity not in identities:
            records.append(record)
        else:
            records = [record if existing.identity == record.identity else existing for existing in records]
        self._write_records(records)

    def latest_green_exact(self, key: BaselineKey) -> BaselineRecord | None:
        candidates = [record for record in self._read_records() if record.status == GREEN_STATUS and record.key == key]
        if not candidates:
            return None
        return max(candidates, key=lambda record: record.created_at)

    def latest_green_for_comparison(self, key: BaselineKey) -> BaselineRecord | None:
        exact = self.latest_green_exact(key)
        if exact is not None:
            return exact

        candidates = [
            record
            for record in self._read_records()
            if record.status == GREEN_STATUS and record.key.scope_tuple() == key.scope_tuple()
        ]
        if not candidates:
            return None

        def rank(record: BaselineRecord) -> tuple[int, float, str]:
            comparable_matches = sum(
                (
                    record.key.condition == key.condition,
                    record.key.model_version == key.model_version,
                    record.key.catalog_context == key.catalog_context,
                )
            )
            mean_score = sum(record.evaluator_means.values()) / len(record.evaluator_means) if record.evaluator_means else 0.0
            return (comparable_matches, mean_score, record.created_at)

        return max(candidates, key=rank)

    def all_records(self) -> list[BaselineRecord]:
        return self._read_records()
