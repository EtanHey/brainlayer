#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jsonschema

REPO = Path(__file__).resolve().parents[1]
FIXTURES = REPO / "tests/fixtures/observability"


@dataclass(frozen=True)
class Grade:
    case_id: str
    field_mismatches: list[str]
    mock_green: list[str]
    traceability: list[str]

    @property
    def passed(self) -> bool:
        return not (self.field_mismatches or self.mock_green or self.traceability)


def _cases(root: Path = FIXTURES) -> list[dict[str, Any]]:
    return json.loads((root / "cases.json").read_text(encoding="utf-8"))["cases"]


def load_case(case_id: str, root: Path = FIXTURES) -> dict[str, Any]:
    return next(case for case in _cases(root) if case["case_id"] == case_id)


def load_golden(case_id: str, root: Path = FIXTURES, heldout_root: Path | None = None) -> dict[str, Any]:
    case = load_case(case_id, root)
    base = heldout_root if case["split"] == "heldout" and heldout_root is not None else root
    return json.loads((base / case["golden"]).read_text(encoding="utf-8"))


def declared_inputs(case: dict[str, Any]) -> list[str]:
    return case["declared_inputs"]


def _field_diff(expected: Any, actual: Any, path: str = "$") -> list[str]:
    if isinstance(expected, dict) and isinstance(actual, dict):
        result = []
        for key in sorted(expected.keys() | actual.keys()):
            child = f"{path}.{key}"
            if key not in actual:
                result.append(f"{child}: missing (expected {expected[key]!r})")
            elif key not in expected:
                result.append(f"{child}: unexpected {actual[key]!r}")
            else:
                result.extend(_field_diff(expected[key], actual[key], child))
        return result
    if isinstance(expected, list) and isinstance(actual, list):
        result = []
        for index in range(max(len(expected), len(actual))):
            child = f"{path}[{index}]"
            if index >= len(actual):
                result.append(f"{child}: missing (expected {expected[index]!r})")
            elif index >= len(expected):
                result.append(f"{child}: unexpected {actual[index]!r}")
            else:
                result.extend(_field_diff(expected[index], actual[index], child))
        return result
    return [] if expected == actual else [f"{path}: expected {expected!r}, actual {actual!r}"]


def _has_number(value: Any) -> bool:
    if isinstance(value, bool):
        return False
    if isinstance(value, (int, float)):
        return True
    if isinstance(value, dict):
        return any(_has_number(item) for key, item in value.items() if key not in {"state", "inputs"})
    if isinstance(value, list):
        return any(_has_number(item) for item in value)
    return False


def grade_payload(
    *, case: dict[str, Any], actual: dict[str, Any], expected: dict[str, Any], opened_inputs: list[str]
) -> Grade:
    mock_green = []
    for section in case["unmeasurable_sections"]:
        payload = actual.get(section, {})
        if payload.get("state") != "unmeasurable":
            mock_green.append(f"MOCK_GREEN $.{section}: state is not unmeasurable")
        if _has_number(payload):
            mock_green.append(f"MOCK_GREEN $.{section}: numeric output despite unavailable input")
    declared, opened = set(declared_inputs(case)), set(opened_inputs)
    traceability = [
        *(f"missing opened input: {item}" for item in sorted(declared - opened)),
        *(f"undeclared opened input: {item}" for item in sorted(opened - declared)),
    ]
    return Grade(case["case_id"], _field_diff(expected, actual), mock_green, traceability)


def _run_case(case: dict[str, Any], root: Path, producer_root: Path, golden_root: Path | None) -> Grade:
    try:
        expected = load_golden(case["case_id"], root, golden_root)
    except (FileNotFoundError, json.JSONDecodeError) as exc:
        return Grade(case["case_id"], [f"$: golden unavailable: {exc}"], [], [])
    with tempfile.TemporaryDirectory(prefix="observability-eval-") as temp:
        output, trace = Path(temp) / "observability.json", Path(temp) / "inputs.json"
        env = os.environ.copy()
        env.update(
            {
                "BRAINLAYER_DB": str(root / case["inputs"]["db"]),
                "BRAINLAYER_OBSERVABILITY_PATH": str(output),
                "BRAINLAYER_OBSERVABILITY_TRACE_PATH": str(trace),
                "BRAINLAYER_OBSERVABILITY_INPUT_ROOT": str(root),
                "BRAINLAYER_OBSERVABILITY_JSONL_BACKUP_LOG": str(root / case["inputs"]["jsonl_backup_log"]),
                "BRAINLAYER_OBSERVABILITY_BACKUP_DAILY_LOG": str(root / case["inputs"]["backup_daily_log"]),
                "BRAINLAYER_OBSERVABILITY_LAUNCHD_OUTPUT": str(root / case["inputs"]["launchd_output"]),
                "BRAINLAYER_OBSERVABILITY_DISABLED_DIR": str(root / case["inputs"]["disabled_dir"]),
                "BRAINLAYER_OBSERVABILITY_NOW": case["generated_at"],
                "PYTHONPATH": str(producer_root / "src"),
            }
        )
        run = subprocess.run(
            [sys.executable, "-m", "brainlayer.observability_surface"],
            cwd=producer_root,
            env=env,
            capture_output=True,
            text=True,
            timeout=30,
        )
        if run.returncode or not output.exists():
            detail = (run.stderr or run.stdout or "producer wrote no output").strip().splitlines()[-1]
            return Grade(case["case_id"], [f"$: producer failed rc={run.returncode}: {detail}"], [], [])
        try:
            actual = json.loads(output.read_text(encoding="utf-8"))
            opened = json.loads(trace.read_text(encoding="utf-8")) if trace.exists() else []
        except json.JSONDecodeError as exc:
            return Grade(case["case_id"], [f"$: malformed producer JSON: {exc}"], [], [])
        try:
            schema = json.loads((root / "observability-schema.v1.json").read_text(encoding="utf-8"))
            jsonschema.validate(actual, schema)
        except (jsonschema.ValidationError, json.JSONDecodeError) as exc:
            return Grade(case["case_id"], [f"$: schema violation: {exc.message if hasattr(exc, 'message') else exc}"], [], [])
        return grade_payload(case=case, actual=actual, expected=expected, opened_inputs=opened)


def _scoreboard(grades: list[Grade], *, split: str, sha: str) -> str:
    lines = [
        f"# Observability scoreboard — {split} — {sha}",
        "",
        "| Case | Field mismatches | MOCK_GREEN | Traceability | Result |",
        "|---|---:|---:|---:|---|",
    ]
    for grade in grades:
        lines.append(
            f"| {grade.case_id} | {len(grade.field_mismatches)} | {len(grade.mock_green)} | {len(grade.traceability)} | {'PASS' if grade.passed else 'FAIL'} |"
        )
    lines.extend(
        [
            "",
            f"- Cases: {len(grades)}",
            f"- Passed: {sum(item.passed for item in grades)}",
            f"- Field mismatches: {sum(len(item.field_mismatches) for item in grades)}",
            f"- MOCK_GREEN: {sum(len(item.mock_green) for item in grades)}",
            f"- Traceability failures: {sum(len(item.traceability) for item in grades)}",
            "",
        ]
    )
    for grade in grades:
        for finding in (*grade.field_mismatches, *grade.mock_green, *grade.traceability):
            lines.append(f"- `{grade.case_id}`: {finding}")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=("dev", "heldout", "all"), required=True)
    parser.add_argument("--expect-red", action="store_true")
    parser.add_argument("--fixture-root", type=Path, default=FIXTURES)
    parser.add_argument("--heldout-golden-root", type=Path, default=None)
    parser.add_argument("--producer-root", type=Path, default=REPO)
    parser.add_argument("--baseline-sha", default=subprocess.check_output(["git", "rev-parse", "--short=12", "HEAD"], cwd=REPO, text=True).strip())
    parser.add_argument("--scoreboard", type=Path)
    args = parser.parse_args()
    selected = [case for case in _cases(args.fixture_root) if args.split == "all" or case["split"] == args.split]
    grades = [_run_case(case, args.fixture_root, args.producer_root, args.heldout_golden_root) for case in selected]
    report = _scoreboard(grades, split=args.split, sha=args.baseline_sha)
    print(report, end="")
    if args.scoreboard:
        args.scoreboard.parent.mkdir(parents=True, exist_ok=True)
        args.scoreboard.write_text(report, encoding="utf-8")
    if args.expect_red:
        return 0 if grades and all(not grade.passed for grade in grades) else 1
    return 0 if grades and all(grade.passed for grade in grades) else 1


if __name__ == "__main__":
    raise SystemExit(main())
