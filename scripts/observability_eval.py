#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import jsonschema

# fmt: off
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
def _heldout_digest(root: Path, cases: list[dict[str, Any]]) -> str:
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode="w", format=tarfile.PAX_FORMAT) as archive:
        for name in sorted(case["golden"] for case in cases if case["split"] == "heldout"):
            payload = (root / name).read_bytes()
            info = tarfile.TarInfo(name)
            info.size, info.mtime, info.uid, info.gid = len(payload), 20260913, 0, 0
            info.uname, info.gname, info.mode = "", "", 0o644
            archive.addfile(info, io.BytesIO(payload))
    return hashlib.sha256(stream.getvalue()).hexdigest()
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
    declared, opened = set(case["declared_inputs"]), set(opened_inputs)
    traceability = [
        *(f"missing opened input: {item}" for item in sorted(declared - opened)),
        *(f"undeclared opened input: {item}" for item in sorted(opened - declared)),
    ]
    return Grade(case["case_id"], _field_diff(expected, actual), mock_green, traceability)


def _stage_case_inputs(case: dict[str, Any], source_root: Path, staged_root: Path) -> None:
    source_root, staged_root = source_root.resolve(), staged_root.resolve()
    declared = case["declared_inputs"]
    mtimes = case.get("input_mtimes", {})
    missing = sorted(set(declared) - set(mtimes))
    if missing:
        raise ValueError(f"missing input_mtimes for declared inputs: {', '.join(missing)}")
    unexpected = sorted(set(mtimes) - set(declared))
    if unexpected:
        raise ValueError(f"input_mtimes contains undeclared inputs: {', '.join(unexpected)}")
    for relative in declared:
        source = (source_root / relative).resolve()
        target = (staged_root / relative).resolve()
        if source_root not in source.parents and source != source_root:
            raise ValueError(f"declared input escapes fixture root: {relative}")
        if staged_root not in target.parents and target != staged_root:
            raise ValueError(f"declared input escapes staging root: {relative}")
        if source.is_dir():
            shutil.copytree(source, target)
        elif source.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
        if target.exists():
            timestamp = datetime.fromisoformat(mtimes[relative].replace("Z", "+00:00")).timestamp()
            os.utime(target, (timestamp, timestamp))


def _run_case(
    case: dict[str, Any], root: Path, producer_root: Path, golden_root: Path | None, *, stage_inputs: bool = True
) -> Grade:
    root, producer_root = root.resolve(), producer_root.resolve()
    golden_root = golden_root.resolve() if golden_root is not None else None
    try:
        expected = load_golden(case["case_id"], root, golden_root)
    except (FileNotFoundError, json.JSONDecodeError) as exc:
        return Grade(case["case_id"], [f"$: golden unavailable: {exc}"], [], [])
    with tempfile.TemporaryDirectory(prefix="observability-eval-") as temp:
        output, trace = Path(temp) / "observability.json", Path(temp) / "inputs.json"
        input_root = root
        if stage_inputs:
            input_root = Path(temp) / "inputs"
            input_root.mkdir()
            try:
                _stage_case_inputs(case, root, input_root)
            except (OSError, ValueError) as exc:
                return Grade(case["case_id"], [f"$: input staging failed: {exc}"], [], [])
        env = {key: os.environ[key] for key in ("HOME", "PATH") if key in os.environ}
        env.update({
            "BRAINLAYER_DB": str(input_root / case["inputs"]["db"]), "BRAINLAYER_OBSERVABILITY_PATH": str(output),
            "BRAINLAYER_OBSERVABILITY_TRACE_PATH": str(trace), "BRAINLAYER_OBSERVABILITY_INPUT_ROOT": str(input_root),
            "BRAINLAYER_OBSERVABILITY_JSONL_BACKUP_LOG": str(input_root / case["inputs"]["jsonl_backup_log"]),
            "BRAINLAYER_OBSERVABILITY_BACKUP_DAILY_LOG": str(input_root / case["inputs"]["backup_daily_log"]),
            "BRAINLAYER_OBSERVABILITY_LAUNCHD_OUTPUT": str(input_root / case["inputs"]["launchd_output"]),
            "BRAINLAYER_OBSERVABILITY_DISABLED_DIR": str(input_root / case["inputs"]["disabled_dir"]),
            "BRAINLAYER_OBSERVABILITY_NOW": case["generated_at"], "PYTHONPATH": str(producer_root / "src"),
            "BRAINLAYER_OBSERVABILITY_PRODUCER_ROOT": str(producer_root),
        })
        try:
            run = subprocess.run([sys.executable, "-m", "brainlayer.observability_surface"], cwd=producer_root,
                env=env, capture_output=True, text=True, timeout=30)
        except subprocess.TimeoutExpired:
            return Grade(case["case_id"], ["$: producer timed out after 30 seconds"], [], [])
        except OSError as exc:
            return Grade(case["case_id"], [f"$: producer could not start: {exc}"], [], [])
        if run.returncode or not output.exists():
            details = (run.stderr or run.stdout or "").strip().splitlines()
            detail = details[-1] if details else "producer failed without output"
            return Grade(case["case_id"], [f"$: producer failed rc={run.returncode}: {detail}"], [], [])
        try:
            actual = json.loads(output.read_text(encoding="utf-8"))
            opened = json.loads(trace.read_text(encoding="utf-8")) if trace.exists() else []
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            return Grade(case["case_id"], [f"$: malformed producer JSON: {exc}"], [], [])
        if not isinstance(opened, list) or not all(isinstance(item, str) for item in opened):
            return Grade(case["case_id"], ["$: malformed input trace: expected a JSON list of strings"], [], [])
        try:
            schema = json.loads((root / "observability-schema.v1.json").read_text(encoding="utf-8"))
            jsonschema.validate(actual, schema)
        except (jsonschema.ValidationError, json.JSONDecodeError) as exc:
            return Grade(
                case["case_id"], [f"$: schema violation: {exc.message if hasattr(exc, 'message') else exc}"], [], []
            )
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
    absolute_path = lambda value: Path(value).resolve()  # noqa: E731
    parser.add_argument("--fixture-root", type=absolute_path, default=FIXTURES)
    parser.add_argument("--heldout-golden-root", type=absolute_path, default=None)
    parser.add_argument("--producer-root", type=absolute_path, default=REPO)
    parser.add_argument("--baseline-sha")
    parser.add_argument("--scoreboard", type=Path)
    args = parser.parse_args()
    if not (args.producer_root / "src/brainlayer").is_dir():
        parser.error(f"producer root has no src/brainlayer package: {args.producer_root}")
    if args.baseline_sha is None:
        args.baseline_sha = subprocess.check_output(["git", "rev-parse", "--short=12", "HEAD"], cwd=REPO, text=True).strip()
    manifest = json.loads((args.fixture_root / "cases.json").read_text(encoding="utf-8"))
    selected = [case for case in manifest["cases"] if args.split == "all" or case["split"] == args.split]
    heldout_selected = any(case["split"] == "heldout" for case in selected)
    if heldout_selected and args.heldout_golden_root is None:
        parser.error("held-out golden root required for held-out cases")
    if heldout_selected:
        try:
            digest = _heldout_digest(args.heldout_golden_root, manifest["cases"])
        except OSError as exc:
            parser.error(f"held-out golden seal unavailable: {exc}")
        if digest != manifest["heldout_goldens_sha256"]:
            parser.error(f"held-out golden seal mismatch: expected {manifest['heldout_goldens_sha256']}, got {digest}")
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
# fmt: on
