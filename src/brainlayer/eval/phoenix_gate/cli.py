"""CLI entrypoint for cron/CI-style Phoenix regression checks."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from brainlayer.eval.phoenix_gate.baseline_store import DEFAULT_BASELINE_STORE_PATH, JsonBaselineStore
from brainlayer.eval.phoenix_gate.models import HarnessFault
from brainlayer.eval.phoenix_gate.phoenix_client import DEFAULT_BASE_URL, PhoenixRestClient
from brainlayer.eval.phoenix_gate.regression_gate import RegressionGate
from brainlayer.eval.phoenix_gate.triggers import diff_rerun_triggers


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Check a Phoenix experiment against stored GREEN baselines.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    check = subparsers.add_parser("check", help="Fetch and score a Phoenix experiment.")
    check.add_argument("--experiment-id", required=True, help="Phoenix REST experiment ID, e.g. RXhwZXJpbWVudDoxMA==")
    check.add_argument(
        "--expected-evaluator",
        action="append",
        dest="expected_evaluators",
        required=True,
        help="Expected evaluator name. Repeat for every evaluator in the suite.",
    )
    check.add_argument("--baseline-store", default=str(DEFAULT_BASELINE_STORE_PATH), help="JSON baseline store path.")
    check.add_argument("--base-url", default=DEFAULT_BASE_URL, help="Phoenix base URL.")
    check.add_argument("--threshold", type=float, default=0.0, help="Allowed score drop before alarm.")

    triggers = subparsers.add_parser("triggers", help="Compare two trigger manifests.")
    triggers.add_argument("--previous-manifest", required=True, help="Path to previous trigger manifest JSON.")
    triggers.add_argument("--current-manifest", required=True, help="Path to current trigger manifest JSON.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "triggers":
        try:
            previous = json.loads(Path(args.previous_manifest).read_text())
            current = json.loads(Path(args.current_manifest).read_text())
            diff = diff_rerun_triggers(previous, current)
        except (HarnessFault, json.JSONDecodeError, OSError) as exc:
            print(json.dumps({"status": "HARNESS_FAULT", "error": str(exc)}, sort_keys=True), file=sys.stderr)
            return 2
        print(json.dumps(diff.to_dict(), indent=2, sort_keys=True))
        return 1 if diff.rerun_required else 0

    if args.command != "check":
        raise AssertionError(f"unhandled command {args.command!r}")

    try:
        client = PhoenixRestClient(base_url=args.base_url)
        score = client.load_experiment_score(
            args.experiment_id,
            expected_evaluators=set(args.expected_evaluators),
        )
        gate = RegressionGate(JsonBaselineStore(Path(args.baseline_store)), threshold=args.threshold)
        verdict = gate.evaluate(score)
    except HarnessFault as exc:
        print(json.dumps({"status": "HARNESS_FAULT", "error": str(exc)}, sort_keys=True), file=sys.stderr)
        return 2

    print(json.dumps(verdict.to_dict(), indent=2, sort_keys=True))
    return 1 if verdict.alarm else 0


if __name__ == "__main__":
    raise SystemExit(main())
