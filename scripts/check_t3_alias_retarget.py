"""Validate immutable existing aliases and lifecycle-aware read precedence after replay."""

import argparse
import json
from pathlib import Path


def check_alias_retargets(result: dict, cases: list[dict]) -> int:
    """No existing alias may change; every active own row must resolve to itself."""
    if result["new_t3_aliases"] or result["deleted_alias_rows"] or result["changed_existing_alias_rows"]:
        raise ValueError("new T3 alias or an existing alias changed/deleted")
    if result["active_own_rows_checked"] != len(cases) or len({case["id"] for case in cases}) != len(cases):
        raise ValueError("missing or duplicate active-own resolution evidence")
    for case in cases:
        if not case["active"] or any(
            case[field] != case["id"] for field in ("resolved_id", "get_chunk_id", "expanded_id")
        ):
            raise ValueError("an active own row resolved through its alias")
    return len(cases)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("receipt", type=Path)
    args = parser.parse_args()
    receipt = json.loads(args.receipt.read_text())
    count = check_alias_retargets(receipt["result"], receipt["active_own_reads"])
    print(f"PASS: 0 alias deletions/changes; 0 new T3 aliases; {count} active own rows resolve to themselves")


if __name__ == "__main__":
    main()
