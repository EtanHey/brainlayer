"""Check every alias delta and newly shadowed id in a session replay receipt."""

import argparse
import json
from pathlib import Path


def check_alias_retargets(result: dict, cases: list[dict]) -> int:
    """Only retire a redirect to materialize its exact incoming id; never re-home it."""
    if result["new_t3_aliases"] or result["new_shadowed_ids"]:
        raise ValueError("new T3 alias or live session id shadowed by an alias")
    if result["changed_alias_rows"] != len(cases) or len({case["id"] for case in cases}) != len(cases):
        raise ValueError("unaccounted or duplicate alias changes")
    retired = 0
    for case in cases:
        before, after = case["before"], case["after"]
        if before is None:
            if case["target_source"] == "t3":
                raise ValueError("new T3 alias")
        elif after is not None:
            if before[0] != after[0]:
                raise ValueError("existing alias retargeted by text instead of incoming id")
        else:
            own, incoming = case["own_row"], case["incoming"]
            if not (
                case["preimage"] == [case["id"], *before]
                and own["id"] == incoming["id"] == case["id"]
                and own["source"] != "t3"
                and own["conversation_id"] == incoming["conversation_id"]
                and own["source_file"] == incoming["source_file"]
                and case["resolved_id"] == case["id"]
            ):
                raise ValueError("retired alias lacks its exact incoming session row and reversible preimage")
            retired += 1
    return retired


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("receipt", type=Path)
    args = parser.parse_args()
    receipt = json.loads(args.receipt.read_text())
    count = check_alias_retargets(receipt["result"], receipt["alias_changes"])
    print(f"PASS: 0 new T3 aliases; 0 new shadows; {count} id-matched reversible retirements; 0 retargets")


if __name__ == "__main__":
    main()
