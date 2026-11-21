"""Validate a replay receipt's only permitted change to existing T3 aliases.

Receipts contain row identities, fingerprints and decisions, never session text.
The original before/after comparison must count *all* changed existing T3 aliases.
"""

import argparse
import json
from pathlib import Path


def check_alias_retargets(result: dict, cases: list[dict]) -> int:
    """Fail closed unless every changed alias moves to its T3 row's session twin."""
    if result["new_t3_aliases"] != 0 or result["changed_existing_t3_aliases"] != len(cases):
        raise ValueError("unexpected T3 alias creation or unaccounted existing-alias change")
    if len({case["alias_id"] for case in cases}) != len(cases):
        raise ValueError("duplicate alias evidence cannot account for separate changes")
    for case in cases:
        old, new = case["rows"]["main"], case["rows"]["branch"]
        old_id, new_id = case["main_alias_target"], case["branch_alias_target"]
        session = old.get("provider_session_id")
        if not (
            old["id"] == old_id
            and new["id"] == new_id
            and old_id != new_id
            and old["source"] == "t3"
            and new["source"] == "realtime_watcher"
            and session
            and new["conversation_id"] == session
            and Path(new["source_file"]).stem == session
            and all(
                old.get(key) is not None and old[key] == new.get(key)
                for key in ("source_class", "project", "content_type", "dedupe_hash", "content_hash")
            )
        ):
            raise ValueError("alias did not move from T3 to its exact provider-session twin")
        for clone in ("main", "branch"):
            for code, expected in (("main", old_id), ("branch", new_id)):
                if case["same_input_decisions"][f"{clone}_clone_{code}_code"] != expected:
                    raise ValueError("same-input decision does not reproduce the alias transition")
            audit = case[f"{clone}_latest_audit"]
            expected = old_id if clone == "main" else new_id
            if (audit["chunk_id_dropped"], audit["chunk_id_kept"], audit["mechanism"], audit["hamming_distance"]) != (
                case["alias_id"],
                expected,
                "sha256",
                0,
            ):
                raise ValueError("alias transition lacks an exact-match replay audit")
    return len(cases)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result", type=Path)
    parser.add_argument("cases", type=Path, nargs="+")
    args = parser.parse_args()
    result = json.loads(args.result.read_text())["branch"]
    count = check_alias_retargets(result, [json.loads(path.read_text()) for path in args.cases])
    print(f"PASS: 0 new T3 aliases; {count} exact T3-to-own-provider-session retarget(s); 0 unexplained changes")


if __name__ == "__main__":
    main()
