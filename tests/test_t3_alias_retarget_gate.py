"""The replay gate accepts only a proven T3-to-own-session-twin transition."""

import copy
import runpy
from pathlib import Path

import pytest

check_alias_retargets = runpy.run_path(str(Path(__file__).parents[1] / "scripts/check_t3_alias_retarget.py"))[
    "check_alias_retargets"
]


def case():
    common = dict(
        source_class="cli-agent",
        project="test",
        content_type="user_message",
        dedupe_hash="hash",
        content_hash="exact-hash",
    )
    return {
        "alias_id": "later-session-alias",
        "main_alias_target": "t3:mirror",
        "branch_alias_target": "session-twin",
        "rows": {
            "main": dict(common, id="t3:mirror", source="t3", provider_session_id="native"),
            "branch": dict(
                common,
                id="session-twin",
                source="realtime_watcher",
                conversation_id="native",
                source_file="/synthetic/native.jsonl",
            ),
        },
        "same_input_decisions": {
            f"{clone}_clone_{code}_code": target
            for clone in ("main", "branch")
            for code, target in (("main", "t3:mirror"), ("branch", "session-twin"))
        },
        **{
            f"{clone}_latest_audit": dict(
                chunk_id_dropped="later-session-alias", chunk_id_kept=target, mechanism="sha256", hamming_distance=0
            )
            for clone, target in (("main", "t3:mirror"), ("branch", "session-twin"))
        },
    }


def test_gate_accepts_only_accounted_exact_session_twin_retargets():
    assert check_alias_retargets(dict(new_t3_aliases=0, changed_existing_t3_aliases=0), []) == 0
    assert check_alias_retargets(dict(new_t3_aliases=0, changed_existing_t3_aliases=1), [case()]) == 1


@pytest.mark.parametrize(
    "mutation",
    [
        "new-alias",
        "missing-case",
        "duplicate-case",
        "wrong-session",
        "wrong-file",
        "wrong-class",
        "wrong-hash",
        "wrong-project",
        "t3-target",
        "near-match",
        "wrong-decision",
        "missing-audit",
    ],
)
def test_gate_rejects_unexplained_or_non_twin_changes(mutation):
    result = dict(new_t3_aliases=0, changed_existing_t3_aliases=1)
    item = copy.deepcopy(case())
    cases = [item]
    if mutation == "new-alias":
        result["new_t3_aliases"] = 1
    elif mutation == "missing-case":
        result["changed_existing_t3_aliases"] = 2
    elif mutation == "duplicate-case":
        cases.append(copy.deepcopy(item))
        result["changed_existing_t3_aliases"] = 2
    elif mutation in {"wrong-session", "wrong-file", "wrong-class", "wrong-hash", "wrong-project", "t3-target"}:
        field = {
            "wrong-session": "conversation_id",
            "wrong-file": "source_file",
            "wrong-class": "source_class",
            "wrong-hash": "content_hash",
            "wrong-project": "project",
            "t3-target": "source",
        }[mutation]
        item["rows"]["branch"][field] = "different"
    elif mutation == "near-match":
        item["branch_latest_audit"]["mechanism"] = "simhash"
    elif mutation == "wrong-decision":
        item["same_input_decisions"]["main_clone_main_code"] = "session-twin"
    else:
        item.pop("branch_latest_audit")
    with pytest.raises((ValueError, KeyError)):
        check_alias_retargets(result, cases)
