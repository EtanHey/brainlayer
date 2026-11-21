"""Replay validation includes unchanged redirects shadowing newly inserted rows."""

import runpy
from pathlib import Path

import pytest

check_alias_retargets = runpy.run_path(str(Path(__file__).parents[1] / "scripts/check_t3_alias_retarget.py"))[
    "check_alias_retargets"
]


def retirement():
    own = dict(id="incoming-id", source="realtime_watcher", conversation_id="native", source_file="native.jsonl")
    return dict(
        id="incoming-id",
        before=["t3:mirror", "original-date"],
        after=None,
        old_target_source="t3",
        own_row=own,
        incoming=dict(own),
        resolved_id="incoming-id",
        preimage=["incoming-id", "t3:mirror", "original-date"],
    )


def test_gate_accepts_only_id_matched_reversible_retirement():
    result = dict(new_t3_aliases=0, new_shadowed_ids=0, changed_alias_rows=1)
    assert check_alias_retargets(result, [retirement()]) == 1


@pytest.mark.parametrize(
    "mutation",
    [
        "shadow",
        "new-t3",
        "unaccounted",
        "duplicate",
        "retarget",
        "wrong-id",
        "wrong-session",
        "wrong-file",
        "wrong-preimage",
        "still-shadowed",
    ],
)
def test_gate_rejects_unsafe_alias_changes(mutation):
    result = dict(new_t3_aliases=0, new_shadowed_ids=0, changed_alias_rows=1)
    case = retirement()
    cases = [case]
    if mutation == "shadow":
        result.update(changed_alias_rows=0, new_shadowed_ids=1)
        cases = []
    elif mutation == "new-t3":
        result["new_t3_aliases"] = 1
    elif mutation == "unaccounted":
        result["changed_alias_rows"] = 2
    elif mutation == "duplicate":
        cases.append(retirement())
        result["changed_alias_rows"] = 2
    elif mutation == "retarget":
        case["after"] = ["unrelated-older-session", "today"]
    elif mutation == "wrong-preimage":
        case["preimage"][1] = "other"
    elif mutation == "still-shadowed":
        case["resolved_id"] = "t3:mirror"
    else:
        field = {"wrong-id": "id", "wrong-session": "conversation_id", "wrong-file": "source_file"}[mutation]
        case["own_row"][field] = "other"
    with pytest.raises(ValueError):
        check_alias_retargets(result, cases)
