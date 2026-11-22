"""The gate preserves every existing alias and checks effective active-own reads."""

import runpy
from pathlib import Path

import pytest

check_alias_retargets = runpy.run_path(str(Path(__file__).parents[1] / "scripts/check_t3_alias_retarget.py"))[
    "check_alias_retargets"
]


def receipt():
    result = dict(new_t3_aliases=0, deleted_alias_rows=0, changed_existing_alias_rows=0, active_own_rows_checked=1)
    cases = [dict(id="own", active=True, resolved_id="own", get_chunk_id="own", expanded_id="own")]
    return result, cases


def test_gate_accepts_unchanged_alias_and_active_own_reads():
    result, cases = receipt()
    assert check_alias_retargets(result, cases) == 1


@pytest.mark.parametrize(
    "mutation",
    ["new-t3", "delete", "change", "missing", "duplicate", "inactive", "resolved_id", "get_chunk_id", "expanded_id"],
)
def test_gate_rejects_alias_mutation_or_wrong_read_precedence(mutation):
    result, cases = receipt()
    if mutation in {"new-t3", "delete", "change"}:
        field = {"new-t3": "new_t3_aliases", "delete": "deleted_alias_rows", "change": "changed_existing_alias_rows"}[
            mutation
        ]
        result[field] = 1
    elif mutation == "missing":
        result["active_own_rows_checked"] = 2
    elif mutation == "duplicate":
        cases.append(dict(cases[0]))
        result["active_own_rows_checked"] = 2
    elif mutation == "inactive":
        cases[0]["active"] = False
    else:
        cases[0][mutation] = "t3:mirror"
    with pytest.raises(ValueError):
        check_alias_retargets(result, cases)
