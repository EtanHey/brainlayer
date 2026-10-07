"""The native ratchet refuses the ordinary suite's capability skips."""

import pytest

from scripts.retirement_native import require_native_matrix


def native_log():
    names = [
        "testNativeNetworkBoundaryIsArmed",
        "testEveryPaletteRejectsRetiredDispatch",
        "testActualTemporaryStoreDigestAndSearchRemainLocal",
    ]
    return "\n".join(f"Test Case '{name}' passed" for name in names) + "\nExecuted 3 tests, with 0 failures"


def test_complete_matrix_requires_three_passed_and_zero_skipped():
    assert require_native_matrix(native_log()) == {"tests": 3, "skipped": 0}


@pytest.mark.parametrize(
    "change",
    [
        lambda log: log.replace(
            "testNativeNetworkBoundaryIsArmed' passed", "testNativeNetworkBoundaryIsArmed' skipped"
        ),
        lambda log: log.replace("with 0 failures", "with 1 test skipped and 0 failures"),
        lambda log: log.replace("Executed 3", "Executed 2"),
    ],
)
def test_skipped_or_incomplete_native_matrix_is_red(change):
    with pytest.raises(ValueError):
        require_native_matrix(change(native_log()))
