"""Clock cleanup must also cover modules imported while a fixture is active."""

import datetime as dt
import sys
import time
from types import ModuleType

import pytest

from tests.calendar_clock import pin_wall_clock


@pytest.mark.parametrize("name", ["brainlayer._clock_alias_probe", "_third_party_clock_alias_probe"])
def test_clock_restores_aliases_imported_after_setup(name):
    original_datetime, original_date, original_time = dt.datetime, dt.date, time.time
    late = ModuleType(name)
    with pytest.MonkeyPatch.context() as patch:
        restore = pin_wall_clock(patch, dt.datetime(2026, 10, 18, 12, tzinfo=dt.UTC))
        exec("from datetime import datetime, date; from time import time", late.__dict__)
        patch.setitem(sys.modules, late.__name__, late)
        if restore is not None:
            restore()
        patch.undo()
        assert late.datetime is original_datetime
        assert late.date is original_date
        assert late.time is original_time
