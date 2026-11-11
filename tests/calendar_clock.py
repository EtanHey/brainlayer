"""A ticking wall clock; elapsed-time and process-custody clocks remain real."""

import datetime as dt
import sys
import time


def pin_wall_clock(monkeypatch, instant):
    original_datetime, original_date = dt.datetime, dt.date
    started = time.monotonic()

    class DateTimeMeta(type):
        def __instancecheck__(cls, instance):
            return isinstance(instance, original_datetime)

    class DateMeta(type):
        def __instancecheck__(cls, instance):
            return isinstance(instance, original_date)

    class DateTime(original_datetime, metaclass=DateTimeMeta):
        @classmethod
        def now(cls, tz=None):
            return cls.fromtimestamp(instant.timestamp() + time.monotonic() - started, tz)

        @classmethod
        def utcnow(cls):
            return cls.now(dt.UTC).replace(tzinfo=None)

    class Date(original_date, metaclass=DateMeta):
        @classmethod
        def today(cls):
            return DateTime.now().date()

    # Also cover `from datetime import datetime` bindings made during collection.
    for module in tuple(sys.modules.values()):
        if module is None or not getattr(module, "__name__", "").startswith(("brainlayer.", "tests.")):
            continue
        for name, value in tuple(vars(module).items()):
            if value is original_datetime:
                monkeypatch.setattr(module, name, DateTime)
            elif value is original_date:
                monkeypatch.setattr(module, name, Date)
    monkeypatch.setattr(dt, "datetime", DateTime)
    monkeypatch.setattr(dt, "date", Date)
    monkeypatch.setattr(time, "time", lambda: instant.timestamp() + time.monotonic() - started)
