"""Opt-in wall clock for isolated pytest children; monotonic deadlines remain real."""

import os

if os.environ.get("BRAINLAYER_TEST_CLOCK"):
    import datetime as dt
    import time

    base = dt.datetime.fromisoformat(os.environ["BRAINLAYER_TEST_CLOCK"])
    real_dt, real_date = dt.datetime, dt.date
    epoch, mono = base.timestamp(), time.monotonic()

    class ClockDateTime(real_dt):
        @classmethod
        def now(cls, tz=None):
            current = base + dt.timedelta(seconds=time.monotonic() - mono)
            value = current.astimezone(tz) if tz else current.astimezone().replace(tzinfo=None)
            return cls.fromisoformat(value.isoformat())

        @classmethod
        def utcnow(cls):
            return cls.now(dt.timezone.utc).replace(tzinfo=None)

    class ClockDate(real_date):
        @classmethod
        def today(cls):
            return ClockDateTime.now().date()

    dt.datetime, dt.date = ClockDateTime, ClockDate
    time.time = lambda: epoch + time.monotonic() - mono

# A clock run must preserve the subprocess socket guard when this file is first on PYTHONPATH.
if os.environ.get("BRAINLAYER_FORBID_BRAINBAR_SOCKET") == "1":
    import runpy
    from pathlib import Path

    runpy.run_path(str(Path(__file__).resolve().parents[1] / "socket_hygiene/sitecustomize.py"))
