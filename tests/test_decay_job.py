"""Decay job archives via archived_at only. Behavioral coverage lives in test_decay.py."""

from brainlayer.decay_job import run_decay_job


def test_decay_job_module_exports_run():
    assert callable(run_decay_job)
