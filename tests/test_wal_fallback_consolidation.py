"""Fallback ownership through scratch launchd migration, never live launchctl."""

from brainlayer.supervisor_jobs import load_components
from tests.test_consolidation_plan import NEW, ROOT
from tests.test_consolidation_plan import setup as legacy_install  # noqa: F401
from tests.test_launchd_consolidation import Launchctl, invoke


def test_wal_fallback_has_one_consolidated_owner_and_no_legacy_schedule(request):
    fixture_setup = request.getfixturevalue("legacy_install")
    home, agents, _ = fixture_setup
    old = "com.brainlayer.wal-checkpoint"
    (agents / f"{old}.plist").write_bytes((ROOT / f"scripts/launchd/{old}.plist").read_bytes())
    runner = Launchctl()
    runner.loaded.add(old)
    invoke(fixture_setup, runner)
    control = home / ".local/share/brainlayer/supervisor"
    components = load_components(control / "jobs.json", control / "control.sock")
    assert "wal-checkpoint-fallback" in components and "wal-checkpoint" not in components
    assert old not in runner.loaded and not (agents / f"{old}.plist").exists()
    assert not (ROOT / "scripts/launchd/com.brainlayer.wal-checkpoint-fallback.plist").exists()
    assert runner.loaded == {NEW, "com.brainlayer.maintenance"}
