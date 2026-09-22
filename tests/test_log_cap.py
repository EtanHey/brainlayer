"""Bound launchd logs without changing the inode held by a running job."""

import os
import plistlib
import subprocess
import sys
from pathlib import Path

import pytest

from brainlayer.log_cap import cap_job_logs


def _agent(agents_dir: Path, name: str, log_path: Path) -> None:
    (agents_dir / f"com.brainlayer.{name}.plist").write_bytes(
        plistlib.dumps({"Label": f"com.brainlayer.{name}", "StandardErrorPath": str(log_path)})
    )


def test_cap_keeps_recent_tail_and_open_append_writer(tmp_path):
    agents_dir = tmp_path / "agents"
    logs_dir = tmp_path / "logs"
    agents_dir.mkdir()
    logs_dir.mkdir()
    log_path = logs_dir / "hotlane-brainbar.err.log"
    log_path.write_bytes(b"old line\n" * 30 + b"recent error\n")
    _agent(agents_dir, "hotlane-brainbar", log_path)
    before_inode = log_path.stat().st_ino

    writer = os.open(log_path, os.O_WRONLY | os.O_APPEND)
    try:
        assert cap_job_logs(agents_dir, max_bytes=100, keep_bytes=60) == [log_path]
        os.write(writer, b"next cycle\n")
    finally:
        os.close(writer)

    assert log_path.stat().st_ino == before_inode
    assert log_path.stat().st_size <= 100
    assert log_path.read_bytes().endswith(b"recent error\nnext cycle\n")


def test_cap_does_not_overwrite_append_after_truncate(tmp_path, monkeypatch):
    agents_dir = tmp_path / "agents"
    agents_dir.mkdir()
    log_path = tmp_path / "job.err.log"
    log_path.write_bytes(b"old\n" * 50 + b"tail\n")
    _agent(agents_dir, "job", log_path)
    writer = os.open(log_path, os.O_WRONLY | os.O_APPEND)
    truncate = os.ftruncate

    def append_during_truncate(fd, size):
        truncate(fd, size)
        os.write(writer, b"new write\n")

    monkeypatch.setattr(os, "ftruncate", append_during_truncate)
    try:
        cap_job_logs(agents_dir, max_bytes=100, keep_bytes=50)
    finally:
        os.close(writer)

    assert b"new write\n" in log_path.read_bytes()
    assert log_path.read_bytes().endswith(b"tail\n")


def test_cap_covers_installed_external_log_and_ignores_unregistered_files(tmp_path):
    agents_dir = tmp_path / "agents"
    logs_dir = tmp_path / "logs"
    agents_dir.mkdir()
    logs_dir.mkdir()
    outside = tmp_path / "brainlayer-gemini-loopback.log"
    outside.write_bytes(b"loopback" * 100)
    unrelated = logs_dir / "manual.log"
    unrelated.write_bytes(b"manual" * 100)
    _agent(agents_dir, "outside", outside)
    personal = tmp_path / "personal.log"
    personal.write_bytes(b"private" * 100)
    (agents_dir / "com.brainlayer.fake.plist").write_bytes(
        plistlib.dumps({"Label": "com.otherapp", "StandardErrorPath": str(personal)})
    )

    assert cap_job_logs(agents_dir, max_bytes=100, keep_bytes=50) == [outside]
    assert outside.stat().st_size == 50
    assert unrelated.stat().st_size == 600
    assert personal.stat().st_size == 700


def test_cap_covers_both_stdout_and_stderr_from_installed_job(tmp_path):
    agents_dir = tmp_path / "agents"
    agents_dir.mkdir()
    stdout = tmp_path / "job.out.log"
    stderr = tmp_path / "job.err.log"
    stdout.write_bytes(b"o" * 200)
    stderr.write_bytes(b"e" * 200)
    (agents_dir / "com.brainlayer.job.plist").write_bytes(
        plistlib.dumps(
            {
                "Label": "com.brainlayer.job",
                "StandardOutPath": str(stdout),
                "StandardErrorPath": str(stderr),
            }
        )
    )

    assert cap_job_logs(agents_dir, max_bytes=100, keep_bytes=50) == [stderr, stdout]
    assert stdout.read_bytes() == b"o" * 50
    assert stderr.read_bytes() == b"e" * 50


def test_cap_refuses_symlink_instead_of_following_it(tmp_path):
    agents_dir = tmp_path / "agents"
    logs_dir = tmp_path / "logs"
    agents_dir.mkdir()
    logs_dir.mkdir()
    target = tmp_path / "personal.log"
    target.write_bytes(b"personal" * 100)
    link = logs_dir / "hotlane.err.log"
    link.symlink_to(target)
    _agent(agents_dir, "hotlane", link)

    with pytest.raises(ValueError, match="symlink"):
        cap_job_logs(agents_dir, max_bytes=100, keep_bytes=50)
    assert target.stat().st_size == 800


def test_cap_reports_missing_job_inventory(tmp_path):
    with pytest.raises(FileNotFoundError, match="LaunchAgents directory missing"):
        cap_job_logs(tmp_path / "missing")

    agents_dir = tmp_path / "agents"
    agents_dir.mkdir()
    with pytest.raises(RuntimeError, match="no BrainLayer job log paths"):
        cap_job_logs(agents_dir)


@pytest.mark.parametrize("bad_kind", ["malformed", "unreadable"])
def test_bad_plist_reports_error_after_capping_valid_logs(tmp_path, bad_kind):
    agents_dir = tmp_path / "agents"
    agents_dir.mkdir()
    log_path = tmp_path / "valid.err.log"
    log_path.write_bytes(b"v" * 200)
    _agent(agents_dir, "valid", log_path)
    bad = agents_dir / "com.brainlayer.bad.plist"
    if bad_kind == "malformed":
        bad.write_bytes(b'<?xml version="1.0"?><plist><dict>')
    else:
        bad.mkdir()  # Opening this matching path raises IsADirectoryError.

    with pytest.raises(RuntimeError, match=r"com\.brainlayer\.bad: (ExpatError|IsADirectoryError)"):
        cap_job_logs(agents_dir, max_bytes=100, keep_bytes=50)
    assert log_path.read_bytes() == b"v" * 50


def test_bad_only_plist_still_reports_missing_valid_inventory(tmp_path):
    agents_dir = tmp_path / "agents"
    agents_dir.mkdir()
    (agents_dir / "com.brainlayer.bad.plist").write_bytes(b"not a plist")

    with pytest.raises(RuntimeError, match="no BrainLayer job log paths") as error:
        cap_job_logs(agents_dir)
    assert "com.brainlayer.bad: InvalidFileException" in str(error.value)


def test_non_plist_siblings_are_not_parsed(tmp_path):
    agents_dir = tmp_path / "agents"
    agents_dir.mkdir()
    log_path = tmp_path / "valid.err.log"
    log_path.write_bytes(b"v" * 200)
    _agent(agents_dir, "valid", log_path)
    for suffix in (".plist.loaded-keg", ".plist.bak-20260922", ".plist.PAUSED-20260922"):
        (agents_dir / f"com.brainlayer.bad{suffix}").write_bytes(b"not a plist")

    assert cap_job_logs(agents_dir, max_bytes=100, keep_bytes=50) == [log_path]


def test_cli_caps_valid_log_and_exits_nonzero_for_each_bad_plist(tmp_path):
    agents_dir = tmp_path / "agents"
    agents_dir.mkdir()
    log_path = tmp_path / "valid.err.log"
    log_path.write_bytes(b"v" * 200)
    _agent(agents_dir, "valid", log_path)
    for name in ("bad-one", "bad-two"):
        (agents_dir / f"com.brainlayer.{name}.plist").write_bytes(b"not a plist")

    script = Path(__file__).resolve().parents[1] / "src/brainlayer/log_cap.py"
    result = subprocess.run(
        [sys.executable, str(script), "--agents-dir", str(agents_dir), "--max-bytes", "100", "--keep-bytes", "50"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 1
    assert "com.brainlayer.bad-one: InvalidFileException" in result.stderr
    assert "com.brainlayer.bad-two: InvalidFileException" in result.stderr
    assert log_path.read_bytes() == b"v" * 50


def test_cap_job_is_installed_with_the_all_mode():
    root = Path(__file__).resolve().parents[1]
    plist = plistlib.loads((root / "scripts/launchd/com.brainlayer.log-cap.plist").read_bytes())
    installer = (root / "scripts/launchd/install.sh").read_text()

    assert plist["Label"] == "com.brainlayer.log-cap"
    assert plist["StartInterval"] <= 300
    assert plist["ProgramArguments"][-2:] == ["-m", "brainlayer.log_cap"]
    assert "install_plist log-cap" in installer
    assert "install_many maintenance-nightly maintenance-weekly health-check log-cap" in installer
    assert "remove_plist log-cap" in installer
