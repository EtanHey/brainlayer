"""R3: the watcher's operational logs must actually reach launchd's log files.

`brainlayer watch` is the repo's one long-running daemon CLI command, and it was the
only one that never configured logging. Every `logger.info` in `watcher.py` -- including
the 60-second liveness heartbeat and the started/stopped markers -- was dropped on the
floor, because an unconfigured root logger sends nothing below WARNING to stderr.

The visible symptom was three months of `watch.out.log` in which 1032 of 1200 runs
recorded the startup banner (an `rprint` to stdout) and nothing else, which is equally
consistent with 1032 healthy runs and 1032 silent deaths. These tests exist so that
question is answerable from the logs next time.
"""

import logging
import subprocess
import sys
import time

from brainlayer import watcher as watcher_module


def test_watch_command_configures_logging():
    """The watch command must configure logging, or every logger.info is discarded."""
    import inspect

    from brainlayer.cli import watch

    source = inspect.getsource(watch)
    assert "basicConfig" in source, (
        "brainlayer watch does not configure logging, so watcher.py's logger.info calls "
        "(heartbeat, started, stopped) never reach StandardErrorPath"
    )


def test_watcher_heartbeat_is_emitted_at_info(caplog):
    """The 60s liveness heartbeat must be emitted on the brainlayer.watcher logger."""
    with caplog.at_level(logging.INFO, logger="brainlayer.watcher"):
        logging.getLogger("brainlayer.watcher").info("Watcher alive: %d sessions tracked, %d chunks indexed", 3, 7)
    assert any("Watcher alive" in r.getMessage() for r in caplog.records)


def test_watcher_module_logs_liveness_at_info_not_debug():
    """Heartbeat/started/stopped must be INFO, so a default INFO config surfaces them."""
    import inspect

    source = inspect.getsource(watcher_module)
    for marker in ("JSONL watcher started", "Watcher alive", "JSONL watcher stopped"):
        assert f'logger.info(\n                    "{marker}' in source or f'logger.info("{marker}' in source, (
            f"{marker!r} must be logged at INFO so operators see liveness"
        )


def test_watch_subprocess_writes_startup_line_to_stderr(tmp_path):
    """Live check: the real CLI must emit its startup line to stderr, not just stdout.

    This is the regression that mattered -- the banner went to stdout while every
    operational log line went nowhere at all.

    Stderr is redirected to a file and polled for the marker rather than read with a single
    fixed `communicate` timeout: under parallel load the subprocess can take many seconds to
    import and start, and a fixed timeout made this test flake in a full-suite run.
    """
    watch_dir = tmp_path / "projects"
    watch_dir.mkdir()
    err_path = tmp_path / "stderr.log"
    out_path = tmp_path / "stdout.log"
    env = {
        "PATH": "/usr/bin:/bin:/usr/sbin:/sbin",
        "HOME": str(tmp_path),
        "BRAINLAYER_DB": str(tmp_path / "test.db"),
        "BRAINLAYER_ARBITRATED": "1",
        "BRAINLAYER_QUEUE_DIR": str(tmp_path / "queue"),
        "BRAINLAYER_WATCHER_HEALTH_PATH": str(tmp_path / "health.json"),
        "PYTHONUNBUFFERED": "1",
        "PYTHONPATH": "src",
        "BRAINLAYER_FORBID_EMBEDDING_MODEL": "1",
    }
    marker = "JSONL watcher started"
    with open(out_path, "w") as out_fh, open(err_path, "w") as err_fh:
        proc = subprocess.Popen(
            [
                sys.executable,
                "-c",
                "from brainlayer.cli import app; app()",
                "watch",
                "--source",
                str(watch_dir),
                "--poll",
                "30",
            ],
            stdout=out_fh,
            stderr=err_fh,
            env=env,
            text=True,
        )
        try:
            deadline = time.monotonic() + 90
            while time.monotonic() < deadline:
                if marker in err_path.read_text(errors="replace"):
                    break
                if proc.poll() is not None:
                    break
                time.sleep(0.5)
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=30)

    stderr = err_path.read_text(errors="replace")
    assert marker in stderr, (
        "watcher startup line never reached stderr.\n"
        f"--- stdout ---\n{out_path.read_text(errors='replace')[-2000:]}\n"
        f"--- stderr ---\n{stderr[-2000:]}"
    )
