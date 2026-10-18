"""Calendar fixture's resident-watch readiness assertion."""

import time


def _fixture_diagnostics(path):
    """Read only known private fixture artifacts, never sockets or symlink targets."""
    artifacts = {}
    for name in (
        "calendar.stderr",
        "calendar/runs.jsonl",
        "errors",
        "step",
        "watch",
        "drain",
        "descendant",
        "guard",
        "pause",
    ):
        candidate = path / name
        try:
            if candidate.is_symlink() or candidate.parent.is_symlink():
                artifacts[name] = "symlink (not read)"
            elif candidate.is_file():
                with candidate.open(encoding="utf-8", errors="replace") as stream:
                    artifacts[name] = stream.read(16384)
            else:
                artifacts[name] = "missing or not a regular file"
        except OSError as error:
            artifacts[name] = f"unreadable: {error!r}"
    artifacts["control.sock exists"] = (path / "control.sock").exists()
    return artifacts


def wait_for_calendar_watch(query, path, *, monotonic=time.monotonic, sleep=time.sleep):
    """Retry status timeouts within 3 s; the query retains its 0.6 s socket timeout."""
    deadline = monotonic() + 3
    last_timeout, last_components = None, None
    while monotonic() < deadline:
        try:
            response = query()
        except TimeoutError as error:
            last_timeout = error
        else:
            last_components = response["components"]
            if monotonic() < deadline and last_components["watch"]["pid"]:
                return
        remaining = deadline - monotonic()
        if remaining > 0:
            sleep(min(0.02, remaining))
    # No final blocking request: diagnostics use artifacts and already-returned state only.
    raise AssertionError(
        f"calendar resident watch not ready within 3s; last_timeout={last_timeout!r}; "
        f"last_components={last_components!r}; fixture={path}; artifacts={_fixture_diagnostics(path)!r}"
    ) from last_timeout
