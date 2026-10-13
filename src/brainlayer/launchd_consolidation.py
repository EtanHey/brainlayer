"""Explicit launchd migration with durable archive custody and checked rollback."""

import hashlib
import json
import os
import plistlib
import re
import stat
import time
import uuid
from pathlib import Path

from .component_jobs import _legacy_configured
from .consolidation_plan import NEW, REVIVERS, disabled_labels, read_owned, rendered_launchers, snapshot_legacy
from .migration_custody import migration_lock, write_launch_agent
from .migration_custody import write_new_file as write_file
from .migration_journal import archive_replace as update
from .migration_journal import recover_staging
from .migration_journal import sync_directory as sync
from .migration_runtime import LaunchdRuntime, deadline_after, pid_alive, wait_supervisor
from .supervisor_adapters import SupervisorCommands
from .supervisor_guard import Custody, boot_time, group_exists, start_time
from .supervisor_jobs import load_components


def publication_paths(home):
    agents, control = home / "Library/LaunchAgents", home / ".local/share/brainlayer/supervisor"
    return (
        control / "migration.json",
        control / "jobs.json",
        *(agents / (label + suffix) for label in NEW for suffix in (".plist", ".plist.loaded-keg")),
    )


def wait_custody(directory, *, clock=time, grace=30):
    """Require the guard's lock and every recorded group to be quiescent; never signal."""
    if not os.path.lexists(directory):
        return
    if directory.resolve() != directory.absolute():
        raise ValueError("linked supervisor custody refused")
    deadline = deadline_after(clock, grace) + 2
    while True:
        journal = directory / "groups.json"
        if os.path.lexists(journal):
            document = json.loads(read_owned(journal))
            if journal.stat().st_mode & 0o077 or document.get("boot_time", boot_time()) != boot_time():
                raise RuntimeError("supervisor custody identity refused")
        try:
            with Custody(directory) as custody:
                live = False
                for key, item in custody.groups.items():
                    pgid, birth = item["pgid"], item["start_time"]
                    if type(pgid) is not int or pgid <= 1 or key != str(pgid) or not isinstance(birth, str):
                        raise RuntimeError("invalid supervisor custody group")
                    if group_exists(pgid):
                        if start_time(pgid) != birth:
                            raise RuntimeError("foreign supervisor custody group")
                        live = True
                if not live:
                    return
        except BlockingIOError:
            pass
        if clock.monotonic() >= deadline:
            raise RuntimeError("supervisor custody did not exit")
        clock.sleep(0.1)


def migrate(*, home, **options):
    with migration_lock(home) as control:
        receipt, *destinations = publication_paths(home)
        recover_staging(receipt, control, destinations=destinations)
        return _migrate(home=home, **options)


def _migrate(
    *, home, templates, env_file, python, keg, domain, run, alive=pid_alive, verify=None, clock=time, retry=None
):
    agents, control = home / "Library/LaunchAgents", home / ".local/share/brainlayer/supervisor"
    receipt, manifest = control / "migration.json", control / "jobs.json"
    runtime = LaunchdRuntime(domain, run, alive=alive, clock=clock)
    prepared = rendered_launchers(templates, home, python, env_file)
    launcher_env = plistlib.loads(prepared[NEW[0]])["EnvironmentVariables"]
    if not _legacy_configured(launcher_env, env_file, contents=read_owned(env_file).decode()):
        raise RuntimeError("new daemon is configured OFF; refusing handoff")

    def observe(label):
        state = runtime.observe(label)
        return state["loaded"], state["pid"]

    def stop(label, expected=None):
        runtime.stop(label, expected=expected)
        clock.sleep(0.1)
        runtime.stop(label, grace=0.1, expected={"loaded": False, "pid": None})
        if label in NEW:
            wait_custody(control if label == NEW[0] else control / "calendar", clock=clock)

    disabled = disabled_labels(runtime._checked("print-disabled", domain).stdout)
    if NEW[0] in disabled:
        raise RuntimeError("new daemon is disabled; refusing handoff")

    def journal_update(journal, expected):
        update(receipt, expected, json.dumps(journal).encode(), archive, write_file)

    started = {}

    def bootstrap(label, path):
        if label not in disabled:
            if label in NEW and runtime.observe(label)["loaded"]:
                raise RuntimeError(f"unowned replacement appeared: {label}")
            if label in NEW:
                wait_custody(control if label == NEW[0] else control / "calendar", clock=clock)
            runtime._checked("bootstrap", domain, path)
            current = runtime.observe(label)
            if not current["loaded"]:
                raise RuntimeError(f"replacement did not load: {label}")
            if label in NEW:
                started[label] = current

    def verify_daemon():
        if NEW[0] not in disabled:
            if verify is not None:
                verify()
            else:
                client = SupervisorCommands(control / "control.sock", run)
                wait_supervisor(client.request, load_components(manifest, client.path), alive=alive, clock=clock)

    if os.path.lexists(receipt) and retry is None:
        previous_bytes = read_owned(receipt)
        previous = json.loads(previous_bytes)
        archive = Path(previous["archive"])
        if (
            archive.parent != agents
            or not re.fullmatch(r"\.archive-[0-9]{8}T[0-9]{6}Z-[a-f0-9]{8}", archive.name)
            or archive.resolve() != archive.absolute()
            or archive.stat().st_uid != os.getuid()
            or stat.S_IMODE(archive.stat().st_mode) != 0o700
        ):
            raise RuntimeError("migration archive custody changed")
        recover_staging(receipt, archive, destinations=publication_paths(home)[1:])
        if previous.get("phase") in {"rolled_back", "reverted"}:
            restored = snapshot_legacy(agents, env_file, domain, run)
            if (
                os.path.lexists(manifest)
                or any(runtime.observe(label)["loaded"] for label in NEW)
                or restored["files"] != previous["files"]
                or any(
                    restored["states"][label][key] != state[key]
                    for label, state in previous["states"].items()
                    for key in ("loaded", "disabled_override")
                )
            ):
                raise RuntimeError("rolled-back preimage changed; explicit recovery required")
            return _migrate(
                home=home,
                templates=templates,
                env_file=env_file,
                python=python,
                keg=keg,
                domain=domain,
                run=run,
                alive=alive,
                verify=verify,
                clock=clock,
                retry=previous_bytes,
            )
        if previous.get("phase") != "migrated":
            raise RuntimeError("unfinished migration journal; explicit recovery required")
        if any(os.path.lexists(agents / name) for name in previous["files"]):
            raise RuntimeError("legacy files reappeared after consolidation")
        for name, original in previous["files"].items():
            if Path(name).name != name or read_owned(archive / name) != bytes.fromhex(original):
                raise RuntimeError("migration archive bytes changed")
        if any(runtime.observe(label)["loaded"] for label in previous["states"] if label not in NEW):
            raise RuntimeError("legacy service loaded after consolidation")
        if any(read_owned(agents / f"{label}.plist") != raw for label, raw in prepared.items()):
            raise RuntimeError("launcher configuration changed; explicit replan required")
        if hashlib.sha256(read_owned(manifest)).hexdigest() != previous["manifest_sha256"]:
            raise RuntimeError("component manifest changed; explicit replan required")
        if any(observe(label)[0] != (label not in disabled) for label in NEW):
            raise RuntimeError("replacement runtime differs from journal")
        for label in NEW:
            marker = agents / f"{label}.plist.loaded-keg"
            if label not in disabled and read_owned(marker) != (previous["keg"] + "\n").encode():
                raise RuntimeError("replacement marker differs from journal")
        if previous["keg"] == keg:
            return previous
        _, calendar_pid = observe(NEW[1])
        if calendar_pid is not None and alive(calendar_pid):
            raise RuntimeError("calendar is active; defer keg restart")
        original_state = runtime.observe(NEW[0])
        restarting = previous | dict(phase="migrating", requested_keg=keg)
        restarting_bytes = json.dumps(restarting).encode()
        completed = previous | dict(keg=keg)
        completed_bytes = json.dumps(completed).encode()
        marker_paths = [agents / f"{label}.plist.loaded-keg" for label in NEW if label not in disabled]
        old_marker, new_marker = (previous["keg"] + "\n").encode(), (keg + "\n").encode()
        attempted, stopping, stopped_daemon = [], False, False
        try:
            journal_update(restarting, previous_bytes)
            stopping = True
            stop(NEW[0], original_state)
            stopped_daemon = True
            bootstrap(NEW[0], agents / f"{NEW[0]}.plist")
            verify_daemon()
            for marker in marker_paths:
                attempted.append(marker)
                update(marker, old_marker, new_marker, archive, write_launch_agent)
            journal_update(completed, restarting_bytes)
            return completed
        except Exception as failure:
            try:
                original_kept = False
                if stopping:
                    current_state = runtime.observe(NEW[0])
                    if NEW[0] in started:
                        stop(NEW[0], started[NEW[0]])
                    elif current_state["loaded"]:
                        if not stopped_daemon and current_state == original_state:
                            original_kept = True
                        else:
                            raise RuntimeError("unowned replacement prevents restart rollback")
                    if not original_kept and original_state["pid"] is not None and alive(original_state["pid"]):
                        raise RuntimeError("original supervisor still lives; refusing restart bootstrap")
                for marker in reversed(attempted):
                    current = read_owned(marker)
                    if current != old_marker:
                        if current != new_marker:
                            raise RuntimeError("foreign marker prevents restart rollback")
                        update(marker, new_marker, old_marker, archive, write_launch_agent)
                if stopping:
                    if not original_kept:
                        bootstrap(NEW[0], agents / f"{NEW[0]}.plist")
                    verify_daemon()
                current = read_owned(receipt)
                if current != previous_bytes:
                    if current not in (restarting_bytes, completed_bytes):
                        raise RuntimeError("foreign journal prevents restart rollback")
                    update(receipt, current, previous_bytes, archive, write_file)
            except Exception as rollback:
                raise RuntimeError(f"keg restart failed; rollback incomplete: {rollback}") from failure
            raise RuntimeError("keg restart failed; markers/journal and supervisor readiness restored") from failure

    if os.path.lexists(manifest) or any(observe(label)[0] for label in NEW):
        raise RuntimeError("unowned replacement state; refusing migration")
    wait_custody(control, clock=clock)
    wait_custody(control / "calendar", clock=clock)
    snapshot = snapshot_legacy(agents, env_file, domain, run)
    components = snapshot["components"]
    files = {name: bytes.fromhex(raw) for name, raw in snapshot["files"].items()}
    states = snapshot["states"]
    refused = [
        label
        for label, state in states.items()
        if state["loaded"]
        and (
            not state["plist_present"]
            or state["disabled"]
            or state["disabled_override"] is None
            and plistlib.loads(files[f"{label}.plist"]).get("Disabled", False)
        )
    ]
    if refused:
        raise RuntimeError(f"loaded preimage is not revertible without changing operator state: {refused}")
    archive = agents / (".archive-" + time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()) + "-" + uuid.uuid4().hex[:8])
    archive.mkdir(mode=0o700)
    staged_manifest = archive / "jobs.json"
    write_file(staged_manifest, json.dumps(components).encode())
    load_components(staged_manifest, control / "control.sock")  # Validate every argv/resource key before bootout.
    journal = snapshot | dict(
        phase="migrating",
        keg=keg,
        archive=str(archive),
        group_custody={label: str(control if label == NEW[0] else control / "calendar") for label in NEW},
    )
    initial_receipt = json.dumps(journal).encode()
    if retry is None:
        write_file(receipt, initial_receipt)
    else:
        update(receipt, retry, initial_receipt, archive, write_file)
    moved, created, stopped = [], [], set()
    completed_receipt = None

    def publish(path, content, writer):
        entry = [path, content]
        created.append(entry)
        try:
            writer(path, content)
        except FileExistsError:
            entry[1] = None  # Equal bytes do not establish ownership of a concurrent arrival.
            raise

    try:
        for label in sorted(REVIVERS):
            if states[label]["loaded"]:
                stop(label, states[label])
                stopped.add(label)
        for name, content in files.items():
            source = agents / name
            if read_owned(source) != content:
                raise RuntimeError(f"legacy file changed during preflight: {name}")
            source.rename(archive / name)
            moved.append(name)
            sync(archive)
            sync(agents)
            if read_owned(archive / name) != content:
                raise RuntimeError(f"legacy file changed while archiving: {name}")
        for label, state in states.items():
            if label not in REVIVERS and state["loaded"]:
                stop(label, state)
                stopped.add(label)
        manifest_bytes = read_owned(staged_manifest)
        publish(manifest, manifest_bytes, write_file)
        for label, raw in prepared.items():
            path = agents / f"{label}.plist"
            if os.path.lexists(path):
                raise RuntimeError(f"replacement definition appeared during migration: {label}")
            publish(path, raw, write_launch_agent)
            bootstrap(label, path)
        verify_daemon()
        for label in NEW:
            if label not in disabled:
                marker = agents / f"{label}.plist.loaded-keg"
                publish(marker, (keg + "\n").encode(), write_launch_agent)
        journal.update(phase="migrated", manifest_sha256=hashlib.sha256(read_owned(manifest)).hexdigest())
        completed_receipt = json.dumps(journal).encode()
        journal_update(journal, initial_receipt)
        return journal
    except Exception as failure:
        try:
            for label in reversed(NEW):
                if label in started:
                    stop(label, started[label])
                elif runtime.observe(label)["loaded"]:
                    raise RuntimeError(f"unowned replacement prevents rollback: {label}")
            for path, expected in reversed(created):
                if os.path.lexists(path):
                    if expected is None or read_owned(path) != expected:
                        raise RuntimeError(f"replacement changed; refusing rollback unlink: {path}")
                    path.unlink()
            for name in reversed(moved):
                destination = agents / name
                if os.path.lexists(destination):
                    raise RuntimeError(f"rollback destination occupied: {name}")
                if read_owned(archive / name) != files[name]:
                    raise RuntimeError(f"rollback archive changed: {name}")
                os.link(archive / name, destination)
                (archive / name).unlink()
                sync(agents)
                sync(archive)
            for label, state in states.items():
                loaded = observe(label)[0]
                if loaded and (label in stopped or not state["loaded"]):
                    raise RuntimeError(f"unowned legacy service prevents rollback: {label}")
                if state["loaded"] and not loaded:
                    if not state["plist_present"] or state["disabled"]:
                        raise RuntimeError(f"original loaded service has no restorable definition/state: {label}")
                    if any(alive(pid) for pid in runtime.observed_pids.get(label, {state["pid"]} - {None})):
                        raise RuntimeError(f"original writer still lives; refusing rollback bootstrap: {label}")
                    bootstrap(label, agents / f"{label}.plist")
            journal["phase"] = "rolled_back"
            current_receipt = read_owned(receipt)
            if current_receipt not in (initial_receipt, completed_receipt):
                raise RuntimeError("foreign journal prevents rollback")
            journal_update(journal, current_receipt)
        except Exception as rollback:
            raise RuntimeError(f"migration failed; rollback incomplete: {rollback}") from failure
        raise RuntimeError("migration failed; original files/states restored") from failure
