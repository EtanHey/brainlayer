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
from .consolidation_plan import (
    NEW,
    REVIVERS,
    UNMANAGED,
    disabled_labels,
    disabled_overrides,
    read_owned,
    rendered_launchers,
    snapshot_legacy,
)
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
        try:
            # Inspect under the kernel lock without clearing previous-boot groups.
            with Custody(directory, reset_previous_boot=False) as custody:
                journal = directory / "groups.json"
                if os.path.lexists(journal):
                    document = json.loads(read_owned(journal))
                    if journal.stat().st_mode & 0o077 or (
                        document.get("boot_time", boot_time()) != boot_time() and document.get("groups") != {}
                    ):
                        raise RuntimeError("supervisor custody identity refused")
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


def revert(home, journal, domain, run, **options):
    """Public installer API for the separately implemented post-success revert."""
    from .launchd_revert import revert as restore

    return restore(home, journal, domain, run, **options)


def _migrate(
    *, home, templates, env_file, python, keg, domain, run, alive=pid_alive, verify=None, clock=time, retry=None
):
    agents, control = home / "Library/LaunchAgents", home / ".local/share/brainlayer/supervisor"
    receipt, manifest = control / "migration.json", control / "jobs.json"
    runtime = LaunchdRuntime(domain, run, alive=alive, clock=clock)
    previous_bytes = read_owned(receipt) if os.path.lexists(receipt) and retry is None else None
    previous = json.loads(previous_bytes) if previous_bytes is not None else None
    if previous is not None:
        permitted = {f"{label}{suffix}" for label in NEW for suffix in (".plist", ".plist.loaded-keg")}
        if (
            not set(previous["published_files"]).issubset(permitted)
            or not {f"{label}.plist" for label in NEW}.issubset(previous["published_files"])
            or previous["published_sha256"]
            != {
                name: hashlib.sha256(bytes.fromhex(raw)).hexdigest()
                for name, raw in previous["published_files"].items()
            }
        ):
            raise RuntimeError("invalid published-file custody; operator reconciliation required")
    pending = previous is not None and previous.get("phase") == "migrating"
    if pending:
        prepared = {label: bytes.fromhex(previous["published_files"][f"{label}.plist"]) for label in NEW}
    else:
        prepared = rendered_launchers(templates, home, python, env_file)
        launcher_env = plistlib.loads(prepared[NEW[0]])["EnvironmentVariables"]
        if not _legacy_configured(launcher_env, env_file, contents=read_owned(env_file).decode()):
            raise RuntimeError("new daemon is configured OFF; refusing handoff")

    def pinned_daemon(raw):
        document = plistlib.loads(raw)
        interpreter = Path(document["ProgramArguments"][1])
        # Resolve opt/keg directories, retaining the venv's bin/python symlink and pyvenv.cfg.
        document["ProgramArguments"][1] = str(interpreter.parent.resolve() / interpreter.name)
        document["EnvironmentVariables"]["BRAINLAYER_REPO_ROOT"] = str(
            Path(document["EnvironmentVariables"]["BRAINLAYER_REPO_ROOT"]).resolve()
        )
        return plistlib.dumps(document, sort_keys=False)

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

    known_receipts = {previous_bytes} if previous_bytes is not None else set()

    def journal_update(journal, expected):
        content = json.dumps(journal).encode()
        known_receipts.add(content)
        update(receipt, expected, content, archive, write_file)

    def preserve_refusal(journal, failure):
        if "unowned" not in str(failure) and "replacement changed; refusing" not in str(failure):
            return
        try:
            current = read_owned(receipt)
            if current not in known_receipts:
                raise RuntimeError("journal ownership changed")
            journal = journal | dict(
                phase="migrating", recovery_refusal="ownership refused; operator reconciliation required"
            )
            journal_update(journal, current)
        except BaseException as recording:
            failure.add_note(f"ownership-refusal recording failed: {recording}")

    started = {}

    def remember_started(label, current):
        started[label] = current
        if current["pid"] is not None:
            runtime.observed_pids.setdefault(label, set()).add(current["pid"])

    def bootstrap(label, path):
        if label not in disabled:
            if label in NEW and runtime.observe(label)["loaded"]:
                raise RuntimeError(f"unowned replacement appeared: {label}")
            if label in NEW:
                wait_custody(control if label == NEW[0] else control / "calendar", clock=clock)
            expected_definition = read_owned(path) if label in NEW else None
            try:
                runtime._checked("bootstrap", domain, path)
            except BaseException as failure:
                # A signal can arrive after a successful bootstrap but before its observation.
                if label in NEW and not isinstance(failure, Exception) and read_owned(path) == expected_definition:
                    current = runtime.observe(label)
                    if current["loaded"]:
                        remember_started(label, current)
                raise
            current = runtime.observe(label)
            if not current["loaded"]:
                raise RuntimeError(f"replacement did not load: {label}")
            if label in NEW:
                remember_started(label, current)

    def verify_daemon():
        if NEW[0] not in disabled:
            if verify is not None:
                verify()
            else:
                client = SupervisorCommands(control / "control.sock", run)
                wait_supervisor(client.request, load_components(manifest, client.path), alive=alive, clock=clock)

    def check_overrides(journal):
        overrides = disabled_overrides(runtime._checked("print-disabled", domain).stdout)
        if any(overrides.get(label) != state["disabled_override"] for label, state in journal["states"].items()):
            raise RuntimeError("operator disabled state changed; recovery refused")

    def original_locations(journal, publications):
        locations = {}
        for name, encoded in journal["files"].items():
            if Path(name).name != name:
                raise RuntimeError("invalid archived filename")
            expected = bytes.fromhex(encoded)
            source, copy = agents / name, archive / name
            present, archived = os.path.lexists(source), os.path.lexists(copy)
            if present and archived:
                left, right = source.lstat(), copy.lstat()
                if (left.st_dev, left.st_ino) == (right.st_dev, right.st_ino):
                    if not stat.S_ISREG(left.st_mode) or left.st_uid != os.getuid() or left.st_nlink != 2:
                        raise RuntimeError(f"rollback hardlink custody refused: {name}")
                    fd = os.open(source, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
                    with os.fdopen(fd, "rb") as stream:
                        if stream.read() != expected:
                            raise RuntimeError(f"rollback archive changed: {name}")
                    locations[name] = "linked"
                    continue
            current = read_owned(source) if present else None
            backup = read_owned(copy) if archived else None
            if backup is not None and backup != expected:
                raise RuntimeError(f"rollback archive changed: {name}")
            if current == expected and not archived:
                locations[name] = "live"
            elif backup == expected and (not present or current == publications.get(source)):
                locations[name] = "archive"
            else:
                raise RuntimeError(f"rollback destination occupied or preimage missing: {name}")
        return locations

    def restore_first(journal, expected_receipt, *, recovering, created=(), stopped=()):
        if journal.get("recovery_refusal"):
            raise RuntimeError("prior ownership refusal requires operator reconciliation")
        check_overrides(journal)
        publications = {agents / name: bytes.fromhex(raw) for name, raw in journal["published_files"].items()}
        publications[manifest] = bytes.fromhex(journal["manifest_bytes"])
        locations = original_locations(journal, publications)
        created_map = {path: content for path, content in created}
        for path, content in publications.items():
            if not os.path.lexists(path) or locations.get(path.name) in {"live", "linked"} and path.parent == agents:
                continue
            if read_owned(path) != content or not recovering and created_map.get(path) != content:
                raise RuntimeError(f"replacement changed; refusing rollback unlink: {path}")
        permitted = (
            set(journal["files"])
            | set(journal["published_files"])
            | {label + suffix for label in UNMANAGED for suffix in (".plist", ".plist.loaded-keg")}
        )
        for path in agents.iterdir():
            if (
                "brainlayer" in path.name.casefold()
                and path.name.casefold().endswith((".plist", ".loaded-keg"))
                and path.name not in permitted
            ):
                raise RuntimeError("foreign legacy files prevent rollback")
        current_states = {label: runtime.observe(label) for label in journal["states"]}
        legacy_refusal = None
        for label, current in current_states.items():
            if label in NEW or not current["loaded"]:
                continue
            original = journal["states"][label]
            resumed = journal.get("recovery_bootstrap", {}).get(label)
            if (
                not original["loaded"]
                or not recovering
                and label in stopped
                or recovering
                and current["pid"] != original["pid"]
                and (resumed is None or resumed.get("pid") not in (None, current["pid"]))
            ):
                legacy_refusal = f"unowned legacy service prevents rollback: {label}"
        if journal["phase"] != "migrating":
            journal["phase"] = "migrating"
            journal_update(journal, expected_receipt)
            expected_receipt = json.dumps(journal).encode()
        for label in reversed(NEW):
            current = current_states[label]
            if current["loaded"]:
                path = agents / f"{label}.plist"
                if not os.path.lexists(path) or read_owned(path) != publications[path]:
                    raise RuntimeError(f"unowned replacement prevents rollback: {label}")
                if not recovering and label not in started:
                    raise RuntimeError(f"unowned replacement prevents rollback: {label}")
                stop(label, current)
            else:
                wait_custody(control if label == NEW[0] else control / "calendar", clock=clock)
        if legacy_refusal:
            raise RuntimeError(legacy_refusal)
        for path, content in publications.items():
            if os.path.lexists(path) and not (path.parent == agents and locations.get(path.name) in {"live", "linked"}):
                if read_owned(path) != content:
                    raise RuntimeError(f"replacement changed; refusing rollback unlink: {path}")
                path.unlink()
                sync(path.parent)
        for name, location in locations.items():
            source, copy = agents / name, archive / name
            if location == "archive":
                os.link(copy, source)
            if location in {"archive", "linked"}:
                copy.unlink()
                sync(agents)
                sync(archive)
        for label, state in journal["states"].items():
            if label in NEW:
                continue
            current = runtime.observe(label)
            if state["loaded"] and not current["loaded"]:
                if any(alive(pid) for pid in runtime.observed_pids.get(label, set()) | ({state["pid"]} - {None})):
                    raise RuntimeError(f"original writer still lives; refusing rollback bootstrap: {label}")
                journal.setdefault("recovery_bootstrap", {})[label] = {"pid": None}
                journal_update(journal, expected_receipt)
                expected_receipt = json.dumps(journal).encode()
                bootstrap(label, agents / f"{label}.plist")
                journal["recovery_bootstrap"][label] = runtime.observe(label)
                journal_update(journal, expected_receipt)
                expected_receipt = json.dumps(journal).encode()
            elif current["loaded"] != state["loaded"]:
                raise RuntimeError(f"unowned legacy service prevents rollback: {label}")
        journal["phase"] = "rolled_back"
        journal_update(journal, expected_receipt)
        return journal

    def replace_known(path, allowed, target, writer):
        current = read_owned(path)
        if current not in allowed:
            raise RuntimeError(f"foreign migration file prevents restart recovery: {path}")
        if current != target:
            update(path, current, target, archive, writer)

    def restart_daemon(restarting, expected_receipt, *, recovering, publication_failure=None):
        if restarting.get("recovery_refusal"):
            raise RuntimeError("prior ownership refusal requires operator reconciliation")
        previous = restarting["restart_preimage"]
        check_overrides(previous)
        for name, raw in previous["files"].items():
            if os.path.lexists(agents / name) or read_owned(archive / name) != bytes.fromhex(raw):
                raise RuntimeError("legacy/archive custody changed; restart recovery refused")
        if any(runtime.observe(label)["loaded"] for label in previous["states"] if label not in NEW):
            raise RuntimeError("unowned legacy service prevents restart recovery")
        if hashlib.sha256(read_owned(manifest)).hexdigest() != previous["manifest_sha256"]:
            raise RuntimeError("foreign manifest prevents restart recovery")
        daemon_path = agents / f"{NEW[0]}.plist"
        old_daemon = bytes.fromhex(previous["published_files"][daemon_path.name])
        declared_daemon = requested_daemon = bytes.fromhex(restarting["requested_daemon"])
        requested = plistlib.loads(requested_daemon)
        old_python = Path(requested["ProgramArguments"][1])
        if old_python != python:
            relative_root = Path(requested["EnvironmentVariables"]["BRAINLAYER_REPO_ROOT"]).relative_to(
                old_python.parents[1]
            )
            requested["ProgramArguments"][1] = str(python)
            requested["EnvironmentVariables"]["BRAINLAYER_REPO_ROOT"] = str(python.parents[1] / relative_root)
            requested_daemon = plistlib.dumps(requested, sort_keys=False)
            if hashlib.sha256(requested_daemon).hexdigest() != previous["configuration_sha256"][NEW[0]]:
                raise RuntimeError("requested launcher configuration changed; explicit replan required")
        fallback_daemon = bytes.fromhex(previous["daemon_pinned"])
        allowed_daemons = {old_daemon, declared_daemon, requested_daemon, fallback_daemon}
        old_marker, new_marker = (previous["keg"] + "\n").encode(), (restarting["requested_keg"] + "\n").encode()
        marker_paths = [agents / f"{label}.plist.loaded-keg" for label in NEW if label not in disabled]
        for name, raw in previous["published_files"].items():
            current = read_owned(agents / name)
            allowed = (
                allowed_daemons
                if name == daemon_path.name
                else ({old_marker, new_marker} if name.endswith(".loaded-keg") else {bytes.fromhex(raw)})
            )
            if current not in allowed:
                raise RuntimeError(f"foreign migration file prevents restart recovery: {name}")
        initial_state = runtime.observe(NEW[0])
        original_state = restarting["restart_state"]
        if not recovering and initial_state != original_state:
            raise RuntimeError("unowned replacement prevents restart recovery")
        if original_state["pid"] is not None:
            runtime.observed_pids.setdefault(NEW[0], set()).add(original_state["pid"])
        stopping, stopped_daemon = False, False
        receipt_versions = {expected_receipt}
        missing_previous = ""

        def completed(keg_value, daemon_bytes, fallback=False):
            result = previous | dict(phase="migrated", keg=keg_value, daemon_pinned=pinned_daemon(daemon_bytes).hex())
            files = dict(previous["published_files"])
            files[daemon_path.name] = daemon_bytes.hex()
            for marker in marker_paths:
                files[marker.name] = (keg_value + "\n").encode().hex()
            result.update(
                published_files=files,
                published_sha256={name: hashlib.sha256(bytes.fromhex(raw)).hexdigest() for name, raw in files.items()},
            )
            result.pop("restart_fallback", None)
            if fallback:
                result["restart_fallback"] = {"requested_keg": restarting["requested_keg"], "restored_keg": keg_value}
            current_receipt = read_owned(receipt)
            if current_receipt not in receipt_versions:
                raise RuntimeError("foreign journal prevents restart rollback")
            receipt_versions.add(json.dumps(result).encode())
            journal_update(result, current_receipt)
            return result

        try:
            if publication_failure is not None:
                raise publication_failure
            stopping = True
            stop(NEW[0], initial_state)
            stopped_daemon = True
            wait_custody(control / "calendar", clock=clock)
            replace_known(daemon_path, allowed_daemons, requested_daemon, write_launch_agent)
            bootstrap(NEW[0], daemon_path)
            verify_daemon()
            for marker in marker_paths:
                replace_known(marker, {old_marker, new_marker}, new_marker, write_launch_agent)
            return completed(restarting["requested_keg"], requested_daemon)
        except BaseException as failure:
            try:
                kept = False
                current = runtime.observe(NEW[0])
                fallback_python = Path(plistlib.loads(fallback_daemon)["ProgramArguments"][1])
                if not fallback_python.is_file() or not os.access(fallback_python, os.X_OK):
                    # Keep/recover the runnable current definition; never publish a deleted keg.
                    missing_previous = (
                        "previous keg removed by brew cleanup; re-run `install.sh consolidate` on the current keg"
                    )
                    definition = read_owned(daemon_path)
                    if definition not in allowed_daemons:
                        raise RuntimeError("unowned daemon definition prevents restart rollback")
                    if current["loaded"] and NEW[0] not in started and (stopped_daemon or current != initial_state):
                        raise RuntimeError("unowned replacement prevents restart rollback")
                    requested_python = Path(plistlib.loads(requested_daemon)["ProgramArguments"][1])
                    if not requested_python.is_file() or not os.access(requested_python, os.X_OK):
                        raise RuntimeError(missing_previous + "; current interpreter is also unavailable")
                    if current["loaded"] and definition != requested_daemon:
                        stop(NEW[0], current)
                        current = runtime.observe(NEW[0])
                    if not current["loaded"]:
                        if any(alive(pid) for pid in runtime.observed_pids.get(NEW[0], set())):
                            raise RuntimeError(missing_previous + "; supervisor still lives; refusing bootstrap")
                        wait_custody(control / "calendar", clock=clock)
                        replace_known(daemon_path, allowed_daemons, requested_daemon, write_launch_agent)
                        bootstrap(NEW[0], daemon_path)
                    raise RuntimeError(missing_previous)
                if NEW[0] in started:
                    stop(NEW[0], current)
                elif current["loaded"]:
                    if not stopped_daemon and current == initial_state:
                        kept = True
                    else:
                        raise RuntimeError("unowned replacement prevents restart rollback")
                if not kept and any(alive(pid) for pid in runtime.observed_pids.get(NEW[0], set())):
                    raise RuntimeError("original supervisor still lives; refusing restart bootstrap")
                replace_known(daemon_path, allowed_daemons, fallback_daemon, write_launch_agent)
                for marker in marker_paths:
                    replace_known(marker, {old_marker, new_marker}, old_marker, write_launch_agent)
                if not kept:
                    bootstrap(NEW[0], daemon_path)
                verify_daemon()
                restored = completed(previous["keg"], fallback_daemon, fallback=recovering)
            except BaseException as rollback:
                preserve_refusal(restarting, rollback)
                detail = str(rollback)
                if missing_previous and missing_previous not in detail:
                    detail = missing_previous + "; " + detail
                if not isinstance(failure, Exception):
                    failure.add_note(f"keg restart rollback incomplete: {detail}")
                    raise failure
                raise RuntimeError(f"keg restart failed; rollback incomplete: {detail}") from failure
            if not isinstance(failure, Exception):
                raise
            if recovering:
                return restored
            raise RuntimeError("keg restart failed; markers/journal and supervisor readiness restored") from failure

    if os.path.lexists(receipt) and retry is None:
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
        if pending:
            try:
                if "requested_keg" in previous:
                    return restart_daemon(previous, previous_bytes, recovering=True)
                return restore_first(previous, previous_bytes, recovering=True)
            except BaseException as failure:
                if not isinstance(failure, Exception):
                    raise
                raise RuntimeError(
                    f"interrupted migration; rollback incomplete; operator reconciliation: {failure}"
                ) from failure
        if previous.get("phase") != "migrated":
            raise RuntimeError("unknown migration phase; operator reconciliation required")
        if any(os.path.lexists(agents / name) for name in previous["files"]):
            raise RuntimeError("legacy files reappeared after consolidation")
        for name, original in previous["files"].items():
            if Path(name).name != name or read_owned(archive / name) != bytes.fromhex(original):
                raise RuntimeError("migration archive bytes changed")
        if any(runtime.observe(label)["loaded"] for label in previous["states"] if label not in NEW):
            raise RuntimeError("legacy service loaded after consolidation")
        if any(
            hashlib.sha256(read_owned(agents / name)).hexdigest() != digest
            for name, digest in previous["published_sha256"].items()
        ):
            raise RuntimeError("replacement marker/launcher differs from journal")
        if any(
            hashlib.sha256(raw).hexdigest() != previous["configuration_sha256"][label]
            for label, raw in prepared.items()
        ):
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
        restarting = previous | dict(
            phase="migrating",
            requested_keg=keg,
            restart_preimage=previous,
            restart_state=runtime.observe(NEW[0]),
            requested_daemon=prepared[NEW[0]].hex(),
        )
        restarting_bytes = json.dumps(restarting).encode()
        try:
            journal_update(restarting, previous_bytes)
        except BaseException as failure:
            current_receipt = read_owned(receipt)
            if current_receipt == restarting_bytes:
                return restart_daemon(restarting, current_receipt, recovering=False, publication_failure=failure)
            if current_receipt != previous_bytes:
                raise RuntimeError("keg restart publication failed; foreign journal; rollback incomplete") from failure
            if not isinstance(failure, Exception):
                raise
            raise RuntimeError("keg restart failed; original journal and supervisor readiness restored") from failure
        return restart_daemon(restarting, restarting_bytes, recovering=False)

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
    published_files = {f"{label}.plist": raw.hex() for label, raw in prepared.items()}
    published_files.update(
        {f"{label}.plist.loaded-keg": (keg + "\n").encode().hex() for label in NEW if label not in disabled}
    )
    journal = snapshot | dict(
        phase="migrating",
        keg=keg,
        archive=str(archive),
        group_custody={label: str(control if label == NEW[0] else control / "calendar") for label in NEW},
        published_files=published_files,
        published_sha256={
            name: hashlib.sha256(bytes.fromhex(raw)).hexdigest() for name, raw in published_files.items()
        },
        configuration_sha256={label: hashlib.sha256(raw).hexdigest() for label, raw in prepared.items()},
        daemon_pinned=pinned_daemon(prepared[NEW[0]]).hex(),
        manifest_bytes=read_owned(staged_manifest).hex(),
    )
    initial_receipt = json.dumps(journal).encode()
    known_receipts.add(initial_receipt)
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
    except BaseException as failure:
        try:
            current_receipt = read_owned(receipt)
            if current_receipt not in (initial_receipt, completed_receipt):
                raise RuntimeError("foreign journal prevents rollback")
            restore_first(journal, current_receipt, recovering=False, created=created, stopped=stopped)
        except BaseException as rollback:
            preserve_refusal(journal, rollback)
            if not isinstance(failure, Exception):
                failure.add_note(f"migration rollback incomplete: {rollback}")
                raise failure
            raise RuntimeError(f"migration failed; rollback incomplete: {rollback}") from failure
        if not isinstance(failure, Exception):
            raise
        raise RuntimeError("migration failed; original files/states restored") from failure
