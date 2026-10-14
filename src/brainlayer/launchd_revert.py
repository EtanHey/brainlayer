"""Explicit, replayable post-success revert; no installer activation on import.

Archived legacy definitions can name a keg removed by brew cleanup. Refuse that
preimage before stopping the runnable consolidation; restoring bytes cannot make
a deleted interpreter available. Retain the journal/archive for reconciliation.
"""

import hashlib
import json
import os
import plistlib
import re
import stat
import time
from pathlib import Path

from .consolidation_plan import ALIASES, ARCHIVED, NEW, UNMANAGED, disabled_overrides, read_owned
from .launchd_consolidation import publication_paths, wait_custody
from .migration_custody import _write_exclusive, migration_lock, write_new_file
from .migration_journal import archive_replace, recover_staging, sync_directory
from .migration_runtime import LaunchdRuntime, pid_alive


def revert(home, journal, domain, run, *, alive=pid_alive, clock=time):
    """Restore the recorded legacy state under the shared migration lock.

    Interrupted calls retain phase ``reverting`` and resume through this API.
    Disabled overrides must still equal the original tri-state: never enable a
    job to conceal an operator change. Completed replay performs no mutations.
    """
    with migration_lock(home) as control:
        receipt, manifest, *new_paths = publication_paths(home)
        if journal != receipt:
            raise ValueError("revert requires the canonical migration journal Path")
        agents = home / "Library/LaunchAgents"
        # Recover the journal first; legacy staging peers are only known after
        # validating its filename inventory. Do not prematurely scan LaunchAgents.
        recover_staging(receipt, control, destinations=(manifest,))
        expected = read_owned(receipt)
        document = json.loads(expected)
        phase = document.get("phase")
        if phase not in {"migrated", "reverting", "reverted"} or document.get("recovery_refusal"):
            raise RuntimeError("migration is not a completed/reverting transaction; reconciliation required")
        archive = Path(document["archive"])
        if (
            archive.parent != agents
            or not re.fullmatch(r"\.archive-[0-9]{8}T[0-9]{6}Z-[a-f0-9]{8}", archive.name)
            or archive.resolve() != archive.absolute()
            or archive.stat().st_uid != os.getuid()
            or stat.S_IMODE(archive.stat().st_mode) != 0o700
        ):
            raise RuntimeError("migration archive custody changed")
        states = document["states"]
        legacy = set(ALIASES) | ARCHIVED
        if not set(states).issubset(legacy | set(NEW)) or not set(NEW).issubset(states):
            raise ValueError("invalid journaled service labels")
        permitted = {label + suffix for label in legacy for suffix in (".plist", ".plist.loaded-keg", ".loaded-keg")}
        if not set(document["files"]).issubset(permitted):
            raise ValueError("invalid archived filenames")
        originals = {agents / name: bytes.fromhex(raw) for name, raw in document["files"].items()}
        permitted_new = {p.name for p in new_paths}
        published = document["published_files"]
        if (
            not set(published).issubset(permitted_new)
            or not {f"{label}.plist" for label in NEW}.issubset(published)
            or document["published_sha256"]
            != {name: hashlib.sha256(bytes.fromhex(raw)).hexdigest() for name, raw in published.items()}
        ):
            raise ValueError("invalid published-file custody hashes")
        publications = {agents / name: bytes.fromhex(raw) for name, raw in published.items()}
        publications[manifest] = bytes.fromhex(document["manifest_bytes"])
        if hashlib.sha256(publications[manifest]).hexdigest() != document["manifest_sha256"]:
            raise ValueError("invalid manifest custody hash")
        custody = {label: control if label == NEW[0] else control / "calendar" for label in NEW}
        if document["group_custody"] != {label: str(path) for label, path in custody.items()}:
            raise ValueError("invalid journaled group custody")
        recover_staging(receipt, archive, destinations=(manifest, *new_paths, *originals))
        runtime = LaunchdRuntime(domain, run, alive=alive, clock=clock)

        def check_overrides():
            overrides = disabled_overrides(runtime._checked("print-disabled", domain).stdout)
            if any(overrides.get(label) != state["disabled_override"] for label, state in states.items()):
                raise RuntimeError("operator disabled state changed; revert refused")

        def save():
            nonlocal expected
            content = json.dumps(document).encode()
            archive_replace(receipt, expected, content, archive, write_new_file)
            expected = content

        observe = runtime.observe

        def durable_observe(label):
            state = observe(label)
            if phase == "reverting" and label in NEW:
                pids = document["revert_pids"][label]
                if any(type(pid) is not int or pid <= 1 for pid in pids):
                    raise ValueError("invalid retained replacement PID")
                if state["pid"] is not None and state["pid"] not in pids:
                    pids.append(state["pid"])
                    save()  # Also covers PID rotation while LaunchdRuntime.stop polls.
                runtime.observed_pids.setdefault(label, set()).update(pids)
            return state

        runtime.observe = durable_observe

        def check_files():
            for path, raw in originals.items():
                if read_owned(archive / path.name) != raw:
                    raise RuntimeError(f"migration archive changed: {path.name}")
                if os.path.lexists(path) and (phase == "migrated" or read_owned(path) != raw):
                    raise RuntimeError(f"legacy destination occupied: {path.name}")
                if phase == "reverted" and not os.path.lexists(path):
                    raise RuntimeError(f"reverted preimage missing: {path.name}")
            for path, raw in publications.items():
                if os.path.lexists(path):
                    if read_owned(path) != raw or phase == "reverted":
                        raise RuntimeError(f"foreign published file prevents revert unlink: {path.name}")
                elif phase == "migrated":
                    raise RuntimeError(f"published file missing: {path.name}")
            names = {p.name for p in originals} | set(published)
            names |= {
                label + suffix for label in UNMANAGED for suffix in (".plist", ".loaded-keg", ".plist.loaded-keg")
            }
            for path in agents.iterdir():
                if (
                    "brainlayer" in path.name.casefold()
                    and path.name.casefold().endswith((".plist", ".loaded-keg"))
                    and path.name not in names
                ):
                    raise RuntimeError(f"foreign legacy definition prevents revert: {path.name}")

        def legacy_available(label):
            path = agents / f"{label}.plist"
            if path not in originals:
                raise RuntimeError(f"loaded legacy preimage has no definition: {label}")
            definition = plistlib.loads(originals[path])
            if definition.get("Label") != label:
                raise ValueError("archived service identity changed")
            args = definition.get("ProgramArguments", [])
            executable = definition.get("Program") or (args[0] if args else "")
            candidates = [
                executable,
                *(
                    arg
                    for arg in args
                    if isinstance(arg, str) and ("/Cellar/" in arg or Path(arg).name.startswith("python"))
                ),
            ]
            if any(
                not Path(arg).is_absolute() or not Path(arg).is_file() or not os.access(arg, os.X_OK)
                for arg in candidates
            ):
                raise RuntimeError(
                    "previous keg removed by brew cleanup or legacy executable unavailable; restore the legacy keg before reverting"
                )
            if (
                states[label]["disabled_override"] == "disabled"
                or states[label]["disabled_override"] is None
                and definition.get("Disabled", False)
            ):
                raise RuntimeError("legacy bootstrap would change disabled state; revert refused")

        check_overrides()
        check_files()
        current = {label: runtime.observe(label) for label in states}
        resumed = document.get("revert_bootstrap", {})
        for label in states.keys() - set(NEW):
            if states[label]["loaded"]:
                legacy_available(label)
            if current[label]["loaded"] and (
                phase == "migrated"
                or not states[label]["loaded"]
                or label not in resumed
                or resumed[label].get("pid") not in (None, current[label]["pid"])
            ):
                raise RuntimeError(f"unowned legacy service prevents revert: {label}")
        if phase == "reverted":
            if any(
                current[label]["loaded"] != (False if label in NEW else state["loaded"])
                for label, state in states.items()
            ):
                raise RuntimeError("reverted runtime changed; reconciliation required")
            for path in custody.values():
                wait_custody(path, clock=clock)
            return document

        # Persist the ownership handoff before the first bootout. Every subsequent
        # cut re-enters through the same journal; the old archive is never consumed.
        if phase == "migrated":
            document.update(
                phase="reverting",
                revert_pids={
                    label: [current[label]["pid"]] if current[label]["pid"] is not None else [] for label in NEW
                },
            )
            save()
            phase = "reverting"
        for label in reversed(NEW):
            state = runtime.observe(label)
            if state["loaded"]:
                path = agents / f"{label}.plist"
                if not os.path.lexists(path) or read_owned(path) != publications[path]:
                    raise RuntimeError("unowned replacement prevents revert bootout")
            runtime.stop(label, expected=state)
            clock.sleep(0.1)
            runtime.stop(label, grace=0.1, expected={"loaded": False, "pid": None})
            wait_custody(custody[label], clock=clock)
        check_overrides()
        check_files()
        for path, raw in publications.items():
            if os.path.lexists(path):
                # Ownership is established by the recorded published SHA, never
                # by re-rendering today's templates against a different keg.
                if hashlib.sha256(read_owned(path)).hexdigest() != hashlib.sha256(raw).hexdigest():
                    raise RuntimeError("published file changed before unlink")
                path.unlink()
                sync_directory(path.parent)
        for path, raw in originals.items():
            if not os.path.lexists(path):
                try:
                    _write_exclusive(path, raw, 0o644, private_parent=False)
                except FileExistsError:
                    document["recovery_refusal"] = "concurrent legacy publication; operator reconciliation required"
                    save()
                    raise RuntimeError(document["recovery_refusal"])
            elif read_owned(path) != raw:
                raise RuntimeError("legacy destination changed during restore")
        for label, state in states.items():
            if label in NEW:
                continue
            check_overrides()
            loaded = runtime.observe(label)
            if loaded["loaded"]:
                intent = document.get("revert_bootstrap", {}).get(label)
                if not state["loaded"] or intent is None or intent.get("pid") not in (None, loaded["pid"]):
                    raise RuntimeError("unowned legacy service appeared during revert")
            elif state["loaded"]:
                for replacement in NEW:
                    if runtime.observe(replacement)["loaded"]:
                        raise RuntimeError("replacement revived; refusing legacy bootstrap")
                    wait_custody(custody[replacement], clock=clock)
                if any(alive(pid) for pid in runtime.observed_pids.get(label, set()) | ({state["pid"]} - {None})):
                    raise RuntimeError("legacy writer still lives; refusing bootstrap")
                legacy_available(label)
                document.setdefault("revert_bootstrap", {})[label] = {"pid": None}
                save()
                runtime._checked("bootstrap", domain, agents / f"{label}.plist")
                loaded = runtime.observe(label)
                if not loaded["loaded"]:
                    raise RuntimeError("legacy service failed to load")
            if state["loaded"]:
                document["revert_bootstrap"][label] = loaded
                save()
        check_overrides()
        check_files()
        if any(read_owned(path) != raw for path, raw in originals.items()):
            raise RuntimeError("restored legacy bytes changed")
        if any(
            runtime.observe(label)["loaded"] != (False if label in NEW else state["loaded"])
            for label, state in states.items()
        ):
            raise RuntimeError("runtime changed before revert completion")
        document["phase"] = "reverted"
        save()
        return document
