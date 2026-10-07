"""Actual Swift router fixture, OS network denial plus native attempt ledger."""

import json
import platform
import re
import subprocess
from pathlib import Path

from scripts.retirement_artifact import command, digest, private_env


def require_native_matrix(log):
    required = (
        "testNativeNetworkBoundaryIsArmed",
        "testEveryPaletteRejectsRetiredDispatch",
        "testActualTemporaryStoreDigestAndSearchRemainLocal",
    )
    if (
        not all(any(name in line and "passed" in line for line in log.splitlines()) for name in required)
        or not re.search(r"Executed 3 tests, with (?:0 tests? skipped and )?0 failures", log)
        or any("Test Case" in line and "skipped" in line for line in log.splitlines())
    ):
        raise ValueError("Native tests must execute 3 required tests with 0 skipped")
    return {"tests": 3, "skipped": 0}


def native_probe(source: Path, work: Path) -> dict:
    if platform.system() != "Darwin" or not Path("/usr/bin/sandbox-exec").is_file():
        raise RuntimeError("Native macOS gate unavailable")
    env = private_env(work / "native-home")
    scratch = work / "swift-build"
    package = source / "brain-bar"
    command(
        ["swift", "build", "--build-tests", "--scratch-path", str(scratch)],
        package,
        env,
        work / "native-build.log",
        1800,
    )
    binaries = [
        path
        for name in ("BrainBarPackageTests", "BrainBarTests")
        for path in scratch.glob(f"**/{name}.xctest/Contents/MacOS/{name}")
    ]
    # Swift's debug symlink may name the same binary a second time.
    binaries = sorted({path.resolve() for path in binaries})
    if len(binaries) != 1:
        raise ValueError("Native test product missing/ambiguous")
    host = subprocess.check_output(["xcrun", "--find", "xctest"], env=env, text=True).strip()
    if not Path(host).is_file():
        raise ValueError("Actual XCTest host unavailable")
    dylib = work / "retirement-interpose.dylib"
    command(
        ["clang", "-dynamiclib", str(source / "scripts/retirement_interpose.c"), "-o", str(dylib)],
        work,
        env,
        work / "native-guard-build.log",
    )
    events = work / "native-events.jsonl"
    env.update(
        DYLD_INSERT_LIBRARIES=str(dylib),
        RETIREMENT_EVENTS=str(events),
        RETIREMENT_PHASE="candidate",
        RETIREMENT_NATIVE_BOUNDARY="1",
    )
    command(
        [
            "/usr/bin/sandbox-exec",
            "-p",
            "(version 1)(allow default)(deny network*)",
            "/usr/bin/env",
            "DYLD_INSERT_LIBRARIES=" + str(dylib),
            host,
            "-XCTest",
            "BrainBarTests.RetirementRatchetTests",
            str(binaries[0].parents[2]),
        ],
        package,
        env,
        work / "native-tests.log",
        300,
    )
    log = (work / "native-tests.log").read_text()
    matrix = require_native_matrix(log)
    ledger = [json.loads(line) for line in events.read_text().splitlines()]
    if ledger != [{"kind": "native.connect", "phase": "control"}]:
        raise ValueError("Native guard control missing or candidate attempted a send")
    return {
        "status": "PASS",
        **matrix,
        "product_sha256": digest(binaries[0]),
        "guard_sha256": digest(dylib),
        "events": ledger,
        "boundary": "OS deny network + connect/sendto interposition",
        "method": "actual Swift XCTest product; real router dispatch and temporary SQLite data, no production socket",
    }
