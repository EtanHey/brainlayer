"""Actual Swift router fixture, OS network denial plus native attempt ledger."""

import json
import platform
from pathlib import Path

from scripts.retirement_artifact import command, digest, private_env


def native_probe(source: Path, work: Path) -> dict:
    if platform.system() != "Darwin" or not Path("/usr/bin/sandbox-exec").is_file():
        raise RuntimeError("Native macOS gate unavailable")
    env = private_env(work / "native-home")
    scratch = work / "swift-build"
    package = source / "brain-bar"
    command(
        ["swift", "test", "--build-tests", "--scratch-path", str(scratch)],
        package,
        env,
        work / "native-build.log",
        1800,
    )
    binaries = list(scratch.glob("**/BrainBarPackageTests.xctest/Contents/MacOS/BrainBarPackageTests"))
    # Swift's debug symlink may name the same binary a second time.
    binaries = sorted({path.resolve() for path in binaries})
    if len(binaries) != 1:
        raise ValueError("Native test product missing/ambiguous")
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
            str(binaries[0]),
            "BrainBarTests.RetirementRatchetTests",
        ],
        package,
        env,
        work / "native-tests.log",
        300,
    )
    log = (work / "native-tests.log").read_text()
    required = (
        "testNativeNetworkBoundaryIsArmed",
        "testEveryPaletteRejectsRetiredDispatch",
        "testActualTemporaryStoreDigestAndSearchRemainLocal",
    )
    if (
        not all(any(name in line and "passed" in line for line in log.splitlines()) for name in required)
        or "Executed 3 tests, with 0 failures" not in log
    ):
        raise ValueError("Native tests did not execute the complete required matrix")
    ledger = [json.loads(line) for line in events.read_text().splitlines()]
    if ledger != [{"kind": "native.connect", "phase": "control"}]:
        raise ValueError("Native guard control missing or candidate attempted a send")
    return {
        "status": "PASS",
        "tests": 3,
        "product_sha256": digest(binaries[0]),
        "guard_sha256": digest(dylib),
        "events": ledger,
        "boundary": "OS deny network + connect/sendto interposition",
        "method": "actual Swift XCTest product; real router dispatch and temporary SQLite data, no production socket",
    }
