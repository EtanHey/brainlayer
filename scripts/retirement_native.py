"""Actual Swift router fixture, OS network denial plus native attempt ledger."""

import json
import platform
import re
import struct
import subprocess
from pathlib import Path

from scripts.retirement_artifact import command, digest, private_env


def _architecture(cpu, subtype):
    if cpu == 0x1000007:
        return "x86_64"
    if cpu == 0x100000C and subtype & 0xFFFFFF in (0, 1, 2):
        return "arm64e" if subtype & 0xFFFFFF == 2 else "arm64"
    raise ValueError("Unsupported native Mach-O architecture")


def macho_architectures(path: Path) -> set[str]:
    data = path.read_bytes()
    magic = data[:4]
    fat = {
        b"\xca\xfe\xba\xbe": (">", 20),
        b"\xbe\xba\xfe\xca": ("<", 20),
        b"\xca\xfe\xba\xbf": (">", 32),
        b"\xbf\xba\xfe\xca": ("<", 32),
    }
    thin = {b"\xcf\xfa\xed\xfe": "<", b"\xfe\xed\xfa\xcf": ">"}
    if magic in fat:
        endian, width = fat[magic]
        if len(data) < 8:
            raise ValueError("Truncated native fat header")
        count = struct.unpack_from(endian + "I", data, 4)[0]
        if not 0 < count <= 32 or len(data) < 8 + count * width:
            raise ValueError("Incomplete native architecture table")
        pairs = [struct.unpack_from(endian + "II", data, 8 + i * width) for i in range(count)]
    elif magic in thin and len(data) >= 32:
        pairs = [struct.unpack_from(thin[magic] + "II", data, 4)]
    else:
        raise ValueError("Native executable/guard is not a supported Mach-O file")
    return {_architecture(cpu, subtype) for cpu, subtype in pairs}


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
    host_arches = macho_architectures(Path(host))
    product_arches = macho_architectures(binaries[0])
    required_arches = host_arches | product_arches
    dylib = work / "retirement-interpose.dylib"
    command(
        [
            "clang",
            "-dynamiclib",
            *[flag for arch in sorted(required_arches) for flag in ("-arch", arch)],
            str(source / "scripts/retirement_interpose.c"),
            "-o",
            str(dylib),
        ],
        work,
        env,
        work / "native-guard-build.log",
    )
    guard_arches = macho_architectures(dylib)
    if not required_arches <= guard_arches:
        raise ValueError("Native guard is missing an actual host/product slice")
    events = work / "native-events.jsonl"
    process = work / "native-process.json"
    env.update(
        RETIREMENT_EVENTS=str(events),
        RETIREMENT_PHASE="candidate",
        RETIREMENT_NATIVE_BOUNDARY="1",
        RETIREMENT_PROCESS_IDENTITY=str(process),
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
    identity = json.loads(process.read_text())
    executable = Path(bytes.fromhex(identity["executable_hex"]).decode())
    if (
        type(identity["pid"]) is not int
        or identity["pid"] <= 0
        or not executable.samefile(host)
        or _architecture(identity["cpu_type"], identity["cpu_subtype"]) not in host_arches & guard_arches
    ):
        raise ValueError("Native guard did not identify the actual XCTest process/slice")
    return {
        "status": "PASS",
        **matrix,
        "product_sha256": digest(binaries[0]),
        "guard_sha256": digest(dylib),
        "events": ledger,
        "boundary": "OS deny network + connect/sendto interposition",
        "method": "actual Swift XCTest product; real router dispatch and temporary SQLite data, no production socket",
        "process_identity": identity,
        "host_architectures": sorted(host_arches),
        "product_architectures": sorted(product_arches),
        "guard_architectures": sorted(guard_arches),
    }
