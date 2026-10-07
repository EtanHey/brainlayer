import json
import struct
from pathlib import Path

import pytest

from scripts import retirement_native as native


def thin(cpu, subtype):
    return struct.pack("<8I", 0xFEEDFACF, cpu, subtype, 2, 0, 0, 0, 0)


@pytest.mark.parametrize(
    ("cpu", "subtype", "expected"),
    [(0x100000C, 0, "arm64"), (0x100000C, 0x80000002, "arm64e"), (0x1000007, 3, "x86_64")],
)
def test_architectures_come_from_actual_macho_header(tmp_path, cpu, subtype, expected):
    path = tmp_path / "host"
    path.write_bytes(thin(cpu, subtype))
    assert native.macho_architectures(path) == {expected}


@pytest.mark.parametrize(("endian", "wide"), [(">", False), ("<", False), (">", True), ("<", True)])
def test_fat_header_retains_arm64e_and_arm64(tmp_path, endian, wide):
    path = tmp_path / "host"
    width, fmt = (32, "IIQQII") if wide else (20, "5I")
    offset = 8 + 2 * width
    extra = (0,) if wide else ()
    path.write_bytes(
        struct.pack(endian + "2I", 0xCAFEBABF if wide else 0xCAFEBABE, 2)
        + struct.pack(endian + fmt, 0x100000C, 0, offset, 32, 0, *extra)
        + struct.pack(endian + fmt, 0x100000C, 2, offset + 32, 32, 0, *extra)
        + thin(0x100000C, 0)
        + thin(0x100000C, 2)
    )
    assert native.macho_architectures(path) == {"arm64", "arm64e"}


@pytest.mark.parametrize("data", [b"not-macho", struct.pack(">2I", 0xCAFEBABE, 4), thin(12, 0)])
def test_unknown_or_incomplete_architecture_refuses(tmp_path, data):
    path = tmp_path / "host"
    path.write_bytes(data)
    with pytest.raises(ValueError):
        native.macho_architectures(path)


@pytest.mark.parametrize("defect", [None, "missing-slice", "candidate-send", "missing-identity", "wrong-process"])
def test_guard_covers_host_and_product_and_starts_only_in_xctest(tmp_path, monkeypatch, defect):
    source, work = tmp_path / "source", tmp_path / "work"
    source.mkdir()
    work.mkdir()
    host = tmp_path / "xctest"
    host.write_bytes(thin(0x100000C, 2))
    monkeypatch.setattr(native.platform, "system", lambda: "Darwin")
    original_is_file = Path.is_file
    monkeypatch.setattr(Path, "is_file", lambda p: str(p) == "/usr/bin/sandbox-exec" or original_is_file(p))
    monkeypatch.setattr(native.subprocess, "check_output", lambda *_a, **_k: str(host) + "\n")

    def command(args, cwd, env, log, *unused):
        if args[0] == "swift":
            product = work / "swift-build/private/BrainBarTests.xctest/Contents/MacOS/BrainBarTests"
            product.parent.mkdir(parents=True)
            product.write_bytes(thin(0x100000C, 0))
        elif args[0] == "clang":
            arches = {args[i + 1] for i, value in enumerate(args[:-1]) if value == "-arch"}
            assert arches == {"arm64", "arm64e"}, "compile for the actual host and product slices"
            guard = work / "retirement-interpose.dylib"
            guard.write_bytes(
                struct.pack(">2I", 0xCAFEBABE, 2)
                + struct.pack(">5I", 0x100000C, 0, 48, 32, 0)
                + struct.pack(">5I", 0x100000C, 2, 80, 32, 0)
                + thin(0x100000C, 0)
                + thin(0x100000C, 2)
            )
            if defect == "missing-slice":
                guard.write_bytes(thin(0x100000C, 0))
        else:
            assert "DYLD_INSERT_LIBRARIES" not in env, "do not inject into sandbox/env wrappers"
            assert args[:4] == [
                "/usr/bin/sandbox-exec",
                "-p",
                "(version 1)(allow default)(deny network*)",
                "/usr/bin/env",
            ]
            assert "DYLD_INSERT_LIBRARIES=" + str(work / "retirement-interpose.dylib") in args
            assert env["RETIREMENT_NATIVE_BOUNDARY"] == "1"
            (work / "native-events.jsonl").write_text('{"kind":"native.connect","phase":"control"}\n')
            if defect == "candidate-send":
                (work / "native-events.jsonl").write_text('{"kind":"native.connect","phase":"candidate"}\n')
            if defect != "missing-identity":
                executable = source if defect == "wrong-process" else host
                (work / "native-process.json").write_text(
                    json.dumps(
                        {
                            "pid": 42,
                            "cpu_type": 0x100000C,
                            "cpu_subtype": 2,
                            "executable_hex": str(executable).encode().hex(),
                        }
                    )
                )
            names = [
                "testNativeNetworkBoundaryIsArmed",
                "testEveryPaletteRejectsRetiredDispatch",
                "testActualTemporaryStoreDigestAndSearchRemainLocal",
            ]
            log.write_text(
                "\n".join("Test Case '" + n + "' passed" for n in names) + "\nExecuted 3 tests, with 0 failures"
            )

    monkeypatch.setattr(native, "command", command)
    if defect:
        with pytest.raises((ValueError, FileNotFoundError)):
            native.native_probe(source, work)
    else:
        result = native.native_probe(source, work)
        assert result["tests"] == 3 and result["skipped"] == 0
