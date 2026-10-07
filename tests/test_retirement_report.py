"""Stale, incomplete and tampered receipts must never give a green row."""

import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

from scripts.retirement_report import SCOPE, validate

SHA = "a" * 40
NOW = datetime(2026, 10, 7, tzinfo=timezone.utc)


def valid():
    files = {"src/brainlayer/__init__.py": "b" * 64, "brain-bar/Sources/BrainBar/MCPRouter.swift": "c" * 64}
    fingerprint = hashlib.sha256(
        "\n".join(f"{name}:{value}" for name, value in sorted(files.items())).encode()
    ).hexdigest()
    profiles = [
        {
            "profile": name,
            "status": "PASS",
            "cli": "PASS",
            "source_sha": SHA,
            "wheel_sha256": "d" * 64,
            "origin": "/fixture/venv/site/brainlayer/__init__.py",
            "prefix": "/fixture/venv",
            "modules": ["brainlayer._build", "brainlayer"],
            "verified_members": 2,
            "controls": 12,
            "events": [
                {"phase": "control", "kind": kind}
                for kind in [
                    "socket.connect",
                    "socket.connect_ex",
                    "socket.sendto",
                    "dns",
                    "dns",
                    "model_or_retired_import",
                    "model_or_retired_import",
                    "model_or_retired_import",
                    "subprocess",
                    "http_send",
                    "http_send",
                    "async_http_send",
                ]
            ],
            "sdk": {"status": "PASS", "importable": [], "declared": []},
            "guard": "private sitecustomize + OS deny network",
            "core": "tested",
            "local_adapter": "tested",
            "surfaces": [
                "cli-retirement",
                "library-mcp-retirement",
                "installed-hotlane-vector",
                "historical-replay",
                "saved-result-drain",
                "watcher-flush",
                "drive-oauth-imports",
            ],
        }
        for name in ("default", "dev")
    ]
    return {
        "schema_version": 1,
        "scope": SCOPE,
        "status": "PASS",
        "source_sha": SHA,
        "transport_inventory": {
            "sites": {"fixture-site": {"path": "src/brainlayer/__init__.py"}},
            "sha256": hashlib.sha256(b'{"fixture-site":{"path":"src/brainlayer/__init__.py"}}').hexdigest(),
            "unclassified": [],
            "source_inventory": files.copy(),
            "policy_sha256": "9" * 64,
        },
        "timestamp_utc": NOW.isoformat(),
        "failures": [],
        "dependency_sha": "e" * 40,
        "dependency_ref": "origin/main",
        "dependency_resolution": "merge-base",
        "dependency_is_ancestor": True,
        "source": {
            "sha": SHA,
            "tree": "f" * 40,
            "archive_sha256": "a" * 64,
            "files": {**files, "scripts/retirement_policy.json": "9" * 64},
        },
        "scan": {"inventory": files, "inventory_sha256": fingerprint, "findings": [], "errors": []},
        "wheel": {"sha256": "d" * 64, "members": ["brainlayer/_build.py", "brainlayer/__init__.py"]},
        "profiles": profiles,
        "native": {
            "status": "PASS",
            "tests": 3,
            "skipped": 0,
            "product_sha256": "a" * 64,
            "guard_sha256": "b" * 64,
            "events": [{"kind": "native.connect", "phase": "control"}],
            "boundary": "OS deny network + connect/sendto interposition",
        },
    }


def test_complete_report_validates():
    assert validate(valid(), SHA, NOW)["native_tests"] == 3


@pytest.mark.parametrize(
    "path,value",
    [
        (("status",), "PENDING_R10B"),
        (("source_sha",), "b" * 40),
        (("dependency_is_ancestor",), False),
        (("dependency_resolution",), None),
        (("dependency_ref",), None),
        (("failures",), ["timeout"]),
        (("timestamp_utc",), "2026-10-06T00:00:00+00:00"),
        (("timestamp_utc",), "2026-10-08T00:00:00+00:00"),
        (("scan", "inventory_sha256"), "0" * 64),
        (("scan", "findings"), [{"target": "restored sender"}]),
        (("profiles", 0, "wheel_sha256"), "e" * 64),
        (("profiles", 0, "origin"), "/checkout/src/brainlayer/__init__.py"),
        (("profiles", 0, "controls"), 11),
        (("profiles", 0, "surfaces"), []),
        (("transport_inventory", "source_inventory"), {}),
        (("transport_inventory", "policy_sha256"), "0" * 64),
        (("profiles", 0, "events"), [{"kind": "caught model send", "phase": "candidate"}]),
        (("profiles", 0, "sdk", "importable"), ["google.genai"]),
        (("native", "status"), "SKIP"),
        (("native", "events"), []),
        (("native", "tests"), 0),
        (("native", "skipped"), 1),
        (("native", "skipped"), None),
        (("native", "skipped"), False),
    ],
)
def test_invalid_receipt_fails(path, value):
    report = valid()
    target = report
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    with pytest.raises((ValueError, KeyError, TypeError)):
        validate(report, SHA, NOW)


@pytest.mark.parametrize("key", ["scope", "source", "scan", "wheel", "profiles", "native", "dependency_sha"])
def test_missing_leg_fails(key):
    report = valid()
    del report[key]
    with pytest.raises((ValueError, KeyError, TypeError)):
        validate(report, SHA, NOW)


def test_receipt_validation_is_still_armed_under_optimized_python(tmp_path):
    report = valid()
    report["status"] = "FAIL"
    fixture = tmp_path / "invalid.json"
    fixture.write_text(json.dumps(report))
    code = "import sys,json;sys.path.insert(0,sys.argv[1]);from scripts.retirement_report import validate;validate(json.load(open(sys.argv[2])),sys.argv[3])"
    process = subprocess.run(
        [sys.executable, "-I", "-O", "-c", code, str(Path(__file__).resolve().parents[1]), str(fixture), SHA],
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert process.returncode != 0 and "global gate did not pass" in process.stderr


def test_scan_cannot_drop_a_file_and_rehash_inventory():
    report = valid()
    report["scan"]["inventory"] = {"src/brainlayer/__init__.py": "b" * 64}
    report["scan"]["inventory_sha256"] = hashlib.sha256(b"src/brainlayer/__init__.py:" + b"b" * 64).hexdigest()
    with pytest.raises(ValueError, match="incomplete or unbound"):
        validate(report, SHA, NOW)
