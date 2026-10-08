"""Validate global-retirement evidence; checks survive python -O."""

import hashlib
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from scripts.retirement_source import EXTENSIONS, ROOTS

SCOPE = "no cloud model call reachable anywhere"


def need(condition, reason):
    if not condition:
        raise ValueError(reason)


def identity(value, length):
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{" + str(length) + "}", value) is not None


def validate(report, expected_sha, now=None):
    need(isinstance(report, dict), "report is not an object")
    need(identity(expected_sha, 40), "expected source SHA missing")
    need(type(report.get("schema_version")) is int and report["schema_version"] == 1, "unknown schema")
    need(report.get("scope") == SCOPE and report.get("status") == "PASS", "global gate did not pass")
    need(report.get("source_sha") == expected_sha, "stale/wrong source head")
    need(report.get("failures") == [], "failure ledger missing or nonempty")
    need(identity(report.get("dependency_sha"), 40), "R10b dependency gate not armed")
    need(
        report.get("dependency_resolution") == "merge-base" and bool(report.get("dependency_ref")),
        "dependency reference resolution missing",
    )
    need(report.get("dependency_is_ancestor") is True, "R10b is not in the measured source")
    timestamp = datetime.fromisoformat(report["timestamp_utc"])
    need(timestamp.tzinfo is not None, "timestamp is not timezone-aware")
    age = ((now or datetime.now(timezone.utc)) - timestamp).total_seconds()
    need(0 <= age <= 21600, "report stale or from future")
    source, scan, wheel = report["source"], report["scan"], report["wheel"]
    need(source["sha"] == expected_sha and identity(source["tree"], 40), "source provenance incomplete")
    need(identity(source["archive_sha256"], 64), "source archive identity missing")
    files = source["files"]
    need(
        isinstance(files, dict) and files and all(identity(value, 64) for value in files.values()),
        "source manifest invalid",
    )
    need(any(name.startswith("src/brainlayer/") for name in files), "package source absent")
    need(any(name.startswith("brain-bar/Sources/") for name in files), "native source absent")
    need(scan["findings"] == [] and scan["errors"] == [], "static source scan is RED")
    inventory = scan["inventory"]
    need(
        isinstance(inventory, dict) and inventory and all(identity(value, 64) for value in inventory.values()),
        "scan inventory invalid",
    )
    required = {
        name
        for name in files
        if Path(name).suffix in EXTENSIONS
        and any(name.startswith(directory + "/") for directory in ROOTS)
        and not set(Path(name).parts) & {"node_modules", ".next", ".build", ".swiftpm", "__pycache__"}
    }
    need(set(inventory) == required, "source scan inventory is incomplete or unbound")
    links = source.get("links", {})
    for name, value in inventory.items():
        target = name
        if name in links:
            need(
                hashlib.sha256(links[name]["target"].encode()).hexdigest() == files[name],
                "source symlink identity mismatch",
            )
            target = links[name]["resolved"]
        need(value == files[target], "scanned source bytes are not from the manifest")
    digest = hashlib.sha256(
        "\n".join(f"{name}:{value}" for name, value in sorted(inventory.items())).encode()
    ).hexdigest()
    need(digest == scan["inventory_sha256"], "source scan inventory was tampered")
    transport = report["transport_inventory"]
    from scripts.retirement_inventory import fingerprint

    need(transport["sites"] and transport["unclassified"] == [], "transport inventory incomplete/unclassified")
    need(fingerprint(transport["sites"]) == transport["sha256"], "transport inventory hash mismatch")
    need(transport["source_inventory"] == inventory, "transport corpus is incomplete/unbound")
    need(transport["policy_sha256"] == files["scripts/retirement_policy.json"], "transport policy is unbound")
    need(all(site["path"] in inventory for site in transport["sites"].values()), "transport site outside corpus")
    need(identity(wheel["sha256"], 64) and "brainlayer/_build.py" in wheel["members"], "wheel identity/stamp missing")
    package_members = [name for name in wheel["members"] if name.startswith("brainlayer/") and not name.endswith("/")]
    required_modules = sorted(
        {name[:-3].replace("/", ".").removesuffix(".__init__") for name in package_members if name.endswith(".py")}
    )
    profiles = report["profiles"]
    need(isinstance(profiles, list) and len(profiles) == 2, "dependency profiles incomplete")
    need({profile["profile"] for profile in profiles} == {"default", "dev"}, "dependency profile names invalid")
    for profile in profiles:
        need(profile["status"] == "PASS" and profile["cli"] == "PASS", "installed/CLI failure")
        need(
            profile["source_sha"] == expected_sha and profile["wheel_sha256"] == wheel["sha256"],
            "installed artifact identity mismatch",
        )
        prefix, origin = Path(profile["prefix"]), Path(profile["origin"])
        need(
            prefix.is_absolute() and origin.is_absolute() and origin.is_relative_to(prefix),
            "installed origin outside fixture",
        )
        modules = profile["modules"]
        need(
            isinstance(modules, list) and modules and len(set(modules)) == len(modules),
            "module inventory missing/duplicated",
        )
        need(sorted(modules) == required_modules, "installed module inventory incomplete")
        need(
            type(profile["verified_members"]) is int and profile["verified_members"] == len(package_members),
            "installed file checks incomplete",
        )
        need(profile["controls"] == 12, "positive controls incomplete")
        events = profile["events"]
        need(
            len(events) == 12 and all(event["phase"] == "control" for event in events),
            "candidate attempted a send or ledger incomplete",
        )
        expected_controls = Counter(
            {
                "socket.connect": 1,
                "socket.connect_ex": 1,
                "socket.sendto": 1,
                "dns": 2,
                "model_or_retired_import": 3,
                "subprocess": 1,
                "http_send": 2,
                "async_http_send": 1,
            }
        )
        need(Counter(event["kind"] for event in events) == expected_controls, "control transport matrix incomplete")
        need(profile["sdk"] == {"status": "PASS", "importable": [], "declared": []}, "SDK absence not measured")
        need(profile["guard"] == "private sitecustomize + OS deny network", "OS boundary absent")
        need(bool(profile["core"]) and bool(profile["local_adapter"]), "preservation/local endpoint checks missing")
        need(
            set(profile["surfaces"])
            == {
                "cli-retirement",
                "library-mcp-retirement",
                "installed-hotlane-vector",
                "historical-replay",
                "saved-result-drain",
                "watcher-flush",
                "drive-oauth-imports",
            },
            "installed entrypoint matrix incomplete",
        )
    native = report["native"]
    need(native["status"] == "PASS" and native["tests"] == 3, "native matrix incomplete")
    need(type(native.get("skipped")) is int and native["skipped"] == 0, "native matrix skipped a required test")
    need(
        identity(native["product_sha256"], 64) and identity(native["guard_sha256"], 64), "native artifacts unidentified"
    )
    need(native["events"] == [{"kind": "native.connect", "phase": "control"}], "native attempt/control ledger invalid")
    need(native["boundary"] == "OS deny network + connect/sendto interposition", "native boundary missing")
    return {
        "source_files": len(inventory),
        "wheel_sha256": wheel["sha256"],
        "installed_modules": sum(len(profile["modules"]) for profile in profiles),
        "native_tests": 3,
    }
