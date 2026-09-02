#!/usr/bin/env python3
"""Executable zero-regression gate for the BrainLayer sprint.

How to get proof_eligible (``--require-code-under-test``)
---------------------------------------------------------
"Code under test" means the served package was BUILT FROM this working tree's sha, not that
it lives under this path. Two modes; ``provenance.provenance_mode`` says which one ran:

* ``dev-tree`` -- BrainBar's hybrid helper imports ``brainlayer`` from this repo's ``src/`` (an
  editable install, or BrainBar launched with ``BRAINLAYER_SOURCE_FALLBACK=1`` and
  ``BRAINLAYER_REPO_ROOT`` pointing here) AND ``git status --porcelain`` is empty AND the helper
  process started AFTER the newest file mtime under ``src/brainlayer/`` (a helper that loaded
  commit A and survived a checkout to commit B still serves A). Package data counts too: a branch
  that changes only ``lexical_defense_dictionary.json`` changes search behaviour. ``ps`` reports
  start times at whole-second resolution, so a helper restarted in the same second a file was
  written is refused (``helper_older_than_tree``) rather than accepted: the window errs closed.
  Only ``src/brainlayer`` counts: a stale copy under ``.venv/.../site-packages`` lives under this
  repo too but is a keg, not the tree.
* ``keg`` -- the served package (e.g. the brew Cellar venv) exposes ``brainlayer.__build_sha__``
  equal to ``git rev-parse HEAD`` AND the tree is clean. No release build stamps the sha yet
  (planned for 1.5.11), so every keg today refuses with ``served_build_sha_missing``.

Version equality alone is never accepted: an installed 1.5.10 and 1.5.10-plus-a-branch share a
string while being different code. A refusal names every predicate that fired: ``version``,
``package_path_outside_tree`` (the keg-mode qualifier: dev-tree cannot apply, so the build sha
must), ``served_build_sha_missing``, ``build_sha_mismatch``, ``helper_older_than_tree``,
``working_tree_dirty``. Replay (``--fixture``) under the flag is REJECTED by design: a fixture
is not evidence about any served code, so replay never shells out to ``git`` at all. Live
provenance that cannot be resolved (no helper, ``lsof``, ``ps`` or ``git`` failure) is reported as
``provenance_error`` in the normal payload shape, never as a traceback.
"""

from __future__ import annotations

import argparse
import calendar
import json
import math
import os
import platform
import re
import shlex
import socket
import sqlite3
import subprocess
import sys
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = (ROOT / "src").resolve()
PACKAGE = SRC / "brainlayer"
sys.path.insert(0, str(SRC))
from brainlayer import __version__
from brainlayer.paths import get_db_path

CORPUS = ROOT / "tests" / "fixtures" / "sprint_gate" / "corpus.json"
SUCCESS_STATUSES = {"STORED", "DUPLICATE", "MERGED", "DEFERRED"}
CHECKS = ("search_latency", "mcp_roundtrip", "resource_budget", "wal_bound")
PS_LSTART_FORMAT = "%a %b %d %H:%M:%S %Y"
PLACEHOLDER_TREE = {"working_tree_version": __version__, "working_tree_sha": None, "working_tree_dirty": None}


def wal_size(path: Path) -> int:
    try:
        return path.stat().st_size
    except FileNotFoundError:
        return 0


def contains_key(value, key: str) -> bool:
    if isinstance(value, dict):
        return key in value or any(contains_key(item, key) for item in value.values())
    return isinstance(value, list) and any(contains_key(item, key) for item in value)


class MCPClient:
    def __init__(self, path: str, timeout: float):
        self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.sock.settimeout(timeout)
        self.sock.connect(path)
        self.stream = self.sock.makefile("rwb")
        self.request_id = 0

    def close(self) -> None:
        self.stream.close()
        self.sock.close()

    def request(self, method: str, params: dict | None = None) -> dict:
        self.request_id += 1
        request_id = self.request_id
        body = {"jsonrpc": "2.0", "id": request_id, "method": method, "params": params or {}}
        self.stream.write(json.dumps(body, separators=(",", ":")).encode() + b"\n")
        self.stream.flush()
        while line := self.stream.readline():
            response = json.loads(line)
            if response.get("id") == request_id:
                if "error" in response:
                    raise RuntimeError(str(response["error"]))
                return response["result"]
        raise RuntimeError("MCP socket closed before response")

    def initialize(self) -> dict:
        result = self.request(
            "initialize",
            {
                "protocolVersion": "2025-06-18",
                "capabilities": {},
                "clientInfo": {"name": "sprint-gate", "version": "1"},
            },
        )
        self.stream.write(b'{"jsonrpc":"2.0","method":"notifications/initialized"}\n')
        self.stream.flush()
        return result

    def call(self, name: str, arguments: dict) -> dict:
        result = self.request("tools/call", {"name": name, "arguments": arguments})
        if result.get("isError"):
            raise RuntimeError(tool_text(result) or f"{name} returned isError=true")
        return result


class WalMonitor:
    def __init__(self, interval: float = 0.25):
        self.path = Path(f"{get_db_path()}-wal")
        self.interval = interval
        self.samples: list[int] = []
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        while not self.stop_event.is_set():
            self.samples.append(wal_size(self.path))
            self.stop_event.wait(self.interval)

    def start(self) -> None:
        self.thread.start()

    def stop(self) -> list[int]:
        self.stop_event.set()
        self.thread.join()
        self.samples.append(wal_size(self.path))
        return self.samples


def tool_text(result: dict) -> str:
    return "\n".join(item.get("text", "") for item in result.get("content", []) if item.get("type") == "text")


def search_result_rows(text: str) -> list[str]:
    header = re.search(r"^## Search results for .+ - (\d+) of \d+ shown$", text, re.MULTILINE)
    if not header or int(header.group(1)) < 1:
        return []
    body = text[header.end() :]
    return re.findall(r"^### \d+\..*?(?=^### \d+\.|\Z)", body, re.MULTILINE | re.DOTALL)


def search_visible(client: MCPClient, marker: str, timeout: float) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        text = tool_text(client.call("brain_search", {"query": marker, "num_results": 5}))
        if any(marker in row for row in search_result_rows(text)):
            return True
        time.sleep(0.25)
    return False


def percentile(samples: list[float], fraction: float) -> float:
    ordered = sorted(samples)
    return ordered[max(0, math.ceil(len(ordered) * fraction) - 1)]


def status(name: str, passed: bool, **details) -> dict:
    return {"name": name, "status": "PASS" if passed else "FAIL", "details": details}


def check_search(config: dict) -> dict:
    samples = config.get("latency_samples_ms")
    if samples is None:
        client = MCPClient(config["socket_path"], config["mcp_timeout_seconds"])
        try:
            client.initialize()
            samples = []
            for query in config["queries"]:
                started = time.perf_counter()
                client.call("brain_search", {"query": query, "num_results": 1})
                samples.append(round((time.perf_counter() - started) * 1000, 3))
        finally:
            client.close()
    baseline = config["latency_baseline_ms"]
    regression = config["thresholds"]["latency_regression_fraction"]
    measured = {"p50": percentile(samples, 0.50), "p95": percentile(samples, 0.95)}
    limits = {key: baseline[key] * (1 + regression) for key in measured}
    passed = all(measured[key] <= limits[key] for key in measured)
    return status(
        "search_latency", passed, samples_ms=samples, measured_ms=measured, baseline_ms=baseline, limits_ms=limits
    )


def validate_tools(result: dict) -> tuple[bool, dict]:
    tools = result.get("tools", [])
    missing = [tool.get("name", "unknown") for tool in tools if not tool.get("description")]
    truncated = [tool.get("name", "unknown") for tool in tools if tool.get("description", "").endswith("…[truncated]")]
    notice = result.get("_meta", {}).get("brainlayer/descriptionsTruncated")
    notice_names = set(notice.get("tools", [])) if isinstance(notice, dict) else set()
    missing_annotations = [tool.get("name", "unknown") for tool in tools if "annotations" not in tool]
    missing_input_schema_prose = [
        tool.get("name", "unknown") for tool in tools if not contains_key(tool.get("inputSchema"), "description")
    ]
    truncation_evidenced = bool(truncated) or isinstance(notice, dict)
    truncation_order_valid = not truncation_evidenced or (
        len(missing_annotations) == len(tools) and len(missing_input_schema_prose) == len(tools)
    )
    intact = bool(tools) and not missing and notice_names == set(truncated) and truncation_order_valid
    return intact, {
        "tool_count": len(tools),
        "missing_descriptions": missing,
        "truncated_descriptions": truncated,
        "missing_annotations": missing_annotations,
        "missing_input_schema_prose": missing_input_schema_prose,
        "truncation_evidenced": truncation_evidenced,
        "truncation_order_valid": truncation_order_valid,
    }


def check_mcp(config: dict) -> dict:
    if fixture := config.get("mcp_fixture"):
        return status(
            "mcp_roundtrip",
            bool(fixture["tools_intact"] and fixture["store_status"] in SUCCESS_STATUSES and fixture["planted_hit"]),
            **fixture,
        )
    client = MCPClient(config["socket_path"], config["mcp_timeout_seconds"])
    try:
        client.initialize()
        try:
            client.call("expand_palette", {})
        except RuntimeError as error:
            if "-32601" not in str(error) or "Unknown tool" not in str(error):
                raise
        intact, tool_details = validate_tools(client.request("tools/list"))
        marker = f"SPRINT-GATE-TEST-{socket.gethostname()}-{time.time_ns()}"
        stored = client.call(
            "brain_store",
            {
                "content": f"SPRINT GATE TEST PAYLOAD; safe to archive; marker={marker}",
                "project": "brainlayer",
                "tags": ["sprint-gate-test"],
                "importance": 1,
            },
        )
        store_status = stored.get("status")
        wait_budget = config["thresholds"]["deferred_visibility_wait_seconds"] if store_status == "DEFERRED" else 1.25
        wait_started = time.monotonic()
        hit = search_visible(client, marker, wait_budget)
        observed_wait = round(time.monotonic() - wait_started, 3)
        passed = intact and store_status in SUCCESS_STATUSES and hit
        return status(
            "mcp_roundtrip",
            passed,
            **tool_details,
            store_status=store_status,
            planted_hit=hit,
            planted_hit_wait_seconds=observed_wait,
        )
    finally:
        client.close()


def ps_sample(patterns: dict[str, str]) -> dict:
    output = subprocess.run(
        ["ps", "-axo", "pid=,rss=,pcpu=,command="], check=True, capture_output=True, text=True
    ).stdout
    sample = {name: {"cpu_pct": 0.0, "rss_bytes": 0, "pids": []} for name in patterns}
    for line in output.splitlines():
        parts = line.strip().split(None, 3)
        if len(parts) != 4:
            continue
        pid, rss_kib, cpu, command = parts
        for name, pattern in patterns.items():
            if pattern in command and "sprint_gate.py" not in command:
                sample[name]["cpu_pct"] += float(cpu)
                sample[name]["rss_bytes"] += int(rss_kib) * 1024
                sample[name]["pids"].append(int(pid))
    return sample


def resource_sample_count(thresholds: dict) -> int:
    return math.floor(thresholds["resource_window_seconds"] / thresholds["resource_sample_interval_seconds"]) + 1


def check_resource(config: dict) -> dict:
    samples = config.get("resource_samples")
    if samples is None:
        thresholds = config["thresholds"]
        count = resource_sample_count(thresholds)
        samples = []
        for index in range(count):
            samples.append(ps_sample(config["process_patterns"]))
            if index + 1 < count:
                time.sleep(thresholds["resource_sample_interval_seconds"])
    names = config["process_patterns"]
    cpu = {name: sum(sample[name]["cpu_pct"] for sample in samples) / len(samples) for name in names}
    helper_rss = max(sample["helper"]["rss_bytes"] for sample in samples)
    required_missing = [name for name in config.get("required_processes", []) if not samples[-1][name]["pids"]]
    thresholds = config["thresholds"]
    passed = all(value < thresholds["cpu_percent"] for value in cpu.values())
    passed = passed and helper_rss < thresholds["helper_rss_bytes"] and not required_missing
    return status(
        "resource_budget",
        passed,
        sample_count=len(samples),
        average_cpu_pct=cpu,
        helper_max_rss_bytes=helper_rss,
        required_missing=required_missing,
    )


def check_wal(config: dict, samples: list[int] | None = None) -> dict:
    samples = config.get("wal_samples_bytes", samples or [])
    ceiling = config["thresholds"]["wal_ceiling_bytes"]
    maximum = max(samples, default=0)
    return status("wal_bound", maximum < ceiling, max_bytes=maximum, ceiling_bytes=ceiling, sample_count=len(samples))


def merge(base: dict, override: dict) -> dict:
    result = dict(base)
    for key, value in override.items():
        result[key] = (
            merge(result[key], value) if isinstance(value, dict) and isinstance(result.get(key), dict) else value
        )
    return result


def working_tree_provenance() -> dict:
    sha = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()
    dirty = bool(
        subprocess.run(
            ["git", "status", "--porcelain"], cwd=ROOT, check=True, capture_output=True, text=True
        ).stdout.strip()
    )
    return {"working_tree_version": __version__, "working_tree_sha": sha, "working_tree_dirty": dirty}


def warmup_query() -> str:
    """A nonce, never a corpus query: hybrid_search caches identical requests for 60 s, so warming
    with ``queries[0]`` would hand check_search a cache hit as its first timed sample."""
    return f"warmup-{uuid.uuid4()}"


def warm_helper(config: dict) -> None:
    """The hybrid helper is spawned on demand by the first search; idle == 0 helpers is normal."""
    client = MCPClient(config["socket_path"], config["mcp_timeout_seconds"])
    try:
        client.initialize()
        client.call("brain_search", {"query": warmup_query(), "num_results": 1})
    finally:
        client.close()


def find_helper() -> tuple[int, Path]:
    processes = subprocess.run(
        ["ps", "-axo", "pid=,command="], check=True, capture_output=True, text=True
    ).stdout.splitlines()
    helpers = []
    for line in processes:
        fields = line.strip().split(None, 1)
        if len(fields) != 2 or "brainlayer.brainbar_hybrid_helper" not in fields[1]:
            continue
        arguments = shlex.split(fields[1])
        if "--db-path" in arguments:
            helpers.append((int(fields[0]), Path(arguments[arguments.index("--db-path") + 1])))
    if len(helpers) != 1:
        raise RuntimeError(f"expected one serving hybrid helper, found {len(helpers)}")
    return helpers[0]


def helper_started_at(helper_pid: int) -> float:
    """Epoch seconds the helper process started.

    ``ps -o lstart=`` prints a naive wall-clock in the zone IT runs in, so both sides are pinned
    to UTC: ``TZ=UTC`` for ps and ``calendar.timegm`` for the parse. Parsing in the parent's zone
    would date every helper hours into the future for a ``TZ`` east of the system zone and the
    predicate would silently stop firing (pair review, #749).
    """
    started = subprocess.run(
        ["ps", "-o", "lstart=", "-p", str(helper_pid)],
        check=True,
        capture_output=True,
        text=True,
        env={"LC_ALL": "C", "TZ": "UTC", "PATH": os.environ.get("PATH", os.defpath)},
    ).stdout.strip()
    if not started:
        raise RuntimeError(f"helper pid {helper_pid} has no start time (exited?)")
    return float(calendar.timegm(time.strptime(started, PS_LSTART_FORMAT)))


def newest_source_mtime() -> float:
    """Newest mtime of any file under ``src/brainlayer/`` -- code and package data a fresh helper
    would load. ``__pycache__`` is excluded: bytecode is written AFTER the helper starts importing."""
    files = (path for path in PACKAGE.rglob("*") if path.is_file() and "__pycache__" not in path.parts)
    return max((path.stat().st_mtime for path in files), default=0.0)


def iso_utc(epoch: float | None) -> str | None:
    return None if epoch is None else datetime.fromtimestamp(epoch, tz=timezone.utc).isoformat()


def helper_python(helper_pid: int) -> Path:
    open_files = subprocess.run(
        ["lsof", "-p", str(helper_pid), "-Fn"], check=True, capture_output=True, text=True
    ).stdout.splitlines()
    site_dirs = {
        Path(line[1:].split("/site-packages/", 1)[0] + "/site-packages")
        for line in open_files
        if line.startswith("n") and "/site-packages/" in line
    }
    if len(site_dirs) != 1:
        raise RuntimeError(f"could not resolve one site-packages for helper pid {helper_pid}")
    return next(iter(site_dirs)).parents[2] / "bin" / "python"


def served_package(python: Path) -> dict:
    return json.loads(
        subprocess.run(
            [
                str(python),
                "-I",
                "-c",
                "import brainlayer,json; print(json.dumps({'version':brainlayer.__version__,"
                "'path':brainlayer.__file__,'build_sha':getattr(brainlayer,'__build_sha__',None)}))",
            ],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    )


def resolve_served(config: dict) -> tuple[dict, Path]:
    warm_helper(config)
    helper_pid, db_path = find_helper()
    served = served_package(helper_python(helper_pid))
    served["helper_started_at"] = helper_started_at(helper_pid)
    return served, db_path


def db_provenance(db_path: Path) -> dict:
    db_path = db_path.expanduser().resolve()
    connection = sqlite3.connect(f"{db_path.as_uri()}?mode=ro", uri=True)
    try:
        connection.execute("PRAGMA query_only=ON")
        chunk_count = connection.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
    finally:
        connection.close()
    return {"db_path": str(db_path), "db_size_bytes": db_path.stat().st_size, "chunk_count": chunk_count}


def eligibility(served: dict, tree: dict, source_newest_mtime: float) -> tuple[str, list[str]]:
    """Return (provenance_mode, refusals). Eligible iff refusals is empty. Never version-equality alone.

    Pure: ``source_newest_mtime`` is measured once by the caller and the same number is recorded in
    the payload, so the evidence published is the evidence that decided. dev-tree means the served
    file is under ``src/brainlayer`` specifically: ``.venv/.../site-packages`` sits under ROOT as
    well and must not skip the build-sha check (Macroscope, #749).
    """
    mode = "dev-tree" if Path(served["path"]).resolve().is_relative_to(PACKAGE) else "keg"
    refusals = []
    if served["version"] != tree["working_tree_version"]:
        refusals.append("version")
    if mode == "keg":
        if served["build_sha"] is None:
            refusals += ["package_path_outside_tree", "served_build_sha_missing"]
        elif served["build_sha"] != tree["working_tree_sha"]:
            refusals += ["package_path_outside_tree", "build_sha_mismatch"]
    elif served["helper_started_at"] <= source_newest_mtime:  # must be strictly newer than the tree
        refusals.append("helper_older_than_tree")
    if tree["working_tree_dirty"]:
        refusals.append("working_tree_dirty")
    return mode, refusals


def unresolved_provenance(error: str) -> dict:
    tree = dict(PLACEHOLDER_TREE)
    try:
        tree = working_tree_provenance()
    except Exception:
        pass
    return {
        "served_version": None,
        "served_package_path": None,
        **tree,
        "served_build_sha": None,
        "helper_started_at": None,
        "source_newest_mtime": None,
        "provenance_mode": None,
        "proof_refusals": ["provenance_unresolved"],
        "served_matches_working_tree": False,
        "db_path": None,
        "db_size_bytes": 0,
        "chunk_count": 0,
        "provenance_error": error,
    }


def collect_live_provenance(config: dict) -> dict:
    """Never raises: refusing to know is a provenance answer, a traceback is not."""
    try:
        served, db_path = resolve_served(config)
        tree = working_tree_provenance()
        source_newest = newest_source_mtime()
        mode, refusals = eligibility(served, tree, source_newest)
        return {
            "served_version": served["version"],
            "served_package_path": str(Path(served["path"]).resolve()),
            **tree,
            "served_build_sha": served["build_sha"],
            "helper_started_at": iso_utc(served["helper_started_at"]),
            "source_newest_mtime": iso_utc(source_newest),
            "provenance_mode": mode,
            "proof_refusals": refusals,
            "served_matches_working_tree": not refusals,
            **db_provenance(db_path),
            "provenance_error": None,
        }
    except Exception as exc:
        return unresolved_provenance(f"{type(exc).__name__}: {exc}")


def refusal_message(provenance: dict) -> str:
    if error := provenance.get("provenance_error"):
        return f"served code could not be resolved: {error}"
    return (
        f"not proof-eligible [{provenance['provenance_mode']}]: {', '.join(provenance['proof_refusals'])} "
        f"(served {provenance['served_version']} build_sha={provenance['served_build_sha']} "
        f"from {provenance['served_package_path']}; working tree {provenance['working_tree_version']} "
        f"at {provenance['working_tree_sha']}, dirty={provenance['working_tree_dirty']})"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON")
    parser.add_argument("--fixture", type=Path, help="Replay a deterministic RED/GREEN fixture")
    parser.add_argument(
        "--require-code-under-test",
        action="store_true",
        help=(
            "Refuse (rc=1, structured payload, no checks run) unless the served package is provably built "
            "from this working tree: dev-tree mode (served path under src/brainlayer, helper started after "
            "the newest source mtime, clean tree) or keg mode (served brainlayer.__build_sha__ == HEAD, clean tree). "
            "Replay is always refused."
        ),
    )
    args = parser.parse_args(argv)
    config = json.loads(CORPUS.read_text(encoding="utf-8"))
    if args.fixture:
        config = merge(config, json.loads(args.fixture.read_text(encoding="utf-8")))
    machine = {"hostname": socket.gethostname(), "os": platform.system(), "architecture": platform.machine()}
    provenance = (
        {
            # Replay never touches git: a fixture is not evidence, so its tree metadata is a placeholder.
            "served_version": "fixture",
            "served_package_path": None,
            **PLACEHOLDER_TREE,
            "served_build_sha": None,
            "helper_started_at": None,
            "source_newest_mtime": None,
            "provenance_mode": "replay",
            "proof_refusals": ["replay"],
            "served_matches_working_tree": False,
            "db_path": None,
            "db_size_bytes": 0,
            "chunk_count": 0,
            "provenance_error": None,
        }
        if args.fixture
        else collect_live_provenance(config)
    )
    proof_eligible = provenance["served_matches_working_tree"]

    def fail(error: str) -> int:
        payload = {
            "mode": "replay" if args.fixture else "live",
            "fixture": str(args.fixture) if args.fixture else None,
            "status": "FAIL",
            "machine": machine,
            "provenance": provenance,
            "proof_eligible": proof_eligible,
            "checks": [],
            "error": error,
        }
        print(json.dumps(payload, indent=None if args.json else 2, sort_keys=True))
        return 1

    if args.require_code_under_test and args.fixture:
        return fail("replay is never proof-eligible: a fixture is not evidence about any served code")
    if args.require_code_under_test and not proof_eligible:
        return fail(refusal_message(provenance))

    machine_target = config.get("machine_target")
    if not args.fixture and machine_target is None:
        return fail("machine target is missing")
    if not args.fixture and not isinstance(machine_target, dict):
        return fail("machine target is invalid")
    if not args.fixture and not {"os", "architecture"}.issubset(machine_target):
        return fail("machine target is incomplete")
    if not args.fixture and any(key not in machine or machine[key] != value for key, value in machine_target.items()):
        return fail("machine target mismatch")
    selected = config.get("checks", list(CHECKS))
    latency_baseline = config.get("latency_baseline_ms")
    if not args.fixture and "search_latency" in selected and latency_baseline is None:
        return fail("latency baseline is missing")
    if not args.fixture and latency_baseline is not None and not isinstance(latency_baseline, dict):
        return fail("latency baseline is invalid")
    calibrated_hostname = latency_baseline.get("hostname") if isinstance(latency_baseline, dict) else None
    if not args.fixture and "search_latency" in selected and not calibrated_hostname:
        return fail("latency baseline is missing its calibrated hostname")
    wal_monitor = None
    if "wal_bound" in selected and "wal_samples_bytes" not in config:
        wal_monitor = WalMonitor()
        wal_monitor.start()
    results = []
    runners = {"search_latency": check_search, "mcp_roundtrip": check_mcp, "resource_budget": check_resource}
    for name in selected:
        if name == "wal_bound":
            continue
        if name == "search_latency" and not args.fixture and machine["hostname"] != calibrated_hostname:
            results.append(
                {
                    "name": name,
                    "status": "SKIPPED",
                    "details": {
                        "reason": "uncalibrated host",
                        "running_hostname": machine["hostname"],
                        "calibrated_hostname": calibrated_hostname,
                    },
                }
            )
            continue
        try:
            results.append(runners[name](config))
        except Exception as exc:
            results.append(status(name, False, error=f"{type(exc).__name__}: {exc}"))
    if "wal_bound" in selected:
        samples = wal_monitor.stop() if wal_monitor else None
        results.append(check_wal(config, samples))
    payload = {
        "mode": "replay" if args.fixture else "live",
        "fixture": str(args.fixture) if args.fixture else None,
        # Option (b): rc stays 0; release consumers must reject skipped checks (w6-REPORT.md).
        "status": "PASS" if all(item["status"] in {"PASS", "SKIPPED"} for item in results) else "FAIL",
        "machine": machine,
        "provenance": provenance,
        "proof_eligible": proof_eligible,
        "checks": results,
        "skipped": [item["name"] for item in results if item["status"] == "SKIPPED"],
    }
    print(json.dumps(payload, indent=None if args.json else 2, sort_keys=True))
    return 0 if payload["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
