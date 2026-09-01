#!/usr/bin/env python3
"""Executable zero-regression gate for the BrainLayer sprint."""

from __future__ import annotations

import argparse
import json
import math
import platform
import re
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from brainlayer.paths import get_db_path

CORPUS = ROOT / "docs.local" / "plans" / "2026-09-01-sprint-gate" / "corpus.json"
SUCCESS_STATUSES = {"STORED", "DUPLICATE", "MERGED", "DEFERRED"}
CHECKS = ("search_latency", "mcp_roundtrip", "resource_budget", "wal_bound")


def wal_size(path: Path) -> int:
    try:
        return path.stat().st_size
    except FileNotFoundError:
        return 0


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
    intact = bool(tools) and not missing and notice_names == set(truncated)
    return intact, {"tool_count": len(tools), "missing_descriptions": missing, "truncated_descriptions": truncated}


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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON")
    parser.add_argument("--fixture", type=Path, help="Replay a deterministic RED/GREEN fixture")
    args = parser.parse_args(argv)
    config = json.loads(CORPUS.read_text(encoding="utf-8"))
    if args.fixture:
        config = merge(config, json.loads(args.fixture.read_text(encoding="utf-8")))
    selected = config.get("checks", list(CHECKS))
    wal_monitor = None
    if "wal_bound" in selected and "wal_samples_bytes" not in config:
        wal_monitor = WalMonitor()
        wal_monitor.start()
    results = []
    runners = {"search_latency": check_search, "mcp_roundtrip": check_mcp, "resource_budget": check_resource}
    for name in selected:
        if name == "wal_bound":
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
        "status": "PASS" if all(item["status"] == "PASS" for item in results) else "FAIL",
        "machine": {"hostname": socket.gethostname(), "os": platform.system(), "architecture": platform.machine()},
        "checks": results,
    }
    print(json.dumps(payload, indent=None if args.json else 2, sort_keys=True))
    return 0 if payload["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
