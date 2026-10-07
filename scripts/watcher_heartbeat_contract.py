"""Reviewed heartbeat evidence contract shared by producer and table consumer."""

import hashlib
import re
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CASES = {
    "fresh",
    "stale",
    "unrelated_store",
    "refresh_recovery",
    "missing",
    "malformed",
    "stopped",
    "boundary_300",
    "boundary_over_300",
    "future",
    "timezone",
    "scan_in_progress",
    "scan_recovery",
}
SOURCES = [
    "src/brainlayer/watcher.py",
    "src/brainlayer/watcher_bridge.py",
    "src/brainlayer/queue_io.py",
    "brain-bar/Sources/BrainBar/Dashboard/WatcherHealthStatus.swift",
    "brain-bar/Sources/BrainBar/Dashboard/DashboardMetricFormatter.swift",
    "brain-bar/Sources/BrainBar/Dashboard/PipelineState.swift",
    "scripts/watcher_heartbeat_probe.swift",
    "scripts/watcher_heartbeat_ratchet.py",
    "scripts/watcher_heartbeat_contract.py",
]


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def store_content(index: int) -> str:
    return f"Synthetic independent heartbeat store {index}; never revives watcher health."


def validate_stores(stores: list[dict], *, complete: bool = True) -> None:
    if not isinstance(stores, list) or not 1 <= len(stores) <= 4 or (complete and len(stores) != 4):
        raise ValueError("four persisted independent stores required")
    paths, identities = set(), set()
    for index, item in enumerate(stores):
        path, event = item["path"], item["event"]
        if not isinstance(path, str) or not re.fullmatch(r"mcp-[0-9]+-[0-9a-f]{32}\.jsonl", path):
            raise ValueError("invalid private store path")
        if path in paths or event["chunk_id"] in identities:
            raise ValueError("duplicate independent store identity")
        if (
            event["kind"] != "store_memory"
            or event["source"] != "mcp"
            or event["memory_type"] != "note"
            or event["content"] != store_content(index)
            or not re.fullmatch(r"manual-[0-9a-f]{16}", event["chunk_id"])
        ):
            raise ValueError("persisted store event does not match stimulus")
        paths.add(path)
        identities.add(event["chunk_id"])


def parse_clock(value: str) -> datetime:
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("health clock requires timezone")
    return parsed


def validate_completion(completed: dict, previous: int) -> None:
    count = completed["poll_count"]
    if (
        type(count) is not int
        or count != previous + 1
        or type(completed["actual_poll_count"]) is not int
        or count != completed["actual_poll_count"]
        or type(completed["previous_poll_count"]) is not int
        or completed["previous_poll_count"] != previous
    ):
        raise ValueError("published completed poll count did not advance")
    timestamp = parse_clock(completed["updated_at"])
    if not (
        parse_clock(completed["previous_updated_at"]) < timestamp
        and parse_clock(completed["completion_started_at"])
        <= timestamp
        <= parse_clock(completed["completion_finished_at"])
    ):
        raise ValueError("published completion timestamp outside advancing completion clocks")


def process_declaration() -> str:
    text = (ROOT / "brain-bar/Sources/BrainBar/Dashboard/PipelineState.swift").read_text()
    declaration = text.split("enum WatcherProcessProbeResult:", 1)[1].split("\n}", 1)[0]
    return "import Foundation\nenum WatcherProcessProbeResult:" + declaration + "\n}\n"


def validate_artifact(data: dict, evidence_root: Path) -> None:
    # Only fixed job-produced paths below the caller-declared root; never execute receipt paths.
    artifact = evidence_root / "heartbeat-evidence/heartbeat-consumer"
    enum_file = evidence_root / "heartbeat-evidence/ProcessEvidence.swift"
    for path in (artifact, enum_file):
        if not path.resolve().is_relative_to(evidence_root.resolve()) or not path.is_file():
            raise ValueError("missing or escaped compiled consumer artifact")
    if data["binary_sha256"] != digest(artifact):
        raise ValueError("compiled consumer artifact digest mismatch")
    expected = {
        p: data["sources"][p] for p in SOURCES if p.endswith(".swift") and not p.endswith("PipelineState.swift")
    }
    expected["ProcessEvidence.swift"] = digest(enum_file)
    if enum_file.read_text() != process_declaration() or data["compile_sources"] != expected:
        raise ValueError("compiled consumer source identity mismatch")
