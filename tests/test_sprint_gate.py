import json
import subprocess
import sys
from pathlib import Path

import pytest

from scripts import sprint_gate

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "sprint_gate.py"
CORPUS = ROOT / "docs.local" / "plans" / "2026-09-01-sprint-gate" / "corpus.json"
FIXTURES = ROOT / "tests" / "fixtures" / "sprint_gate"
QUERIES = (
    "watch bootout typing lag|hybrid helper CPU respawn|brainlayer sprint plan ready milestone|v1.5.10 release install.sh python package|tag filter exact match test socat|enrichment pause sentinel 2026-08-04|cmuxlayer tight-loop contract smaller PRs|how did I implement authentication|agada-bench recall placebo|deploy-local-prod M1 receipts"
).split("|")


def run_fixture(name: str) -> tuple[subprocess.CompletedProcess[str], dict]:
    assert SCRIPT.is_file(), "sprint gate executable is missing"
    fixture = FIXTURES / f"{name}.json"
    assert fixture.is_file(), f"RED fixture is missing: {fixture}"
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--json", "--fixture", str(fixture)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=10,
    )
    return result, json.loads(result.stdout)


def test_corpus_freezes_the_ten_verbatim_queries():
    assert CORPUS.is_file(), "frozen corpus is missing"
    corpus = json.loads(CORPUS.read_text(encoding="utf-8"))
    assert corpus["queries"] == QUERIES
    assert corpus["thresholds"]["latency_regression_fraction"] == 0.10
    assert corpus["thresholds"]["resource_window_seconds"] == 60
    assert corpus["thresholds"]["cpu_percent"] == 30
    assert corpus["thresholds"]["helper_rss_bytes"] == 2 * 1024**3
    assert sprint_gate.resource_sample_count(corpus["thresholds"]) == 61
    assert corpus["machine_target"] == {"os": "Darwin", "architecture": "arm64"}


def test_truncation_notice_and_marker_must_match():
    result = json.loads(
        '{"tools":[{"name":"brain_search","description":"short"}],"_meta":{"brainlayer/descriptionsTruncated":{"tools":["brain_search"]}}}'
    )
    assert sprint_gate.validate_tools(result)[0] is False


def test_missing_wal_is_zero(tmp_path: Path):
    assert sprint_gate.wal_size(tmp_path / "gone") == 0


@pytest.mark.parametrize(
    ("fixture_name", "check_name"),
    [
        ("search_latency_red", "search_latency"),
        ("mcp_roundtrip_red", "mcp_roundtrip"),
        ("resource_budget_red", "resource_budget"),
        ("wal_bound_red", "wal_bound"),
    ],
)
def test_red_fixture_makes_its_check_fail(fixture_name: str, check_name: str):
    result, payload = run_fixture(fixture_name)
    assert result.returncode == 1
    assert payload["status"] == "FAIL"
    assert len(payload["checks"]) == 1
    assert payload["checks"][0]["name"] == check_name
    assert payload["checks"][0]["status"] == "FAIL"
    if fixture_name == "resource_budget_red":
        assert payload["checks"][0]["details"]["helper_max_rss_bytes"] > 2 * 1024**3


def test_green_fixture_proves_all_four_checks_can_pass():
    result, payload = run_fixture("all_green")
    assert result.returncode == 0
    assert payload["mode"] == "replay" and payload["fixture"].endswith("all_green.json")
    assert payload["status"] == "PASS"
    assert [check["name"] for check in payload["checks"]] == [
        "search_latency",
        "mcp_roundtrip",
        "resource_budget",
        "wal_bound",
    ]
    assert {check["status"] for check in payload["checks"]} == {"PASS"}


def test_json_failure_is_machine_readable_without_traceback(tmp_path: Path):
    fixture = tmp_path / "bad.json"
    fixture.write_text(
        json.dumps({"checks": ["mcp_roundtrip"], "socket_path": str(tmp_path / "missing.sock")}), encoding="utf-8"
    )
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--json", "--fixture", str(fixture)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=10,
    )
    payload = json.loads(result.stdout)
    assert result.returncode == 1
    assert payload["checks"][0]["status"] == "FAIL"
    assert "Traceback" not in result.stderr
