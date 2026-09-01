import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import sprint_gate

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "sprint_gate.py"
CORPUS = ROOT / "tests" / "fixtures" / "sprint_gate" / "corpus.json"
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


def deterministic_live_config() -> dict:
    config = json.loads(CORPUS.read_text(encoding="utf-8"))
    config.update(
        {
            "checks": list(sprint_gate.CHECKS),
            "latency_samples_ms": [100],
            "mcp_fixture": {"tools_intact": True, "store_status": "STORED", "planted_hit": True},
            "resource_samples": [
                {
                    "daemon": {"cpu_pct": 0, "rss_bytes": 0, "pids": [1]},
                    "helper": {"cpu_pct": 0, "rss_bytes": 0, "pids": []},
                    "watcher": {"cpu_pct": 0, "rss_bytes": 0, "pids": []},
                    "drain": {"cpu_pct": 0, "rss_bytes": 0, "pids": []},
                }
            ],
            "wal_samples_bytes": [0],
        }
    )
    return config


def run_live_config(monkeypatch, capsys, tmp_path: Path, config: dict, hostname: str | None = None) -> tuple[int, dict]:
    live_config = tmp_path / "live.json"
    live_config.write_text(json.dumps(config), encoding="utf-8")
    monkeypatch.setattr(sprint_gate, "CORPUS", live_config)
    target = config.get("machine_target", {})
    for key, attribute in (("os", "system"), ("architecture", "machine")):
        if key in target:
            monkeypatch.setattr(sprint_gate.platform, attribute, lambda key=key: target[key])
    if hostname:
        monkeypatch.setattr(sprint_gate.socket, "gethostname", lambda: hostname)
    return sprint_gate.main(["--json"]), json.loads(capsys.readouterr().out)


def test_corpus_freezes_the_ten_verbatim_queries():
    assert CORPUS.is_file(), "frozen corpus is missing"
    corpus = json.loads(CORPUS.read_text(encoding="utf-8"))
    assert corpus["queries"] == QUERIES
    assert corpus["thresholds"]["latency_regression_fraction"] == 0.10
    assert corpus["thresholds"]["resource_window_seconds"] == 60
    assert corpus["thresholds"]["cpu_percent"] == 30
    assert corpus["thresholds"]["helper_rss_bytes"] == 2 * 1024**3
    assert corpus["thresholds"]["deferred_visibility_wait_seconds"] == 5
    assert sprint_gate.resource_sample_count(corpus["thresholds"]) == 61
    assert corpus["machine_target"] == {"os": "Darwin", "architecture": "arm64"}


def test_truncation_notice_and_marker_must_match():
    result = json.loads(
        '{"tools":[{"name":"brain_search","description":"short"}],"_meta":{"brainlayer/descriptionsTruncated":{"tools":["brain_search"]}}}'
    )
    assert sprint_gate.validate_tools(result)[0] is False


def test_missing_wal_is_zero(tmp_path: Path):
    assert sprint_gate.wal_size(tmp_path / "gone") == 0


def test_search_visibility_rejects_zero_result_header_echo(monkeypatch):
    ticks = iter([0.0, 0.0, 1.0])
    monkeypatch.setattr(sprint_gate.time, "monotonic", lambda: next(ticks))
    monkeypatch.setattr(sprint_gate.time, "sleep", lambda _: None)
    response = '## Entity: related context\n\n## Search results for "MARKER" - 0 of 0 shown\n\nNo results found.'
    client = SimpleNamespace(call=lambda _name, _arguments: {"content": [{"type": "text", "text": response}]})
    assert sprint_gate.search_visible(client, "MARKER", 0.5) is False


def test_deferred_roundtrip_uses_configured_wait_and_reports_observed(monkeypatch):
    waits = []
    client = SimpleNamespace(initialize=lambda: None, call=lambda *_: {"status": "DEFERRED"}, close=lambda: None)
    client.request = lambda _method: {"tools": [{"name": "brain_search", "description": "Search memory"}]}
    monkeypatch.setattr(sprint_gate, "MCPClient", lambda _path, _timeout: client)
    monkeypatch.setattr(sprint_gate, "search_visible", lambda _client, _marker, timeout: waits.append(timeout) or True)
    ticks = iter([10.0, 12.5])
    monkeypatch.setattr(sprint_gate.time, "monotonic", lambda: next(ticks))

    config = {"socket_path": "unused", "mcp_timeout_seconds": 20, "thresholds": {"deferred_visibility_wait_seconds": 5}}
    result = sprint_gate.check_mcp(config)

    assert result["status"] == "PASS"
    assert waits == [5]
    assert result["details"]["planted_hit_wait_seconds"] == 2.5


def test_live_gate_rejects_wrong_machine(monkeypatch, capsys):
    monkeypatch.setattr(sprint_gate, "CHECKS", ())
    monkeypatch.setattr(sprint_gate.platform, "machine", lambda: "wrong")
    assert sprint_gate.main(["--json"]) == 1
    assert "machine target mismatch" in json.loads(capsys.readouterr().out)["error"]


def test_live_gate_skips_latency_on_uncalibrated_host(monkeypatch, capsys, tmp_path: Path):
    config = deterministic_live_config()
    returncode, payload = run_live_config(monkeypatch, capsys, tmp_path, config, "uncalibrated.local")
    assert returncode == 0
    assert [check["name"] for check in payload["checks"]] == list(sprint_gate.CHECKS)
    assert payload["skipped"] == ["search_latency"]
    assert payload["checks"][0] == {
        "name": "search_latency",
        "status": "SKIPPED",
        "details": {
            "reason": "uncalibrated host",
            "running_hostname": "uncalibrated.local",
            "calibrated_hostname": "MacBook-Pro.local",
        },
    }
    assert {check["status"] for check in payload["checks"][1:]} == {"PASS"}


def test_live_gate_rejects_missing_baseline_hostname(monkeypatch, capsys, tmp_path: Path):
    config = deterministic_live_config()
    del config["latency_baseline_ms"]["hostname"]
    returncode, payload = run_live_config(monkeypatch, capsys, tmp_path, config)
    assert returncode == 1
    assert payload["error"] == "latency baseline is missing its calibrated hostname"


def test_live_gate_rejects_unknown_machine_target_key(monkeypatch, capsys, tmp_path: Path):
    config = deterministic_live_config()
    config["machine_target"]["model"] = "Mac14,7"
    returncode, payload = run_live_config(monkeypatch, capsys, tmp_path, config)
    assert returncode == 1
    assert payload["error"] == "machine target mismatch"


@pytest.mark.parametrize(
    ("missing_key", "error"),
    [
        ("machine_target", "machine target is missing"),
        ("latency_baseline_ms", "latency baseline is missing"),
    ],
)
def test_live_gate_rejects_missing_config_block(monkeypatch, capsys, tmp_path: Path, missing_key: str, error: str):
    config = deterministic_live_config()
    del config[missing_key]
    returncode, payload = run_live_config(monkeypatch, capsys, tmp_path, config)
    assert returncode == 1
    assert payload["error"] == error


def test_live_gate_does_not_require_baseline_without_latency(monkeypatch, capsys, tmp_path: Path):
    config = deterministic_live_config()
    config["checks"] = ["mcp_roundtrip"]
    del config["latency_baseline_ms"]
    returncode, payload = run_live_config(monkeypatch, capsys, tmp_path, config)
    assert returncode == 0
    assert payload["checks"][0]["status"] == "PASS"


def test_live_gate_runs_latency_on_calibrated_host(monkeypatch, capsys, tmp_path: Path):
    config = deterministic_live_config()
    hostname = config["latency_baseline_ms"]["hostname"]
    returncode, payload = run_live_config(monkeypatch, capsys, tmp_path, config, hostname)
    assert returncode == 0
    assert payload["skipped"] == []
    assert payload["checks"][0]["status"] == "PASS"


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
