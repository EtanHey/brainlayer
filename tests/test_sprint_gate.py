import json
import os
import subprocess
import sys
import time
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


def run_live_config(
    monkeypatch,
    capsys,
    tmp_path: Path,
    config: dict,
    hostname: str | None = None,
    extra_args: list[str] | None = None,
    provenance: dict | None = None,
) -> tuple[int, dict]:
    live_config = tmp_path / "live.json"
    live_config.write_text(json.dumps(config), encoding="utf-8")
    monkeypatch.setattr(sprint_gate, "CORPUS", live_config)
    target = config.get("machine_target", {})
    for key, attribute in (("os", "system"), ("architecture", "machine")):
        if key in target:
            monkeypatch.setattr(sprint_gate.platform, attribute, lambda key=key: target[key])
    if hostname:
        monkeypatch.setattr(sprint_gate.socket, "gethostname", lambda: hostname)
    monkeypatch.setattr(
        sprint_gate,
        "collect_live_provenance",
        lambda _config: (
            provenance
            or {
                "served_version": "1.5.10",
                "served_package_path": str(ROOT / "src" / "brainlayer" / "__init__.py"),
                "working_tree_version": "1.5.10",
                "working_tree_sha": "abc123",
                "working_tree_dirty": False,
                "served_build_sha": None,
                "provenance_mode": "dev-tree",
                "proof_refusals": [],
                "served_matches_working_tree": True,
                "db_path": "/tmp/test.db",
                "db_size_bytes": 1,
                "chunk_count": 796_098,
            }
        ),
    )
    return sprint_gate.main(["--json", *(extra_args or [])]), json.loads(capsys.readouterr().out)


CLEAN_TREE = {"working_tree_version": "1.5.10", "working_tree_sha": "abc123", "working_tree_dirty": False}
KEG_PATH = "/opt/homebrew/Cellar/brainlayer/1.5.10/libexec/venv/lib/python3.13/site-packages/brainlayer/__init__.py"
DEV_PATH = str(ROOT / "src" / "brainlayer" / "__init__.py")
VENV_PATH = str(ROOT / ".venv" / "lib" / "python3.13" / "site-packages" / "brainlayer" / "__init__.py")
FRESH_HELPER = 4102444800.0  # 2100-01-01: a helper that started after every source file was written
MTIME = 1_000_000.0  # the tree's newest source mtime as handed to eligibility()


def run_live_with_served(monkeypatch, capsys, tmp_path: Path, served: dict, tree: dict, extra_args=None):
    """Drive main() with only the I/O leaves faked; eligibility and payload assembly run for real."""
    live_config = tmp_path / "live.json"
    config = deterministic_live_config()
    live_config.write_text(json.dumps(config), encoding="utf-8")
    monkeypatch.setattr(sprint_gate, "CORPUS", live_config)
    monkeypatch.setattr(sprint_gate.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(sprint_gate.platform, "machine", lambda: "arm64")
    # Host-independent: main() SKIPs search_latency on any host but the corpus's calibrated one (Codex, #749).
    monkeypatch.setattr(sprint_gate.socket, "gethostname", lambda: config["latency_baseline_ms"]["hostname"])
    monkeypatch.setattr(sprint_gate, "resolve_served", lambda _config: (served, Path("/tmp/test.db")))
    monkeypatch.setattr(sprint_gate, "working_tree_provenance", lambda: dict(tree))
    monkeypatch.setattr(
        sprint_gate, "db_provenance", lambda path: {"db_path": str(path), "db_size_bytes": 1, "chunk_count": 7}
    )
    return sprint_gate.main(["--json", *(extra_args or [])]), json.loads(capsys.readouterr().out)


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


def test_validate_tools_reports_absent_optional_fields_without_failing_rung_zero():
    result = {
        "tools": [
            {
                "name": "brain_search",
                "description": "Search memory",
                "inputSchema": {"type": "object", "properties": {"query": {"type": "string"}}},
            }
        ]
    }

    intact, details = sprint_gate.validate_tools(result)

    assert intact is True
    assert details["truncation_order_valid"] is True
    assert details["missing_annotations"] == ["brain_search"]
    assert details["missing_input_schema_prose"] == ["brain_search"]


def test_validate_tools_rejects_description_truncation_before_first_rung():
    result = {
        "tools": [
            {
                "name": "brain_search",
                "description": "Search…[truncated]",
                "annotations": {},
                "inputSchema": {"properties": {"query": {"description": "Search query"}}},
            }
        ],
        "_meta": {"brainlayer/descriptionsTruncated": {"tools": ["brain_search"]}},
    }

    intact, details = sprint_gate.validate_tools(result)

    assert intact is False
    assert details["truncation_order_valid"] is False


def test_validate_tools_accepts_correctly_ordered_description_truncation():
    result = {
        "tools": [
            {
                "name": "brain_search",
                "description": "Search…[truncated]",
                "inputSchema": {"type": "object"},
            }
        ],
        "_meta": {"brainlayer/descriptionsTruncated": {"tools": ["brain_search"]}},
    }

    intact, details = sprint_gate.validate_tools(result)

    assert intact is True
    assert details["truncation_order_valid"] is True


def test_missing_wal_is_zero(tmp_path: Path):
    assert sprint_gate.wal_size(tmp_path / "gone") == 0


def test_search_visibility_rejects_zero_result_header_echo(monkeypatch):
    ticks = iter([0.0, 0.0, 1.0])
    monkeypatch.setattr(sprint_gate.time, "monotonic", lambda: next(ticks))
    monkeypatch.setattr(sprint_gate.time, "sleep", lambda _: None)
    response = '## Entity: related context\n\n## Search results for "MARKER" - 0 of 0 shown\n\nNo results found.'
    client = SimpleNamespace(call=lambda _name, _arguments: {"content": [{"type": "text", "text": response}]})
    assert sprint_gate.search_visible(client, "MARKER", 0.5) is False


@pytest.mark.parametrize("expand_error", [None, RuntimeError("{'code': -32601, 'message': 'Unknown tool'}")])
def test_deferred_roundtrip_uses_configured_wait_and_reports_observed(monkeypatch, expand_error):
    waits = []
    calls = []

    def call(name, *_):
        calls.append(name)
        if name == "expand_palette" and expand_error:
            raise expand_error
        return {"status": "DEFERRED"}

    client = SimpleNamespace(
        initialize=lambda: None,
        call=call,
        close=lambda: None,
    )
    client.request = lambda _method: {
        "tools": [
            {
                "name": "brain_search",
                "description": "Search memory",
                "annotations": {},
                "inputSchema": {"properties": {"query": {"description": "Search query"}}},
            }
        ]
    }
    monkeypatch.setattr(sprint_gate, "MCPClient", lambda _path, _timeout: client)
    monkeypatch.setattr(sprint_gate, "search_visible", lambda _client, _marker, timeout: waits.append(timeout) or True)
    ticks = iter([10.0, 12.5])
    monkeypatch.setattr(sprint_gate.time, "monotonic", lambda: next(ticks))

    config = {"socket_path": "unused", "mcp_timeout_seconds": 20, "thresholds": {"deferred_visibility_wait_seconds": 5}}
    result = sprint_gate.check_mcp(config)

    assert result["status"] == "PASS"
    assert calls == ["expand_palette", "brain_store"]
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
    ("machine_target", "error"),
    [
        ({}, "machine target is incomplete"),
        ({"os": "Darwin"}, "machine target is incomplete"),
        ("Darwin/arm64", "machine target is invalid"),
    ],
)
def test_live_gate_rejects_invalid_machine_target(
    monkeypatch, capsys, tmp_path: Path, machine_target: object, error: str
):
    config = deterministic_live_config()
    config["machine_target"] = machine_target
    returncode, payload = run_live_config(monkeypatch, capsys, tmp_path, config)
    assert returncode == 1
    assert payload["status"] == "FAIL"
    assert payload["checks"] == []
    assert payload["error"] == error


def test_live_gate_rejects_non_object_latency_baseline(monkeypatch, capsys, tmp_path: Path):
    config = deterministic_live_config()
    config["latency_baseline_ms"] = "not-an-object"
    returncode, payload = run_live_config(monkeypatch, capsys, tmp_path, config)
    assert returncode == 1
    assert payload["status"] == "FAIL"
    assert payload["checks"] == []
    assert payload["error"] == "latency baseline is invalid"


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


def test_mismatch_is_not_proof_without_requirement(monkeypatch, capsys, tmp_path: Path):
    config = deterministic_live_config()
    mismatch = {
        "served_version": "1.5.9",
        "served_package_path": "/opt/homebrew/Cellar/brainlayer/1.5.9/brainlayer/__init__.py",
        "working_tree_version": "1.5.10",
        "working_tree_sha": "abc123",
        "working_tree_dirty": False,
        "served_matches_working_tree": False,
        "db_path": "/tmp/test.db",
        "db_size_bytes": 1,
        "chunk_count": 796_098,
    }
    returncode, payload = run_live_config(monkeypatch, capsys, tmp_path, config, provenance=mismatch)

    assert returncode == 0
    assert payload["proof_eligible"] is False


def test_require_code_under_test_fails_before_checks_on_mismatch(monkeypatch, capsys, tmp_path: Path):
    config = deterministic_live_config()
    mismatch = {
        "served_version": "1.5.9",
        "served_package_path": "/opt/homebrew/Cellar/brainlayer/1.5.9/brainlayer/__init__.py",
        "working_tree_version": "1.5.10",
        "working_tree_sha": "abc123",
        "working_tree_dirty": False,
        "served_build_sha": None,
        "provenance_mode": "keg",
        "proof_refusals": ["version", "package_path_outside_tree", "served_build_sha_missing"],
        "served_matches_working_tree": False,
        "db_path": "/tmp/test.db",
        "db_size_bytes": 1,
        "chunk_count": 796_098,
    }
    returncode, payload = run_live_config(
        monkeypatch, capsys, tmp_path, config, extra_args=["--require-code-under-test"], provenance=mismatch
    )

    assert returncode == 1
    assert payload["status"] == "FAIL"
    assert payload["checks"] == []
    assert payload["provenance"] == mismatch
    assert payload["proof_eligible"] is False
    assert payload["error"] == (
        "not proof-eligible [keg]: version, package_path_outside_tree, served_build_sha_missing "
        "(served 1.5.9 build_sha=None from /opt/homebrew/Cellar/brainlayer/1.5.9/brainlayer/__init__.py; "
        "working tree 1.5.10 at abc123, dirty=False)"
    )


# --- Finding 6: pin the predicate itself, not the wire it hangs on -------------------------------


def test_eligibility_keg_without_build_sha_is_refused_even_when_versions_match():
    served = {"version": "1.5.10", "path": KEG_PATH, "build_sha": None, "helper_started_at": FRESH_HELPER}
    assert sprint_gate.eligibility(served, CLEAN_TREE, MTIME) == (
        "keg",
        ["package_path_outside_tree", "served_build_sha_missing"],
    )


def test_eligibility_keg_with_matching_build_sha_is_eligible():
    served = {"version": "1.5.10", "path": KEG_PATH, "build_sha": "abc123", "helper_started_at": FRESH_HELPER}
    assert sprint_gate.eligibility(served, CLEAN_TREE, MTIME) == ("keg", [])


def test_eligibility_keg_with_foreign_build_sha_names_the_mismatch():
    served = {"version": "1.5.10", "path": KEG_PATH, "build_sha": "def456", "helper_started_at": FRESH_HELPER}
    assert sprint_gate.eligibility(served, CLEAN_TREE, MTIME) == (
        "keg",
        ["package_path_outside_tree", "build_sha_mismatch"],
    )


def test_eligibility_dev_tree_clean_is_eligible_and_dirty_alone_blocks():
    served = {"version": "1.5.10", "path": DEV_PATH, "build_sha": None, "helper_started_at": FRESH_HELPER}
    assert sprint_gate.eligibility(served, CLEAN_TREE, MTIME) == ("dev-tree", [])
    assert sprint_gate.eligibility(served, {**CLEAN_TREE, "working_tree_dirty": True}, MTIME) == (
        "dev-tree",
        ["working_tree_dirty"],
    )


def test_eligibility_site_packages_under_root_is_keg_not_dev_tree():
    """Macroscope #749: ROOT/.venv/.../site-packages is under ROOT but is NOT the source tree."""
    served = {"version": "1.5.10", "path": VENV_PATH, "build_sha": None, "helper_started_at": FRESH_HELPER}
    assert sprint_gate.eligibility(served, CLEAN_TREE, MTIME) == (
        "keg",
        ["package_path_outside_tree", "served_build_sha_missing"],
    )


def test_eligibility_dev_tree_refuses_helper_older_than_tree():
    """Codex #749: a helper that loaded commit A and survived a checkout to B still serves A."""
    stale = {"version": "1.5.10", "path": DEV_PATH, "build_sha": None, "helper_started_at": MTIME - 1}
    assert sprint_gate.eligibility(stale, CLEAN_TREE, MTIME) == ("dev-tree", ["helper_older_than_tree"])
    same_second = {**stale, "helper_started_at": MTIME}
    assert sprint_gate.eligibility(same_second, CLEAN_TREE, MTIME) == ("dev-tree", ["helper_older_than_tree"])
    fresh = {**stale, "helper_started_at": MTIME + 1}
    assert sprint_gate.eligibility(fresh, CLEAN_TREE, MTIME) == ("dev-tree", [])
    # Lead ruling (#749 round 1): keg mode keeps the sha predicate only; process age is dev-tree's.
    keg = {"version": "1.5.10", "path": KEG_PATH, "build_sha": "abc123", "helper_started_at": MTIME - 1}
    assert sprint_gate.eligibility(keg, CLEAN_TREE, MTIME) == ("keg", [])


def test_newest_source_mtime_counts_package_data_but_not_bytecode(monkeypatch, tmp_path):
    package = tmp_path / "brainlayer"
    (package / "__pycache__").mkdir(parents=True)
    (package / "core.py").write_text("x = 1")
    os.utime(package / "core.py", (100, 100))
    (package / "taxonomy.json").write_text("{}")
    os.utime(package / "taxonomy.json", (200, 200))
    (package / "__pycache__" / "core.cpython-313.pyc").write_bytes(b"")
    os.utime(package / "__pycache__" / "core.cpython-313.pyc", (300, 300))
    monkeypatch.setattr(sprint_gate, "PACKAGE", package)

    assert sprint_gate.newest_source_mtime() == 200.0


def test_helper_started_at_parses_ps_lstart(monkeypatch):
    seen = {}

    def run(argv, **kwargs):
        seen["argv"], seen["env"] = argv, kwargs.get("env")
        return subprocess.CompletedProcess(argv, 0, stdout="Wed Sep  2 14:46:57 2026\n", stderr="")

    monkeypatch.setattr(sprint_gate.subprocess, "run", run)
    started = sprint_gate.helper_started_at(4242)
    assert seen["argv"] == ["ps", "-o", "lstart=", "-p", "4242"]
    assert seen["env"]["LC_ALL"] == "C" and seen["env"]["TZ"] == "UTC"
    assert started == 1788360417.0, "parsed as UTC, whatever the parent's TZ"

    monkeypatch.setattr(
        sprint_gate.subprocess, "run", lambda argv, **_: subprocess.CompletedProcess(argv, 0, stdout="\n", stderr="")
    )
    with pytest.raises(RuntimeError, match="no start time"):
        sprint_gate.helper_started_at(4242)


def test_helper_started_at_parses_real_ps_output_independent_of_parent_tz(monkeypatch):
    """Mock-green is not live-green: real `ps`, and the answer must not move with the parent's TZ."""
    started = sprint_gate.helper_started_at(os.getpid())
    assert 0 <= time.time() - started < 3600
    monkeypatch.setenv("TZ", "Pacific/Kiritimati")  # UTC+14: the fail-open direction
    time.tzset()
    try:
        assert sprint_gate.helper_started_at(os.getpid()) == started
    finally:
        monkeypatch.undo()
        time.tzset()


def test_stale_helper_blocks_dev_tree_proof_and_payload_records_both_timestamps(monkeypatch, capsys, tmp_path):
    monkeypatch.setattr(sprint_gate, "newest_source_mtime", lambda: 1_000_000.0)
    served = {"version": "1.5.10", "path": DEV_PATH, "build_sha": None, "helper_started_at": 999_999.0}
    returncode, payload = run_live_with_served(
        monkeypatch, capsys, tmp_path, served, CLEAN_TREE, ["--require-code-under-test"]
    )

    assert returncode == 1
    assert payload["checks"] == []
    assert payload["provenance"]["provenance_mode"] == "dev-tree"
    assert payload["provenance"]["proof_refusals"] == ["helper_older_than_tree"]
    assert payload["provenance"]["helper_started_at"] == "1970-01-12T13:46:39+00:00"
    assert payload["provenance"]["source_newest_mtime"] == "1970-01-12T13:46:40+00:00"
    assert payload["error"].startswith("not proof-eligible [dev-tree]: helper_older_than_tree")


def test_warmup_query_is_a_nonce_never_a_corpus_query(monkeypatch):
    """Codex #749: hybrid_search caches identical requests for 60 s; warming with queries[0] would
    hand check_search a cache hit as its first timed sample."""
    calls = []
    client = SimpleNamespace(
        initialize=lambda: None, call=lambda name, arguments: calls.append((name, arguments)), close=lambda: None
    )
    monkeypatch.setattr(sprint_gate, "MCPClient", lambda _path, _timeout: client)
    config = json.loads(CORPUS.read_text(encoding="utf-8"))

    sprint_gate.warm_helper(config)
    sprint_gate.warm_helper(config)

    assert [name for name, _ in calls] == ["brain_search", "brain_search"]
    first, second = (arguments["query"] for _, arguments in calls)
    assert first.startswith("warmup-") and second.startswith("warmup-")
    assert first != second
    assert first not in config["queries"] and second not in config["queries"]
    assert all(arguments["num_results"] == 1 for _, arguments in calls)


def test_version_equality_alone_is_not_proof_and_message_names_the_path_predicate(monkeypatch, capsys, tmp_path):
    served = {"version": "1.5.10", "path": KEG_PATH, "build_sha": None, "helper_started_at": FRESH_HELPER}
    returncode, payload = run_live_with_served(
        monkeypatch, capsys, tmp_path, served, CLEAN_TREE, ["--require-code-under-test"]
    )

    assert returncode == 1
    assert payload["checks"] == []
    assert payload["proof_eligible"] is False
    assert payload["provenance"]["provenance_mode"] == "keg"
    assert payload["provenance"]["served_matches_working_tree"] is False
    assert payload["provenance"]["proof_refusals"] == ["package_path_outside_tree", "served_build_sha_missing"]
    assert "package_path_outside_tree" in payload["error"]
    assert "served_build_sha_missing" in payload["error"]
    assert "1.5.10 does not match" not in payload["error"]


def test_dirty_tree_alone_blocks_dev_tree_proof(monkeypatch, capsys, tmp_path):
    served = {"version": "1.5.10", "path": DEV_PATH, "build_sha": None, "helper_started_at": FRESH_HELPER}
    dirty = {**CLEAN_TREE, "working_tree_dirty": True}
    returncode, payload = run_live_with_served(
        monkeypatch, capsys, tmp_path, served, dirty, ["--require-code-under-test"]
    )

    assert returncode == 1
    assert payload["provenance"]["provenance_mode"] == "dev-tree"
    assert payload["provenance"]["proof_refusals"] == ["working_tree_dirty"]
    assert payload["error"].startswith("not proof-eligible [dev-tree]: working_tree_dirty")


def test_keg_built_from_this_sha_is_proof_eligible(monkeypatch, capsys, tmp_path):
    served = {"version": "1.5.10", "path": KEG_PATH, "build_sha": "abc123", "helper_started_at": FRESH_HELPER}
    returncode, payload = run_live_with_served(
        monkeypatch, capsys, tmp_path, served, CLEAN_TREE, ["--require-code-under-test"]
    )

    assert returncode == 0
    assert payload["proof_eligible"] is True
    assert payload["provenance"]["provenance_mode"] == "keg"
    assert payload["provenance"]["served_build_sha"] == "abc123"
    assert [check["status"] for check in payload["checks"]] == ["PASS"] * 4


def fake_ps(helper_count: int, calls: list[str] | None = None):
    helper = "  4242 /venv/bin/python -m brainlayer.brainbar_hybrid_helper --db-path /tmp/x.db --socket /tmp/h.sock"
    lines = ["     1 /sbin/launchd", *([helper] * helper_count)]

    def run(argv, **_kwargs):
        assert argv[0] == "ps", f"collector reached {argv[0]} with {helper_count} helpers"
        if calls is not None:
            calls.append("ps")
        return subprocess.CompletedProcess(argv, 0, stdout="\n".join(lines) + "\n", stderr="")

    return run


@pytest.mark.parametrize("helper_count", [0, 2])
def test_collector_failure_emits_structured_payload_not_traceback(monkeypatch, capsys, tmp_path, helper_count):
    live_config = tmp_path / "live.json"
    live_config.write_text(json.dumps(deterministic_live_config()), encoding="utf-8")
    monkeypatch.setattr(sprint_gate, "CORPUS", live_config)
    monkeypatch.setattr(sprint_gate.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(sprint_gate.platform, "machine", lambda: "arm64")
    calls: list[str] = []
    monkeypatch.setattr(sprint_gate, "warm_helper", lambda _config: calls.append("warm"))
    monkeypatch.setattr(sprint_gate, "working_tree_provenance", lambda: dict(CLEAN_TREE))
    monkeypatch.setattr(sprint_gate.subprocess, "run", fake_ps(helper_count, calls))

    returncode, payload = sprint_gate.main(["--json", "--require-code-under-test"]), json.loads(capsys.readouterr().out)

    assert calls == ["warm", "ps"], "helper must be warmed over the socket before ps resolves it"
    assert returncode == 1
    assert payload["checks"] == []
    assert payload["proof_eligible"] is False
    provenance = payload["provenance"]
    assert provenance["served_version"] is None
    assert provenance["served_package_path"] is None
    assert provenance["served_matches_working_tree"] is False
    assert provenance["provenance_mode"] is None
    assert provenance["proof_refusals"] == ["provenance_unresolved"]
    assert provenance["working_tree_sha"] == "abc123"
    assert provenance["provenance_error"] == f"RuntimeError: expected one serving hybrid helper, found {helper_count}"
    assert payload["error"] == f"served code could not be resolved: {provenance['provenance_error']}"


def test_collector_failure_without_requirement_keeps_rc_and_records_error(monkeypatch, capsys, tmp_path):
    live_config = tmp_path / "live.json"
    live_config.write_text(json.dumps(deterministic_live_config()), encoding="utf-8")
    monkeypatch.setattr(sprint_gate, "CORPUS", live_config)
    monkeypatch.setattr(sprint_gate.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(sprint_gate.platform, "machine", lambda: "arm64")
    monkeypatch.setattr(sprint_gate, "warm_helper", lambda _config: None)
    monkeypatch.setattr(sprint_gate, "working_tree_provenance", lambda: dict(CLEAN_TREE))
    monkeypatch.setattr(sprint_gate.subprocess, "run", fake_ps(0))

    returncode, payload = sprint_gate.main(["--json"]), json.loads(capsys.readouterr().out)

    assert returncode == 0
    assert payload["status"] == "PASS"
    assert payload["proof_eligible"] is False
    assert payload["provenance"]["provenance_error"] == "RuntimeError: expected one serving hybrid helper, found 0"


def test_replay_under_requirement_is_rejected_and_labelled_as_replay():
    fixture = FIXTURES / "all_green.json"
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--json", "--fixture", str(fixture), "--require-code-under-test"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    payload = json.loads(result.stdout)

    assert result.returncode == 1
    assert "Traceback" not in result.stderr
    assert payload["mode"] == "replay"
    assert payload["fixture"].endswith("all_green.json")
    assert payload["checks"] == []
    assert payload["proof_eligible"] is False
    assert payload["provenance"]["provenance_mode"] == "replay"
    assert payload["error"] == "replay is never proof-eligible: a fixture is not evidence about any served code"


def test_replay_never_needs_git(tmp_path: Path):
    """Macroscope #749: replay built its provenance via `git` before fail() could refuse; with no
    `git` on PATH it tracebacked instead of replaying. A fixture is not evidence, so replay
    carries placeholder tree metadata and never shells out."""
    empty_path = tmp_path / "nobin"
    empty_path.mkdir()
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--json", "--fixture", str(FIXTURES / "all_green.json")],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
        env={**os.environ, "PATH": str(empty_path)},
    )

    assert result.returncode == 0, result.stderr
    assert "Traceback" not in result.stderr
    payload = json.loads(result.stdout)
    assert payload["status"] == "PASS"
    assert payload["provenance"]["provenance_mode"] == "replay"
    assert payload["provenance"]["working_tree_sha"] is None
    assert payload["provenance"]["working_tree_dirty"] is None
    assert payload["provenance"]["working_tree_version"] == sprint_gate.__version__


def test_live_git_failure_is_a_structured_refusal_not_a_traceback(monkeypatch, capsys, tmp_path):
    live_config = tmp_path / "live.json"
    live_config.write_text(json.dumps(deterministic_live_config()), encoding="utf-8")
    monkeypatch.setattr(sprint_gate, "CORPUS", live_config)
    monkeypatch.setattr(sprint_gate.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(sprint_gate.platform, "machine", lambda: "arm64")
    served = {"version": "1.5.10", "path": DEV_PATH, "build_sha": None, "helper_started_at": FRESH_HELPER}
    monkeypatch.setattr(sprint_gate, "resolve_served", lambda _config: (served, Path("/tmp/test.db")))

    def no_git():
        raise FileNotFoundError("git")

    monkeypatch.setattr(sprint_gate, "working_tree_provenance", no_git)

    returncode = sprint_gate.main(["--json", "--require-code-under-test"])
    payload = json.loads(capsys.readouterr().out)

    assert returncode == 1
    assert payload["checks"] == []
    assert payload["provenance"]["proof_refusals"] == ["provenance_unresolved"]
    assert payload["provenance"]["working_tree_sha"] is None
    assert payload["provenance"]["provenance_error"] == "FileNotFoundError: git"
    assert payload["error"] == "served code could not be resolved: FileNotFoundError: git"


def test_package_exposes_build_sha_slot_for_keg_mode():
    import brainlayer

    assert hasattr(brainlayer, "__build_sha__")
    assert brainlayer.__build_sha__ is None or isinstance(brainlayer.__build_sha__, str)


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
