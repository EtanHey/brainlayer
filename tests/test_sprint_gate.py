import json
import os
import pwd
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
    default = fake_provenance(
        served_version="1.5.10",
        served_package_path=str(ROOT / "src" / "brainlayer" / "__init__.py"),
        working_tree_version="1.5.10",
        working_tree_sha="abc123",
        working_tree_dirty=False,
        provenance_mode="dev-tree",
        proof_refusals=[],
        served_matches_working_tree=True,
        db_path="/tmp/test.db",
        db_size_bytes=1,
        chunk_count=796_098,
        provenance_error=None,
    )
    monkeypatch.setattr(sprint_gate, "collect_live_provenance", lambda _config: provenance or default)
    return sprint_gate.main(["--json", *(extra_args or [])]), json.loads(capsys.readouterr().out)


CLEAN_TREE = {"working_tree_version": "1.5.10", "working_tree_sha": "abc123", "working_tree_dirty": False}
KEG_PATH = "/opt/homebrew/Cellar/brainlayer/1.5.10/libexec/venv/lib/python3.13/site-packages/brainlayer/__init__.py"
DEV_PATH = str(ROOT / "src" / "brainlayer" / "__init__.py")
VENV_PATH = str(ROOT / ".venv" / "lib" / "python3.13" / "site-packages" / "brainlayer" / "__init__.py")
FRESH_HELPER = 4102444800.0  # 2100-01-01: a helper that started after every source file was written
MTIME = 1_000_000.0  # the tree's newest source mtime as handed to eligibility()


def fake_served(path: str, **overrides) -> dict:
    """The full `served` shape resolve_served() produces, so a key the code reads cannot be missing."""
    served = {
        "version": "1.5.10",
        "path": path,
        "build_sha": None,
        "helper_started_at": FRESH_HELPER,
        "pythonpath": None,  # today's case: BrainBar sets neither PYTHONPATH nor BRAINLAYER_REPO_ROOT
        "repo_root": None,
        "package_newest_mtime": MTIME,
    }
    assert set(overrides) <= set(served), overrides
    return {**served, **overrides}


def fake_provenance(**overrides) -> dict:
    """A payload-shaped provenance dict derived from the real shape (CodeRabbit, #749)."""
    base = sprint_gate.unresolved_provenance("")
    assert set(overrides) <= set(base), overrides
    return {**base, **overrides}


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
    # Ratchet (c): the flat 10% is gone. Latency limits are measured bands from attested green main runs.
    assert "latency_regression_fraction" not in corpus["thresholds"]
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


def test_truncation_notice_that_names_nobody_and_explains_nothing_is_not_intact():
    """A notice must ACCOUNT for itself: name what it shortened, or say why it shortened nothing.

    Two empty sets used to satisfy `notice_names == set(truncated)` by both being empty, so a
    notice could assert that descriptions were shortened and be believed without naming one.
    """
    result = {
        "tools": [
            {
                "name": "brain_search",
                "description": "Search memory",
                "inputSchema": {"type": "object"},
            }
        ],
        "_meta": {"brainlayer/descriptionsTruncated": {"tools": []}},
    }

    intact, details = sprint_gate.validate_tools(result)

    assert intact is False
    assert details["truncation_notice_unexplained"] is True
    assert details["truncation_notice_names"] == []


def test_the_over_limit_notice_with_no_shortened_description_is_accepted():
    """BrainBar's documented over-limit shape is a LIVE response, not a finding.

    `BrainBarServer.responseTruncatingDescriptions(forceNotice: true)` emits the notice with
    `tools: []` and a `reason` saying every description is already at the floor, and ships the
    response over-limit with its contract intact. Rejecting an empty `tools` list outright turned
    that into a false RED against a legitimate response (Macroscope, #755).
    """
    result = {
        "tools": [
            {
                "name": "brain_search",
                "description": "Search memory",
                "inputSchema": {"type": "object"},
            }
        ],
        "_meta": {
            "brainlayer/descriptionsTruncated": {
                "tools": [],
                "reason": "raw newline tools/list exceeds the client's chunk limit even though every "
                "description is already at or below the floor",
                "fullDescriptionsAvailableOver": "Content-Length framing",
            }
        },
    }

    intact, details = sprint_gate.validate_tools(result)

    assert intact is True
    assert details["truncation_notice_unexplained"] is False
    assert details["truncation_notice_names"] == []
    assert "already at or below the floor" in details["truncation_notice_reason"]


def test_truncation_notice_naming_its_tools_is_reported_verbatim():
    result = {
        "tools": [
            {
                "name": "brain_search",
                "description": "Search\u2026[truncated]",
                "inputSchema": {"type": "object"},
            }
        ],
        "_meta": {"brainlayer/descriptionsTruncated": {"tools": ["brain_search"]}},
    }

    intact, details = sprint_gate.validate_tools(result)

    assert intact is True
    assert details["truncation_notice_names"] == ["brain_search"]
    assert details["truncation_notice_unexplained"] is False


def _roundtrip_client(store_result: dict, archive_error: Exception | None = None) -> tuple[SimpleNamespace, list]:
    calls: list[tuple[str, dict]] = []

    def call(name, arguments=None):
        calls.append((name, arguments or {}))
        if name == "brain_archive" and archive_error is not None:
            raise archive_error
        if name == "brain_store":
            return store_result
        return {}

    client = SimpleNamespace(initialize=lambda: None, call=call, close=lambda: None)
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
    return client, calls


ROUNDTRIP_CONFIG = {
    "socket_path": "unused",
    "mcp_timeout_seconds": 20,
    "thresholds": {"deferred_visibility_wait_seconds": 5},
}


def test_roundtrip_archives_the_probe_chunk_it_planted(monkeypatch):
    """The gate must not grow the corpus it measures by one chunk per run."""
    client, calls = _roundtrip_client({"status": "STORED", "chunk_id": "chunk-42"})
    monkeypatch.setattr(sprint_gate, "MCPClient", lambda _path, _timeout: client)
    monkeypatch.setattr(sprint_gate, "search_visible", lambda *_args, **_kwargs: True)
    monkeypatch.setattr(sprint_gate.time, "monotonic", lambda: 0.0)

    result = sprint_gate.check_mcp(ROUNDTRIP_CONFIG)

    assert result["status"] == "PASS"
    assert ("brain_archive", {"chunk_id": "chunk-42", "reason": "sprint-gate probe"}) in calls
    assert result["details"]["probe_retired"] is True
    assert result["details"]["probe_chunk_id"] == "chunk-42"


def test_roundtrip_fails_when_it_cannot_retire_its_own_probe_chunk(monkeypatch):
    """A gate that cannot undo its own write does not get to call the run green."""
    client, _calls = _roundtrip_client(
        {"status": "STORED", "chunk_id": "chunk-42"}, archive_error=RuntimeError("archive refused")
    )
    monkeypatch.setattr(sprint_gate, "MCPClient", lambda _path, _timeout: client)
    monkeypatch.setattr(sprint_gate, "search_visible", lambda *_args, **_kwargs: True)
    monkeypatch.setattr(sprint_gate.time, "monotonic", lambda: 0.0)

    result = sprint_gate.check_mcp(ROUNDTRIP_CONFIG)

    assert result["status"] == "FAIL"
    assert result["details"]["probe_retired"] is False
    assert "archive refused" in result["details"]["probe_retire_error"]


@pytest.mark.parametrize("dedupe_status", ["DUPLICATE", "MERGED"])
def test_roundtrip_never_archives_a_chunk_the_gate_did_not_create(monkeypatch, dedupe_status):
    """A deduped probe means the returned chunk is SOMEBODY ELSE'S MEMORY. Never archive it.

    `brain_store` answers DUPLICATE/MERGED with the id of PRE-EXISTING content. Archiving that
    unconditionally would make the gate silently delete a real user memory on every run where
    BrainLayer deduped the probe -- "never silently degrade, never auto-delete personal data",
    broken by the cleanup routine itself.
    """
    existing = "chunk-somebody-elses-memory"
    client, calls = _roundtrip_client({"status": dedupe_status, "stored_new": False, "chunk_id": existing})
    monkeypatch.setattr(sprint_gate, "MCPClient", lambda _path, _timeout: client)
    monkeypatch.setattr(sprint_gate, "search_visible", lambda *_args, **_kwargs: True)
    monkeypatch.setattr(sprint_gate.time, "monotonic", lambda: 0.0)

    result = sprint_gate.check_mcp(ROUNDTRIP_CONFIG)

    # The assertion that proves the blocker is closed: brain_archive is never reached at all, and
    # in particular never with the pre-existing chunk id.
    assert [name for name, _ in calls] == ["expand_palette", "brain_store"]
    assert all(arguments.get("chunk_id") != existing for _name, arguments in calls)
    # The gate created nothing, so there is nothing outstanding -- and it still passes.
    assert result["status"] == "PASS"
    assert result["details"]["probe_retired"] is True
    assert result["details"]["probe_chunk_id"] is None
    assert result["details"]["probe_reused_existing_chunk"] == existing


def test_a_stored_probe_is_still_archived_when_stored_new_is_reported(monkeypatch):
    """The other direction: a chunk the gate DID create is retired exactly as before."""
    client, calls = _roundtrip_client({"status": "STORED", "stored_new": True, "chunk_id": "chunk-42"})
    monkeypatch.setattr(sprint_gate, "MCPClient", lambda _path, _timeout: client)
    monkeypatch.setattr(sprint_gate, "search_visible", lambda *_args, **_kwargs: True)
    monkeypatch.setattr(sprint_gate.time, "monotonic", lambda: 0.0)

    result = sprint_gate.check_mcp(ROUNDTRIP_CONFIG)

    assert ("brain_archive", {"chunk_id": "chunk-42", "reason": "sprint-gate probe"}) in calls
    assert result["status"] == "PASS"
    assert result["details"]["probe_chunk_id"] == "chunk-42"


def test_probe_ownership_reads_stored_new_before_falling_back_to_status():
    """`stored_new` is authoritative where BrainBar sends it; status is the queued-path fallback."""
    assert sprint_gate.probe_is_gate_created({"status": "STORED", "stored_new": True}) is True
    assert sprint_gate.probe_is_gate_created({"status": "STORED", "stored_new": False}) is False
    assert sprint_gate.probe_is_gate_created({"status": "DUPLICATE", "stored_new": False}) is False
    assert sprint_gate.probe_is_gate_created({"status": "MERGED", "stored_new": False}) is False
    # The queued path (`queuedBrainStoreOutput`) sends no `stored_new`; a DEFERRED write is ours.
    assert sprint_gate.probe_is_gate_created({"status": "DEFERRED"}) is True
    assert sprint_gate.probe_is_gate_created({"status": "STORED"}) is True
    assert sprint_gate.probe_is_gate_created({"status": "DUPLICATE"}) is False


def test_roundtrip_retires_its_probe_chunk_even_when_the_search_raises(monkeypatch):
    """A gate run that BLOWS UP still cleans up. Otherwise every failing run grows the corpus."""
    client, calls = _roundtrip_client({"status": "STORED", "chunk_id": "chunk-42"})
    monkeypatch.setattr(sprint_gate, "MCPClient", lambda _path, _timeout: client)

    def explode(*_args, **_kwargs):
        raise RuntimeError("socket died mid-search")

    monkeypatch.setattr(sprint_gate, "search_visible", explode)
    monkeypatch.setattr(sprint_gate.time, "monotonic", lambda: 0.0)

    with pytest.raises(RuntimeError, match="socket died mid-search"):
        sprint_gate.check_mcp(ROUNDTRIP_CONFIG)

    assert ("brain_archive", {"chunk_id": "chunk-42", "reason": "sprint-gate probe"}) in calls


def test_roundtrip_reports_a_store_that_returned_no_chunk_id(monkeypatch):
    client, calls = _roundtrip_client({"status": "STORED"})
    monkeypatch.setattr(sprint_gate, "MCPClient", lambda _path, _timeout: client)
    monkeypatch.setattr(sprint_gate, "search_visible", lambda *_args, **_kwargs: True)
    monkeypatch.setattr(sprint_gate.time, "monotonic", lambda: 0.0)

    result = sprint_gate.check_mcp(ROUNDTRIP_CONFIG)

    assert result["status"] == "FAIL"
    assert result["details"]["probe_retire_error"] == (
        "the store returned no chunk_id, so the probe chunk cannot be retired"
    )
    assert [name for name, _ in calls] == ["expand_palette", "brain_store"]


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
        # BrainBar's queued store returns a chunk_id (`queuedBrainStoreOutput`), so the fake does
        # too -- otherwise this test would be measuring a response shape the daemon never sends.
        return {"status": "DEFERRED", "chunk_id": "chunk-deferred"}

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
    assert calls == ["expand_palette", "brain_store", "brain_archive"]
    assert waits == [5]
    assert result["details"]["planted_hit_wait_seconds"] == 2.5
    assert result["details"]["probe_chunk_id"] == "chunk-deferred"


def test_empty_check_list_is_unmeasured_never_pass(monkeypatch, capsys, tmp_path: Path):
    """`checks: []` measured nothing, so it cannot report PASS.

    `all([])` is True, so an empty selection rendered a green payload with rc 0 -- the same
    vacuous-PASS shape #752 legislated out of the ratchet table, reached here through an empty
    selection instead of an empty measurement.
    """
    config = deterministic_live_config()
    config["checks"] = []

    rc, payload = run_live_config(monkeypatch, capsys, tmp_path, config)

    assert rc == 1
    assert payload["status"] == sprint_gate.UNMEASURED
    assert payload["checks"] == []
    assert "measured nothing" in payload["error"]


def test_a_selection_that_runs_one_check_still_passes(monkeypatch, capsys, tmp_path: Path):
    """The counterpart: UNMEASURED is about measuring nothing, not about measuring little."""
    config = deterministic_live_config()
    config["checks"] = ["wal_bound"]

    rc, payload = run_live_config(monkeypatch, capsys, tmp_path, config)

    assert rc == 0
    assert payload["status"] == "PASS"
    assert [check["name"] for check in payload["checks"]] == ["wal_bound"]


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
    # It RAN (not skipped) and reported its samples; with no attested main runs handed over there
    # is no band to judge them against, so the verdict is UNMEASURED, not a PASS nobody measured.
    assert payload["checks"][0]["status"] == "UNMEASURED"
    assert payload["checks"][0]["details"]["measured_ms"] == {"p50": 100, "p95": 100}


def test_mismatch_is_not_proof_without_requirement(monkeypatch, capsys, tmp_path: Path):
    config = deterministic_live_config()
    mismatch = fake_provenance(
        served_version="1.5.9",
        served_package_path="/opt/homebrew/Cellar/brainlayer/1.5.9/brainlayer/__init__.py",
        working_tree_version="1.5.10",
        working_tree_sha="abc123",
        working_tree_dirty=False,
        provenance_mode="keg",
        proof_refusals=["version", "package_path_outside_tree", "served_build_sha_missing"],
        served_matches_working_tree=False,
        db_path="/tmp/test.db",
        db_size_bytes=1,
        chunk_count=796_098,
        provenance_error=None,
    )
    returncode, payload = run_live_config(monkeypatch, capsys, tmp_path, config, provenance=mismatch)

    assert returncode == 0
    assert payload["proof_eligible"] is False


def test_require_code_under_test_fails_before_checks_on_mismatch(monkeypatch, capsys, tmp_path: Path):
    config = deterministic_live_config()
    mismatch = fake_provenance(
        served_version="1.5.9",
        served_package_path="/opt/homebrew/Cellar/brainlayer/1.5.9/brainlayer/__init__.py",
        working_tree_version="1.5.10",
        working_tree_sha="abc123",
        working_tree_dirty=False,
        provenance_mode="keg",
        proof_refusals=["version", "package_path_outside_tree", "served_build_sha_missing"],
        served_matches_working_tree=False,
        db_path="/tmp/test.db",
        db_size_bytes=1,
        chunk_count=796_098,
        provenance_error=None,
    )
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
    served = fake_served(KEG_PATH)
    assert sprint_gate.eligibility(served, CLEAN_TREE, MTIME) == (
        "keg",
        ["package_path_outside_tree", "served_build_sha_missing"],
    )


def test_eligibility_keg_with_matching_build_sha_is_eligible():
    served = fake_served(KEG_PATH, build_sha="abc123")
    assert sprint_gate.eligibility(served, CLEAN_TREE, MTIME) == ("keg", [])


def test_eligibility_keg_with_foreign_build_sha_names_the_mismatch():
    served = fake_served(KEG_PATH, build_sha="def456")
    assert sprint_gate.eligibility(served, CLEAN_TREE, MTIME) == (
        "keg",
        ["package_path_outside_tree", "build_sha_mismatch"],
    )


def test_eligibility_dev_tree_clean_is_eligible_and_dirty_alone_blocks():
    served = fake_served(DEV_PATH)
    assert sprint_gate.eligibility(served, CLEAN_TREE, MTIME) == ("dev-tree", [])
    assert sprint_gate.eligibility(served, {**CLEAN_TREE, "working_tree_dirty": True}, MTIME) == (
        "dev-tree",
        ["working_tree_dirty"],
    )


def test_eligibility_site_packages_under_root_is_keg_not_dev_tree():
    """Macroscope #749: ROOT/.venv/.../site-packages is under ROOT but is NOT the source tree."""
    served = fake_served(VENV_PATH)
    assert sprint_gate.eligibility(served, CLEAN_TREE, MTIME) == (
        "keg",
        ["package_path_outside_tree", "served_build_sha_missing"],
    )


def test_eligibility_dev_tree_refuses_helper_older_than_tree():
    """Codex #749: a helper that loaded commit A and survived a checkout to B still serves A."""
    stale = fake_served(DEV_PATH, helper_started_at=MTIME - 1)
    assert sprint_gate.eligibility(stale, CLEAN_TREE, MTIME) == ("dev-tree", ["helper_older_than_tree"])
    same_second = {**stale, "helper_started_at": MTIME}
    assert sprint_gate.eligibility(same_second, CLEAN_TREE, MTIME) == ("dev-tree", ["helper_older_than_tree"])
    fresh = {**stale, "helper_started_at": MTIME + 1}
    assert sprint_gate.eligibility(fresh, CLEAN_TREE, MTIME) == ("dev-tree", [])


def test_eligibility_keg_refuses_helper_older_than_keg():
    """CodeRabbit #749: the sha proves the keg on disk; a helper started before the keg was replaced
    still serves the old build. The keg's own files are the bound, not the source tree."""
    fresh_keg = fake_served(KEG_PATH, build_sha="abc123", helper_started_at=MTIME + 1, package_newest_mtime=MTIME)
    assert sprint_gate.eligibility(fresh_keg, CLEAN_TREE, MTIME + 500) == ("keg", []), "tree mtime is irrelevant"
    stale = {**fresh_keg, "helper_started_at": MTIME}
    assert sprint_gate.eligibility(stale, CLEAN_TREE, MTIME) == ("keg", ["helper_older_than_keg"])
    stale_and_unstamped = {**stale, "build_sha": None}
    assert sprint_gate.eligibility(stale_and_unstamped, CLEAN_TREE, MTIME) == (
        "keg",
        ["package_path_outside_tree", "served_build_sha_missing", "helper_older_than_keg"],
    )


def test_newest_mtime_counts_package_data_and_extensions_but_not_bytecode(monkeypatch, tmp_path):
    package = tmp_path / "brainlayer"
    (package / "__pycache__").mkdir(parents=True)
    (package / "core.py").write_text("x = 1")
    os.utime(package / "core.py", (100, 100))
    (package / "taxonomy.json").write_text("{}")
    os.utime(package / "taxonomy.json", (200, 200))
    (package / "_native.cpython-313-darwin.so").write_bytes(b"")
    os.utime(package / "_native.cpython-313-darwin.so", (250, 250))
    (package / "__pycache__" / "core.cpython-313.pyc").write_bytes(b"")
    os.utime(package / "__pycache__" / "core.cpython-313.pyc", (300, 300))
    monkeypatch.setattr(sprint_gate, "PACKAGE", package)

    assert sprint_gate.newest_mtime(package) == 250.0
    assert sprint_gate.newest_source_mtime() == 250.0


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
    served = fake_served(DEV_PATH, helper_started_at=999_999.0)
    returncode, payload = run_live_with_served(
        monkeypatch, capsys, tmp_path, served, CLEAN_TREE, ["--require-code-under-test"]
    )

    assert returncode == 1
    assert payload["checks"] == []
    assert payload["provenance"]["provenance_mode"] == "dev-tree"
    assert payload["provenance"]["proof_refusals"] == ["helper_older_than_tree"]
    assert payload["provenance"]["helper_started_at"] == "1970-01-12T13:46:39+00:00"
    assert payload["provenance"]["source_newest_mtime"] == "1970-01-12T13:46:40+00:00"
    assert payload["provenance"]["served_package_newest_mtime"] == "1970-01-12T13:46:40+00:00"
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
    served = fake_served(KEG_PATH)
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
    served = fake_served(DEV_PATH)
    dirty = {**CLEAN_TREE, "working_tree_dirty": True}
    returncode, payload = run_live_with_served(
        monkeypatch, capsys, tmp_path, served, dirty, ["--require-code-under-test"]
    )

    assert returncode == 1
    assert payload["provenance"]["provenance_mode"] == "dev-tree"
    assert payload["provenance"]["proof_refusals"] == ["working_tree_dirty"]
    assert payload["error"].startswith("not proof-eligible [dev-tree]: working_tree_dirty")


def test_keg_built_from_this_sha_is_proof_eligible(monkeypatch, capsys, tmp_path):
    served = fake_served(KEG_PATH, build_sha="abc123")
    returncode, payload = run_live_with_served(
        monkeypatch, capsys, tmp_path, served, CLEAN_TREE, ["--require-code-under-test"]
    )

    assert returncode == 0
    assert payload["proof_eligible"] is True
    assert payload["provenance"]["provenance_mode"] == "keg"
    assert payload["provenance"]["served_build_sha"] == "abc123"
    assert payload["provenance"]["served_pythonpath"] is None
    # search_latency ran and is UNMEASURED: proof-eligibility is about the served code, not about
    # whether a latency band exists yet, and no attested main runs were handed to this run.
    assert [check["status"] for check in payload["checks"]] == ["UNMEASURED", "PASS", "PASS", "PASS"]


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
    served = fake_served(DEV_PATH)
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


# --- Round 3 (Codex P1): probe under the helper's own PYTHONPATH -------------------------------


def fake_ps_env(plain: str, with_env: str):
    def run(argv, **_kwargs):
        assert argv[:2] == ["ps", "-p"] and argv[-2:] == ["-o", "command="]
        return subprocess.CompletedProcess(argv, 0, stdout=(with_env if "-E" in argv else plain) + "\n", stderr="")

    return run


def test_helper_env_is_the_ps_E_block_and_stitches_spaced_values(monkeypatch):
    plain = "/venv/bin/python -m brainlayer.brainbar_hybrid_helper --db-path /tmp/x.db"
    block = " PATH=/usr/bin:/bin HOME=/Users/e PYTHONPATH=/Users/e/My Gits/brainlayer/src BRAINLAYER_REPO_ROOT=/Users/e/My Gits/brainlayer"
    monkeypatch.setattr(sprint_gate.subprocess, "run", fake_ps_env(plain, plain + block))

    env = sprint_gate.helper_env(4242)

    assert env["PYTHONPATH"] == "/Users/e/My Gits/brainlayer/src"
    assert env["BRAINLAYER_REPO_ROOT"] == "/Users/e/My Gits/brainlayer"
    assert env["PATH"] == "/usr/bin:/bin"


def test_helper_env_unreadable_is_an_error_not_a_guess(monkeypatch):
    plain = "/venv/bin/python -m brainlayer.brainbar_hybrid_helper --db-path /tmp/x.db"
    monkeypatch.setattr(sprint_gate.subprocess, "run", fake_ps_env(plain, plain))
    with pytest.raises(RuntimeError, match="could not read the environment of helper pid 4242"):
        sprint_gate.helper_env(4242)


@pytest.mark.skipif(sys.platform != "darwin", reason="ps -E is a BSD flag; Linux ps rejects it")
def test_helper_env_reads_a_real_process():
    """`ps -E` reports the LAUNCH-time environment (pytest's conftest rewrites HOME afterwards),
    which is exactly the env BrainBar handed the helper."""
    env = sprint_gate.helper_env(os.getpid())
    assert env["USER"] == pwd.getpwuid(os.getuid()).pw_name
    assert "PATH" in env


def fake_package(tmp_path: Path) -> Path:
    package = tmp_path / "shadow" / "brainlayer"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("__version__ = '9.9.9'\n__build_sha__ = 'shadow'\n")
    return package.parent


def test_probe_honours_exactly_the_helpers_pythonpath(tmp_path):
    """(a)/(b): a source-fallback helper imports via PYTHONPATH only; the probe must see the same copy."""
    shadow = fake_package(tmp_path)
    served = sprint_gate.served_package(Path(sys.executable), str(shadow))
    assert served == {"version": "9.9.9", "path": str(shadow / "brainlayer" / "__init__.py"), "build_sha": "shadow"}

    served = sprint_gate.served_package(Path(sys.executable), str(ROOT / "src"))
    assert Path(served["path"]).resolve().is_relative_to(sprint_gate.PACKAGE)
    served.update(helper_started_at=FRESH_HELPER)
    assert sprint_gate.eligibility(served, CLEAN_TREE, MTIME)[0] == "dev-tree"


def test_probe_without_helper_pythonpath_never_inherits_the_gates_own(monkeypatch, tmp_path):
    shadow = fake_package(tmp_path)
    monkeypatch.setenv("PYTHONPATH", str(shadow))  # the gate's own env must not leak into the probe
    served = sprint_gate.served_package(Path(sys.executable), None)
    assert not Path(served["path"]).is_relative_to(shadow)
    assert served["version"] != "9.9.9"


def test_probe_mirrors_brainbar_launch_no_isolation_flags_and_neutral_cwd(monkeypatch):
    seen = {}

    def run(argv, **kwargs):
        seen["argv"], seen["env"], seen["cwd"] = argv, kwargs["env"], kwargs["cwd"]
        return subprocess.CompletedProcess(argv, 0, stdout='{"version":"1","path":"/p","build_sha":null}', stderr="")

    monkeypatch.setattr(sprint_gate.subprocess, "run", run)
    sprint_gate.served_package(Path("/venv/bin/python"), None)
    assert seen["argv"][1] == "-c", "no -I/-s/-E: BrainBar launches the helper with no isolation flags"
    assert "PYTHONPATH" not in seen["env"] and seen["cwd"] == "/"
    sprint_gate.served_package(Path("/venv/bin/python"), "/x/src")
    assert seen["env"]["PYTHONPATH"] == "/x/src"


def test_resolve_served_records_pythonpath_and_repo_root(monkeypatch):
    monkeypatch.setattr(sprint_gate, "warm_helper", lambda _config: None)
    monkeypatch.setattr(sprint_gate, "find_helper", lambda: (4242, Path("/tmp/x.db")))
    monkeypatch.setattr(sprint_gate, "helper_python", lambda _pid: Path("/venv/bin/python"))
    monkeypatch.setattr(sprint_gate, "helper_started_at", lambda _pid: FRESH_HELPER)
    monkeypatch.setattr(sprint_gate, "helper_env", lambda _pid: {"PYTHONPATH": "/r/src", "BRAINLAYER_REPO_ROOT": "/r"})
    seen = {}

    def probe(python, pythonpath):
        seen["pp"] = pythonpath
        return {"version": "1.5.10", "path": "/r/src/brainlayer/__init__.py", "build_sha": None}

    monkeypatch.setattr(sprint_gate, "served_package", probe)
    monkeypatch.setattr(sprint_gate, "newest_mtime", lambda directory: seen.update(dir=directory) or 42.0)

    served, db_path = sprint_gate.resolve_served({})

    assert seen["pp"] == "/r/src"
    assert served["pythonpath"] == "/r/src" and served["repo_root"] == "/r"
    assert seen["dir"] == Path("/r/src/brainlayer") and served["package_newest_mtime"] == 42.0
    assert db_path == Path("/tmp/x.db")


def test_unreadable_helper_env_is_a_structured_refusal(monkeypatch, capsys, tmp_path):
    """(c): the gate must say it could not read the helper, not probe blind."""
    live_config = tmp_path / "live.json"
    live_config.write_text(json.dumps(deterministic_live_config()), encoding="utf-8")
    monkeypatch.setattr(sprint_gate, "CORPUS", live_config)
    monkeypatch.setattr(sprint_gate.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(sprint_gate.platform, "machine", lambda: "arm64")
    monkeypatch.setattr(sprint_gate, "warm_helper", lambda _config: None)
    monkeypatch.setattr(sprint_gate, "find_helper", lambda: (4242, Path("/tmp/x.db")))
    monkeypatch.setattr(sprint_gate, "working_tree_provenance", lambda: dict(CLEAN_TREE))
    plain = "/venv/bin/python -m brainlayer.brainbar_hybrid_helper --db-path /tmp/x.db"
    monkeypatch.setattr(sprint_gate.subprocess, "run", fake_ps_env(plain, plain))

    returncode = sprint_gate.main(["--json", "--require-code-under-test"])
    payload = json.loads(capsys.readouterr().out)

    assert returncode == 1
    assert payload["checks"] == []
    assert payload["provenance"]["proof_refusals"] == ["provenance_unresolved"]
    assert payload["provenance"]["served_pythonpath"] is None
    assert (
        payload["provenance"]["provenance_error"] == "RuntimeError: could not read the environment of helper pid 4242"
    )


def test_payload_records_served_pythonpath(monkeypatch, capsys, tmp_path):
    served = fake_served(DEV_PATH, pythonpath=str(ROOT / "src"), repo_root=str(ROOT))
    returncode, payload = run_live_with_served(monkeypatch, capsys, tmp_path, served, CLEAN_TREE)
    assert returncode == 0
    assert payload["provenance"]["served_pythonpath"] == str(ROOT / "src")
    assert payload["provenance"]["served_repo_root"] == str(ROOT)


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


# --- ratchet (c): margins are measured from attested green main runs, never a flat fraction ------------

# Sprint-gate `search_latency` values recorded on green main, 2026-09-01..02, socket-measured on
# MacBook-Pro.local: resign-1.5.10.log, w12b-REPORT.md (×2), w12c-REPORT.md, w8-REPORT.md (×2),
# a5-bench-m4-2026-09-02.log. Real numbers; the point of the tests below is what they do to the gate.
GREEN_MAIN_P50_MS = [185.0, 210.0, 214.6, 98.382, 291.416, 293.901, 281.4]
GREEN_MAIN_P95_MS = [1920.0, 1924.0, 1895.1, 1808.074, 2242.013, 1946.418, 2084.9]


def attestation(index: int, **measured) -> dict:
    return {
        "schema": 1,
        "run_id": 1000 + index,
        "run_attempt": 1,
        "main_sha": f"{index:040x}",
        "measured_at": f"2026-09-0{1 + index % 5}T12:00:00Z",
        "workflow": ".github/workflows/ratchet-attest.yml",
        "measured": measured,
    }


def real_history() -> list[dict]:
    return [
        attestation(index, **{"latency_baseline_ms.p50": p50, "latency_baseline_ms.p95": p95})
        for index, (p50, p95) in enumerate(zip(GREEN_MAIN_P50_MS, GREEN_MAIN_P95_MS, strict=True))
    ]


def replay(tmp_path: Path, capsys, fixture: dict) -> tuple[int, dict]:
    path = tmp_path / "fixture.json"
    path.write_text(json.dumps(fixture), encoding="utf-8")
    returncode = sprint_gate.main(["--json", "--fixture", str(path)])
    return returncode, json.loads(capsys.readouterr().out)


def latency_fixture(samples_ms: list[float], attestations: list[dict], baseline: dict | None = None) -> dict:
    return {
        "checks": ["search_latency"],
        "latency_baseline_ms": baseline or {"p50": 225.0, "p95": 1974.4},
        "latency_samples_ms": samples_ms,
        "attestations": attestations,
    }


def test_a_value_inside_measured_variance_is_no_longer_red(tmp_path: Path, capsys):
    """FAILS on 3ee7c279. There, 294 ms against a 225 ms baseline × 1.10 = 247.5 ms was RED -- and
    294 ms is a value green main actually produced (w8-REPORT.md, 2026-09-02). Against the band the
    seven attested runs measure (limit 463.7 ms), it is what it was: normal."""
    returncode, payload = replay(tmp_path, capsys, latency_fixture([294.0] * 10, real_history()))
    assert returncode == 0 and payload["status"] == "PASS"
    check = payload["checks"][0]
    assert check["status"] == "PASS"
    assert check["details"]["limits_ms"]["p50"] == pytest.approx(463.7, abs=0.2)
    assert check["details"]["attested_runs"] == {"p50": 7, "p95": 7}
    assert "n=7 attested green main runs" in check["details"]["margins"]["p50"]
    assert "limit 463.7 ms" in check["details"]["margins"]["p50"]


def test_a_two_fold_regression_no_longer_hides_behind_a_stale_baseline(tmp_path: Path, capsys):
    """FAILS on 3ee7c279. There, 950 ms against the corpus's 911.887 ms × 1.10 = 1003 ms was GREEN,
    while every green main run on record sits between 98 and 294 ms. The measured band says RED."""
    corpus = json.loads(CORPUS.read_text(encoding="utf-8"))
    assert corpus["latency_baseline_ms"]["p50"] == pytest.approx(911.887)
    fixture = latency_fixture([950.0] * 10, real_history(), baseline=corpus["latency_baseline_ms"])
    returncode, payload = replay(tmp_path, capsys, fixture)
    assert returncode == 1 and payload["status"] == "FAIL"
    check = payload["checks"][0]
    assert check["status"] == "FAIL"
    assert check["details"]["limits_ms"]["p50"] < 950.0 < corpus["latency_baseline_ms"]["p50"] * 1.10


def test_fewer_than_five_attested_runs_is_unmeasured_never_a_verdict(tmp_path: Path, capsys):
    returncode, payload = replay(tmp_path, capsys, latency_fixture([950.0] * 10, real_history()[:4]))
    assert returncode == 0
    assert payload["status"] == "PASS"  # nothing FAILED; and the payload says what it could not judge
    assert payload["unmeasured"] == ["search_latency"]
    check = payload["checks"][0]
    assert check["status"] == "UNMEASURED"
    assert check["details"]["measured_ms"]["p50"] == 950.0  # the value is real and is reported
    assert "limits_ms" not in check["details"]  # the verdict is not, and no number stands in for it
    assert check["details"]["attested_runs"] == {"p50": 4, "p95": 4}
    assert "4 of the 5 attested green main runs" in check["details"]["margins"]["p50"]


def test_the_band_needs_five_runs_for_each_percentile_separately(tmp_path: Path, capsys):
    history = real_history()
    for document in history[3:]:
        del document["measured"]["latency_baseline_ms.p95"]
    returncode, payload = replay(tmp_path, capsys, latency_fixture([100.0] * 10, history))
    assert returncode == 0  # unmeasured is not a failure; it is the absence of a verdict, stated
    check = payload["checks"][0]
    assert check["status"] == "UNMEASURED"
    assert check["details"]["attested_runs"] == {"p50": 7, "p95": 3}


def test_a_malformed_inline_attestation_refuses_the_replay(tmp_path: Path, capsys):
    history = real_history()
    history[2]["main_sha"] = "not-a-sha"
    returncode, payload = replay(tmp_path, capsys, latency_fixture([100.0] * 10, history))
    assert returncode == 1 and payload["checks"] == []
    assert "fixture attestation 2" in payload["error"] and "40-hex" in payload["error"]


def test_inline_attestations_are_replay_only(monkeypatch, capsys, tmp_path: Path):
    config = deterministic_live_config()
    config["attestations"] = real_history()
    hostname = config["latency_baseline_ms"]["hostname"]
    returncode, payload = run_live_config(monkeypatch, capsys, tmp_path, config, hostname)
    assert returncode == 1 and payload["checks"] == []
    assert payload["error"].startswith("inline `attestations` are replay-only")


def write_attestations(root: Path, documents: list[dict]) -> Path:
    for document in documents:
        run_dir = root / str(document["run_id"])
        run_dir.mkdir(parents=True)
        (run_dir / "attestation.json").write_text(json.dumps(document), encoding="utf-8")
    return root


def test_live_gate_reads_attested_runs_from_the_path_it_is_handed(monkeypatch, capsys, tmp_path: Path):
    root = write_attestations(tmp_path / "attestations", real_history())
    config = deterministic_live_config()  # one 100 ms sample, inside the 463.7 ms band
    hostname = config["latency_baseline_ms"]["hostname"]
    returncode, payload = run_live_config(
        monkeypatch, capsys, tmp_path, config, hostname, extra_args=["--attestations", str(root)]
    )
    assert returncode == 0 and payload["status"] == "PASS" and payload["unmeasured"] == []
    check = payload["checks"][0]
    assert check["name"] == "search_latency" and check["status"] == "PASS"
    assert check["details"]["attested_runs"] == {"p50": 7, "p95": 7}


def test_live_gate_without_attestations_renders_latency_unmeasured(monkeypatch, capsys, tmp_path: Path):
    config = deterministic_live_config()
    hostname = config["latency_baseline_ms"]["hostname"]
    returncode, payload = run_live_config(monkeypatch, capsys, tmp_path, config, hostname)
    assert returncode == 0 and payload["unmeasured"] == ["search_latency"]
    check = payload["checks"][0]
    assert check["status"] == "UNMEASURED"
    assert "0 of the 5 attested green main runs" in check["details"]["margins"]["p50"]


def test_live_gate_refuses_an_attestation_path_it_cannot_read(monkeypatch, capsys, tmp_path: Path):
    config = deterministic_live_config()
    hostname = config["latency_baseline_ms"]["hostname"]
    returncode, payload = run_live_config(
        monkeypatch, capsys, tmp_path, config, hostname, extra_args=["--attestations", str(tmp_path / "absent")]
    )
    assert returncode == 1 and payload["checks"] == []
    assert "absent" in payload["error"] and "neither a file nor a directory" in payload["error"]


def resource_fixture(daemon_cpu_pct: float, attestations: list[dict]) -> dict:
    idle = {"cpu_pct": 0, "rss_bytes": 0, "pids": []}
    return {
        "checks": ["resource_budget"],
        "resource_samples": [
            {
                "daemon": {"cpu_pct": daemon_cpu_pct, "rss_bytes": 1, "pids": [1]},
                "helper": idle,
                "watcher": idle,
                "drain": idle,
            }
        ],
        "attestations": attestations,
    }


def test_idle_cpu_goes_red_beyond_the_measured_band_even_under_the_ceiling(tmp_path: Path, capsys):
    """R3's worked example. Five attested green main runs at 4.0-5.0% idle give a band with limit
    ~6.2%. A run at 20% is far under the ratified 30% ceiling and was PASS on 3ee7c279; it is a
    four-fold drift from what green main does, and the ratchet says so."""
    history = [attestation(i, **{"idle_cpu_pct.daemon": value}) for i, value in enumerate([4.0, 4.5, 5.0, 4.8, 4.2])]
    returncode, payload = replay(tmp_path, capsys, resource_fixture(20.0, history))
    assert returncode == 1
    check = payload["checks"][0]
    assert check["status"] == "FAIL"
    assert check["details"]["cpu_over_measured_band"] == ["daemon"]
    assert check["details"]["cpu_margins"]["daemon"].startswith("limit 6.2 % = mean 4.5 %")
    # ...and the same band lets a run that green main would recognise through.
    returncode, payload = replay(tmp_path, capsys, resource_fixture(5.5, history))
    assert returncode == 0 and payload["checks"][0]["status"] == "PASS"


def test_idle_cpu_ceiling_alone_decides_while_its_band_is_unmeasured(tmp_path: Path, capsys):
    # The 30% ceiling is ratified and does not become unmeasured; the band's absence is stated per process.
    returncode, payload = replay(tmp_path, capsys, resource_fixture(20.0, []))
    check = payload["checks"][0]
    assert returncode == 0 and check["status"] == "PASS"
    assert check["details"]["cpu_over_measured_band"] == []
    assert check["details"]["cpu_margins"]["daemon"].startswith("margin unmeasured — 0 of the 5")
    returncode, payload = replay(tmp_path, capsys, resource_fixture(31.0, []))
    assert returncode == 1 and payload["checks"][0]["status"] == "FAIL"
