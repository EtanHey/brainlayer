"""Checker probes. subprocess.run is intercepted, so TLC never starts."""

import importlib.util
import os
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("backup_verdict_check", HERE / "check.py")
checker = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(checker)
sys.path.insert(0, str(HERE))
import replay  # noqa: E402

PASS_LOG = "TLC2 Version 2.19 of 08 August 2024\nModel checking completed. No error has been found.\n12 distinct states found\n"
TRACE_LOG = """TLC2 Version 2.19 of 08 August 2024
Error: Invariant NoFalseGreen is violated.
Error: The behavior up to this point is:
State 1: <Initial predicate>
/\\ now = 0
/\\ st = [off |-> "none", snap |-> "none"]

State 2: <Tick line 1, col 1 to line 2, col 2 of module BackupVerdict>
/\\ now = 0
/\\ st = [ off |-> "red",
  snap |-> "green" ]

7 distinct states found
"""


def test_every_case_has_a_config_and_known_module():
    for case in checker.CASES:
        assert (checker.CONFIGS / f"{case['name']}.cfg").is_file(), case["name"]
        assert (HERE / f"{case['module']}.tla").is_file(), case["module"]


def test_every_config_belongs_to_a_case():
    names = {case["name"] for case in checker.CASES}
    assert {p.stem for p in checker.CONFIGS.glob("*.cfg")} == names


def test_every_bug_case_names_its_pr_and_breaks_a_named_rule():
    for case in checker.CASES:
        if case["name"].startswith("bug-"):
            assert case["pr"].startswith(("#", "S1a-")), case["name"]
            assert case["expect"] != "pass", case["name"]
        if case["name"].startswith("current-"):
            assert case["expect"] == "pass", case["name"]


@pytest.mark.parametrize(
    ("text", "code", "expected"),
    [
        (PASS_LOG, 0, "pass"),
        (TRACE_LOG, 12, "NoFalseGreen"),
        ("Error: Temporal properties were violated.\n", 13, "temporal"),
        ("***Parse Error***\n", 255, "error"),
        ("Model checking completed. No error has been found.\n", 1, "error"),
    ],
)
def test_outcome_classification(text, code, expected):
    assert checker.outcome(text, code) == expected


def test_trace_parser_joins_wrapped_values():
    states = checker.parse_trace(TRACE_LOG)
    assert [action for action, _ in states] == ["Initial", "Tick"]
    assert states[1][1]["st"] == '[ off |-> "red", snap |-> "green" ]'
    assert "st = " in checker.render_trace(states)


def test_launch_uses_one_worker_and_no_load_wait(tmp_path, monkeypatch):
    monkeypatch.setattr(checker, "wait_for_load", lambda: {})
    monkeypatch.delenv("TLC_JAR", raising=False)
    monkeypatch.setenv("TLC", "/opt/rig/tlc")
    monkeypatch.delenv("TLC_WORKERS", raising=False)
    case = next(c for c in checker.CASES if c["name"] == "current-dot")
    done = subprocess.CompletedProcess([], 0, PASS_LOG, "")
    with patch.object(checker.subprocess, "run", return_value=done) as launch:
        result = checker.run_case(case, tmp_path)
    command = launch.call_args.args[0]
    assert command[0] == "/opt/rig/tlc"
    assert command[command.index("-workers") + 1] == "1"
    assert result["matched"] and result["tlc_version"] == "2.19"


def test_tlc_jar_runs_java_with_a_bounded_heap(monkeypatch, tmp_path):
    monkeypatch.setenv("TLC_JAR", str(tmp_path / "tla2tools.jar"))
    command = checker.tlc_command()
    assert command[0] == "java" and "-Xmx512m" in command and command[-1] == "tlc2.TLC"


def test_unexpected_pass_of_a_bug_case_is_a_mismatch(tmp_path, monkeypatch):
    monkeypatch.setattr(checker, "wait_for_load", lambda: {})
    case = next(c for c in checker.CASES if c["name"] == "bug-1583-B4-drive-latch")
    with patch.object(checker.subprocess, "run", return_value=subprocess.CompletedProcess([], 0, PASS_LOG, "")):
        assert not checker.run_case(case, tmp_path)["matched"]


def test_crash_is_incomplete_not_a_pass(capsys):
    with patch.object(checker, "main", side_effect=KeyError("sources")):
        assert checker.cli() == 2
    assert "could not complete" in capsys.readouterr().err


def test_sync_report_flags_a_changed_source(tmp_path, monkeypatch):
    for path in checker.COVERED:
        (tmp_path / path).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / path).write_text("changed\n")
    monkeypatch.setattr(checker, "REPO", tmp_path)
    assert checker.sync_report() == list(checker.COVERED)


@pytest.mark.parametrize(
    ("kind", "fields"),
    [
        ("local_ok", {"local_verified": True, "local_integrity": "ok"}),
        ("upload_ok", {"uploaded": True, "verified": True}),
        ("revoke", {"drive_md5_match": False}),
        ("removed", {"local_removed": True}),
        ("abandon", {"upload_state": "abandoned", "local_verified": False}),
        ("supersede", {"upload_state": "superseded", "local_verified": False}),
    ],
)
def test_replay_receipts_use_backup_daily_field_names(kind, fields):
    body = replay.receipt({"kind": kind, "snap": 1, "hour": 2, "created": 1, "exact": True})
    assert body["backup_log_provenance"] == "real"
    assert fields.items() <= body.items()


def test_filename_only_revocation_omits_the_exact_time():
    body = replay.receipt({"kind": "revoke", "snap": 1, "hour": 2, "created": 1, "exact": False})
    assert "snapshot_created_at" not in body


def test_replay_runs_real_source_without_the_package(tmp_path):
    src = checker.REPO / "src" / "brainlayer"
    payload = {
        "constants": {"RedSnap": 2, "RedOff": 3, "Grace": 1, "Gap": 1},
        "ops": [
            {"op": "receipt", "hour": 0, "kind": "upload_ok", "snap": 1, "created": 0, "exact": True},
            {"op": "tick", "hour": 1, "readable": True, "model": {"snap": "green", "off": "green"}},
            {"op": "receipt", "hour": 1, "kind": "revoke", "snap": 1, "created": 0, "exact": True},
            {"op": "tick", "hour": 2, "readable": True, "model": {"snap": "red", "off": "red"}},
        ],
    }
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    proc = subprocess.run(
        [sys.executable, "-S", str(HERE / "replay.py"), "--run", str(src)],
        input=__import__("json").dumps(payload),
        capture_output=True,
        text=True,
        cwd=tmp_path,
        env=env,
        check=True,
    )
    ticks = __import__("json").loads(proc.stdout)
    assert [t["src"] for t in ticks] == [{"snap": "green", "off": "green"}, {"snap": "red", "off": "red"}]


@pytest.mark.parametrize(
    ("fixed_here", "prefix", "src", "ok"),
    [
        ("yes", (True, True), (False, False), True),  # pre-fix ref reproduces tick for tick; fixed here
        ("yes", (True, True), (True, True), False),  # the fix regressed in this tree
        ("no", (True, True), (True, True), True),  # fix not merged yet: still reproduces here
        ("no", (True, True), (True, False), False),  # still here, but this tree disagrees on a tick
        ("no", (True, True), (False, False), False),  # fix landed: flip fixed_here
        ("yes", (False, True), (False, False), False),  # the model no longer describes the pre-fix code
        ("yes", (True, False), (False, False), False),  # divergent historical prefix (REV-FTLA1-R1 B2)
    ],
)
def test_replay_ratchet(fixed_here, prefix, src, ok):
    case = {"prefix": "abc", "fixed_here": fixed_here}
    replays = {
        "prefix": {"reproduces": prefix[0], "conforms": prefix[1]},
        "src": {"reproduces": src[0], "conforms": src[1]},
    }
    assert checker.replay_matches(case, replays) is ok


def test_a_required_replay_cannot_be_skipped():
    """REV-FTLA1-R1 B2: replay_matches({}) must not pass a case that names a pre-fix ref."""
    assert checker.replay_matches({"prefix": "abc", "fixed_here": "yes"}, {}) is False
    assert checker.replay_matches({}, {}) is True  # no replay required, none given
    error = {"prefix": {"error": "x"}, "src": {"reproduces": False, "conforms": False}}
    assert checker.replay_matches({"prefix": "abc", "fixed_here": "yes"}, error) is False


def test_absent_counterexample_trace_fails_a_replayed_case(tmp_path, monkeypatch):
    """TLC reports the violation but no trace parses: the replayed bug case must not match."""
    monkeypatch.setattr(checker, "wait_for_load", lambda: {})
    case = next(c for c in checker.CASES if c["name"] == "bug-1596-abandoned")
    text = "TLC2 Version 2.19\nError: Invariant NoFalseGreen is violated.\n"
    with patch.object(checker.subprocess, "run", return_value=subprocess.CompletedProcess([], 12, text, "")):
        result = checker.run_case(case, tmp_path)
    assert result["got"] == "NoFalseGreen"
    assert result["matched"] is False
    assert "error" in result["replay"]["prefix"]


def _simulation_case():
    return next(c for c in checker.CASES if c["name"] == "conform-ingest-main")


def test_simulation_with_zero_files_is_an_error(tmp_path):
    got, detail = checker.simulate_conformance(_simulation_case(), tmp_path, "Finished", 0, tmp_path)
    assert got == "error" and "no trace file" in detail["error"]


def test_simulation_with_zero_ticks_is_an_error(tmp_path):
    state = (
        "STATE_{n} ==\n/\\ now = {n}\n/\\ lastTick = -1\n"
        '/\\ lastObs = [hb |-> "unobs", canary |-> "unobs", drive |-> "unobs", queue |-> "unobs", emb |-> "unobs"]\n'
        '/\\ st = [ingest |-> "none", drive |-> "none", stores |-> "none", embedding |-> "none"]\n'
        "/\\ hb = 0\n\n"
    )
    (tmp_path / "tr_0_0").write_text("".join(state.format(n=n) for n in (0, 1, 2)))
    got, detail = checker.simulate_conformance(_simulation_case(), tmp_path, "Finished", 0, tmp_path)
    assert got == "error" and "no health tick" in str(detail["error"])


def test_simulation_file_that_parses_to_nothing_is_an_error(tmp_path):
    (tmp_path / "tr_0_0").write_text("---- MODULE tr ----\n====\n")
    got, _ = checker.simulate_conformance(_simulation_case(), tmp_path, "Finished", 0, tmp_path)
    assert got == "error"


def test_backup_replay_receipts_carry_iso_times():
    """A shadowed helper once turned attempted_at into a list, and the reader saw no receipt."""
    body = replay.receipt({"kind": "local_ok", "snap": 1, "hour": 2, "created": 1, "exact": True})
    assert isinstance(body["attempted_at"], str) and body["attempted_at"].startswith("2026-11-02T")
    assert isinstance(body["snapshot_created_at"], str)


def test_every_backup_bug_case_is_replayed():
    for case in checker.CASES:
        if case["name"].startswith("bug-") and case["module"] == "BackupVerdict":
            assert case.get("prefix") and case.get("fixed_here") in ("yes", "no"), case["name"]
        if case.get("simulate"):
            assert case["expect"] in ("conforms", "diverges") and not case.get("prefix"), case["name"]
