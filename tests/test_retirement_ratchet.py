"""The row must be RED when measurement/report handoffs fail."""

from pathlib import Path

import yaml

from scripts import ci_ratchet_table as ratchet
from tests.test_ci_ratchet_table import HEAD, retirement_receipt


def test_required_row_rejects_missing_or_malformed_report(tmp_path):
    assert ratchet.row_retirement(None, HEAD).status == ratchet.RED
    path = tmp_path / "bad.json"
    path.write_text("not JSON")
    assert ratchet.row_retirement(path, HEAD).status == ratchet.RED
    assert ratchet.row_retirement(path, None).status == ratchet.RED


def test_row_binds_proof_to_exact_head(tmp_path):
    path = retirement_receipt(tmp_path, HEAD)
    assert ratchet.row_retirement(path, HEAD).status == ratchet.GREEN
    assert ratchet.row_retirement(path, "b" * 40).status == ratchet.RED


def test_workflow_has_unconditional_failure_handoff():
    workflow = yaml.safe_load((Path(__file__).resolve().parents[1] / ".github/workflows/ratchet.yml").read_text())
    jobs = workflow["jobs"]
    assert "retirement" in jobs["table"]["needs"]
    native = jobs["retirement"]
    assert native["runs-on"] == "macos-15"
    command = next(step["run"] for step in native["steps"] if "retirement_run.py" in step.get("run", ""))
    assert "--dependency-sha" in command and "pull_request.head.sha" in command
    upload = next(step for step in native["steps"] if step.get("uses", "").startswith("actions/upload-artifact"))
    assert "!cancelled()" in upload["if"] and upload["with"]["if-no-files-found"] == "error"
    collect = next(step for step in jobs["table"]["steps"] if step.get("name") == "Collect ratchet rows")
    assert "--retirement-report" in collect["run"]
    assert "RETIREMENT_RESULT" in collect["env"] and "!= success" in collect["run"]
