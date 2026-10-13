"""Real scratch installer errors reach maintenance without leaking captured secrets."""

import json
import shutil
import subprocess
from pathlib import Path

import pytest


@pytest.mark.parametrize("prefix", ["", "x" * 5000], ids=["short", "bounded-tail"])
def test_installer_stderr_tail_survives_resume_failure_and_job_alert(tmp_path, monkeypatch, capsys, prefix):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("BRAINLAYER_SUPERVISOR_SOCKET", raising=False)
    from brainlayer import maintenance

    script = tmp_path / "scripts/launchd/install.sh"
    script.parent.mkdir(parents=True)
    script.write_text("#!/bin/sh\nprintf '%s\\n' '" + prefix + "ERROR: X' >&2\necho stdout-context\nexit 1\n")
    script.chmod(0o700)
    monkeypatch.setattr(maintenance, "_maintenance_keep_down_services", lambda: set())
    monkeypatch.setattr(maintenance, "_service_is_deliberately_paused", lambda service: False)
    failures = maintenance._resume_services(tmp_path, ("index",), {"index": True})
    assert len(failures) == 1
    reason = maintenance._format_resume_failures(failures)
    assert "ERROR: X" in reason
    assert "stdout-context" in reason
    assert len(reason.split("stderr tail: ", 1)[1]) <= 2048
    assert_surfaces(reason, monkeypatch, capsys)


def assert_surfaces(reason, monkeypatch, capsys):
    from brainlayer import job_alerts, maintenance

    alerts, logs = [], []

    def abort(*args, **kwargs):
        raise maintenance.MaintenanceAbort(reason)

    monkeypatch.setattr(maintenance, "run_maintenance", abort)
    monkeypatch.setattr(maintenance, "_write_log", lambda path, record: logs.append(record))
    monkeypatch.setattr(job_alerts, "report", lambda key, message: alerts.append((key, message)))
    assert maintenance.main(["--light"]) == 75
    assert json.loads(capsys.readouterr().out)["reason"] == reason
    assert logs[0]["reason"] == reason
    assert alerts[0][0] == "maintenance-light" and reason in alerts[0][1]


def test_real_installer_missing_config_stdout_reaches_reason(tmp_path, monkeypatch, capsys):
    from brainlayer import maintenance

    source = Path(__file__).resolve().parents[1] / "scripts/launchd"
    launchd = tmp_path / "scripts/launchd"
    shutil.copytree(source, launchd)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("PATH", "/usr/bin:/bin")
    monkeypatch.setenv("PYTHON_BIN", "/usr/bin/true")
    monkeypatch.setenv("BRAINLAYER_BIN", "/usr/bin/true")
    monkeypatch.setenv("BRAINLAYER_ENV_FILE", str(tmp_path / "missing.env"))
    monkeypatch.delenv("BRAINLAYER_SUPERVISOR_SOCKET", raising=False)
    monkeypatch.setattr(maintenance, "_maintenance_keep_down_services", lambda: set())
    monkeypatch.setattr(maintenance, "_service_is_deliberately_paused", lambda service: False)
    failures = maintenance._resume_services(tmp_path, ("index",), {"index": True})
    assert len(failures) == 1 and "config file not found" in failures[0][1].stdout
    reason = maintenance._format_resume_failures(failures)
    assert "config file not found" in reason and "stdout" in reason
    assert_surfaces(reason, monkeypatch, capsys)


@pytest.mark.parametrize("channel", ["stdout", "stderr"])
def test_provider_and_quarantined_tokens_never_reach_surfaces(channel, monkeypatch, capsys):
    from brainlayer import maintenance
    from brainlayer.pipeline.secret_scrub import scrub_secrets

    provider = "AIza" + "x" * 35
    bare = "mF9qP2xL7vR8sK4nT6yB3cD5eG7hJ9kL2mN4pQ6r"
    assert [token.value for token in scrub_secrets(bare).quarantine] == [bare]
    error = subprocess.CalledProcessError(
        1, ["install.sh"], **{"output" if channel == "stdout" else "stderr": f"ERROR: {provider} {bare}"}
    )
    reason = maintenance._format_resume_failures([("index", error)])
    assert provider not in reason and bare not in reason
    assert "REDACTED" in reason
    assert_surfaces(reason, monkeypatch, capsys)


def test_scrub_failure_withholds_all_output_and_exception_text(monkeypatch, capsys):
    from brainlayer import maintenance
    from brainlayer.pipeline import secret_scrub

    def fail(text):
        raise RuntimeError("scrubber failed with private output")

    monkeypatch.setattr(secret_scrub, "scrub_secrets", fail)
    error = subprocess.CalledProcessError(1, ["private-command"], stderr="private-output")
    reason = maintenance._format_resume_failures([("index", error)])
    assert "output withheld: scrub failed" in reason
    assert "private" not in reason
    assert_surfaces(reason, monkeypatch, capsys)


def test_total_reason_is_bounded_and_reports_omitted_failures():
    from brainlayer.maintenance import _format_resume_failures

    failures = [
        (f"job-{n}", subprocess.CalledProcessError(1, ["install.sh"], output="x" * 3000, stderr="y" * 3000))
        for n in range(40)
    ]
    reason = _format_resume_failures(failures)
    assert len(reason) <= 4096 and "+" in reason and " more" in reason


def test_real_env_sourcing_does_not_print_malformed_key(tmp_path):
    source = Path(__file__).resolve().parents[1] / "scripts/launchd/install.sh"
    text = source.read_text()
    function = (
        "verify_gemini_env_file() {" + text.split("verify_gemini_env_file() {", 1)[1].split("\n}\n", 1)[0] + "\n}\n"
    )
    env = tmp_path / "brainlayer.env"
    token = "AIza" + "x" * 35
    env.write_text("GOOGLE_API_KEY= " + token + "\n")
    script = tmp_path / "probe.sh"
    script.write_text(function + 'BRAINLAYER_ENV_FILE="$1"\nverify_gemini_env_file\n')
    result = subprocess.run(["/bin/bash", str(script), str(env)], capture_output=True, text=True, timeout=5)
    assert result.returncode == 1
    assert token not in result.stdout + result.stderr


@pytest.mark.parametrize("channel", ["output", "stderr"])
def test_scrubbing_precedes_tail_truncation_even_for_byte_channels(channel, monkeypatch, capsys):
    from brainlayer import maintenance

    provider = "AIza" + "x" * 1200
    error = subprocess.CalledProcessError(1, ["install.sh"], **{channel: (provider + "\nERROR: bounded").encode()})
    reason = maintenance._format_resume_failures([("index", error)])
    assert "ERROR: bounded" in reason and "x" * 40 not in reason
    assert_surfaces(reason, monkeypatch, capsys)
