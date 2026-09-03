"""The ratchet table prints a measured value or a named `n/a` — never a guessed number (w14)."""

from __future__ import annotations

import json
import os
import re
import subprocess
import zipfile
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import pytest
import yaml

from scripts import ci_ratchet_table as ratchet


def _clean_git_env() -> dict[str, str]:
    # An inherited GIT_DIR/GIT_WORK_TREE (a git hook runs the suite) overrides `-C`, so the decoy
    # repo's commits would land in the REAL repo. Same guard as tests/test_build_sha.py and
    # src/brainlayer/deploy_drift.py.
    return {key: value for key, value in os.environ.items() if not key.startswith("GIT_")}


ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / ".github" / "workflows" / "ratchet.yml"
CORPUS = json.loads((ROOT / "tests" / "fixtures" / "sprint_gate" / "corpus.json").read_text(encoding="utf-8"))
HEAD = "a" * 40
NOW = datetime(2026, 9, 3, 12, 0, tzinfo=timezone.utc)


def make_wheel(tmp_path: Path, stamp: str | None) -> Path:
    tmp_path.mkdir(parents=True, exist_ok=True)
    wheel = tmp_path / "brainlayer-0.0.0-py3-none-any.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr("brainlayer/__init__.py", "")
        if stamp is not None:
            archive.writestr(ratchet.STAMP_MEMBER, f'BUILD_SHA = "{stamp}"\n')
    return wheel


def linux_probe(tmp_path: Path, **overrides) -> ratchet.Probe:
    """A GitHub ubuntu runner: no socket, no DB, no keg, a freshly built wheel."""
    base = ratchet.Probe(
        os_name="Linux",
        architecture="x86_64",
        hostname="fv-az123-456",
        socket_path=tmp_path / "absent.sock",
        db_path=tmp_path / "absent.db",
        wheel=make_wheel(tmp_path, HEAD),
        head_sha=HEAD,
        tree_dirty=False,
    )
    return replace(base, **overrides)


def mac_probe(tmp_path: Path, **overrides) -> ratchet.Probe:
    """An installed Mac with everything the socket rows need present."""
    sock = tmp_path / "brainbar.sock"
    sock.touch()
    db = tmp_path / "brainlayer.db"
    db.touch()
    ready = {
        "os_name": "Darwin",
        "architecture": "arm64",
        "hostname": CORPUS["latency_baseline_ms"]["hostname"],
        "socket_path": sock,
        "db_path": db,
    }
    return replace(linux_probe(tmp_path), **{**ready, **overrides})


def workflow_document() -> dict:
    return yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))


def workflow_jobs() -> dict[str, dict]:
    return workflow_document()["jobs"]


def workflow_steps(job: str = "table") -> dict[str, dict]:
    """One job's steps parsed as YAML, keyed by name.

    Counting `if: always()` occurrences is not a contract — it says nothing about WHICH step carries
    it, and it rejected the correct fix. These assertions read the real conditions.
    """
    return {step["name"]: step for step in workflow_jobs()[job]["steps"] if "name" in step}


def signature_report(**overrides) -> ratchet.SignatureReport:
    base = {
        "status": ratchet.SIGNATURE_MEASURED,
        "valid": 442,
        "invalid": 0,
        "keg": "brainlayer 1.5.11",
        "runner": "macos-15 · arm64",
        "install_outcome": "success",
    }
    return ratchet.SignatureReport(**{**base, **overrides})


def write_report(tmp_path: Path, payload) -> Path:
    path = tmp_path / "signature.json"
    path.write_text(payload if isinstance(payload, str) else json.dumps(payload), encoding="utf-8")
    return path


def workflow_code() -> str:
    """The workflow with whole-line comments stripped.

    These assertions are about what the job DOES. A comment explaining why `set +e` and
    `$(ls dist/*.whl)` were removed must not read as the job still using them.
    """
    lines = WORKFLOW.read_text(encoding="utf-8").splitlines()
    return "\n".join(line for line in lines if not line.lstrip().startswith("#"))


def row(rows: list[ratchet.Row], name: str) -> ratchet.Row:
    (found,) = [item for item in rows if item.name == name]
    return found


# --- provenance: the one row a runner can actually measure ----------------------------------


def test_provenance_is_green_when_the_wheel_stamp_matches_head(tmp_path: Path) -> None:
    result = ratchet.row_provenance(linux_probe(tmp_path), CORPUS)
    assert result.status == ratchet.GREEN
    assert HEAD[:12] in result.value


def test_provenance_is_red_when_the_wheel_ships_no_stamp(tmp_path: Path) -> None:
    probe = linux_probe(tmp_path, wheel=make_wheel(tmp_path / "bare", stamp=None))
    result = ratchet.row_provenance(probe, CORPUS)
    assert result.status == ratchet.RED
    assert ratchet.STAMP_MEMBER in result.value


def test_provenance_is_red_when_the_wheel_is_unreadable(tmp_path: Path) -> None:
    # A truncated wheel used to crash the collector, costing the whole table (Macroscope, #752).
    corrupt = tmp_path / "corrupt" / "brainlayer-0.0.0-py3-none-any.whl"
    corrupt.parent.mkdir(parents=True)
    corrupt.write_bytes(b"PK\x03\x04 not actually a zip")
    result = ratchet.row_provenance(linux_probe(tmp_path, wheel=corrupt), CORPUS)
    assert result.status == ratchet.RED
    # Unreadable is its own finding, never blurred into "ships no stamp".
    assert "could not be read" in result.value and "ships no" not in result.value


def test_provenance_is_red_when_the_stamp_declares_no_sha(tmp_path: Path) -> None:
    odd = tmp_path / "odd" / "brainlayer-0.0.0-py3-none-any.whl"
    odd.parent.mkdir(parents=True)
    with zipfile.ZipFile(odd, "w") as archive:
        archive.writestr(ratchet.STAMP_MEMBER, "BUILD_SHA = None\n")
    result = ratchet.row_provenance(linux_probe(tmp_path, wheel=odd), CORPUS)
    assert result.status == ratchet.RED
    assert "declares no 40-hex BUILD_SHA" in result.value


def test_collect_still_renders_a_table_when_the_wheel_is_unreadable(tmp_path: Path) -> None:
    corrupt = tmp_path / "corrupt2" / "brainlayer-0.0.0-py3-none-any.whl"
    corrupt.parent.mkdir(parents=True)
    corrupt.write_bytes(b"")
    probe = linux_probe(tmp_path, wheel=corrupt)
    table = ratchet.render(ratchet.collect(probe, CORPUS), probe, None, NOW)
    assert table.startswith(ratchet.MARKER)
    assert "1 RED row(s) to clear:" in table


def test_provenance_is_red_when_the_stamp_is_a_different_commit(tmp_path: Path) -> None:
    probe = linux_probe(tmp_path, wheel=make_wheel(tmp_path / "skew", stamp="b" * 40))
    result = ratchet.row_provenance(probe, CORPUS)
    assert result.status == ratchet.RED
    assert "b" * 12 in result.value and HEAD[:12] in result.value


def test_provenance_is_red_when_stamping_dirtied_the_tree(tmp_path: Path) -> None:
    # _build.py is gitignored; if it ever stops being, the release stamps a dirty tree.
    result = ratchet.row_provenance(linux_probe(tmp_path, tree_dirty=True), CORPUS)
    assert result.status == ratchet.RED
    assert "gitignored" in result.value


@pytest.mark.parametrize(
    ("overrides", "expected"),
    [
        ({"wheel": None}, "no packaged wheel"),  # nobody asked this job to package anything
        ({"head_sha": None}, "git HEAD"),
        ({"tree_dirty": None}, "git status"),
    ],
)
def test_provenance_says_na_with_a_reason_when_it_cannot_measure(
    tmp_path: Path, overrides: dict, expected: str
) -> None:
    result = ratchet.row_provenance(linux_probe(tmp_path, **overrides), CORPUS)
    assert result.status == ratchet.NA
    assert result.value.startswith("n/a — ")
    assert expected in result.value


def test_git_ignores_an_inherited_GIT_DIR_from_another_repo(tmp_path: Path, monkeypatch) -> None:
    """An inherited GIT_DIR/GIT_WORK_TREE OVERRIDES `-C`, so HEAD would come from the wrong repo — a
    stamp matching it is a false GREEN, one that does not is a false RED. Same guard as
    tests/test_build_sha.py and src/brainlayer/deploy_drift.py.

    The decoy is built with GIT_* stripped for the same reason the code under test strips it. An
    earlier draft of this test inherited the pre-push hook's GIT_DIR and committed into the REAL
    repo — it fell into the exact footgun it exists to catch. The `--git-dir` assertion below is
    what makes that impossible to repeat silently.
    """
    other = tmp_path / "other"
    other.mkdir()
    # Identity via -c, never via GIT_* env: GIT_DIR/GIT_WORK_TREE would redirect these writes.

    def decoy_git(*args: str) -> str:
        return subprocess.run(
            ["git", "-C", str(other), "-c", "user.name=t", "-c", "user.email=t@e", *args],
            check=True,
            capture_output=True,
            text=True,
            env=_clean_git_env(),
        ).stdout.strip()

    decoy_git("init", "-q")
    # Fail loudly if the decoy is not self-contained, rather than writing into whatever repo we are in.
    git_dir = Path(decoy_git("rev-parse", "--absolute-git-dir")).resolve()
    assert git_dir.is_relative_to(tmp_path.resolve()), f"decoy git dir escaped the tmp dir: {git_dir}"

    decoy_git("commit", "-q", "--allow-empty", "-m", "decoy repo")
    decoy = decoy_git("rev-parse", "HEAD")
    assert re.fullmatch(r"[0-9a-f]{40}", decoy), "decoy repo did not produce a commit"

    monkeypatch.setenv("GIT_DIR", str(git_dir))
    monkeypatch.setenv("GIT_WORK_TREE", str(other))
    head = ratchet.git_head()
    assert head is not None and head != decoy, "provenance read HEAD from the inherited GIT_DIR"


def test_git_tree_dirty_sees_untracked_files_when_show_untracked_is_no(tmp_path: Path, monkeypatch) -> None:
    """`status.showUntrackedFiles=no` would hide an untracked `_build.py` and GREEN a dirty stamp."""
    repo = tmp_path / "repo"
    repo.mkdir()

    def repo_git(*args: str) -> str:
        return subprocess.run(
            ["git", "-C", str(repo), "-c", "user.name=t", "-c", "user.email=t@e", *args],
            check=True,
            capture_output=True,
            text=True,
            env=_clean_git_env(),
        ).stdout.strip()

    repo_git("init", "-q")
    repo_git("commit", "-q", "--allow-empty", "-m", "init")
    repo_git("config", "status.showUntrackedFiles", "no")
    (repo / "src-brainlayer-_build.py").write_text("BUILD_SHA = x\n", encoding="utf-8")
    monkeypatch.setattr(ratchet, "ROOT", repo)
    assert ratchet.git_tree_dirty() is True


def test_git_invalid_pathname_bytes_are_a_capability_gap_not_a_crash(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(*_args, **_kwargs):
        raise UnicodeDecodeError("utf-8", b"\xff", 0, 1, "invalid")

    monkeypatch.setattr(ratchet.subprocess, "run", boom)
    assert ratchet.git_head() is None
    assert ratchet.git_tree_dirty() is None


# --- fail-closed wheel selection: a promised wheel that never arrived is RED, not n/a ---


def test_no_wheel_argument_at_all_is_a_capability_gap(tmp_path: Path) -> None:
    assert ratchet.select_wheel(None, None) == ratchet.WheelSelection()


def test_a_glob_matching_nothing_is_a_finding_not_a_capability_gap(tmp_path: Path) -> None:
    selection = ratchet.select_wheel(None, str(tmp_path / "empty" / "*.whl"))
    assert selection.path is None
    assert selection.problem is not None and "should have produced exactly one" in selection.problem


def test_an_ambiguous_glob_is_a_finding(tmp_path: Path) -> None:
    for name in ("brainlayer-1.0-py3-none-any.whl", "brainlayer-2.0-py3-none-any.whl"):
        (tmp_path / name).write_bytes(b"")
    selection = ratchet.select_wheel(None, str(tmp_path / "*.whl"))
    assert selection.path is None
    assert selection.problem is not None and "needs exactly one" in selection.problem


def test_an_explicit_wheel_that_is_not_there_is_a_finding(tmp_path: Path) -> None:
    selection = ratchet.select_wheel(tmp_path / "gone.whl", None)
    assert selection.problem is not None and "produced none" in selection.problem


def test_wheel_and_wheel_glob_together_is_a_finding(tmp_path: Path) -> None:
    # An explicit wheel must not hide a glob that matched zero or many. The CI job only
    # passes --wheel-glob; both together is an ambiguous promise and stays RED.
    wheel = make_wheel(tmp_path / "stale", HEAD)
    selection = ratchet.select_wheel(wheel, str(tmp_path / "empty" / "*.whl"))
    assert selection.path is None
    assert selection.problem is not None and "mutually exclusive" in selection.problem


def test_main_exits_one_when_both_wheel_flags_are_passed(tmp_path: Path, capsys, monkeypatch) -> None:
    monkeypatch.setattr(ratchet, "git_tree_dirty", lambda: False)
    wheel = make_wheel(tmp_path, HEAD)
    assert ratchet.main(["--wheel", str(wheel), "--wheel-glob", str(tmp_path / "*.whl")]) == 1
    err = capsys.readouterr().err
    assert "::error title=Ratchet RED: provenance::" in err
    assert "mutually exclusive" in err


def test_a_glob_matching_one_wheel_resolves_it(tmp_path: Path) -> None:
    wheel = make_wheel(tmp_path / "one", HEAD)
    assert ratchet.select_wheel(None, str(tmp_path / "one" / "*.whl")) == ratchet.WheelSelection(path=wheel)


def test_a_missing_wheel_makes_provenance_RED_so_the_job_cannot_go_green(tmp_path: Path) -> None:
    # The regression this pins: an empty --wheel used to render `n/a — no packaged wheel`, exit 0,
    # and a GREEN job. The only measurable row would vanish while CI reported success.
    probe = linux_probe(
        tmp_path,
        wheel=None,
        wheel_problem="no wheel matched `dist/*.whl` — the build step should have produced exactly one",
    )
    result = ratchet.row_provenance(probe, CORPUS)
    assert result.status == ratchet.RED
    assert not result.value.startswith("n/a")


def test_main_exits_one_when_the_wheel_glob_matches_nothing(tmp_path: Path, capsys, monkeypatch) -> None:
    monkeypatch.setattr(ratchet, "git_tree_dirty", lambda: False)
    assert ratchet.main(["--wheel-glob", str(tmp_path / "nothing" / "*.whl")]) == 1
    captured = capsys.readouterr()
    assert "::error title=Ratchet RED: provenance::" in captured.err
    assert "1 RED row(s) to clear:" in captured.out


def test_a_glob_matching_a_non_file_is_a_finding(tmp_path: Path) -> None:
    named_like_a_wheel = tmp_path / "brainlayer-0.0.0-py3-none-any.whl"
    named_like_a_wheel.mkdir()
    selection = ratchet.select_wheel(None, str(tmp_path / "*.whl"))
    assert selection.path is None
    assert selection.problem is not None and "not a file" in selection.problem


def test_a_selected_wheel_that_is_gone_is_red_not_na(tmp_path: Path) -> None:
    gone = tmp_path / "vanished.whl"
    probe = linux_probe(tmp_path, wheel=gone, wheel_problem=None)
    result = ratchet.row_provenance(probe, CORPUS)
    assert result.status == ratchet.RED
    assert not result.value.startswith("n/a")
    assert "not a file" in result.value


# --- the four rows a runner cannot measure ---------------------------------------------------


@pytest.mark.parametrize("name", ["mapped bytes", "search p50/p95", "idle CPU"])
def test_socket_rows_blame_the_served_stack_not_the_operating_system(tmp_path: Path, name: str) -> None:
    """These three are FIXTURE-bound, not runner-bound.

    The old reason led with "runner is Linux/x86_64", which read as though a macOS runner would turn
    them green. It would not: they need the BrainBar daemon, its hybrid helper and the indexed
    corpus, and no GitHub-hosted runner of any OS has those. The reason has to say the true blocker
    first, or the table teaches the reader something false.
    """
    result = row(ratchet.collect(linux_probe(tmp_path), CORPUS), name)
    assert result.status == ratchet.NA
    assert result.value.startswith("n/a — no BrainBar daemon at ")
    assert "hybrid helper" in result.value and "indexed corpus" in result.value
    assert "no GitHub-hosted runner" in result.value
    # ...and it points at what WOULD give them one, rather than stopping at the complaint.
    assert "self-hosted" in result.value and "installed Mac" in result.value


@pytest.mark.parametrize("name", ["mapped bytes", "search p50/p95", "idle CPU"])
def test_socket_rows_do_not_lead_with_the_runner_os(tmp_path: Path, name: str) -> None:
    # Pins the ordering, not just the wording: putting the OS check back first would restore the
    # exact misreading ("move to macOS and these go green") that w13 exists to correct.
    result = row(ratchet.collect(linux_probe(tmp_path), CORPUS), name)
    assert not result.value.startswith("n/a — runner is Linux")


def test_the_machine_target_reason_survives_for_an_off_target_mac(tmp_path: Path) -> None:
    # The OS/arch predicate is demoted, not deleted: a served stack on Darwin/x86_64 is still off
    # the gate's target, and the row must say so rather than printing a number from the wrong machine.
    probe = mac_probe(tmp_path, architecture="x86_64")
    result = ratchet.row_idle_cpu(probe, CORPUS)
    assert result.status == ratchet.NA
    assert "the gate's machine target is Darwin/arm64" in result.value


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("mapped bytes", "no runner-side collector for mapped bytes"),
        ("search p50/p95", "no runner-side collector for search latency"),
        ("idle CPU", "no runner-side collector for idle CPU"),
    ],
)
def test_socket_rows_fall_through_to_the_missing_collector_on_a_ready_mac(
    tmp_path: Path, name: str, expected: str
) -> None:
    # Everything the row needs is present, so the reason must be the honest one: nobody wrote the
    # collector yet. A row that blamed the machine here would be hiding w13's whole point.
    result = row(ratchet.collect(mac_probe(tmp_path), CORPUS), name)
    assert result.status == ratchet.NA
    assert expected in result.value
    # It must not promise w13 as the fix any more — w13 shipped, and it deliberately did NOT build
    # these three, because a synthetic corpus measures a different thing under a different method.
    assert "w13" not in result.value
    assert "different method label" in result.value


# --- signature_valid: measured by the macOS parity job, handed here as a report ----------------


def test_signature_row_is_green_when_the_parity_job_found_nothing_invalid(tmp_path: Path) -> None:
    probe = linux_probe(tmp_path, signature=signature_report())
    result = ratchet.row_signature_valid(probe, CORPUS)
    assert result.status == ratchet.GREEN
    assert "442 valid / 0 invalid" in result.value
    assert "brainlayer 1.5.11" in result.value and "macos-15" in result.value
    assert result.method == ratchet.SIGNATURE_METHOD_MEASURED


def test_signature_row_is_red_when_the_parity_job_found_an_invalid_file(tmp_path: Path) -> None:
    probe = linux_probe(
        tmp_path,
        signature=signature_report(
            valid=430,
            invalid=2,
            invalid_files=("cramjam.cpython-313-darwin.so: invalid signature", "PIL/.dylibs/libjpeg.dylib: bad"),
        ),
    )
    result = ratchet.row_signature_valid(probe, CORPUS)
    assert result.status == ratchet.RED
    assert "430 valid / 2 invalid" in result.value
    assert "cramjam" in result.value  # the reader can act without opening the run log
    assert not result.value.startswith("n/a")


def test_a_red_signature_row_fails_the_collector(tmp_path: Path, monkeypatch, capsys) -> None:
    # The whole point of measuring it: `invalid > 0` has to fail the job, exactly as a RED
    # provenance row does. A row nobody has to clear is decoration.
    monkeypatch.setattr(ratchet, "git_tree_dirty", lambda: False)
    wheel = make_wheel(tmp_path, ratchet.git_head() or HEAD)
    report = write_report(
        tmp_path,
        {"status": "measured", "valid": 1, "invalid": 1, "keg": "brainlayer 1.5.11", "runner": "macos-15"},
    )
    assert ratchet.main(["--wheel", str(wheel), "--signature-report", str(report)]) == 1
    assert "::error title=Ratchet RED: signature_valid::" in capsys.readouterr().err


def test_signature_row_is_red_when_the_parity_job_could_not_measure(tmp_path: Path) -> None:
    # The job RAN and produced no measurement. Per the rule this table is built on, that is a
    # finding to clear, never `n/a` — `n/a` means the machine cannot measure it at all.
    probe = linux_probe(
        tmp_path,
        signature=ratchet.SignatureReport(
            status=ratchet.SIGNATURE_FAILED,
            stage="brew install etanhey/layers/brainlayer",
            detail="the keg did not install",
        ),
    )
    result = ratchet.row_signature_valid(probe, CORPUS)
    assert result.status == ratchet.RED
    assert "brew install" in result.value and "the keg did not install" in result.value
    assert not result.value.startswith("n/a")


def test_a_measured_report_that_verified_nothing_is_a_finding(tmp_path: Path) -> None:
    # A keg with zero native extensions cannot prove signatures are valid; reporting GREEN off an
    # empty sweep is the same false green as an empty --wheel used to be.
    probe = linux_probe(tmp_path, signature=signature_report(valid=0, invalid=0))
    result = ratchet.row_signature_valid(probe, CORPUS)
    assert result.status == ratchet.RED
    assert "no native extensions" in result.value


def test_signature_row_says_na_with_the_gate_reason_when_the_job_did_not_run(tmp_path: Path) -> None:
    reason = "this PR touches no release or signing path and carries no `ratchet:signatures` label"
    probe = linux_probe(tmp_path, signature_unavailable=reason)
    result = ratchet.row_signature_valid(probe, CORPUS)
    assert result.status == ratchet.NA
    assert result.value == f"n/a — {reason}"
    assert result.method == ratchet.SIGNATURE_METHOD_NA


def test_signature_row_names_a_reason_even_when_nobody_passed_one(tmp_path: Path) -> None:
    result = ratchet.row_signature_valid(linux_probe(tmp_path), CORPUS)
    assert result.status == ratchet.NA
    assert result.value.startswith("n/a — ") and len(result.value) > len("n/a — ") + 10


def test_the_measured_and_unmeasured_methods_are_different_labels() -> None:
    # A keg installed by a hosted runner and a keg installed on Etan's Mac are different
    # measurements, not better and worse ones. Sharing a method label would blur them.
    assert ratchet.SIGNATURE_METHOD_MEASURED != ratchet.SIGNATURE_METHOD_NA
    assert "runner" in ratchet.SIGNATURE_METHOD_MEASURED
    assert "runner" not in ratchet.SIGNATURE_METHOD_NA


# --- fail-closed signature reports: a promised report that never arrived is RED, not n/a -------


def test_no_signature_argument_at_all_is_a_capability_gap() -> None:
    selection = ratchet.select_signature(None, None)
    assert selection.report is None and selection.problem is None
    assert selection.unavailable is not None


def test_a_promised_signature_report_that_is_missing_is_a_finding(tmp_path: Path) -> None:
    selection = ratchet.select_signature(tmp_path / "gone.json", None)
    assert selection.report is None
    assert selection.problem is not None and "is not a file" in selection.problem


def test_an_unparseable_signature_report_is_a_finding(tmp_path: Path) -> None:
    selection = ratchet.select_signature(write_report(tmp_path, "{not json"), None)
    assert selection.problem is not None and "could not be read" in selection.problem


def test_a_signature_report_with_an_unknown_status_is_a_finding(tmp_path: Path) -> None:
    selection = ratchet.select_signature(write_report(tmp_path, {"status": "probably fine"}), None)
    assert selection.problem is not None and "status" in selection.problem


@pytest.mark.parametrize(
    "payload",
    [
        {"status": "measured", "valid": "many", "invalid": 0},
        {"status": "measured", "valid": 1},
        {"status": "measured", "valid": -1, "invalid": 0},
    ],
)
def test_a_measured_report_without_honest_counts_is_a_finding(tmp_path: Path, payload: dict) -> None:
    # The counts ARE the value cell. A report that cannot supply them must not degrade into `n/a`
    # while the job stays green.
    selection = ratchet.select_signature(write_report(tmp_path, payload), None)
    assert selection.report is None
    assert selection.problem is not None and "counts" in selection.problem


def test_a_failed_report_must_still_say_what_failed(tmp_path: Path) -> None:
    selection = ratchet.select_signature(write_report(tmp_path, {"status": "failed"}), None)
    assert selection.problem is not None and "stage" in selection.problem


def test_signature_report_and_unavailable_together_is_a_finding(tmp_path: Path) -> None:
    # Same ambiguity the wheel flags have: a stale report must not hide a job that never ran.
    selection = ratchet.select_signature(
        write_report(tmp_path, {"status": "failed", "stage": "x", "detail": "y"}), "no"
    )
    assert selection.report is None
    assert selection.problem is not None and "mutually exclusive" in selection.problem


def test_main_reads_a_measured_report_end_to_end(tmp_path: Path, capsys, monkeypatch) -> None:
    monkeypatch.setattr(ratchet, "git_tree_dirty", lambda: False)
    wheel = make_wheel(tmp_path, ratchet.git_head() or HEAD)
    report = write_report(
        tmp_path,
        {"status": "measured", "valid": 442, "invalid": 0, "keg": "brainlayer 1.5.11", "runner": "macos-15 · arm64"},
    )
    assert ratchet.main(["--wheel", str(wheel), "--signature-report", str(report)]) == 0
    assert "442 valid / 0 invalid" in capsys.readouterr().out


def test_main_exits_one_when_a_promised_signature_report_is_missing(tmp_path: Path, capsys, monkeypatch) -> None:
    monkeypatch.setattr(ratchet, "git_tree_dirty", lambda: False)
    wheel = make_wheel(tmp_path, ratchet.git_head() or HEAD)
    assert ratchet.main(["--wheel", str(wheel), "--signature-report", str(tmp_path / "absent.json")]) == 1
    assert "::error title=Ratchet RED: signature_valid::" in capsys.readouterr().err


def test_search_latency_refuses_an_uncalibrated_host(tmp_path: Path) -> None:
    result = ratchet.row_search_latency(mac_probe(tmp_path, hostname="some-other-mac.local"), CORPUS)
    assert "is not the calibrated baseline host" in result.value


def test_every_row_is_measured_or_says_why_not(tmp_path: Path) -> None:
    for result in ratchet.collect(linux_probe(tmp_path), CORPUS):
        assert result.status in {ratchet.GREEN, ratchet.RED, ratchet.NA}
        assert result.value.strip()
        assert result.method.strip()
        if result.status == ratchet.NA:
            # "n/a" alone is a blank cell wearing a hat.
            assert result.value.startswith("n/a — ") and len(result.value) > len("n/a — ") + 10


def test_quoted_baselines_never_leak_into_a_value_cell(tmp_path: Path) -> None:
    # The rule that matters: never print a number the runner did not measure. Baselines live in
    # Notes, attributed to their own machine and date.
    for result in ratchet.collect(linux_probe(tmp_path), CORPUS):
        if result.status == ratchet.NA:
            assert "26.2" not in result.value
            assert str(CORPUS["latency_baseline_ms"]["p50"]) not in result.value
            assert "442" not in result.value  # the signature baseline lives in Notes, with its machine
    notes = row(ratchet.collect(linux_probe(tmp_path), CORPUS), "mapped bytes").notes
    assert "26.2 GB" in notes and "installed Mac" in notes and "Not measured by this run." in notes
    signature_notes = row(ratchet.collect(linux_probe(tmp_path), CORPUS), "signature_valid").notes
    assert "442 valid / 0 invalid" in signature_notes
    assert "M4 Max" in signature_notes and "2026-09-03" in signature_notes


# --- rendering and exit code -----------------------------------------------------------------


def test_render_starts_with_the_marker_the_workflow_greps_for(tmp_path: Path) -> None:
    probe = linux_probe(tmp_path)
    table = ratchet.render(ratchet.collect(probe, CORPUS), probe, "https://example/run/1", NOW)
    assert table.startswith(ratchet.MARKER)
    assert ratchet.MARKER in WORKFLOW.read_text(encoding="utf-8")


def test_render_emits_one_table_row_per_ratchet_row(tmp_path: Path) -> None:
    probe = linux_probe(tmp_path)
    rows = ratchet.collect(probe, CORPUS)
    table = ratchet.render(rows, probe, None, NOW)
    body = [line for line in table.splitlines() if line.startswith("| ") and not line.startswith("| ---")]
    assert len(body) == len(rows) + 1  # header + one per row
    for result in rows:
        assert f"| {result.name} |" in table
    assert "No RED rows." in table


def test_render_calls_out_red_rows_for_the_author_to_clear(tmp_path: Path) -> None:
    probe = linux_probe(tmp_path, wheel=make_wheel(tmp_path / "skew", stamp="b" * 40))
    table = ratchet.render(ratchet.collect(probe, CORPUS), probe, None, NOW)
    assert "1 RED row(s) to clear:" in table
    assert "`provenance`" in table


def test_render_escapes_pipes_so_a_reason_cannot_break_the_table(tmp_path: Path) -> None:
    probe = linux_probe(tmp_path)
    rows = [ratchet.Row("x", ratchet.NA, "n/a — a | b", "m | n", "notes | here")]
    line = [item for item in ratchet.render(rows, probe, None, NOW).splitlines() if item.startswith("| x |")][0]
    assert len(re.findall(r"(?<!\\)\|", line)) == 6  # 5 cells worth of delimiters, none injected
    assert line.count(r"\|") == 3


def test_main_exits_zero_when_rows_are_only_na_or_green(tmp_path: Path, capsys, monkeypatch) -> None:
    out = tmp_path / "table.md"
    monkeypatch.setattr(ratchet, "git_tree_dirty", lambda: False)
    wheel = make_wheel(tmp_path, ratchet.git_head() or HEAD)
    assert ratchet.main(["--wheel", str(wheel), "--out", str(out)]) == 0
    assert out.read_text(encoding="utf-8").startswith(ratchet.MARKER)
    assert "n/a — " in capsys.readouterr().out


def test_main_exits_one_and_annotates_when_a_row_is_red(tmp_path: Path, capsys, monkeypatch) -> None:
    monkeypatch.setattr(ratchet, "git_tree_dirty", lambda: False)
    wheel = make_wheel(tmp_path, "c" * 40)
    assert ratchet.main(["--wheel", str(wheel)]) == 1
    assert "::error title=Ratchet RED: provenance::" in capsys.readouterr().err


# --- the workflow that carries it -------------------------------------------------------------


def test_workflow_can_write_the_comment_and_cannot_double_post() -> None:
    workflow = WORKFLOW.read_text(encoding="utf-8")
    assert "pull-requests: write" in workflow
    assert "concurrency:" in workflow and "ratchet-${{ github.event.pull_request.number }}" in workflow
    # find-then-update, create only if absent
    assert re.search(r"comment_id=", workflow) and "-X PATCH" in workflow and "-X POST" in workflow
    assert "head.repo.fork" in workflow  # fork PRs are handled, not silently broken
    assert "dependabot[bot]" in workflow  # same read-only token as forks


def test_dependabot_prs_do_not_attempt_the_comment() -> None:
    post = workflow_steps()["Post or refresh the one ratchet comment"]["if"]
    skip = workflow_steps()["Skip the comment on fork or Dependabot PRs (read-only token)"]["if"]
    assert "dependabot[bot]" in post and "!=" in post
    assert "dependabot[bot]" in skip


def test_the_workflow_stamps_the_sha_that_provenance_compares_against() -> None:
    """The provenance row is only meaningful if the job stamps the wheel with THIS checkout's HEAD.

    Nothing else pins that: change the stamp step to write a constant and every row-level test still
    passes while the live table reports a GREEN it never earned.
    """
    stamp_step = workflow_steps()["Stamp build sha and build the wheel"]["run"]
    assert 'head="$(git rev-parse HEAD)"' in stamp_step
    assert f"> {ratchet.STAMP_MEMBER.replace('brainlayer/', 'src/brainlayer/')}" in stamp_step
    assert 'printf \'BUILD_SHA = "%s"\\n\' "$head"' in stamp_step
    # ...and the collector must read that wheel through the fail-closed glob, not a shell expansion.
    assert "--wheel-glob 'dist/*.whl'" in workflow_steps()["Collect ratchet rows"]["run"]


def test_collect_runs_even_when_an_earlier_step_failed() -> None:
    """A build failure must not leave the previous commit's GREEN table standing on the PR.

    Without this, GitHub skips collect, which skips the comment steps, which leaves a GREEN
    provenance row this run never measured — the same crime as printing an unmeasured number,
    committed by omission.
    """
    assert workflow_steps()["Collect ratchet rows"]["if"] == "${{ !cancelled() }}"


@pytest.mark.parametrize(
    "step",
    [
        "Collect ratchet rows",
        "Guarantee this run publishes its own table",
        "Publish the table to the run summary",
        "Post or refresh the one ratchet comment",
        "Fail on a RED row",
    ],
)
def test_no_writer_runs_after_this_job_is_cancelled(step: str) -> None:
    """`always()` is TRUE for cancelled jobs, so a superseded run would still reach the comment step
    and could PATCH its stale table over the newer run's. `!cancelled()` still runs on failure."""
    condition = workflow_steps()[step]["if"]
    assert "!cancelled()" in condition
    assert "always()" not in condition


def test_a_run_that_measured_nothing_still_replaces_the_previous_table() -> None:
    guarantee = workflow_steps()["Guarantee this run publishes its own table"]["run"]
    assert ratchet.MARKER in guarantee
    assert "does not apply to this commit" in guarantee
    # It must not be gated on collect having produced output — that is exactly the skipped case.
    assert "steps.collect.outputs.rc" not in workflow_steps()["Guarantee this run publishes its own table"]["if"]


def test_the_comment_is_posted_on_a_red_run_not_only_a_green_one() -> None:
    condition = workflow_steps()["Post or refresh the one ratchet comment"]["if"]
    assert "success()" not in condition  # would silence RED tables entirely


def test_a_red_row_can_still_fail_the_job() -> None:
    fail_step = workflow_steps()["Fail on a RED row"]
    assert "steps.collect.outputs.rc != '0'" in fail_step["if"]
    assert "exit 1" in fail_step["run"]


def test_collect_records_the_exit_code_instead_of_dying_on_it() -> None:
    collect = workflow_steps()["Collect ratchet rows"]["run"]
    assert "|| rc=$?" in collect and 'echo "rc=$rc" >> "$GITHUB_OUTPUT"' in collect
    # `set +e` must never come back: it would swallow a crash before rc is ever read.
    assert "set +e" not in workflow_code()


def test_workflow_resolves_the_wheel_fail_closed_not_through_a_shell_glob() -> None:
    # `--wheel "$(ls dist/*.whl)"` made zero-or-many wheels an empty --wheel, which rendered `n/a`
    # and exited 0: the one measurable row vanished and the job went green anyway.
    workflow = workflow_code()
    assert "--wheel-glob 'dist/*.whl'" in workflow
    assert "$(ls dist/" not in workflow


def test_workflow_never_pipes_a_paginated_stream_into_head() -> None:
    # SIGPIPE under `pipefail` would fail the step on the PATCH path -- the refresh this job exists for.
    workflow = workflow_code()
    assert "head -n1" not in workflow
    assert '--paginate > "${RUNNER_TEMP}/comments.json"' in workflow
    # The gate lists changed files the same way, and must not learn the footgun either.
    assert '''--jq '.[].filename' > "$changed"''' in workflow


# --- the macOS signature-parity job -----------------------------------------------------------


def test_the_signature_job_runs_on_a_macos_arm64_runner() -> None:
    """`codesign` needs macOS, and the gate's machine target is arm64.

    macos-15 is GitHub's arm64 image. An x86_64 image would install a different keg — different
    wheels, different relocation — and the row would be measuring something else under this label.
    """
    assert workflow_jobs()["signatures"]["runs-on"] == "macos-15"


def test_the_signature_job_verifies_the_keg_with_the_release_script() -> None:
    # The same script AGENTS.md makes mandatory at release time. A CI-only reimplementation would
    # be a second definition of "valid", free to drift from the one that gates releases.
    verify = workflow_steps("signatures")["Codesign-verify every native extension in the keg"]["run"]
    assert "scripts/release-verify-signatures.sh" in verify
    install = workflow_steps("signatures")["Install the published BrainLayer keg from the tap"]["run"]
    assert "brew install etanhey/layers/brainlayer" in install


def test_the_signature_job_is_trigger_gated_and_not_charged_to_every_pr() -> None:
    """A GitHub macOS runner bills at ~10x Linux minutes and this job builds a venv from source.

    Running it on every PR would spend that on PRs that cannot possibly change a signature. The
    gate decides; this pins that the decision is actually wired to the job.
    """
    signatures = workflow_jobs()["signatures"]
    assert signatures["needs"] == "gate"
    assert "needs.gate.outputs.signatures == 'true'" in signatures["if"]


def test_the_gate_opts_in_on_release_paths_and_on_the_label() -> None:
    decide = workflow_steps("gate")["Decide whether this PR pays for a macOS runner"]["run"]
    for path in ("pyproject.toml", "scripts/release-", "publish", "ratchet"):
        assert path in decide
    assert "ratchet:signatures" in decide


def test_a_label_added_after_the_pr_opened_actually_re_runs_the_workflow() -> None:
    # Without `labeled`, the opt-in label would sit on the PR doing nothing until the next push --
    # an escape hatch that silently is not one.
    types = workflow_document()[True]["pull_request"]["types"]
    assert "labeled" in types
    # ...and the defaults GitHub would have supplied must be listed back explicitly, or naming
    # `types` at all would stop the workflow running on pushes.
    assert {"opened", "synchronize", "reopened"} <= set(types)


def test_the_signature_job_publishes_its_report_even_when_the_install_failed() -> None:
    """An install that died still has to reach the table, as a finding.

    If the verify step were skipped, the table would print `n/a` for a job that RAN -- the exact
    fail-open this table's second rule forbids.
    """
    verify = workflow_steps("signatures")["Codesign-verify every native extension in the keg"]
    assert "!cancelled()" in verify["if"] and "always()" not in verify["if"]
    assert "steps.install.outcome" in verify["env"]["INSTALL_OUTCOME"]
    assert '"failed"' in verify["run"]


def test_the_sweep_runs_on_an_installed_keg_whatever_brew_exited_with() -> None:
    """The bug this workflow's first run found, and it is the interesting one.

    Homebrew `ofail`s a relocation failure -- `MachO::HeaderPadError` on cramjam, homebrew-layers
    #37 -- so `brew install` exits 1 while STILL installing the keg and running the `post_install`
    codesign sweep. Gating the verification on the install step's exit code therefore refused to
    measure a keg that was sitting right there, and reported `could not measure` for the exact
    post-relocation state this row exists to check.

    What decides measurability is whether there is a native-extension root to sweep.
    """
    verify = workflow_steps("signatures")["Codesign-verify every native extension in the keg"]["run"]
    assert 'if [[ -z "$keg" || ! -d "$keg/libexec/venv" ]]; then' in verify
    # The install outcome may only shape the message and the report field -- never skip the sweep.
    guard = verify.split("release-verify-signatures.sh")[0]
    assert '"$INSTALL_OUTCOME" != "success"' in guard
    assert "::warning title=brew install exited non-zero" in guard
    assert "--arg install_outcome" in verify


def test_a_clean_sweep_after_a_brew_ofail_is_green_but_says_so(tmp_path: Path) -> None:
    # GREEN because the signatures ARE valid, and a clean sweep after an aborted relocation is the
    # #37 post_install fix working. Silent, though, it would read as an unremarkable install.
    probe = linux_probe(tmp_path, signature=signature_report(install_outcome="failure"))
    result = ratchet.row_signature_valid(probe, CORPUS)
    assert result.status == ratchet.GREEN
    assert "442 valid / 0 invalid" in result.value
    assert "exited non-zero" in result.value and "failure" in result.value


def test_a_successful_install_adds_no_noise_to_the_value(tmp_path: Path) -> None:
    result = ratchet.row_signature_valid(
        linux_probe(tmp_path, signature=signature_report(install_outcome="success")), CORPUS
    )
    assert result.value == "442 valid / 0 invalid · brainlayer 1.5.11 · macos-15 · arm64"


def test_no_step_leaves_scratch_in_the_checkout() -> None:
    """An untracked file in the workspace makes the tree dirty, which turns PROVENANCE RED.

    Not hypothetical: this workflow's first run wrote `signature-args.txt` and
    `signature-report.json` into the checkout and the ratchet caught its own author with
    `stamping dirtied the tree`. Scratch belongs in $RUNNER_TEMP. The stamp step is the one
    exception, and only because `src/brainlayer/_build.py` and `dist/` are gitignored.
    """
    scratch = (
        "ratchet.md",
        "signature-report.json",
        "signature-args.txt",
        "comments.json",
        "body.json",
        "changed-files.txt",
        "verify.out",
        "verify.err",
        "invalid-files.txt",
    )
    code = workflow_code()
    for name in scratch:
        # Trailing boundary, or `verify.out` matches inside `steps.verify.outputs.report`.
        for match in re.finditer(re.escape(name) + r"(?![A-Za-z0-9_])", code):
            prefix = code[max(0, match.start() - 18) : match.start()]
            assert "RUNNER_TEMP}/" in prefix, f"`{name}` is used without a $RUNNER_TEMP prefix"


def test_no_redirect_targets_a_relative_path_in_the_checkout() -> None:
    # Belt to the braces above: catches a NEW scratch filename nobody thought to list.
    allowed = {"src/brainlayer/_build.py", "/dev/null"}
    for job in ("gate", "signatures", "table"):
        for name, step in workflow_steps(job).items():
            for raw in re.findall(r">>?\s*(\S+)", step.get("run", "")):
                target = raw.strip('";')
                looks_like_a_path = re.fullmatch(r"[A-Za-z0-9_./-]+", target) and ("/" in target or "." in target)
                if not looks_like_a_path or target.startswith("$") or target in allowed:
                    continue
                assert target.startswith("/"), f"{job} :: {name} writes `{target}` into the checkout"


def test_the_signature_job_fails_on_anything_but_a_clean_measurement() -> None:
    """`invalid > 0` AND `could not measure` both have to fail this job.

    Keying the failure off the count alone let the second case through: an install that worked but
    a verifier that exited without counts published a `failed` report and left this job green,
    which reads as a pass. AGENTS.md blocks release on an invalid `*.so`/`*.dylib`; a sweep that
    never happened proves no less than nothing.
    """
    fail_step = workflow_steps("signatures")["Fail this job on an invalid signature"]
    assert "steps.verify.outputs.verdict != 'clean'" in fail_step["if"]
    assert "exit 1" in fail_step["run"]
    # ...and the two cases are distinguishable in the log, not collapsed into one message.
    assert "Invalid keg signature" in fail_step["run"] and "Signatures unmeasured" in fail_step["run"]
    verify = workflow_steps("signatures")["Codesign-verify every native extension in the keg"]["run"]
    assert "verdict=unmeasured" in verify  # the default, so a step that falls over cannot pass
    assert "verdict=clean" in verify and "verdict=invalid" in verify


def test_the_table_job_waits_for_the_signature_job_without_depending_on_it_running() -> None:
    """`needs` orders them; the job-level `if` is what lets the table publish anyway.

    A skipped `signatures` job would otherwise skip `table` too, leaving the previous commit's
    table standing on the PR -- the same crime as printing an unmeasured number, by omission.
    """
    table = workflow_jobs()["table"]
    assert set(table["needs"]) == {"gate", "signatures"}
    assert "!cancelled()" in table["if"] and "always()" not in table["if"]


@pytest.mark.parametrize("job", ["gate", "signatures", "table"])
def test_no_job_inherits_a_cancelled_run(job: str) -> None:
    condition = workflow_jobs()[job].get("if", "")
    assert "always()" not in condition


def test_a_gate_that_never_decided_is_a_finding_not_a_silent_na() -> None:
    """If the gate failed we do not KNOW whether signatures should have been measured.

    Rendering `n/a -- not triggered` there would state something this run never established. The
    table job hands the collector a `failed` report instead, which renders RED.
    """
    step = workflow_steps()["Hand the signature measurement to the collector"]
    hand_off = step["run"]
    assert "needs.gate.result" in step["env"]["GATE_RESULT"]
    assert '"failed"' in hand_off and "trigger gate" in hand_off
    # ...and a signatures job that ran and published nothing is the same kind of finding.
    assert "published no report" in hand_off


def test_the_collector_is_handed_exactly_one_signature_flag() -> None:
    collect = workflow_steps()["Collect ratchet rows"]["run"]
    assert '"${SIGNATURE_ARGS[@]}"' in collect
    hand_off = workflow_steps()["Hand the signature measurement to the collector"]["run"]
    assert "--signature-report" in hand_off and "--signature-unavailable" in hand_off


def test_the_signature_args_are_read_without_bash_4() -> None:
    # `mapfile` is bash >= 4 and macOS ships bash 3.2, so it would break the signatures job and
    # anyone reproducing a step locally on a Mac. A `read` loop is equivalent and portable.
    assert "mapfile" not in workflow_code()
    collect = workflow_steps()["Collect ratchet rows"]["run"]
    assert "while IFS= read -r" in collect and 'SIGNATURE_ARGS+=("$signature_arg")' in collect


def test_the_gate_never_pipes_a_paginated_stream_into_head() -> None:
    # Same SIGPIPE-under-pipefail footgun as the comment step: `gh api --paginate` lands in a file.
    decide = workflow_steps("gate")["Decide whether this PR pays for a macOS runner"]["run"]
    assert "--paginate" in decide
    assert "| head" not in decide
