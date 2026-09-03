#!/usr/bin/env python3
"""Render the bench-brain ratchet table that CI posts as ONE sticky comment per PR.

The whole point of this file is the rule it enforces: **never print a number the runner did not
measure.** Every row resolves to exactly one of

* a value this run measured, marked ``GREEN`` (within budget) or ``RED`` (a finding the PR author
  must clear -- the collector exits 1 so the job goes red with it), or
* ``n/a -- <specific reason>``, naming the first capability the row needs that this machine does
  not have.

A blank cell, a zero, a figure copied from yesterday, or a plausible-looking invented number is a
defect in this script, not a cosmetic issue. Baselines measured elsewhere may appear in the
**Notes** column ONLY with their machine, method and date attached, and never in the value column.

Rows are labelled by method, because a socket measurement on an installed Mac, a keg installed by
a hosted macOS runner, and an in-process measurement on a GitHub runner are three different
measurements, not better and worse ones.

Two rows have a runner-side method. ``provenance`` measures in-process on the ubuntu job.
``signature_valid`` is measured by the separate macOS parity job in ``ratchet.yml``, which installs
the published tap keg and runs ``scripts/release-verify-signatures.sh`` against it; that job hands
its counts here as a JSON report (``--signature-report``).

The remaining three -- mapped bytes, search latency, idle CPU -- are **fixture-bound, not
runner-bound**. They need the BrainBar daemon, its hybrid helper and the indexed corpus running
together, which no GitHub-hosted runner of any OS provides. Their ``n/a`` reasons say exactly that,
because the earlier wording ("runner is Linux/x86_64") read as though a macOS runner would turn them
green. It would not. A synthetic mini-corpus would measure a *different thing* and would owe the
table a different method label, never a borrowed one.
"""

from __future__ import annotations

import argparse
import glob as glob_module
import json
import os
import platform
import re
import shutil
import socket as socket_module
import subprocess
import sys
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "tests" / "fixtures" / "sprint_gate" / "corpus.json"

# The sticky-comment anchor. .github/workflows/ratchet.yml greps for this exact string to decide
# update-vs-create, so one PR can only ever carry one of these comments; tests pin them together.
MARKER = "<!-- brainlayer-ratchet-table -->"
STAMP_MEMBER = "brainlayer/_build.py"
STAMP_PATTERN = re.compile(r'^BUILD_SHA = "([0-9a-f]{40})"\s*$')
FALLBACK_DB = Path("~/.local/share/brainlayer/brainlayer.db").expanduser()

# What the macOS parity job may say about itself. `measured` carries counts this run produced;
# `failed` carries the stage that broke. There is deliberately no third value: a job that ran and
# cannot say either of those things is a finding, not a new kind of silence.
SIGNATURE_MEASURED = "measured"
SIGNATURE_FAILED = "failed"
SIGNATURE_STATUSES = (SIGNATURE_MEASURED, SIGNATURE_FAILED)
SIGNATURE_UNAVAILABLE_DEFAULT = (
    "no macOS signature-parity report was handed to this run, and this job cannot install a keg itself"
)

GREEN = "GREEN"
RED = "RED"
NA = "n/a"
BADGE = {GREEN: "🟢 GREEN", RED: "🔴 RED", NA: "⚪ n/a"}


@dataclass(frozen=True)
class Row:
    name: str
    status: str
    value: str
    method: str
    notes: str


@dataclass(frozen=True)
class Probe:
    """Everything a row is allowed to know about the machine it is running on."""

    os_name: str
    architecture: str
    hostname: str
    socket_path: Path
    db_path: Path
    wheel: Path | None
    head_sha: str | None
    tree_dirty: bool | None
    # Set when this job MEANT to hand over a wheel and could not. That is a finding to clear, not a
    # capability this machine lacks, so it renders RED and never `n/a`.
    wheel_problem: str | None = None
    # The macOS parity job's own words, and the same three-way split: a measurement it made, the
    # reason nobody was asked to make one, or a promise it broke.
    signature: SignatureReport | None = None
    signature_unavailable: str | None = None
    signature_problem: str | None = None

    @classmethod
    def detect(
        cls,
        corpus: dict,
        wheel: Path | None,
        wheel_glob: str | None,
        signature_report: Path | None = None,
        signature_unavailable: str | None = None,
    ) -> Probe:
        selected = select_wheel(wheel, wheel_glob)
        signature = select_signature(signature_report, signature_unavailable)
        return cls(
            os_name=platform.system(),
            architecture=platform.machine(),
            hostname=socket_module.gethostname(),
            socket_path=Path(corpus["socket_path"]),
            db_path=canonical_db_path(),
            wheel=selected.path,
            head_sha=git_head(),
            tree_dirty=git_tree_dirty(),
            wheel_problem=selected.problem,
            signature=signature.report,
            signature_unavailable=signature.unavailable,
            signature_problem=signature.problem,
        )


@dataclass(frozen=True)
class WheelSelection:
    """Which wheel the provenance row reads, or why the job failed to name exactly one."""

    path: Path | None = None
    problem: str | None = None


def select_wheel(explicit: Path | None, pattern: str | None) -> WheelSelection:
    """Resolve the wheel FAIL-CLOSED.

    A job that packaged nothing passes neither argument and gets `n/a` — a real capability gap. But a
    job that asked for a wheel and cannot produce exactly one has a **finding**: the only row this
    runner can measure would otherwise vanish into a principled-looking `n/a` while CI stayed green,
    which is the false green this whole file exists to prevent.
    """
    if explicit is not None and pattern is not None:
        return WheelSelection(problem="`--wheel` and `--wheel-glob` are mutually exclusive")
    if explicit is None and pattern is None:
        return WheelSelection()
    if explicit is not None:
        if explicit.is_file():
            return WheelSelection(path=explicit)
        return WheelSelection(
            problem=f"`--wheel {explicit}` is not a file — the build was asked for a wheel and produced none"
        )
    matches = sorted(Path(match) for match in glob_module.glob(pattern or ""))
    if not matches:
        return WheelSelection(problem=f"no wheel matched `{pattern}` — the build step should have produced exactly one")
    if len(matches) > 1:
        listed = ", ".join(f"`{match.name}`" for match in matches)
        return WheelSelection(
            problem=f"{len(matches)} wheels matched `{pattern}` ({listed}) — provenance needs exactly one"
        )
    path = matches[0]
    if not path.is_file():
        return WheelSelection(
            problem=(f"`{path.name}` matched `{pattern}` but is not a file — provenance needs exactly one wheel")
        )
    return WheelSelection(path=path)


@dataclass(frozen=True)
class SignatureReport:
    """What the macOS parity job measured, or the stage at which it could not."""

    status: str
    valid: int | None = None
    invalid: int | None = None
    keg: str = ""
    runner: str = ""
    invalid_files: tuple[str, ...] = ()
    stage: str = ""
    detail: str = ""


@dataclass(frozen=True)
class SignatureSelection:
    """Which of the three states `signature_valid` is in, resolved once and fail-closed."""

    report: SignatureReport | None = None
    unavailable: str | None = None
    problem: str | None = None


def select_signature(report_path: Path | None, unavailable: str | None) -> SignatureSelection:
    """Resolve the signature measurement FAIL-CLOSED, on the same rule the wheel follows.

    `--signature-unavailable <reason>` says nobody was asked to measure -- a real capability gap,
    rendered `n/a` with that reason. `--signature-report <path>` says the macOS parity job RAN and
    owes this run a measurement; anything that stops it arriving is a **finding**, because a job
    that ran and produced nothing is not a machine that cannot measure.
    """
    if report_path is not None and unavailable is not None:
        return SignatureSelection(problem="`--signature-report` and `--signature-unavailable` are mutually exclusive")
    if report_path is None and unavailable is None:
        return SignatureSelection(unavailable=SIGNATURE_UNAVAILABLE_DEFAULT)
    if report_path is None:
        return SignatureSelection(unavailable=unavailable)
    if not report_path.is_file():
        return SignatureSelection(
            problem=(
                f"`--signature-report {report_path}` is not a file — the macOS parity job ran and published no report"
            )
        )
    try:
        payload = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        return SignatureSelection(
            problem=(
                f"the macOS parity job's report could not be read ({type(error).__name__}) — "
                "signatures are unverified, not verified"
            )
        )
    if not isinstance(payload, dict) or payload.get("status") not in SIGNATURE_STATUSES:
        return SignatureSelection(
            problem=(
                "the macOS parity job's report carries no known `status` "
                f"(expected one of {', '.join(SIGNATURE_STATUSES)})"
            )
        )
    if payload["status"] == SIGNATURE_FAILED:
        stage = str(payload.get("stage", "")).strip()
        detail = str(payload.get("detail", "")).strip()
        if not stage or not detail:
            return SignatureSelection(
                problem="the macOS parity job reported a failure without naming the `stage` and `detail` that failed"
            )
        return SignatureSelection(report=SignatureReport(status=SIGNATURE_FAILED, stage=stage, detail=detail))
    valid, invalid = payload.get("valid"), payload.get("invalid")
    # bool is an int in Python; a `true` here would otherwise become a count of 1.
    honest = [isinstance(count, int) and not isinstance(count, bool) and count >= 0 for count in (valid, invalid)]
    if not all(honest):
        return SignatureSelection(
            problem=(
                "the macOS parity job reported a measurement without honest `valid`/`invalid` counts "
                f"(got {valid!r}/{invalid!r}) — the counts ARE the value of this row"
            )
        )
    files = payload.get("invalid_files") or []
    return SignatureSelection(
        report=SignatureReport(
            status=SIGNATURE_MEASURED,
            valid=valid,
            invalid=invalid,
            keg=str(payload.get("keg", "")).strip(),
            runner=str(payload.get("runner", "")).strip(),
            invalid_files=tuple(str(name) for name in files if str(name).strip()),
        )
    )


def canonical_db_path() -> Path:
    # The resolver is inside the guard too: an import that succeeds but a call that raises would
    # crash the whole table over a row that was only ever going to say `n/a` on a runner.
    try:
        from brainlayer.paths import get_db_path

        return Path(get_db_path())
    except Exception:
        # The ratchet job does not install brainlayer; the documented canonical path is the answer.
        return FALLBACK_DB


def _clean_git_env() -> dict[str, str]:
    # Same guard as tests/test_build_sha.py and src/brainlayer/deploy_drift.py: an inherited
    # GIT_DIR/GIT_WORK_TREE OVERRIDES `-C`, so HEAD and dirty-tree would answer for another repo --
    # a stamp that matches it is a false GREEN, and a stamp that does not is a false RED.
    return {key: value for key, value in os.environ.items() if not key.startswith("GIT_")}


def _git(*args: str) -> str | None:
    # Resolved absolutely rather than left to PATH: this decides a provenance verdict, so the
    # binary it asks is worth pinning, and a machine without git answers None instead of raising.
    git = shutil.which("git")
    if git is None:
        return None
    try:
        result = subprocess.run(
            [git, "-C", str(ROOT), *args],
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
            env=_clean_git_env(),
        )
    except (OSError, subprocess.SubprocessError, UnicodeDecodeError):
        return None
    return result.stdout.strip()


def git_head() -> str | None:
    sha = _git("rev-parse", "HEAD")
    return sha if sha and re.fullmatch(r"[0-9a-f]{40}", sha) else None


def git_tree_dirty() -> bool | None:
    # `--untracked-files=all` so `status.showUntrackedFiles=no` cannot hide an untracked
    # `_build.py` and turn a dirty stamp into a false GREEN.
    porcelain = _git("status", "--porcelain", "--untracked-files=all")
    return None if porcelain is None else bool(porcelain)


def first_unmet(requirements: list[tuple[bool, str]]) -> str | None:
    """Return the reason for the first capability this machine does not have."""
    for satisfied, reason in requirements:
        if not satisfied:
            return reason
    return None


@dataclass(frozen=True)
class WheelStamp:
    """What ``brainlayer.__build_sha__`` would report from a wheel, or why it would report nothing."""

    sha: str | None = None
    problem: str | None = None


def read_wheel_stamp(wheel: Path) -> WheelStamp:
    # Each failure gets its own sentence. "Unreadable wheel" and "wheel with no stamp" are
    # different findings for the author to clear, and a crash here would cost the whole table.
    try:
        with zipfile.ZipFile(wheel) as archive:
            raw = archive.read(STAMP_MEMBER).decode("utf-8")
    except KeyError:
        return WheelStamp(
            problem=f"wheel ships no `{STAMP_MEMBER}` — a keg built from it can never prove its provenance"
        )
    except (zipfile.BadZipFile, OSError, UnicodeDecodeError) as error:
        return WheelStamp(
            problem=f"wheel could not be read ({type(error).__name__}) — provenance is unprovable, not proven"
        )
    match = STAMP_PATTERN.match(raw.strip() + "\n")
    if match is None:
        return WheelStamp(problem=f"`{STAMP_MEMBER}` declares no 40-hex BUILD_SHA")
    return WheelStamp(sha=match.group(1))


# --------------------------------------------------------------------------------------------
# Rows
# --------------------------------------------------------------------------------------------

PROVENANCE_NOTES = (
    "Sha half of #749 keg-mode provenance: a keg built from this wheel can answer "
    "`__build_sha__`. The helper-age and served-process predicates need a running BrainBar "
    "and are measured only by `scripts/sprint_gate.py` on an installed Mac."
)


def row_provenance(probe: Probe, _corpus: dict) -> Row:
    method = "wheel stamp · in-process · runner"
    wheel, head = probe.wheel, probe.head_sha
    if probe.wheel_problem:
        # Never `n/a`: a wheel this job promised and did not deliver is a finding, not a capability gap.
        return Row("provenance", RED, probe.wheel_problem, method, PROVENANCE_NOTES)
    if wheel is not None and not wheel.is_file():
        # select_wheel already required is_file(); if the path is gone now, that is still a finding.
        # Feeding it to first_unmet would render `n/a` and let the job go green (Macroscope, #752).
        return Row(
            "provenance",
            RED,
            f"selected wheel `{wheel}` is not a file — provenance was promised a wheel and lost it",
            method,
            PROVENANCE_NOTES,
        )
    reason = first_unmet(
        [
            (wheel is not None, "no packaged wheel in this job (pass --wheel or --wheel-glob)"),
            (head is not None, "git HEAD could not be read"),
            (probe.tree_dirty is not None, "`git status --porcelain` could not be read"),
        ]
    )
    # The requirement list above is the guard: past it, wheel and head are both known good.
    if reason or wheel is None or head is None:
        return Row("provenance", NA, f"n/a — {reason or 'provenance inputs unavailable'}", method, PROVENANCE_NOTES)
    stamped = read_wheel_stamp(wheel)
    if stamped.problem or stamped.sha is None:
        return Row("provenance", RED, stamped.problem or "wheel stamp unreadable", method, PROVENANCE_NOTES)
    stamp = stamped.sha
    if stamp != head:
        return Row("provenance", RED, f"stamped `{stamp[:12]}` ≠ HEAD `{head[:12]}`", method, PROVENANCE_NOTES)
    if probe.tree_dirty:
        value = f"stamped `{stamp[:12]}` == HEAD, but stamping dirtied the tree — `{STAMP_MEMBER}` must stay gitignored"
        return Row("provenance", RED, value, method, PROVENANCE_NOTES)
    return Row("provenance", GREEN, f"stamped `{stamp[:12]}` == HEAD, tree clean", method, PROVENANCE_NOTES)


def served_stack_requirements(probe: Probe, corpus: dict) -> list[tuple[bool, str]]:
    """The capabilities every socket-measured row needs before it may print a number.

    The served stack comes FIRST, and the OS/arch check second, on purpose. Leading with
    "runner is Linux/x86_64" made these rows read as runner problems -- as though moving the job to
    a macOS runner would turn them green. It would not, and w13 moved a job to macOS precisely to
    establish that: these three need the daemon, the helper and the indexed corpus, which no hosted
    runner has. The OS check is kept because a served stack on Darwin/x86_64 would still be off the
    gate's target, but it is no longer the headline.
    """
    target = corpus["machine_target"]
    return [
        (
            probe.socket_path.exists(),
            f"no BrainBar daemon at {probe.socket_path}: this row needs the daemon, its hybrid "
            "helper and the indexed corpus running together, and no GitHub-hosted runner has them "
            f"(macOS included) — only a self-hosted {target['os']}/{target['architecture']} runner "
            "on an installed Mac would",
        ),
        (
            probe.os_name == target["os"] and probe.architecture == target["architecture"],
            f"runner is {probe.os_name}/{probe.architecture}; the gate's machine target is "
            f"{target['os']}/{target['architecture']}",
        ),
    ]


def row_mapped_bytes(probe: Probe, corpus: dict) -> Row:
    notes = (
        "Baseline **26.2 GB** — installed Mac, socket, 2026-09-03, after R2 drained 15,070 → 0. "
        "Up from 16.8 GB because the drain left more vectors mapped under the same cap: the change "
        "is the drain, not a leak. Not measured by this run."
    )
    reason = first_unmet(
        served_stack_requirements(probe, corpus)
        + [
            (probe.db_path.exists(), f"no canonical DB at {probe.db_path}"),
            (
                False,
                "no runner-side collector for mapped bytes: a synthetic corpus would measure a "
                "different thing and would owe the table a different method label",
            ),
        ]
    )
    return Row("mapped bytes", NA, f"n/a — {reason}", "socket · installed Mac", notes)


def row_search_latency(probe: Probe, corpus: dict) -> Row:
    baseline = corpus["latency_baseline_ms"]
    notes = (
        f"Baseline p50 {baseline['p50']} ms / p95 {baseline['p95']} ms, captured "
        f"{baseline['captured_at']} on {baseline['hostname']} under {baseline['captured_under']} "
        "(`tests/fixtures/sprint_gate/corpus.json`). Not measured by this run."
    )
    reason = first_unmet(
        served_stack_requirements(probe, corpus)
        + [
            (probe.db_path.exists(), f"no canonical DB at {probe.db_path}"),
            (
                probe.hostname == baseline["hostname"],
                f"host {probe.hostname} is not the calibrated baseline host {baseline['hostname']}",
            ),
            (
                False,
                "no runner-side collector for search latency: a synthetic corpus would measure a "
                "different thing and would owe the table a different method label",
            ),
        ]
    )
    return Row("search p50/p95", NA, f"n/a — {reason}", "socket · installed Mac", notes)


def row_idle_cpu(probe: Probe, corpus: dict) -> Row:
    thresholds = corpus["thresholds"]
    notes = (
        f"Budget: average CPU < {thresholds['cpu_percent']}% over a "
        f"{thresholds['resource_window_seconds']} s window (`resource_budget` in "
        "`scripts/sprint_gate.py`). Needs the BrainBar daemon, helper and watcher actually running. "
        "Not measured by this run."
    )
    reason = first_unmet(
        served_stack_requirements(probe, corpus)
        + [
            (
                False,
                "no runner-side collector for idle CPU: an idle hosted runner measures a different "
                "thing and would owe the table a different method label",
            )
        ]
    )
    return Row("idle CPU", NA, f"n/a — {reason}", "ps sampling · installed Mac", notes)


# Two labels, because they are two measurements. The parity job installs the PUBLISHED tap formula
# on a clean hosted Mac; Etan's release check verifies the keg on his own machine. Sharing one label
# would let a runner result stand in for a release-time one.
SIGNATURE_METHOD_MEASURED = "codesign · brew keg · GitHub macOS runner"
SIGNATURE_METHOD_NA = "codesign · installed keg"

SIGNATURE_NOTES = (
    "`scripts/release-verify-signatures.sh <keg>` codesign-verifies every `*.so`/`*.dylib` under "
    "`libexec/venv`. The macOS parity job installs the **published** tap formula "
    "(`etanhey/layers/brainlayer`), so this row measures the release path — formula, published "
    "sdist and Homebrew's relocation — and **not this PR's tree**. Release-time baseline for the "
    "same keg on a different machine: **442 valid / 0 invalid** — installed Mac (M4 Max), "
    "`brew --prefix brainlayer` 1.5.11, 2026-09-03."
)


def row_signature_valid(probe: Probe, _corpus: dict) -> Row:
    """Report what the macOS parity job measured — this collector never runs codesign itself.

    There is deliberately no OS predicate here any more. The old reason ("runner is Linux; codesign
    verification needs macOS") described the machine rendering the table, which is not the machine
    that measures this row. What decides the row now is whether the parity job ran and what it said.
    """
    if probe.signature_problem:
        # A job that ran and owes a measurement is a finding, never a capability gap.
        return Row("signature_valid", RED, probe.signature_problem, SIGNATURE_METHOD_MEASURED, SIGNATURE_NOTES)
    report = probe.signature
    if report is None:
        reason = probe.signature_unavailable or SIGNATURE_UNAVAILABLE_DEFAULT
        return Row("signature_valid", NA, f"n/a — {reason}", SIGNATURE_METHOD_NA, SIGNATURE_NOTES)
    if report.status == SIGNATURE_FAILED:
        return Row(
            "signature_valid",
            RED,
            f"the macOS parity job could not measure signatures: {report.stage} — {report.detail}",
            SIGNATURE_METHOD_MEASURED,
            SIGNATURE_NOTES,
        )
    valid, invalid = report.valid or 0, report.invalid or 0
    where = " · ".join(part for part in (report.keg, report.runner) if part)
    measured = f"{valid} valid / {invalid} invalid" + (f" · {where}" if where else "")
    if valid + invalid == 0:
        # An empty sweep proves nothing: release-verify-signatures.sh itself treats it as fatal.
        return Row(
            "signature_valid",
            RED,
            f"{measured} — the keg exposed no native extensions to verify, so nothing was proven",
            SIGNATURE_METHOD_MEASURED,
            SIGNATURE_NOTES,
        )
    if invalid:
        named = ", ".join(report.invalid_files[:5]) or "see the parity job log"
        return Row("signature_valid", RED, f"{measured} — {named}", SIGNATURE_METHOD_MEASURED, SIGNATURE_NOTES)
    return Row("signature_valid", GREEN, measured, SIGNATURE_METHOD_MEASURED, SIGNATURE_NOTES)


ROW_BUILDERS = (row_provenance, row_mapped_bytes, row_search_latency, row_idle_cpu, row_signature_valid)


def collect(probe: Probe, corpus: dict) -> list[Row]:
    return [builder(probe, corpus) for builder in ROW_BUILDERS]


# --------------------------------------------------------------------------------------------
# Rendering
# --------------------------------------------------------------------------------------------


def escape_cell(text: str) -> str:
    return text.replace("|", "\\|").replace("\n", " ")


def render(rows: list[Row], probe: Probe, run_url: str | None, now: datetime) -> str:
    head = probe.head_sha[:12] if probe.head_sha else "unknown"
    lines = [
        MARKER,
        "### BrainLayer ratchet",
        "",
        "Every **Value** below was measured by this run. A row this machine cannot measure says "
        "`n/a — <reason>` instead of a number; baselines in **Notes** name their own machine, "
        "method and date and were **not** measured here.",
        "",
        "| Row | Status | Value (measured by this run) | Method | Notes |",
        "| --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        lines.append(
            f"| {escape_cell(row.name)} | {BADGE[row.status]} | {escape_cell(row.value)} "
            f"| {escape_cell(row.method)} | {escape_cell(row.notes)} |"
        )
    reds = [row for row in rows if row.status == RED]
    lines += [
        "",
        f"{BADGE[GREEN]} measured, within budget · {BADGE[RED]} measured, out of budget — a finding "
        f"to clear before merge · {BADGE[NA]} not measurable on this machine, never guessed.",
        "",
    ]
    if reds:
        lines.append(f"**{len(reds)} RED row(s) to clear:** " + ", ".join(f"`{row.name}`" for row in reds) + ".")
    else:
        lines.append("No RED rows.")
    run = f" · [run]({run_url})" if run_url else ""
    lines += [
        "",
        f"_Measured on {probe.os_name}/{probe.architecture} · checked-out HEAD `{head}`"
        f"{run} · updated {now.strftime('%Y-%m-%d %H:%M:%S')} UTC_",
        "",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--wheel", type=Path, help="Wheel built from the checked-out tree (provenance row)")
    parser.add_argument(
        "--wheel-glob",
        help="Glob that must match exactly one built wheel, e.g. 'dist/*.whl'. Zero or several is RED, never n/a.",
    )
    parser.add_argument(
        "--signature-report",
        type=Path,
        help=(
            "JSON written by the macOS signature-parity job. Passing it asserts that job RAN: a "
            "missing or unreadable report is RED, never n/a."
        ),
    )
    parser.add_argument(
        "--signature-unavailable",
        help="Why nobody measured signatures on this run. Renders `n/a — <reason>`.",
    )
    parser.add_argument("--run-url", help="Link back to the workflow run that produced this table")
    parser.add_argument("--out", type=Path, help="Also write the rendered table here")
    args = parser.parse_args(argv)

    corpus = json.loads(CORPUS.read_text(encoding="utf-8"))
    probe = Probe.detect(corpus, args.wheel, args.wheel_glob, args.signature_report, args.signature_unavailable)
    rows = collect(probe, corpus)
    table = render(rows, probe, args.run_url, datetime.now(timezone.utc))

    if args.out:
        args.out.write_text(table, encoding="utf-8")
    print(table)
    for row in rows:
        if row.status == RED:
            print(f"::error title=Ratchet RED: {row.name}::{row.value}", file=sys.stderr)
    # n/a never fails the build; RED always does -- a row nobody has to clear is decoration.
    return 1 if any(row.status == RED for row in rows) else 0


if __name__ == "__main__":
    raise SystemExit(main())
