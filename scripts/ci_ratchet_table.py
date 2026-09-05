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

``commit provenance`` answers the question a reader asks before reading any other row: **which
commit is this table about?** On a ``pull_request`` event ``actions/checkout`` checks out GitHub's
synthetic merge ref, whose sha appears nowhere on the PR -- #759's table printed
``13fa724278bf`` while that PR's head was ``4632f979``, so nobody reading the comment could tell
whether it still described the branch. This row names the PR-head commit the checkout represents and
goes RED when it is no longer the PR's head, because a table measured on a superseded commit must
announce that rather than sit there looking current.

``baseline attestation`` answers the question that comes right after: **what is this table measuring
against, and who says so?** The baseline fields of ``tests/fixtures/sprint_gate/corpus.json`` --
``queries``, ``latency_baseline_ms``, ``thresholds`` -- are what every comparison in this file and in
``scripts/sprint_gate.py`` reads as the reference. Until this row existed, a PR could edit those numbers
and the gate compared against the edited copy. Now the reference is the ``ratchet-attestation``
artifact that a ``push``/``workflow_dispatch`` run of ``ratchet-attest.yml`` on ``main`` publishes,
read back through the Actions API and never from the PR tree: a baseline that differs from it is RED
-- "changed by hand; no CI attestation for these values" -- unless every changed field equals a value
that main run actually **measured**. Same boundary as ``commit provenance``, stated exactly: the
attestation's INPUTS are outside the PR tree (a PR run cannot upload an artifact to a main run), but
the comparator is this file, which the PR checks out and could rewrite. Diff-reviewable, not
tamper-proof.

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
import hashlib
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
CORPUS_RELATIVE = Path("tests") / "fixtures" / "sprint_gate" / "corpus.json"
CORPUS = ROOT / CORPUS_RELATIVE

# The fields of corpus.json that ARE the baseline: what a gate compares a measurement against. The
# rest of the file (socket path, process patterns, machine target, timeouts) is configuration that
# ordinary review covers; freezing it behind a measurement nobody can make would be a lock with no
# key. `queries` is in, because the queries define what the latency baseline measured.
BASELINE_FIELDS = ("queries", "latency_baseline_ms", "thresholds")

# Where an attested baseline lives, and the only writer of it: a `push`/`workflow_dispatch` run of
# this workflow on `main`. ratchet.yml reads the artifact back through the Actions API.
ATTEST_WORKFLOW = ".github/workflows/ratchet-attest.yml"
ATTESTATION_ARTIFACT = "ratchet-attestation"
ATTESTATION_SCHEMA = 1

# The sticky-comment anchor. .github/workflows/ratchet.yml greps for this exact string to decide
# update-vs-create, so one PR can only ever carry one of these comments; tests pin them together.
MARKER = "<!-- brainlayer-ratchet-table -->"
STAMP_MEMBER = "brainlayer/_build.py"
STAMP_PATTERN = re.compile(r'^BUILD_SHA = "([0-9a-f]{40})"\s*$')
SHA_PATTERN = re.compile(r"[0-9a-f]{40}")
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

# What `commit provenance` may say when nobody handed it a commit to describe. This is the only
# genuine capability gap the row has: a local `python scripts/ci_ratchet_table.py` is not a PR and
# has no PR head to be current with. Every other way this row can fail to answer is a finding.
COMMIT_UNAVAILABLE_DEFAULT = (
    "not a pull-request run: no `--measured-sha`/`--pr-head-sha` was handed to this collector, "
    "so there is no PR head for the table to be current with"
)

# What `baseline attestation` may say when nobody handed it an attestation. Like the commit row, a
# local `python scripts/ci_ratchet_table.py` is not a PR and has read nothing from the Actions API.
ATTESTATION_UNAVAILABLE_DEFAULT = (
    "not a pull-request run: no `--attestation` was handed to this collector, so there is no main "
    "attestation for the baseline to be checked against"
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
    # `(HEAD, *parents)` for the checked-out commit. The parents are the point: on a pull_request
    # event the checkout is GitHub's merge ref, and its PR-head parent is the only sha in play that
    # a reviewer can also see on the PR.
    head_lineage: tuple[str, ...] | None = None
    # Which commit this table describes, and which commit the PR head is now. Same three-way split
    # as the signature hand-off: a measurement, a real capability gap, or a promise broken.
    measured_sha: str | None = None
    pr_head_sha: str | None = None
    commit_unresolved: str | None = None
    commit_unavailable: str | None = None
    commit_problem: str | None = None
    # Set when this job MEANT to hand over a wheel and could not. That is a finding to clear, not a
    # capability this machine lacks, so it renders RED and never `n/a`.
    wheel_problem: str | None = None
    # The macOS parity job's own words, and the same three-way split: a measurement it made, the
    # reason nobody was asked to make one, or a promise it broke.
    signature: SignatureReport | None = None
    signature_unavailable: str | None = None
    signature_problem: str | None = None
    # What main says the baseline is, and the same split again: the attestation a main run
    # published, the reason there is none yet (bootstrap: the writer has never run on main), the
    # reason nobody asked (not a PR), or a promise broken (asked, and could not be read).
    attestation: Attestation | None = None
    attestation_bootstrap: str | None = None
    attestation_unavailable: str | None = None
    attestation_problem: str | None = None

    @classmethod
    def detect(
        cls,
        corpus: dict,
        wheel: Path | None,
        wheel_glob: str | None,
        signature_report: Path | None = None,
        signature_unavailable: str | None = None,
        measured_sha: str | None = None,
        pr_head_sha: str | None = None,
        pr_head_unresolved: str | None = None,
        attestation: Path | None = None,
        attestation_bootstrap: str | None = None,
        attestation_unresolved: str | None = None,
    ) -> Probe:
        selected = select_wheel(wheel, wheel_glob)
        signature = select_signature(signature_report, signature_unavailable)
        commit = select_commit(measured_sha, pr_head_sha, pr_head_unresolved)
        attested = select_attestation(attestation, attestation_bootstrap, attestation_unresolved)
        return cls(
            os_name=platform.system(),
            architecture=platform.machine(),
            hostname=socket_module.gethostname(),
            socket_path=Path(corpus["socket_path"]),
            db_path=canonical_db_path(),
            wheel=selected.path,
            head_sha=git_head(),
            tree_dirty=git_tree_dirty(),
            head_lineage=git_head_lineage(),
            measured_sha=commit.measured,
            pr_head_sha=commit.pr_head,
            commit_unresolved=commit.unresolved,
            commit_unavailable=commit.unavailable,
            commit_problem=commit.problem,
            wheel_problem=selected.problem,
            signature=signature.report,
            signature_unavailable=signature.unavailable,
            signature_problem=signature.problem,
            attestation=attested.attestation,
            attestation_bootstrap=attested.bootstrap,
            attestation_unavailable=attested.unavailable,
            attestation_problem=attested.problem,
        )


# --------------------------------------------------------------------------------------------
# Baseline attestation
# --------------------------------------------------------------------------------------------


def baseline_view(corpus: dict) -> dict:
    """The baseline fields of corpus.json, and nothing else -- see BASELINE_FIELDS."""
    return {field: corpus[field] for field in BASELINE_FIELDS if field in corpus}


def baseline_digest(view: dict) -> str:
    # Canonical JSON, so a reformat of the fixture is not a baseline change; the field diff below
    # is what names a real one.
    canonical = json.dumps(view, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


BaselinePath = tuple[str, ...]


def flatten_baseline(value: object, prefix: BaselinePath = ()) -> dict[BaselinePath, object]:
    """`{"a": {"b": 1}, "q": [x, y]}` -> `{("a", "b"): 1, ("q",): [x, y]}`. Lists stay whole: a
    query list is one value, and naming index 3 of it would not help a reader.

    Tuples, not dotted strings. A key that itself contains a dot (`{"a.b": 1}`) flattened to the
    same string as the nested `{"a": {"b": 1}}`, so a PR could replace a nested value with a
    colliding dotted key and make the diff come out empty (Macroscope, #767). A tuple keeps the
    key boundaries; `dotted()` is only how a path is printed.
    """
    if isinstance(value, dict):
        flat: dict[BaselinePath, object] = {}
        for key, inner in value.items():
            flat.update(flatten_baseline(inner, (*prefix, str(key))))
        return flat
    return {prefix: value}


def dotted(path: BaselinePath) -> str:
    return ".".join(path)


def baseline_changes(reference: dict, candidate: dict) -> list[tuple[BaselinePath, object, object]]:
    """Every path whose value differs, as (path, reference value, candidate value).

    A missing side is reported as None: a field that vanished from the baseline is as much a change
    as one that moved. `False != 0` here because a boolean and a count are different claims.
    """
    before, after = flatten_baseline(reference), flatten_baseline(candidate)
    changed = []
    for path in sorted(set(before) | set(after)):
        if path not in before or path not in after:
            # Membership, not `.get()`: a leaf whose value IS null and a leaf that is gone both
            # `.get()` to None, and the second is a change (Macroscope, #766).
            changed.append((path, before.get(path), after.get(path)))
            continue
        old, new = before[path], after[path]
        if old != new or type(old) is not type(new):
            changed.append((path, old, new))
    return changed


@dataclass(frozen=True)
class Attestation:
    """What a main run published about the baseline: the values it saw, and the ones it measured."""

    run_id: int
    run_attempt: int
    main_sha: str
    measured_at: str
    baseline: dict
    digest: str
    # Dotted baseline path -> the value that run MEASURED for it. This is the legitimate path for a
    # baseline to move: a PR may set a field to exactly what main measured, and nothing else. Empty
    # today, and honestly so: no runner-side collector measures any baseline field yet, so today no
    # PR can move one -- the key to this lock is a collector, not a hand.
    measured: dict[str, object]


@dataclass(frozen=True)
class AttestationSelection:
    """Which of the four states `baseline attestation` is in, resolved once and fail-closed."""

    attestation: Attestation | None = None
    bootstrap: str | None = None
    unavailable: str | None = None
    problem: str | None = None


def honest_run_number(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def read_attestation(path: Path) -> tuple[Attestation | None, str | None]:
    """Load a main run's attestation, or say why it cannot be trusted.

    Every failure returns a reason rather than raising, on the same rule as the signature report:
    a crash would cost the whole table over one file the collector was handed. Every check here is
    a check on the ARTIFACT's shape; whether the run that published it was really a main run is
    settled in ratchet.yml against the Actions API, before this file is ever handed over.
    """
    if not path.is_file():
        return None, f"`--attestation {path}` is not a file — the main attestation was promised and not delivered"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        return None, f"the main attestation could not be read ({type(error).__name__}) — the baseline is unattested"
    if not isinstance(payload, dict):
        return None, f"the main attestation is a JSON {type(payload).__name__}, not an object"
    if payload.get("schema") != ATTESTATION_SCHEMA:
        return (
            None,
            f"the main attestation carries schema {payload.get('schema')!r}; this collector reads {ATTESTATION_SCHEMA}",
        )
    run_id, attempt = payload.get("run_id"), payload.get("run_attempt")
    if not (honest_run_number(run_id) and honest_run_number(attempt)):
        return None, f"the main attestation names no honest run (run_id {run_id!r}, attempt {attempt!r})"
    main_sha = payload.get("main_sha")
    if not isinstance(main_sha, str) or SHA_PATTERN.fullmatch(main_sha) is None:
        return None, f"the main attestation's `main_sha` {main_sha!r} is not a 40-hex commit sha"
    baseline, digest, measured = payload.get("baseline"), payload.get("baseline_sha256"), payload.get("measured")
    if not isinstance(baseline, dict) or set(baseline) != set(BASELINE_FIELDS):
        # Exactly the baseline fields: a configuration key inside is not a baseline, and a MISSING
        # field is a reference that does not cover what it claims to (Macroscope, #766).
        return None, "the main attestation's `baseline` is not an object of exactly the baseline fields"
    if digest != baseline_digest(baseline):
        # The digest is what the GREEN message quotes, so it has to be the digest OF this baseline.
        return (
            None,
            "the main attestation's `baseline_sha256` does not match its own `baseline` — the artifact is inconsistent",
        )
    if not isinstance(measured, dict):
        return None, f"the main attestation's `measured` is a {type(measured).__name__}, not an object"
    return (
        Attestation(
            run_id=run_id,
            run_attempt=attempt,
            main_sha=main_sha,
            measured_at=str(payload.get("measured_at", "")).strip(),
            baseline=baseline,
            digest=digest,
            measured={str(key): value for key, value in measured.items()},
        ),
        None,
    )


def select_attestation(path: Path | None, bootstrap: str | None, unresolved: str | None) -> AttestationSelection:
    """Resolve the attestation FAIL-CLOSED, on the same rule the other hand-offs follow.

    `--attestation <path>` says a main run published one and ratchet.yml fetched it; anything
    wrong with it is a finding. `--attestation-bootstrap <reason>` says the Actions API shows NO
    successful attest run on main yet -- the writer has never run there -- which is the one state
    that is neither a measurement nor a finding, and the row then falls back to the base commit.
    `--attestation-unresolved <reason>` says the run was asked to fetch one and could not: RED.
    """
    handed = [flag for flag in (path, bootstrap, unresolved) if flag is not None]
    if len(handed) > 1:
        return AttestationSelection(
            problem="`--attestation`, `--attestation-bootstrap` and `--attestation-unresolved` are mutually exclusive"
        )
    if not handed:
        return AttestationSelection(unavailable=ATTESTATION_UNAVAILABLE_DEFAULT)
    if unresolved is not None:
        return AttestationSelection(
            problem=unresolved.strip() or "the main attestation could not be fetched (no reason given)"
        )
    if bootstrap is not None:
        return AttestationSelection(bootstrap=bootstrap.strip() or "no attest run has completed on main yet")
    attestation, problem = read_attestation(path)  # type: ignore[arg-type]
    if problem is not None or attestation is None:
        return AttestationSelection(problem=problem or "the main attestation could not be read")
    return AttestationSelection(attestation=attestation)


@dataclass(frozen=True)
class CommitClaim:
    """Which commit this table describes, and which commit the PR's head is right now.

    Two shas, from two sources that cannot be the same source. `measured` is the commit the run was
    triggered for -- the event payload's `pull_request.head.sha`, fixed for the life of the run.
    `pr_head` is read live from the REST API at collect time, which is the only way a run can find
    out that it has been overtaken.
    """

    measured: str | None = None
    pr_head: str | None = None
    # Why the live head could not be read. A run that was ASKED to prove it is current and could not
    # is a finding, not a machine that cannot measure -- the same rule the signature report follows.
    unresolved: str | None = None
    # Not a PR at all: the one honest `n/a` this row has.
    unavailable: str | None = None
    problem: str | None = None


def select_commit(measured: str | None, pr_head: str | None, unresolved: str | None) -> CommitClaim:
    """Resolve the commit claim FAIL-CLOSED, on the same rule the wheel and the signature follow."""
    if pr_head is not None and unresolved is not None:
        return CommitClaim(problem="`--pr-head-sha` and `--pr-head-unresolved` are mutually exclusive")
    if measured is None and pr_head is None and unresolved is None:
        return CommitClaim(unavailable=COMMIT_UNAVAILABLE_DEFAULT)
    if measured is None:
        return CommitClaim(problem="a PR head reached this run with no `--measured-sha` for the table to describe")
    if SHA_PATTERN.fullmatch(measured) is None:
        return CommitClaim(problem=f"`--measured-sha {measured}` is not a 40-hex commit sha")
    if pr_head is None and unresolved is None:
        # Naming the measured commit is not the same as showing it is still the PR's head. Without
        # the second sha this row would print a sha and prove nothing -- exactly the shape of table
        # that let #753 be reviewed on a superseded commit.
        return CommitClaim(problem="`--measured-sha` was handed over with no PR head to check it against")
    if unresolved is not None:
        return CommitClaim(measured=measured, unresolved=unresolved.strip() or "no reason given")
    if SHA_PATTERN.fullmatch(pr_head or "") is None:
        return CommitClaim(problem=f"`--pr-head-sha {pr_head}` is not a 40-hex commit sha")
    return CommitClaim(measured=measured, pr_head=pr_head)


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
    # `brew install`'s own step outcome. Homebrew `ofail`s a relocation failure and exits non-zero
    # while still installing the keg, so this is NOT what decides whether the row was measured --
    # but a green row must not hide that brew complained.
    install_outcome: str = ""


@dataclass(frozen=True)
class SignatureSelection:
    """Which of the three states `signature_valid` is in, resolved once and fail-closed."""

    report: SignatureReport | None = None
    unavailable: str | None = None
    problem: str | None = None


def read_signature_payload(report_path: Path) -> tuple[dict | None, str | None]:
    """Load the parity job's report, or say why it cannot be read.

    Every failure here returns a reason rather than raising. A crash would cost the WHOLE table --
    including the provenance row this job did measure -- over a report the collector was handed.
    """
    if not report_path.is_file():
        return None, (
            f"`--signature-report {report_path}` is not a file — the macOS parity job ran and published no report"
        )
    try:
        payload = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        # `ValueError` covers all three ways this line fails, which is why it is listed alone:
        # `json.JSONDecodeError` and `UnicodeDecodeError` both subclass it, and so does the bare
        # ValueError that a count past sys.get_int_max_str_digits() raises out of json.loads. Each
        # has to render RED like any other malformed report instead of killing the whole table.
        return None, (
            f"the macOS parity job's report could not be read ({type(error).__name__}) — "
            "signatures are unverified, not verified"
        )
    if not isinstance(payload, dict):
        return None, f"the macOS parity job's report is a JSON {type(payload).__name__}, not an object"
    return payload, None


def signature_failure(payload: dict) -> SignatureSelection:
    """A report that says the job could not measure. It still owes us WHAT could not."""
    stage = str(payload.get("stage", "")).strip()
    detail = str(payload.get("detail", "")).strip()
    if not stage or not detail:
        return SignatureSelection(
            problem="the macOS parity job reported a failure without naming the `stage` and `detail` that failed"
        )
    return SignatureSelection(report=SignatureReport(status=SIGNATURE_FAILED, stage=stage, detail=detail))


def honest_count(value: object) -> bool:
    # bool is an int in Python; a `true` here would otherwise become a count of 1.
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def signature_measurement(payload: dict) -> SignatureSelection:
    """A report that claims counts. The counts ARE this row's value, so they are checked hard."""
    valid, invalid = payload.get("valid"), payload.get("invalid")
    if not (honest_count(valid) and honest_count(invalid)):
        return SignatureSelection(
            problem=(
                "the macOS parity job reported a measurement without honest `valid`/`invalid` counts "
                f"(got {valid!r}/{invalid!r}) — the counts ARE the value of this row"
            )
        )
    files = payload.get("invalid_files") or []
    if not isinstance(files, list):
        # `invalid_files: 1` used to raise TypeError out of the tuple comprehension and abort the
        # whole table, instead of the fail-closed RED row a malformed hand-off is supposed to give.
        return SignatureSelection(
            problem=(
                f"the macOS parity job's `invalid_files` is a {type(files).__name__}, not a list — "
                "the report is malformed"
            )
        )
    return SignatureSelection(
        report=SignatureReport(
            status=SIGNATURE_MEASURED,
            valid=valid,
            invalid=invalid,
            keg=str(payload.get("keg", "")).strip(),
            runner=str(payload.get("runner", "")).strip(),
            install_outcome=str(payload.get("install_outcome", "")).strip(),
            invalid_files=tuple(str(name) for name in files if str(name).strip()),
        )
    )


def select_signature(report_path: Path | None, unavailable: str | None) -> SignatureSelection:
    """Resolve the signature measurement FAIL-CLOSED, on the same rule the wheel follows.

    `--signature-unavailable <reason>` says nobody was asked to measure -- a real capability gap,
    rendered `n/a` with that reason. `--signature-report <path>` says the macOS parity job RAN and
    owes this run a measurement; anything that stops it arriving is a **finding**, because a job
    that ran and produced nothing is not a machine that cannot measure.

    The per-shape checks live in the helpers above; this function is only the routing, so that
    adding a validation rule does not keep pushing one function's branch count up.
    """
    if report_path is not None and unavailable is not None:
        return SignatureSelection(problem="`--signature-report` and `--signature-unavailable` are mutually exclusive")
    if report_path is None and unavailable is None:
        return SignatureSelection(unavailable=SIGNATURE_UNAVAILABLE_DEFAULT)
    if report_path is None:
        return SignatureSelection(unavailable=unavailable)
    payload, problem = read_signature_payload(report_path)
    if problem is not None or payload is None:
        return SignatureSelection(problem=problem or "the macOS parity job's report could not be read")
    if payload.get("status") not in SIGNATURE_STATUSES:
        return SignatureSelection(
            problem=(
                "the macOS parity job's report carries no known `status` "
                f"(expected one of {', '.join(SIGNATURE_STATUSES)})"
            )
        )
    if payload["status"] == SIGNATURE_FAILED:
        return signature_failure(payload)
    return signature_measurement(payload)


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
    return sha if sha and SHA_PATTERN.fullmatch(sha) else None


def git_head_lineage() -> tuple[str, ...] | None:
    """`(HEAD, *parents)` for the checked-out commit, or None when git cannot say.

    The parents are why this exists. On a `pull_request` event `actions/checkout` checks out
    GitHub's synthetic merge ref, whose sha appears nowhere on the PR: #759's table printed
    `13fa724278bf` while the PR's head was `4632f979`, and that merge commit's parents are
    (base tip, PR head). Matching the run's head sha against this list is what proves the checkout
    really is the commit the table claims, and it reads the commit GRAPH, which GitHub builds rather
    than the PR.

    The boundary, stated exactly, because an earlier draft of this docstring overclaimed it and a
    reviewer (79c16541) disproved the wider version: on a `pull_request` event GitHub runs the
    workflow AND these scripts from the PR's merge ref, so this comparator is the PR's own code and
    an author could rewrite it to answer GREEN unconditionally. What holds is narrower and is still
    the substantive half: **no INPUT to this row comes from the PR's tree** -- the event payload,
    the live REST read and this commit graph are all outside it -- so the row cannot confirm itself
    from material the PR controls. The comparator is diff-reviewable, not tamper-proof.

    Needs `fetch-depth: 2`: at depth 1 the checkout is the shallow boundary and git reports it as
    having no parents at all.
    """
    line = _git("rev-list", "--parents", "-n", "1", "HEAD")
    if not line:
        return None
    parts = tuple(line.split())
    return parts if parts and all(SHA_PATTERN.fullmatch(part) for part in parts) else None


def checkout_stands_for(lineage: tuple[str, ...], measured: str) -> bool:
    """Is the checkout `measured` itself, or GitHub's merge ref built FOR `measured`?

    Two shapes, checked by position -- not `measured in lineage`, which also accepted `lineage[1]`,
    the BASE tip, and proves nothing about the PR head (reviewer 79c16541, F4). GitHub builds the
    merge ref as `Merge <pr head> into <base tip>`, so the PR head is always the SECOND parent:
    verified on #759 (`13fa7242` -> parents `3126ac1b` base, `4632f979` head) and again in a local
    shallow-fetch experiment. A one-parent checkout whose PARENT is `measured` is deliberately NOT
    accepted -- that is a descendant, not a merge ref for it, and it would prove nothing either.
    """
    if lineage[0] == measured:
        return True
    return len(lineage) > 2 and lineage[2] == measured


def git_baseline_at(sha: str) -> tuple[dict | None, str | None]:
    """The baseline fields as committed at `sha`, or why git cannot say.

    Bootstrap only: before the first attest run on main there is nothing published to compare
    against, and the base tip's committed fixture is the next-best reference -- a commit object
    the PR did not write. `git show <sha>:<path>` reads the blob out of the object store, so it
    needs the parent commits that `fetch-depth: 2` brings along.
    """
    raw = _git("show", f"{sha}:{CORPUS_RELATIVE.as_posix()}")
    if raw is None:
        return None, f"git could not show `{CORPUS_RELATIVE.as_posix()}` at base `{sha[:12]}`"
    try:
        corpus = json.loads(raw)
    except ValueError as error:
        return None, f"the baseline at base `{sha[:12]}` is not JSON ({type(error).__name__})"
    if not isinstance(corpus, dict):
        return None, f"the baseline at base `{sha[:12]}` is a JSON {type(corpus).__name__}, not an object"
    missing = [field for field in BASELINE_FIELDS if field not in corpus]
    if missing:
        # The same shape the writer refuses to attest and the reader refuses to read: a base whose
        # fixture lacks a field is not a reference either (Cursor, #767 round 2).
        return None, f"the baseline at base `{sha[:12]}` lacks {', '.join(f'`{field}`' for field in missing)}"
    return baseline_view(corpus), None


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

COMMIT_METHOD = "commit graph + live PR head · in-process · runner"

COMMIT_NOTES = (
    "Which commit this whole table is about. On a `pull_request` event the checkout is GitHub's "
    "synthetic merge ref, whose sha is not on the PR — #759's table printed `13fa724278bf` while "
    "that PR's head was `4632f979` — so this row names the PR-head parent instead, the sha a "
    "reviewer can actually see. The comparison sha is read live from "
    "`repos/{owner}/{repo}/pulls/{n}` when the table is collected, not taken from the event "
    "payload, because the payload cannot know the run has been overtaken. Residual window, stated "
    "rather than papered over: a push landing between that read and the comment being posted is "
    "not caught here — the run for that push refreshes the table."
)


def row_commit_provenance(probe: Probe, _corpus: dict) -> Row:
    """Name the commit this table describes, and go RED the moment it is not the PR's head."""
    if probe.commit_problem:
        return Row("commit provenance", RED, probe.commit_problem, COMMIT_METHOD, COMMIT_NOTES)
    measured = probe.measured_sha
    if measured is None:
        reason = probe.commit_unavailable or COMMIT_UNAVAILABLE_DEFAULT
        return Row("commit provenance", NA, f"n/a — {reason}", COMMIT_METHOD, COMMIT_NOTES)
    lineage = probe.head_lineage
    if lineage is None:
        return Row(
            "commit provenance",
            RED,
            f"this run was triggered for `{measured[:12]}` and git could not say what is checked out, "
            "so nothing here shows the table describes that commit",
            COMMIT_METHOD,
            COMMIT_NOTES,
        )
    if not checkout_stands_for(lineage, measured):
        # The event says one commit; the checkout is another. Whatever the rows below measured, they
        # did not measure the commit this table is about to claim.
        return Row(
            "commit provenance",
            RED,
            f"checkout `{lineage[0][:12]}` is neither `{measured[:12]}` nor a merge of it — this run "
            "was triggered for a commit it does not have checked out",
            COMMIT_METHOD,
            COMMIT_NOTES,
        )
    if probe.commit_unresolved:
        # We know what was measured; we cannot show it is still current. A row that stopped at the
        # first half would print a sha and prove nothing.
        return Row(
            "commit provenance",
            RED,
            f"measured `{measured[:12]}`, and this run could not read the PR's current head, so it "
            f"cannot show the table is still current: {probe.commit_unresolved}",
            COMMIT_METHOD,
            COMMIT_NOTES,
        )
    pr_head = probe.pr_head_sha
    if pr_head is None:
        return Row(
            "commit provenance",
            RED,
            f"measured `{measured[:12]}` with no PR head to check it against",
            COMMIT_METHOD,
            COMMIT_NOTES,
        )
    if measured != pr_head:
        return Row(
            "commit provenance",
            RED,
            f"measured `{measured[:12]}` ≠ PR head `{pr_head[:12]}` — **this table describes a "
            "superseded commit**; every number in it belongs to the older one",
            COMMIT_METHOD,
            COMMIT_NOTES,
        )
    return Row(
        "commit provenance",
        GREEN,
        f"measured `{measured[:12]}` == PR head · checkout `{lineage[0][:12]}`",
        COMMIT_METHOD,
        COMMIT_NOTES,
    )


ATTESTATION_METHOD = "main attestation artifact via Actions API · in-process · runner"
ATTESTATION_METHOD_BOOTSTRAP = "base commit via git (bootstrap) · in-process · runner"

ATTESTATION_NOTES = (
    "What every comparison is measured AGAINST, and who says so. The baseline fields of "
    "`tests/fixtures/sprint_gate/corpus.json` (`queries`, `latency_baseline_ms`, `thresholds`) are "
    "compared to the `ratchet-attestation` artifact of the latest successful `push` or (no-input) "
    "`workflow_dispatch` run of `ratchet-attest.yml` on `main`, fetched through the Actions API — "
    "a PR run cannot write to "
    "another run's artifacts. A field that differs is RED unless that main run **measured** the "
    "new value; today no runner-side collector measures any baseline field, so today the baseline "
    "cannot move by PR at all, and this row says so instead of a hand edit passing. Boundary: the "
    "comparator is this PR's checkout of `ci_ratchet_table.py`, diff-reviewable, not tamper-proof."
)


# A sentinel so a measured `null` and "not measured" cannot be confused inside attested_row.
_UNMEASURED = object()


def describe_change(path: BaselinePath, old: object, new: object) -> str:
    return f"`{dotted(path)}` {json.dumps(old)} → {json.dumps(new)}"


def measured_value(attestation: Attestation, path: BaselinePath) -> object:
    """What the main run measured for `path`, or `_UNMEASURED`.

    The `measured` map is keyed by dotted strings -- that is the contract the margin reader (c)
    consumes -- so a segment that itself contains a dot could collide with a genuinely nested
    measured path. Such a path is never matched: it is unmeasured, and a change to it is RED.
    """
    if any("." in segment for segment in path):
        return _UNMEASURED
    return attestation.measured.get(dotted(path), _UNMEASURED)


def attested_row(attestation: Attestation, corpus: dict, base_tip: str | None) -> Row:
    """Judge the PR's baseline field by field against what the main run saw and measured."""
    where = f"run {attestation.run_id} · main `{attestation.main_sha[:12]}`"
    if attestation.measured_at:
        where += f" · {attestation.measured_at}"
    view = baseline_view(corpus)
    changes = baseline_changes(attestation.baseline, view)
    stale = ""
    if base_tip is not None and base_tip != attestation.main_sha:
        # The attestation is not for this checkout's base. No direction is asserted -- a sha
        # inequality is not an ordering, and a stale PR run could otherwise claim main "moved" to
        # an OLDER commit (Macroscope, #767). What matters is that the base's own committed
        # baseline is checked too: against the artifact alone, a PR that reverts a field to the
        # attested-but-superseded value would read as "no change" (Macroscope, #767).
        stale = (
            f" · the attestation is for main `{attestation.main_sha[:12]}`, not this checkout's base `{base_tip[:12]}`"
        )
        reference, problem = git_baseline_at(base_tip)
        if problem is not None or reference is None:
            return Row(
                "baseline attestation",
                RED,
                f"the attestation ({where}) predates base `{base_tip[:12]}` and {problem or 'the base baseline could not be read'}",
                ATTESTATION_METHOD,
                ATTESTATION_NOTES,
            )
        against_base = baseline_changes(reference, view)
        by_hand = [(path, old, new) for path, old, new in against_base if not licensed(attestation, path, new)]
        if by_hand:
            listed = "; ".join(describe_change(*change) for change in by_hand[:6])
            return Row(
                "baseline attestation",
                RED,
                f"**baseline changed by hand; no CI attestation for these values** — differs from base "
                f"`{base_tip[:12]}`: {listed}; the attestation ({where}) predates that base and measured none of them",
                ATTESTATION_METHOD,
                ATTESTATION_NOTES,
            )
        # The PR carries exactly what main has committed at the base (or values main measured).
        # The stale artifact is not re-applied on top of that: main's committed baseline is the
        # newer truth, and its own attest run will refresh the artifact.
        if against_base:
            listed = "; ".join(describe_change(*change) for change in against_base[:6])
            return Row(
                "baseline attestation",
                GREEN,
                f"baseline moved to values measured by main ({where}) relative to base `{base_tip[:12]}`: {listed}{stale}",
                ATTESTATION_METHOD,
                ATTESTATION_NOTES,
            )
        return Row(
            "baseline attestation",
            GREEN,
            f"baseline `{baseline_digest(reference)[:12]}` unchanged from base `{base_tip[:12]}` by git{stale}",
            ATTESTATION_METHOD,
            ATTESTATION_NOTES,
        )
    if not changes:
        return Row(
            "baseline attestation",
            GREEN,
            f"baseline `{attestation.digest[:12]}` matches the main attestation ({where}){stale}",
            ATTESTATION_METHOD,
            ATTESTATION_NOTES,
        )
    by_hand = [(path, old, new) for path, old, new in changes if not licensed(attestation, path, new)]
    if by_hand:
        listed = "; ".join(describe_change(*change) for change in by_hand[:6])
        more = f" (+{len(by_hand) - 6} more)" if len(by_hand) > 6 else ""
        return Row(
            "baseline attestation",
            RED,
            f"**baseline changed by hand; no CI attestation for these values** — {listed}{more}; "
            f"the latest main attestation ({where}) measured none of them{stale}",
            ATTESTATION_METHOD,
            ATTESTATION_NOTES,
        )
    listed = "; ".join(describe_change(*change) for change in changes[:6])
    return Row(
        "baseline attestation",
        GREEN,
        f"baseline moved to values measured by main ({where}): {listed}{stale}",
        ATTESTATION_METHOD,
        ATTESTATION_NOTES,
    )


def licensed(attestation: Attestation, path: BaselinePath, new: object) -> bool:
    """Did the main run measure exactly this value for this path? Value AND type: `False == 0` in
    Python, and a measured boolean swapped for a count is a hand edit wearing the measurement's
    value (Macroscope, #767)."""
    measured = measured_value(attestation, path)
    return measured is not _UNMEASURED and measured == new and type(measured) is type(new)


def bootstrap_row(reason: str, corpus: dict, lineage: tuple[str, ...] | None) -> Row:
    """No attest run has ever completed on main. The base tip's committed baseline stands in.

    GREEN here means exactly one thing was measured: that this PR does not move the baseline.
    Once one attest run exists this branch is never taken again -- an expired or missing artifact
    after that is `--attestation-unresolved`, a finding, because a writer that has run before and
    published nothing is not a bootstrap.
    """
    base_tip = base_tip_of(lineage)
    if base_tip is None:
        return Row(
            "baseline attestation",
            RED,
            f"bootstrap ({reason}), and this checkout is not a merge ref so git cannot name the base "
            "commit; nothing here shows whether the baseline was changed",
            ATTESTATION_METHOD_BOOTSTRAP,
            ATTESTATION_NOTES,
        )
    reference, problem = git_baseline_at(base_tip)
    if problem is not None or reference is None:
        return Row(
            "baseline attestation",
            RED,
            f"bootstrap ({reason}), and {problem or 'the base baseline could not be read'}",
            ATTESTATION_METHOD_BOOTSTRAP,
            ATTESTATION_NOTES,
        )
    changes = baseline_changes(reference, baseline_view(corpus))
    if changes:
        listed = "; ".join(describe_change(*change) for change in changes[:6])
        return Row(
            "baseline attestation",
            RED,
            f"**baseline changed by hand; no CI attestation for these values** — {listed}; "
            f"bootstrap ({reason}), compared against base `{base_tip[:12]}` by git",
            ATTESTATION_METHOD_BOOTSTRAP,
            ATTESTATION_NOTES,
        )
    return Row(
        "baseline attestation",
        GREEN,
        f"bootstrap: {reason}; baseline `{baseline_digest(reference)[:12]}` unchanged from base "
        f"`{base_tip[:12]}` by git — attestation begins with the first main run of `{ATTEST_WORKFLOW}`",
        ATTESTATION_METHOD_BOOTSTRAP,
        ATTESTATION_NOTES,
    )


def row_baseline_attestation(probe: Probe, corpus: dict) -> Row:
    """Say what the baseline is being checked against, and go RED the moment it was hand-edited."""
    if probe.attestation_problem:
        return Row("baseline attestation", RED, probe.attestation_problem, ATTESTATION_METHOD, ATTESTATION_NOTES)
    if probe.attestation_bootstrap:
        return bootstrap_row(probe.attestation_bootstrap, corpus, probe.head_lineage)
    if probe.attestation is None:
        reason = probe.attestation_unavailable or ATTESTATION_UNAVAILABLE_DEFAULT
        return Row("baseline attestation", NA, f"n/a — {reason}", ATTESTATION_METHOD, ATTESTATION_NOTES)
    return attested_row(probe.attestation, corpus, base_tip_of(probe.head_lineage))


def base_tip_of(lineage: tuple[str, ...] | None) -> str | None:
    """The base branch's tip, which on a `pull_request` merge ref is the FIRST parent.

    Only a merge ref has one. A direct checkout's first parent is the PR head's own parent -- a
    commit on the PR's branch, not main's tip -- and comparing against it would let a baseline
    edit two commits back pass as "unchanged". Same positional rule as `checkout_stands_for`.
    """
    if lineage is None or len(lineage) < 3:
        return None
    return lineage[1]


PROVENANCE_NOTES = (
    "Sha half of #749 keg-mode provenance: a keg built from this wheel can answer "
    "`__build_sha__`. The helper-age and served-process predicates need a running BrainBar "
    "and are measured only by `scripts/sprint_gate.py` on an installed Mac. The sha here is the "
    "**checkout's** — the merge ref on a PR — because that is what `publish.yml` stamps at release "
    "time; the PR-head sha this table describes is the one in `commit provenance` above."
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
    return measured_signature_row(report)


def signature_value(report: SignatureReport) -> str:
    """The counts, where they came from, and whether brew was happy about getting there."""
    valid, invalid = report.valid or 0, report.invalid or 0
    where = " · ".join(part for part in (report.keg, report.runner) if part)
    value = f"{valid} valid / {invalid} invalid" + (f" · {where}" if where else "")
    if report.install_outcome and report.install_outcome != "success":
        # A clean sweep after Homebrew `ofail`ed relocation is the #37 fix WORKING, so the row stays
        # GREEN -- but silently, it would read as an unremarkable install. Say what happened.
        value += (
            f" · `brew install` exited non-zero (outcome: {report.install_outcome}); "
            "the keg installed and the sweep ran after it"
        )
    return value


def measured_signature_row(report: SignatureReport) -> Row:
    value = signature_value(report)
    valid, invalid = report.valid or 0, report.invalid or 0
    if valid + invalid == 0:
        # An empty sweep proves nothing: release-verify-signatures.sh itself treats it as fatal,
        # and it prints `valid: 0 / invalid: 0` BEFORE doing so, so counts alone are not evidence.
        return Row(
            "signature_valid",
            RED,
            f"{value} — the keg exposed no native extensions to verify, so nothing was proven",
            SIGNATURE_METHOD_MEASURED,
            SIGNATURE_NOTES,
        )
    if invalid:
        named = ", ".join(report.invalid_files[:5]) or "see the parity job log"
        return Row("signature_valid", RED, f"{value} — {named}", SIGNATURE_METHOD_MEASURED, SIGNATURE_NOTES)
    return Row("signature_valid", GREEN, value, SIGNATURE_METHOD_MEASURED, SIGNATURE_NOTES)


# `row_commit_provenance` leads: every other row's value belongs to the commit it names, so a
# reader has to see that sha before reading a number measured against it. `baseline attestation`
# is second for the same reason: it names what the numbers are measured AGAINST.
ROW_BUILDERS = (
    row_commit_provenance,
    row_baseline_attestation,
    row_provenance,
    row_mapped_bytes,
    row_search_latency,
    row_idle_cpu,
    row_signature_valid,
)


def collect(probe: Probe, corpus: dict) -> list[Row]:
    return [builder(probe, corpus) for builder in ROW_BUILDERS]


# --------------------------------------------------------------------------------------------
# Rendering
# --------------------------------------------------------------------------------------------


def escape_cell(text: str) -> str:
    return text.replace("|", "\\|").replace("\n", " ")


def render(rows: list[Row], probe: Probe, run_url: str | None, now: datetime) -> str:
    # Every sha labelled for what it IS, because printing one unlabelled `HEAD` is what made
    # #759's table unverifiable: the checkout sha is GitHub's merge ref and is on no PR page.
    #
    # `measured_sha` is the commit this run was TRIGGERED for, which on the superseded path is
    # precisely NOT the PR head. Labelling it `PR head` put a footer four lines under the row
    # contradicting it, and on the unresolved path printed a PR head the run had just said it could
    # not read -- #759's defect re-committed one line lower (reviewer 79c16541, F1).
    checkout = probe.head_sha[:12] if probe.head_sha else "unknown"
    shas = []
    if probe.measured_sha:
        shas.append(f"measured `{probe.measured_sha[:12]}`")
        shas.append(f"PR head `{probe.pr_head_sha[:12]}`" if probe.pr_head_sha else "PR head unread")
    shas.append(f"checkout `{checkout}`")
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
        f"_Measured on {probe.os_name}/{probe.architecture} · " + " · ".join(shas) + f"{run} · "
        f"updated {now.strftime('%Y-%m-%d %H:%M:%S')} UTC_",
        "",
    ]
    return "\n".join(lines)


# --------------------------------------------------------------------------------------------
# Writing an attestation (main runs only)
# --------------------------------------------------------------------------------------------


def row_measurement(row: Row, probe: Probe) -> dict | None:
    """The numeric payload behind a row, for the run history a margin can be derived from.

    Only a row that measured counts carries one; a status string is not a measurement. Today that
    is `signature_valid`, and only when the macOS parity job ran.
    """
    if row.name == "signature_valid" and probe.signature is not None and probe.signature.status == SIGNATURE_MEASURED:
        return {"valid": probe.signature.valid, "invalid": probe.signature.invalid}
    return None


def attestation_payload(
    rows: list[Row], probe: Probe, corpus: dict, main_sha: str, run_id: int, run_attempt: int, now: datetime
) -> dict:
    view = baseline_view(corpus)
    return {
        "schema": ATTESTATION_SCHEMA,
        "run_id": run_id,
        "run_attempt": run_attempt,
        "main_sha": main_sha,
        "measured_at": now.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "workflow": ATTEST_WORKFLOW,
        "baseline": view,
        "baseline_sha256": baseline_digest(view),
        # Dotted baseline path -> value this run measured. Empty until a runner-side collector
        # exists for a baseline field; a PR may only move a field to a value listed here.
        "measured": {},
        "rows": {
            row.name: {
                "status": row.status,
                "value": row.value,
                "method": row.method,
                "measurement": row_measurement(row, probe),
            }
            for row in rows
        },
    }


def attest_flags_problem(args: argparse.Namespace) -> str | None:
    """The four attest flags are one claim; a partial or dishonest set is refused before anything runs."""
    flags = (args.attest_out, args.main_sha, args.run_id, args.run_attempt)
    if any(flag is None for flag in flags):
        return "`--attest-out`, `--main-sha`, `--run-id` and `--run-attempt` go together; an attestation names its run"
    if not (honest_run_number(args.run_id) and honest_run_number(args.run_attempt)):
        return f"`--run-id {args.run_id}` / `--run-attempt {args.run_attempt}` are not positive integers; no reader would accept them"
    if SHA_PATTERN.fullmatch(args.main_sha) is None:
        return f"`--main-sha {args.main_sha}` is not a 40-hex commit sha"
    return None


def attest_checkout_problem(main_sha: str, probe: Probe, rows: list[Row], corpus: dict) -> str | None:
    """Whether THIS checkout may stand behind an attestation for `main_sha`."""
    if probe.head_sha != main_sha:
        # The attestation says "main at this sha saw this baseline". If the checkout is not that
        # sha, the claim is about a commit this run does not have.
        return f"`--main-sha {main_sha[:12]}` is not the checkout (`{probe.head_sha[:12] if probe.head_sha else 'unknown'}`)"
    missing = [field for field in BASELINE_FIELDS if field not in corpus]
    if missing:
        return f"the checkout's baseline lacks {', '.join(f'`{field}`' for field in missing)}; an attestation must cover every baseline field"
    if any(row.status == RED for row in rows):
        return "a run with a RED row does not attest; the finding comes first"
    return None


def attest_refusal(args: argparse.Namespace, probe: Probe, rows: list[Row], corpus: dict) -> str | None:
    """Why this run may NOT publish an attestation. None means it may.

    Every rule here mirrors one in `read_attestation`: an attestation this writer publishes and
    that reader rejects would leave main's baseline unusable for every PR (Macroscope, #766). The
    checks live in the two helpers above so that adding a rule does not keep pushing one
    function's branch count up -- the same split `select_signature` made.
    """
    if all(flag is None for flag in (args.attest_out, args.main_sha, args.run_id, args.run_attempt)):
        return None
    return attest_flags_problem(args) or attest_checkout_problem(args.main_sha, probe, rows, corpus)


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
    parser.add_argument(
        "--measured-sha",
        help=(
            "The PR-head commit this run was triggered for (`github.event.pull_request.head.sha`). "
            "Must be checked out, or be the PR-head parent of the merge ref that is."
        ),
    )
    parser.add_argument(
        "--pr-head-sha",
        help=(
            "The PR's head read LIVE when the table is collected. Passing it asserts this run can "
            "prove it is current: a mismatch with --measured-sha is RED, never n/a."
        ),
    )
    parser.add_argument(
        "--pr-head-unresolved",
        help=(
            "Why the live PR head could not be read. The run was asked to prove it is current and "
            "could not, so this renders RED with that reason — it is not a capability gap."
        ),
    )
    parser.add_argument(
        "--attestation",
        type=Path,
        help=(
            "The `ratchet-attestation` artifact of the latest successful main run of ratchet-attest.yml, "
            "fetched by the workflow through the Actions API. Passing it asserts one exists: a missing or "
            "malformed file is RED, never n/a."
        ),
    )
    parser.add_argument(
        "--attestation-bootstrap",
        help=(
            "Why there is no attestation yet: the Actions API shows no successful attest run on main. "
            "The row then compares against the base commit's committed baseline, by git."
        ),
    )
    parser.add_argument(
        "--attestation-unresolved",
        help="Why the main attestation could not be fetched. Renders RED with that reason — it is not a capability gap.",
    )
    parser.add_argument("--run-url", help="Link back to the workflow run that produced this table")
    parser.add_argument("--out", type=Path, help="Also write the rendered table here")
    attest = parser.add_argument_group(
        "attesting (main runs only)",
        "Write this run's view of the baseline as an attestation. All four flags go together, the "
        "checkout must BE --main-sha, and a run with a RED row writes nothing.",
    )
    attest.add_argument("--attest-out", type=Path, help="Where to write attestation.json")
    attest.add_argument("--main-sha", help="The main commit this run was triggered for (`github.sha` on a push)")
    attest.add_argument("--run-id", type=int, help="`github.run_id`")
    attest.add_argument("--run-attempt", type=int, help="`github.run_attempt`")
    args = parser.parse_args(argv)

    corpus = json.loads(CORPUS.read_text(encoding="utf-8"))
    probe = Probe.detect(
        corpus,
        args.wheel,
        args.wheel_glob,
        args.signature_report,
        args.signature_unavailable,
        args.measured_sha,
        args.pr_head_sha,
        args.pr_head_unresolved,
        args.attestation,
        args.attestation_bootstrap,
        args.attestation_unresolved,
    )
    rows = collect(probe, corpus)
    now = datetime.now(timezone.utc)
    table = render(rows, probe, args.run_url, now)

    if args.out:
        args.out.write_text(table, encoding="utf-8")
    print(table)
    for row in rows:
        if row.status == RED:
            print(f"::error title=Ratchet RED: {row.name}::{row.value}", file=sys.stderr)
    refusal = attest_refusal(args, probe, rows, corpus)
    if refusal is not None:
        print(f"::error title=Ratchet attestation not written::{refusal}", file=sys.stderr)
        return 1
    if args.attest_out is not None:
        payload = attestation_payload(rows, probe, corpus, args.main_sha, args.run_id, args.run_attempt, now)
        args.attest_out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    # n/a never fails the build; RED always does -- a row nobody has to clear is decoration.
    return 1 if any(row.status == RED for row in rows) else 0


if __name__ == "__main__":
    raise SystemExit(main())
