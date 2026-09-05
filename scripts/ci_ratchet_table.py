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

# `python scripts/ci_ratchet_table.py` puts scripts/, not ROOT, on sys.path; the margins module is a sibling.
sys.path.insert(0, str(ROOT))
from scripts import ratchet_margins as margins  # noqa: E402

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
    # Ratchet (c): the attested green main runs every measured margin is derived from, read from
    # the `ratchet-attestation` artifacts of `ratchet-attest.yml` runs on main. None when nobody
    # handed this run a store (margins render `unmeasured` with 0 runs); `attestations_problem` when
    # a store WAS handed over and could not be read -- a finding, on the signature report's rule.
    attestations: tuple[dict, ...] | None = None
    attestations_problem: str | None = None

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
        attestations: Path | None = None,
    ) -> Probe:
        selected = select_wheel(wheel, wheel_glob)
        signature = select_signature(signature_report, signature_unavailable)
        commit = select_commit(measured_sha, pr_head_sha, pr_head_unresolved)
        attested = select_attestations(attestations)
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
            attestations=attested.attestations,
            attestations_problem=attested.problem,
        )


@dataclass(frozen=True)
class AttestationSelection:
    attestations: tuple[dict, ...] | None = None
    problem: str | None = None


def select_attestations(root: Path | None) -> AttestationSelection:
    """Read the attested main runs FAIL-CLOSED: a store handed over and unreadable is a finding."""
    if root is None:
        return AttestationSelection()
    try:
        return AttestationSelection(attestations=tuple(margins.load_attestations(root)))
    except margins.AttestationError as error:
        return AttestationSelection(problem=f"{error} — every measured margin in this table depends on it")


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


def margin_notes(probe: Probe, unit: str, keys: dict[str, tuple[str, ...]]) -> tuple[str, str | None]:
    """The margin each sub-row applies, spelled out, so a reader sees WHY a value would be RED.

    Ratchet (c): a round number hides its reasoning; a measured band exposes it -- mean, spread,
    multiplier, and how many attested green main runs stand behind it. Fewer than five says
    `unmeasured` and prints no limit at all. Returns ``(notes, problem)``: a band that cannot be
    formed from valid attestations (a non-finite limit) is a problem for the row to render RED, not
    an exception for the collector to die on.
    """
    sentences = []
    for name, key in keys.items():
        try:
            margin = margins.margin_for(list(probe.attestations or ()), key)
            sentences.append(f"Margin {name}: {margins.describe(margin, unit=unit)}.")
        except (margins.AttestationError, ValueError) as error:
            # ValueError is describe()'s refusal of an incomplete measured margin -- unreachable with
            # today's measured_margin, but this row follows the signature report's never-abort rule.
            return "", f"margin {name}: {error} — this row cannot say what band it applies"
    return " ".join(sentences), None


def row_search_latency(probe: Probe, corpus: dict) -> Row:
    method = "socket · installed Mac"
    if probe.attestations_problem:
        return Row("search p50/p95", RED, probe.attestations_problem, method, SEARCH_LATENCY_NOTES)
    baseline = corpus["latency_baseline_ms"]
    bands, problem = margin_notes(probe, "ms", margins.LATENCY_KEYS)
    if problem:
        return Row("search p50/p95", RED, problem, method, SEARCH_LATENCY_NOTES)
    notes = (
        f"{bands} Calibrated on {baseline['hostname']} at "
        f"{baseline['captured_at']} under {baseline['captured_under']} "
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
    return Row("search p50/p95", NA, f"n/a — {reason}", method, notes)


SEARCH_LATENCY_NOTES = (
    "Limits are measured bands from attested green main runs (`scripts/ratchet_margins.py`), never a "
    "flat fraction of a captured baseline. Not measured by this run."
)

IDLE_CPU_PROCESSES = ("daemon", "helper", "watcher")


def row_idle_cpu(probe: Probe, corpus: dict) -> Row:
    method = "ps sampling · installed Mac"
    thresholds = corpus["thresholds"]
    budget = (
        f"Ceiling: average CPU < {thresholds['cpu_percent']}% over a "
        f"{thresholds['resource_window_seconds']} s window (`resource_budget` in `scripts/sprint_gate.py`), "
        "ratified and kept as a hard budget."
    )
    if probe.attestations_problem:
        return Row("idle CPU", RED, probe.attestations_problem, method, budget)
    # The ceiling cannot see drift under it: R3's soak measured 4.88% then 6.41% on near-identical
    # code, both far under 30. The ratchet band is what would have, and it is stated per process.
    bands, problem = margin_notes(probe, "%", {name: margins.idle_cpu_key(name) for name in IDLE_CPU_PROCESSES})
    if problem:
        return Row("idle CPU", RED, problem, method, budget)
    notes = (
        f"{budget} {bands} Needs the BrainBar daemon, helper and watcher actually running. Not measured by this run."
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
    return Row("idle CPU", NA, f"n/a — {reason}", method, notes)


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
# reader has to see that sha before reading a number measured against it.
ROW_BUILDERS = (
    row_commit_provenance,
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
        "--attestations",
        type=Path,
        help=(
            "Directory of `attestation.json` files from successful `ratchet-attest.yml` runs on main (one per "
            "run), or one such file. The measured margins in Notes come from these; without them they render "
            "`unmeasured`. A path that cannot be read turns the margin rows RED, never n/a."
        ),
    )
    parser.add_argument("--run-url", help="Link back to the workflow run that produced this table")
    parser.add_argument("--out", type=Path, help="Also write the rendered table here")
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
        args.attestations,
    )
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
