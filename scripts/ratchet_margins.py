"""Ratchet margins measured from attested green main runs — never a round number nobody measured.

Ratchet bolt-down (c). The problem this solves is three days old and it is the watcher: R3's
idle-CPU soak measured 4.88% then 6.41% on near-identical code against a flat ``<5%`` gate. The
run-to-run noise (4.0–6.5%) straddled the threshold, so a single measurement could not resolve pass
from fail, and the worker rightly refused to re-roll soaks until one passed. The gate was a round
number; ``latency_regression_fraction: 0.10`` in ``tests/fixtures/sprint_gate/corpus.json`` was the
same defect, and this module is what replaced it.

The rule here
-------------
* A row's margin is derived from the spread of its value across **attested green main runs** --
  the ``ratchet-attestation`` artifacts that ``.github/workflows/ratchet-attest.yml`` publishes from
  ``main`` (ratchet bolt-down (b) owns that store; this module only READS it). Nothing here has a
  default path and nothing here accepts inline live values: a margin the PR tree can hand-edit is
  the hand-editable baseline all over again.
* **Fewer than five runs is ``unmeasured``**, rendered as such, never as a number. A band from two
  runs is a guess wearing a lab coat.
* The statistic is a **one-sided 99% prediction limit for one future run**::

      limit = mean + t(0.99, n-1) * s * sqrt(1 + 1/n)

  because that is literally the question the gate asks -- "is this one new measurement consistent
  with the green population?" -- and because the multiplier widens on its own for small ``n``
  (k = 4.10 at n=5, 2.96 at n=10, 2.55 at n=30). Five runs give a wide, honest band; the band
  tightens as attested runs accumulate, which is what a ratchet is supposed to do. The alternatives
  were rejected with numbers: "max of the five runs" is exceeded by a sixth run from the same
  distribution one time in six (16.7% false RED per row per run); a fixed ``mean + 2σ`` at n=5 is
  exceeded 7.1% of the time. Both are false-RED generators, and this week found two of those.
* A log-scale band was considered for latency's right skew and rejected: on the seven real green
  main p50 values it widened the limit to ×3.5 instead of ×2.06, because the outlier in that series
  is LOW (98 ms), and idle CPU can legitimately be 0.0, where a log has nothing to say.

The t-quantiles are a fixed table (df 4–29, one-sided 99%) and, past it, the Cornish–Fisher expansion
of the Student-t quantile around the normal one. Both checked against scipy 1.17.0 ``t.ppf(0.99, df)``:
the table to three decimals, the expansion to within 1e-5 for every df ≥ 30. An earlier draft fell
back to the plain normal quantile past the table; a reviewer (Macroscope, #763) pointed out that is
SMALLER than every finite-df t-quantile, so a 31-run series would have been judged against a band
about 5% tighter than the stated 99% -- the false-RED direction. The expansion is why that is gone.
"""

from __future__ import annotations

import json
import math
import os
import re
import statistics
from dataclasses import dataclass
from pathlib import Path

MINIMUM_RUNS = 5
CONFIDENCE = 0.99
MEASURED = "measured"
UNMEASURED = "unmeasured"
SHA_PATTERN = re.compile(r"[0-9a-f]{40}")

# The attestation document (b) publishes, schema 1. Only the fields this reader depends on are
# checked; the rest of the document belongs to the `baseline attestation` row.
ATTESTATION_SCHEMA = 1
ATTESTATION_FILENAME = "attestation.json"

# Where in an attestation each row's per-run value lives. `measured` is (b)'s flat map of
# "dotted baseline path -> the value this main run measured", so latency keys are the corpus's own
# baseline paths, and idle CPU -- which has a budget in the corpus but no baseline value -- gets
# its own key per sampled process. Segments, not one dotted string, so a key that itself contains
# a dot cannot be split wrong.
LATENCY_P50 = ("measured", "latency_baseline_ms.p50")
LATENCY_P95 = ("measured", "latency_baseline_ms.p95")
LATENCY_KEYS = {"p50": LATENCY_P50, "p95": LATENCY_P95}


def idle_cpu_key(process: str) -> tuple[str, ...]:
    return ("measured", f"idle_cpu_pct.{process}")


# One-sided 99% Student-t quantiles by degrees of freedom (n - 1), df 4..29. Past the table, the
# Cornish-Fisher expansion below (never the bare normal quantile: that is tighter than every t).
T_99_ONE_SIDED = {
    4: 3.747,
    5: 3.365,
    6: 3.143,
    7: 2.998,
    8: 2.896,
    9: 2.821,
    10: 2.764,
    11: 2.718,
    12: 2.681,
    13: 2.650,
    14: 2.624,
    15: 2.602,
    16: 2.583,
    17: 2.567,
    18: 2.552,
    19: 2.539,
    20: 2.528,
    21: 2.518,
    22: 2.508,
    23: 2.500,
    24: 2.492,
    25: 2.485,
    26: 2.479,
    27: 2.473,
    28: 2.467,
    29: 2.462,
}
Z_99_ONE_SIDED = 2.326347874  # full precision: the expansion below is sensitive to the third decimal


class AttestationError(ValueError):
    """The attested-run store could not be read as what it claims to be. Fail closed; say why."""


@dataclass(frozen=True)
class Margin:
    """What one row's band is, or why it has none yet.

    ``kind`` is ``measured`` or ``unmeasured``. Every numeric field is ``None`` when unmeasured,
    on purpose: a caller that formats ``margin.limit`` without checking ``kind`` gets ``None`` in
    its output, not a plausible-looking figure.
    """

    kind: str
    n: int
    minimum: int = MINIMUM_RUNS
    mean: float | None = None
    stdev: float | None = None
    k: float | None = None
    limit: float | None = None
    confidence: float = CONFIDENCE


def t_quantile(df: int) -> float:
    if df < min(T_99_ONE_SIDED):
        raise ValueError(f"df={df} is below the {MINIMUM_RUNS}-run minimum this table starts at")
    if df in T_99_ONE_SIDED:
        return T_99_ONE_SIDED[df]
    # Cornish-Fisher expansion of the Student-t quantile in powers of 1/df (Abramowitz & Stegun 26.7.5).
    # |error| <= 1e-5 against scipy for df >= 30, and it approaches Z_99 from ABOVE, as t does.
    z = Z_99_ONE_SIDED
    return (
        z
        + (z**3 + z) / (4 * df)
        + (5 * z**5 + 16 * z**3 + 3 * z) / (96 * df**2)
        + (3 * z**7 + 19 * z**5 + 17 * z**3 - 15 * z) / (384 * df**3)
    )


def measured_margin(values: list[float], *, minimum_runs: int = MINIMUM_RUNS) -> Margin:
    """The band for one row, from the values it took on attested green main runs."""
    n = len(values)
    if n < minimum_runs:
        return Margin(kind=UNMEASURED, n=n, minimum=minimum_runs)
    mean = statistics.fmean(values)
    stdev = statistics.stdev(values)  # sample (n-1): the population is the runs we did NOT see
    k = t_quantile(n - 1) * math.sqrt(1 + 1 / n)
    return Margin(kind=MEASURED, n=n, minimum=minimum_runs, mean=mean, stdev=stdev, k=k, limit=mean + k * stdev)


def within(margin: Margin, value: float) -> bool | None:
    """True/False against a measured band; None when there is no band to be within."""
    if margin.kind != MEASURED or margin.limit is None:
        return None
    return value <= margin.limit


def describe(margin: Margin, *, unit: str) -> str:
    """The sentence a table prints so a reader can see WHY a value is RED, not just that it is."""
    if margin.kind != MEASURED:
        return (
            f"margin unmeasured — {margin.n} of the {margin.minimum} attested green main runs it needs; "
            "no verdict is rendered from fewer"
        )
    if margin.mean is None or margin.stdev is None or margin.k is None or margin.limit is None:
        # Not an assert: under `python -O` an assert vanishes and this would print "None ms" as a limit.
        raise ValueError("a measured margin must carry mean, stdev, k and limit")
    percent = int(round(margin.confidence * 100))
    return (
        f"limit {margin.limit:.1f} {unit} = mean {margin.mean:.1f} {unit} + k {margin.k:.2f} × σ "
        f"{margin.stdev:.1f} {unit} (n={margin.n} attested green main runs; one-sided {percent}% "
        "prediction limit for one run)"
    )


# --------------------------------------------------------------------------------------------
# Reading the attested store: one `attestation.json` per main run, downloaded by the caller
# (`gh run download <id> -n ratchet-attestation -D <root>/<id>`). This module never fetches.
# --------------------------------------------------------------------------------------------


def load_attestations(root: Path) -> list[dict]:
    """Every attestation under ``root`` (or ``root`` itself, if it is one file), validated.

    A root that does not exist is an error: the caller asked for attestations and named a place
    that is not there. An existing directory with none in it is the bootstrap state -- main has not
    attested yet -- and reads as zero runs, which every margin then reports as ``unmeasured``.
    """
    try:
        if root.is_file():
            paths = [root]
        elif root.is_dir():
            paths = find_attestations(root)
        else:
            raise AttestationError(f"attestations root `{root}` is neither a file nor a directory")
    except OSError as error:
        # A root that exists but cannot be scanned (permissions, a dead mount) is the same finding as a
        # missing one, and must not escape as a traceback past the callers' fail-closed handling.
        raise AttestationError(f"attestations root `{root}` could not be scanned ({type(error).__name__})") from error
    attestations = [load_attestation(path) for path in paths]
    seen: dict[object, Path] = {}
    for path, attestation in zip(paths, attestations, strict=True):
        run_id = attestation["run_id"]
        if run_id in seen:
            # The same run twice is not two observations: it would narrow the band (or repeat an
            # outlier) with evidence that was only ever collected once. The store is malformed.
            raise AttestationError(f"attested run {run_id} appears twice: `{seen[run_id]}` and `{path}`")
        seen[run_id] = path
    return attestations


def find_attestations(root: Path) -> list[Path]:
    """Every `attestation.json` under ``root``, or an OSError for a directory that cannot be scanned.

    Not ``Path.rglob``: it swallows PermissionError while walking and returns what it could see, which
    here would read as "main has attested zero runs" -- a silent degrade wearing the bootstrap state's
    clothes. ``os.walk`` with ``onerror`` re-raising makes an unreadable directory a finding instead.
    """

    def fail(error: OSError) -> None:
        raise error

    found: list[Path] = []
    for dirpath, _dirs, files in os.walk(root, onerror=fail):
        found.extend(Path(dirpath) / name for name in files if name == ATTESTATION_FILENAME)
    return sorted(found)


def load_attestation(path: Path) -> dict:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise AttestationError(f"attestation `{path}` could not be parsed ({type(error).__name__})") from error
    return validate_attestation(payload, str(path))


def validate_attestation(payload: object, source: str) -> dict:
    """Refuse anything that is not a schema-1 attestation of a 40-hex main sha with a `measured` map."""
    if not isinstance(payload, dict):
        raise AttestationError(f"attestation `{source}` is a JSON {type(payload).__name__}, not an object")
    if payload.get("schema") != ATTESTATION_SCHEMA:
        raise AttestationError(f"attestation `{source}` has schema {payload.get('schema')!r}, not {ATTESTATION_SCHEMA}")
    if "run_id" not in payload:
        raise AttestationError(f"attestation `{source}` names no `run_id`")
    if not isinstance(payload.get("main_sha"), str) or SHA_PATTERN.fullmatch(payload["main_sha"]) is None:
        raise AttestationError(f"attestation `{source}` has no 40-hex `main_sha`")
    if not isinstance(payload.get("measured_at"), str) or not payload["measured_at"].strip():
        raise AttestationError(f"attestation `{source}` has no `measured_at`")
    if not isinstance(payload.get("measured"), dict):
        raise AttestationError(f"attestation `{source}` has no `measured` object")
    for key, value in payload["measured"].items():
        # Checked at load, not at read: a row builder that formats notes cannot be the place a
        # malformed value first surfaces (Macroscope, #765). `null` means "not measured this run".
        if value is not None and not honest_value(value):
            raise AttestationError(
                f"attestation `{source}` (run {payload['run_id']}): `measured.{key}` is {value!r}, "
                "not a finite non-negative number or null"
            )
    return payload


def honest_value(value: object) -> bool:
    # bool is an int in Python; `true` would otherwise be a measurement of 1.
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value >= 0


def lookup(document: dict, key: tuple[str, ...]) -> object | None:
    node: object = document
    for segment in key:
        if not isinstance(node, dict) or segment not in node:
            return None
        node = node[segment]
    return node


def series(attestations: list[dict], key: tuple[str, ...]) -> list[float]:
    """Every attested run's value at ``key``, in store order. Runs without the key do not count."""
    collected: list[float] = []
    for attestation in attestations:
        value = lookup(attestation, key)
        if value is None:
            continue
        if not honest_value(value):
            # Unreachable for a store that came through validate_attestation; kept for inline lists
            # handed over by callers that did not, so a bad value is never silently coerced.
            raise AttestationError(
                f"attested run {attestation['run_id']}: `{'.'.join(key)}` is {value!r}, "
                "not a finite non-negative number"
            )
        collected.append(float(value))
    return collected


def margin_for(attestations: list[dict] | None, key: tuple[str, ...]) -> Margin:
    """The band at ``key``; with no store handed over at all, ``unmeasured`` with zero runs."""
    if attestations is None:
        return Margin(kind=UNMEASURED, n=0)
    return measured_margin(series(attestations, key))
