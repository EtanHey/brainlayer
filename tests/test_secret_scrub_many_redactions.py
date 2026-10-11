"""The scrubber stays fast when an input yields thousands of redactions.

The regex scan was already linear, but the bookkeeping after it was O(k^2) in the
number of redactions or quarantined tokens: overlap removal, the overlap checks
against existing spans, and quarantine de-duplication. 1 MB of
`"api_key":"<V>",` lines took ~15.6 s. Once BrainBar's store path scrubs, that is
a stall on the MCP hot path.

Timings run in a killable child process with a hard timeout, so RED cannot hang
the suite. Synthetic values only.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

import brainlayer

SRC_ROOT = Path(brainlayer.__file__).resolve().parent.parent
ONE_MB = 1024 * 1024
TIME_BOUND_SECONDS = 3.0
HARD_TIMEOUT_SECONDS = 40

VALUE = "aB3dE5gH7jK9mN1pQ2rS4tU6vW8xY0zC"
UNLABELED = "mF9qP2xL7vR8sK4nT6yB3cD5eG7hJ9kL2mN4pQ6r"

MANY_REDACTION_SHAPES = {
    "quoted-assignment-lines": '"api_key":"' + VALUE + '",',
    "unquoted-assignment-lines": "api_key=" + VALUE + "\n",
    "provider-token-lines": "ghp_" + "0" * 36 + "\n",
    "quarantine-token-lines": UNLABELED + "\n",
}


def _scrub_in_child(text: str) -> dict:
    code = (
        "import json, sys, time\n"
        f"sys.path.insert(0, {str(SRC_ROOT)!r})\n"
        "from brainlayer.pipeline.secret_scrub import scrub_secrets\n"
        "text = sys.stdin.read()\n"
        "start = time.perf_counter()\n"
        "result = scrub_secrets(text)\n"
        "print(json.dumps({'seconds': time.perf_counter() - start,"
        " 'redactions': len(result.redactions), 'quarantine': len(result.quarantine)}))\n"
    )
    try:
        proc = subprocess.run(
            [sys.executable, "-c", code],
            input=text,
            capture_output=True,
            text=True,
            timeout=HARD_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired:
        pytest.fail(f"scrub_secrets did not finish {len(text)} bytes within {HARD_TIMEOUT_SECONDS}s")
    assert proc.returncode == 0, proc.stderr[-500:]
    return json.loads(proc.stdout.strip().splitlines()[-1])


@pytest.mark.parametrize("shape", sorted(MANY_REDACTION_SHAPES))
def test_one_megabyte_with_thousands_of_findings_scrubs_quickly(shape):
    unit = MANY_REDACTION_SHAPES[shape]
    text = (unit * (ONE_MB // len(unit) + 1))[:ONE_MB]

    measured = _scrub_in_child(text)

    assert measured["redactions"] + measured["quarantine"] > 10_000, measured
    assert measured["seconds"] < TIME_BOUND_SECONDS, f"{shape}: {measured}"
