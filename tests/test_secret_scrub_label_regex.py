"""The label-gated assignment rule must be linear-time and must see JSON-quoted labels.

The timing tests run the scrubber in a child process with a hard timeout. The
superlinear shapes below took hours at 64 KB before the fix, and a thread
cannot be killed, so a subprocess is the only way to keep RED from hanging
the suite.

Synthetic values are alphabet walks, not credentials.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

import brainlayer
from brainlayer.pipeline.secret_scrub import scrub_secrets

SRC_ROOT = Path(brainlayer.__file__).resolve().parent.parent
ADVERSARIAL_BYTES = 64 * 1024
TIME_BOUND_SECONDS = 2.0
HARD_TIMEOUT_SECONDS = 20

# Long runs of label characters with a keyword inside and no `=`/`:` after it.
# Before the fix each of these backtracked roughly cubically (4 KB of "key-"
# took ~20 s; 16 KB of "monkey-a-a-…" took ~5.5 s).
ADVERSARIAL_SHAPES = {
    "repeated-keyword": "key-",
    "dotted-keyword": "a.token.",
    "monkey-run": "monkey-" + "a-" * 8,
    "tskey-run": "tskey-a-aaaaaa-" + "a-" * 8,
    "secret-dashes": "secret-" + "b-" * 4,
}

# 32 distinct characters: entropy ~5 bits, 24+ chars, not hex, no path separators.
VALUE = "aB3dE5gH7jK9mN1pQ2rS4tU6vW8xY0zC"


def _time_scrub_in_child(text: str) -> float:
    code = (
        "import json, sys, time\n"
        f"sys.path.insert(0, {str(SRC_ROOT)!r})\n"
        "from brainlayer.pipeline.secret_scrub import scrub_secrets\n"
        "text = sys.stdin.read()\n"
        "start = time.perf_counter()\n"
        "scrub_secrets(text)\n"
        "print(json.dumps(time.perf_counter() - start))\n"
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
        pytest.fail(
            f"scrub_secrets did not finish {len(text)} bytes within {HARD_TIMEOUT_SECONDS}s (catastrophic backtracking)"
        )
    assert proc.returncode == 0, proc.stderr[-500:]
    return float(json.loads(proc.stdout.strip().splitlines()[-1]))


@pytest.mark.parametrize("shape", sorted(ADVERSARIAL_SHAPES))
def test_label_rule_is_linear_on_64kb_adversarial_input(shape):
    unit = ADVERSARIAL_SHAPES[shape]
    text = (unit * (ADVERSARIAL_BYTES // len(unit) + 1))[:ADVERSARIAL_BYTES]

    elapsed = _time_scrub_in_child(text)

    assert elapsed < TIME_BOUND_SECONDS, f"{shape}: {ADVERSARIAL_BYTES} bytes took {elapsed:.2f}s"


@pytest.mark.parametrize(
    "text",
    [
        f'{{"api_key": "{VALUE}"}}',
        f'{{"access_token":"{VALUE}"}}',
        f"{{'client_secret': '{VALUE}'}}",
        f'"password" : "{VALUE}"',
        f'config = {{"auth": {{"token": "{VALUE}"}}}}',
    ],
)
def test_json_quoted_label_value_is_redacted(text):
    result = scrub_secrets(text)

    assert VALUE not in result.text
    assert "[REDACTED:assignment]" in result.text
    assert [redaction.provider for redaction in result.redactions] == ["assignment"]


@pytest.mark.parametrize(
    "text, expected",
    [
        (f"api_key = {VALUE}", "api_key = [REDACTED:assignment]"),
        (f"API_KEY={VALUE}", "API_KEY=[REDACTED:assignment]"),
        (f"export GITHUB_TOKEN='{VALUE}'", "export GITHUB_TOKEN='[REDACTED:assignment]'"),
        (f"db.password: {VALUE}.", "db.password: [REDACTED:assignment]."),
        (f"x-auth-header={VALUE}", "x-auth-header=[REDACTED:assignment]"),
    ],
)
def test_unquoted_label_forms_keep_redacting_exactly_as_before(text, expected):
    assert scrub_secrets(text).text == expected


@pytest.mark.parametrize(
    "text",
    [
        f'{{"name": "{VALUE}"}}',  # no label keyword
        '{"api_key": "short"}',  # below the length floor
        '{"api_key": "0123456789abcdef0123456789abcdef"}',  # hex join key
        '{"token": "/Users/someone/project/file-with-a-long-name.txt"}',  # path
    ],
)
def test_json_quoted_non_secrets_are_left_alone(text):
    result = scrub_secrets(text)

    assert result.text == text
    assert result.redactions == []


@pytest.mark.parametrize(
    "text",
    [
        f"api_key=/path/to/x:secret_token={VALUE}",
        f"auth:/tmp/a:token={VALUE}",
        f"0123456789abcdef0123456789abcdef:keyaccess.:/secretpassword={VALUE}",
    ],
)
def test_a_path_value_does_not_hide_a_later_label(text):
    """A match rejected as a path used to swallow every label inside its value."""
    result = scrub_secrets(text)

    assert VALUE not in result.text
    assert [redaction.provider for redaction in result.redactions] == ["assignment"]


@pytest.mark.parametrize("unit", ["key:/", "key:/key:", "token=/a/b:"])
def test_path_restart_stays_linear_on_64kb_adversarial_input(unit):
    text = (unit * (ADVERSARIAL_BYTES // len(unit) + 1))[:ADVERSARIAL_BYTES]

    elapsed = _time_scrub_in_child(text)

    assert elapsed < TIME_BOUND_SECONDS, f"{unit!r}: {ADVERSARIAL_BYTES} bytes took {elapsed:.2f}s"
