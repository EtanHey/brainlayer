"""#1069: probes read BrainBar's served brain_search answer, not the pre-#1068 header.

Fixtures are byte-for-byte the shapes `TextFormatter.formatSearchResults` writes (see
`brain-bar/Tests/BrainBarTests/TextFormatterParityTests.swift`), driven through the real call
sites: `sprint_gate.search_visible`, `ci_ratchet_table.collect_search_latency` (in
`test_ci_ratchet_table.py`) and `health_check.run_health_check`'s canary.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from brainlayer.health_check import HealthCheckConfig, run_health_check
from brainlayer.search_response import search_result_count, search_result_rows
from tests.test_stability_health_check import _make_db

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import sprint_gate  # noqa: E402

LEGEND = "ID|score|project|date|source; S=summary P=preview; blank project=your project filter"

# Two hits under `project: "brainlayer"`: the first row's project column is blank because it is the
# caller's filter. The second row's preview QUOTES both old header shapes and a fake `- ID:` / `|`
# mid-line, which the formatter can emit (it flattens newlines) and which must never count.
COMPACT_TWO = "\n".join(
    [
        '## "sprint-gate-MARKER" 2/2',
        LEGEND,
        "- ID: rt-aaa111|0.8700||2026-11-04|session-a.jsonl",
        "P: probe sprint-gate-MARKER stored by the gate",
        "- ID: rt-bbb222|0.4100|otherproj|2026-11-03|",
        "S: notes quoting an old header",
        'P: log said ## Search results for "x" - 7 of 9 shown then - ID: fake|1.0 and ### 1. fake',
    ]
)
COMPACT_THREE_WITH_NOTICE = "\n".join(
    [
        '## "q" 3/40',
        LEGEND,
        "- ID: a1|0.9000||2026-11-04|",
        "P: one",
        "- ID: a2|0.8000||2026-11-04|",
        "P: two",
        "- ID: a3|0.7000|elsewhere|2026-11-01|x.jsonl",
        "P: three",
        "Search mode: FTS fallback (helper cold); vector embedding was skipped.",
    ]
)
FULL_TWO = "\n".join(
    [
        '## Search results for "q" - 2 of 2 shown',
        "- ID: b1 | score: 0.5000 | project: brainlayer | date: 2026-11-04",
        "  Source: s.jsonl",
        "  Preview: hello",
        "- ID: b2 | score: 0.4000 | project: unknown | date: unknown",
        "  Preview: sprint-gate-MARKER in full detail",
    ]
)
ZERO = '## Search results for "q" - 0 of 0 shown\n\nNo results found.'
KG_THEN_COMPACT = (
    '## Entity: BrainBar\n\n### KG Facts\n- uses: sqlite\n\n## "q" 1/1\n'
    + LEGEND
    + "\n- ID: k1|0.6||2026-11-04|\nP: hit"
)
HELPER_FULL = '## Search results for "q" - 1 of 1 shown\n\n### 1. relevant result\n- Source: s\n- Date: 2026-11-04'


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        (COMPACT_TWO, 2),
        (COMPACT_THREE_WITH_NOTICE, 3),
        (FULL_TWO, 2),
        (ZERO, 0),
        (KG_THEN_COMPACT, 1),
        (HELPER_FULL, 1),
        ("Found 0 results for 'x'", None),
        ("", None),
    ],
)
def test_counts_rows_of_every_served_shape(text, expected):
    assert search_result_count(text) == expected


def test_rows_keep_the_id_delimiter_invariant_and_drop_trailing_notices():
    rows = search_result_rows(COMPACT_THREE_WITH_NOTICE)
    assert [row.splitlines()[0].removeprefix("- ID: ").split("|", 1)[0] for row in rows] == ["a1", "a2", "a3"]
    assert not any("Search mode" in row for row in rows)
    # The blank project column (implied by the filter) is still one column: 5 fields.
    assert rows[0].splitlines()[0].split("|") == ["- ID: a1", "0.9000", "", "2026-11-04", ""]


def test_structured_content_is_authoritative_over_text():
    result = {"structuredContent": {"results": [{"chunk_id": "x"}, {"chunk_id": "y"}], "total": 9}}
    assert search_result_count("unparseable", result) == 2
    assert search_result_count(COMPACT_TWO, {"structuredContent": {"search_mode": "hybrid"}}) == 2


@pytest.mark.parametrize("text", [COMPACT_TWO, FULL_TWO])
def test_sprint_gate_sees_a_marker_in_the_served_answer(monkeypatch, text):
    monkeypatch.setattr(sprint_gate.time, "sleep", lambda _: None)
    client = SimpleNamespace(call=lambda _name, _args: {"content": [{"type": "text", "text": text}]})
    assert sprint_gate.search_visible(client, "sprint-gate-MARKER", 0.5) is True


def test_sprint_gate_does_not_count_a_marker_quoted_only_in_the_header(monkeypatch):
    ticks = iter([0.0, 0.0, 1.0])
    monkeypatch.setattr(sprint_gate.time, "monotonic", lambda: next(ticks))
    monkeypatch.setattr(sprint_gate.time, "sleep", lambda _: None)
    text = '## "sprint-gate-MARKER" 1/1\n' + LEGEND + "\n- ID: z|0.1||2026-11-04|\nP: unrelated"
    client = SimpleNamespace(call=lambda _name, _args: {"content": [{"type": "text", "text": text}]})
    assert sprint_gate.search_visible(client, "sprint-gate-MARKER", 0.5) is False


def _canary(text: str, structured: dict | None = None):
    result: dict = {"content": [{"type": "text", "text": text}], "isError": False}
    if structured is not None:
        result["structuredContent"] = structured
    return lambda *_args: {"jsonrpc": "2.0", "id": 1, "result": result}


@pytest.mark.parametrize(
    ("text", "structured", "count", "status"),
    [
        (COMPACT_TWO, None, 2, "results"),  # old: 7, read from a header quoted inside a preview
        (COMPACT_THREE_WITH_NOTICE, None, 3, "results"),  # old: 1, the non-empty fallback
        (ZERO, None, 0, "empty"),
        ("unparseable", {"results": [{"chunk_id": "s"}]}, 1, "results"),
        # Current fallback for shapes with no listing header is unchanged.
        ("Found 0 results for 'agentopology'", None, 0, "empty"),
        ("some other non-empty answer", None, 1, "results"),
    ],
)
def test_health_check_canary_counts_the_served_answer(tmp_path, text, structured, count, status):
    db_path = tmp_path / "fixture.db"
    _make_db(db_path, total=1, vector_rows=1)
    result = run_health_check(
        HealthCheckConfig(
            db_path=db_path,
            state_path=tmp_path / "health.json",
            pause_sentinel_path=tmp_path / "pause.sentinel",
            job_alert_path=tmp_path / "alerts.json",
            source_jsonl_globs=[],
        ),
        ps_output_fn=lambda: "",
        socket_request_fn=_canary(text, structured),
        command_runner=lambda _args: SimpleNamespace(returncode=0, stdout="", stderr=""),
    )
    assert result.canary_ok is True
    assert (result.canary_result_count, result.canary_status) == (count, status)
