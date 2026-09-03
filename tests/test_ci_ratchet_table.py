"""The ratchet table prints a measured value or a named `n/a` — never a guessed number (w14)."""

from __future__ import annotations

import json
import re
import zipfile
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import pytest

from scripts import ci_ratchet_table as ratchet

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
        ({"wheel": None}, "no packaged wheel"),
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


# --- the four rows a runner cannot measure ---------------------------------------------------


@pytest.mark.parametrize("name", ["mapped bytes", "search p50/p95", "idle CPU"])
def test_socket_rows_on_a_runner_name_the_machine_target_first(tmp_path: Path, name: str) -> None:
    result = row(ratchet.collect(linux_probe(tmp_path), CORPUS), name)
    assert result.status == ratchet.NA
    assert result.value == "n/a — runner is Linux/x86_64; the gate's machine target is Darwin/arm64"


def test_signature_row_on_a_runner_names_macos(tmp_path: Path) -> None:
    result = row(ratchet.collect(linux_probe(tmp_path), CORPUS), "signature_valid")
    assert result.status == ratchet.NA
    assert result.value == "n/a — runner is Linux; codesign verification needs macOS"


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("mapped bytes", "no runner-side collector for mapped bytes yet — w13 (CI parity)"),
        ("search p50/p95", "no runner-side collector for search latency yet — w13 (CI parity)"),
        ("idle CPU", "no runner-side collector for idle CPU yet — w13 (CI parity)"),
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
    notes = row(ratchet.collect(linux_probe(tmp_path), CORPUS), "mapped bytes").notes
    assert "26.2 GB" in notes and "installed Mac" in notes and "Not measured by this run." in notes


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
