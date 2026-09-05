"""A ratchet margin is measured from attested green main runs, or it is `unmeasured` — never a round number.

Ratchet bolt-down (c). The worked example is three days old and it is the watcher: R3's idle-CPU soak
measured 4.88% then 6.41% on near-identical code against a flat `<5%` gate, so a single measurement
could not resolve pass from fail at all. Every flat threshold in the gate has that defect until its
band is measured.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path

import pytest

from scripts import ratchet_margins as margins

# Seven sprint-gate `search_latency` p50 values recorded on green main between 2026-09-01 and
# 2026-09-02, all socket-measured on MacBook-Pro.local (resign-1.5.10.log, w12b-REPORT.md ×2,
# w12c-REPORT.md, w8-REPORT.md ×2, a5-bench-m4-2026-09-02.log). Real data, not invented.
GREEN_MAIN_P50_MS = [185.0, 210.0, 214.6, 98.382, 291.416, 293.901, 281.4]


def attestation(index: int, **measured) -> dict:
    """A schema-1 attestation as (b)'s `ratchet-attest.yml` publishes it, with only what (c) reads."""
    return {
        "schema": 1,
        "run_id": 1000 + index,
        "run_attempt": 1,
        "main_sha": f"{index:040x}",
        "measured_at": f"2026-09-0{1 + index % 5}T12:00:00Z",
        "workflow": ".github/workflows/ratchet-attest.yml",
        "measured": measured,
    }


def p50_runs(values: list[float]) -> list[dict]:
    return [attestation(i, **{"latency_baseline_ms.p50": value}) for i, value in enumerate(values)]


def write_store(root: Path, attestations: list[dict]) -> Path:
    for document in attestations:
        run_dir = root / str(document["run_id"])
        run_dir.mkdir(parents=True)
        (run_dir / margins.ATTESTATION_FILENAME).write_text(json.dumps(document), encoding="utf-8")
    return root


# --- the statistic ------------------------------------------------------------------------------


def test_fewer_than_five_runs_is_unmeasured_and_carries_no_number() -> None:
    margin = margins.measured_margin([100.0, 101.0, 99.0, 100.5])
    assert margin.kind == margins.UNMEASURED
    assert margin.n == 4
    assert margin.limit is None and margin.mean is None and margin.stdev is None
    text = margins.describe(margin, unit="ms")
    assert "unmeasured" in text
    assert "4 of the 5" in text
    # No digit that could be read as a limit: the count is the only number allowed in the sentence.
    assert not any(token.replace(".", "").isdigit() and token not in {"4", "5"} for token in text.split())


def test_five_runs_is_the_minimum_and_yields_a_measured_band() -> None:
    margin = margins.measured_margin([294.0, 282.0, 283.0, 298.0, 281.0])
    assert margin.kind == margins.MEASURED
    assert margin.n == 5
    # a5-bench-m4-2026-09-02.log, five socket rounds: mean 287.6, s 7.83, t(0.99, 4)=3.747,
    # sqrt(1 + 1/5)=1.0954 -> 287.6 + 3.747 * 7.83 * 1.0954 = 319.7. Checked against scipy.
    assert margin.mean == pytest.approx(287.6, abs=0.05)
    assert margin.stdev == pytest.approx(7.83, abs=0.01)
    assert margin.k == pytest.approx(3.747 * math.sqrt(1.2), abs=0.005)
    assert margin.limit == pytest.approx(319.7, abs=0.1)


def test_the_band_widens_for_small_n_and_tightens_as_runs_accumulate() -> None:
    # Same spread, more runs: the multiplier is what changes, and it must only go DOWN. Five runs are
    # a guess with error bars; thirty runs are a measurement. A fixed k would not know the difference.
    spread = [98.0, 100.0, 102.0, 99.0, 101.0]
    five = margins.measured_margin(spread)
    ten = margins.measured_margin(spread * 2)
    thirty = margins.measured_margin(spread * 6)
    assert five.k > ten.k > thirty.k
    assert five.k == pytest.approx(3.747 * math.sqrt(1 + 1 / 5), abs=0.005)
    assert thirty.k == pytest.approx(2.462 * math.sqrt(1 + 1 / 30), abs=0.005)


@pytest.mark.parametrize(
    ("df", "scipy_t"), [(30, 2.4573), (31, 2.4528), (40, 2.4233), (60, 2.3901), (100, 2.3642), (199, 2.3452)]
)
def test_beyond_the_table_the_quantile_is_the_finite_df_t_not_the_normal(df: int, scipy_t: float) -> None:
    # Reviewer finding (Macroscope, #763): falling back to Z_99 = 2.326 past df=29 is TIGHTER than every
    # finite-df t-quantile, so a 31-run series would have been judged against a band about 5% narrower
    # than the stated 99%. Values are scipy 1.17.0 `t.ppf(0.99, df)`, rounded to 4 places.
    assert margins.t_quantile(df) == pytest.approx(scipy_t, abs=2e-4)
    assert margins.t_quantile(df) > margins.Z_99_ONE_SIDED


def test_the_quantile_is_monotone_across_the_table_boundary() -> None:
    values = [margins.t_quantile(df) for df in range(4, 400)]
    assert values == sorted(values, reverse=True)
    assert values[-1] > margins.Z_99_ONE_SIDED


def test_describe_refuses_a_measured_margin_that_lost_its_numbers() -> None:
    # An assert used to guard this; under `python -O` it would have printed "limit None ms".
    broken = margins.Margin(kind=margins.MEASURED, n=5)
    with pytest.raises(ValueError):
        margins.describe(broken, unit="ms")


def test_zero_variance_is_a_zero_band_not_a_crash() -> None:
    margin = margins.measured_margin([100.0] * 5)
    assert margin.kind == margins.MEASURED
    assert margin.stdev == 0.0
    assert margin.limit == 100.0


def test_verdict_is_red_only_beyond_the_measured_limit() -> None:
    margin = margins.measured_margin([294.0, 282.0, 283.0, 298.0, 281.0])
    assert margins.within(margin, 319.0) is True
    assert margins.within(margin, 320.0) is False
    unmeasured = margins.measured_margin([1.0, 2.0])
    assert margins.within(unmeasured, 0.0) is None


def test_real_green_main_p50_history_makes_the_flat_ten_percent_a_false_red_generator() -> None:
    """The finding, on real numbers: the seven recorded green-main p50 values vary two-fold.

    A flat 10% around their mean (225 ms) would have turned RED on three of the seven runs that
    were, in fact, green main. The measured band (464 ms) contains all seven. And the corpus
    baseline the flat 10% is applied to TODAY (911.887 ms, captured under active sprint load) puts
    the limit at 1003 ms -- more than double the measured band, so a real two-fold regression to
    900 ms would have sailed through GREEN. Both failure modes, one row.
    """
    margin = margins.measured_margin(GREEN_MAIN_P50_MS)
    assert margin.n == 7
    assert margin.mean == pytest.approx(225.0, abs=0.1)
    assert margin.limit == pytest.approx(463.7, abs=0.2)
    flat_limit = margin.mean * 1.10
    false_reds = [value for value in GREEN_MAIN_P50_MS if value > flat_limit]
    assert len(false_reds) == 3, false_reds
    assert all(margins.within(margin, value) for value in GREEN_MAIN_P50_MS)
    todays_limit = 911.887 * 1.10
    assert margins.within(margin, 900.0) is False
    assert 900.0 <= todays_limit  # GREEN under the flat gate as shipped on 3ee7c279


def test_describe_shows_the_reader_why_the_limit_is_what_it_is() -> None:
    text = margins.describe(margins.measured_margin(GREEN_MAIN_P50_MS), unit="ms")
    # Every input to the verdict is in the sentence: mean, spread, multiplier, n, and the limit.
    for fragment in ("mean 225.0 ms", "σ 71.1 ms", "k 3.36", "n=7", "limit 463.7 ms", "99%"):
        assert fragment in text, (fragment, text)


# --- reading the attested store -----------------------------------------------------------------


def test_values_are_collected_per_key_across_attested_runs() -> None:
    runs = [
        attestation(i, **{"latency_baseline_ms.p50": 200.0 + i, "latency_baseline_ms.p95": 1900.0 + i})
        for i in range(6)
    ]
    assert margins.series(runs, margins.LATENCY_P50) == [200.0, 201.0, 202.0, 203.0, 204.0, 205.0]
    assert margins.series(runs, margins.LATENCY_P95)[0] == 1900.0
    assert margins.series(runs, margins.idle_cpu_key("daemon")) == []


def test_a_run_missing_the_key_does_not_count_toward_the_five() -> None:
    runs = p50_runs([200.0] * 4) + [attestation(9, **{"latency_baseline_ms.p95": 1900.0})]
    assert margins.series(runs, margins.LATENCY_P50) == [200.0] * 4
    assert margins.margin_for(runs, margins.LATENCY_P50).kind == margins.UNMEASURED


@pytest.mark.parametrize("bad", [True, "200", None, float("nan"), float("inf"), -1.0])
def test_a_value_that_is_not_a_finite_non_negative_number_is_refused_not_coerced(bad) -> None:
    runs = p50_runs([200.0] * 5) + [attestation(9, **{"latency_baseline_ms.p50": bad})]
    if bad is None:
        # JSON null is "not measured", the same as an absent key; it is skipped, not a finding.
        assert margins.series(runs, margins.LATENCY_P50) == [200.0] * 5
        return
    with pytest.raises(margins.AttestationError) as error:
        margins.series(runs, margins.LATENCY_P50)
    assert "1009" in str(error.value)
    assert "latency_baseline_ms.p50" in str(error.value)


@pytest.mark.parametrize(
    ("payload", "reason"),
    [
        ("[]", "not an object"),
        ('{"schema": 2, "run_id": 1}', "schema 2, not 1"),
        ('{"schema": 1}', "no usable `run_id`"),
        ('{"schema": 1, "run_id": []}', "no usable `run_id`"),
        ('{"schema": 1, "run_id": true}', "no usable `run_id`"),
        ('{"schema": 1, "run_id": " "}', "no usable `run_id`"),
        ('{"schema": 1, "run_id": 1, "main_sha": "abc"}', "no 40-hex `main_sha`"),
        ('{"schema": 1, "run_id": 1, "main_sha": "' + "a" * 40 + '"}', "no `measured_at`"),
        (
            '{"schema": 1, "run_id": 1, "main_sha": "' + "a" * 40 + '", "measured_at": "x", "measured": []}',
            "no `measured` object",
        ),
        ("{not json", "could not be parsed"),
    ],
)
def test_a_malformed_attestation_is_refused_with_the_reason(tmp_path: Path, payload: str, reason: str) -> None:
    path = tmp_path / margins.ATTESTATION_FILENAME
    path.write_text(payload, encoding="utf-8")
    with pytest.raises(margins.AttestationError) as error:
        margins.load_attestations(path)
    assert reason in str(error.value)
    assert str(path) in str(error.value)


def test_a_root_that_does_not_exist_is_refused_not_treated_as_zero_runs(tmp_path: Path) -> None:
    with pytest.raises(margins.AttestationError) as error:
        margins.load_attestations(tmp_path / "absent")
    assert "absent" in str(error.value)


def test_an_empty_root_is_the_bootstrap_state_zero_runs_unmeasured(tmp_path: Path) -> None:
    # main has never attested: `gh run download` fetched nothing. That is a real state, not a fault.
    loaded = margins.load_attestations(tmp_path)
    assert loaded == []
    margin = margins.margin_for(loaded, margins.LATENCY_P50)
    assert margin.kind == margins.UNMEASURED and margin.n == 0


def test_no_store_handed_over_at_all_is_unmeasured_with_zero_runs() -> None:
    margin = margins.margin_for(None, margins.LATENCY_P50)
    assert margin.kind == margins.UNMEASURED
    assert margin.n == 0
    assert "0 of the 5" in margins.describe(margin, unit="ms")


def test_a_directory_of_downloaded_artifacts_round_trips(tmp_path: Path) -> None:
    root = write_store(tmp_path / "attestations", p50_runs(GREEN_MAIN_P50_MS))
    loaded = margins.load_attestations(root)
    assert [document["run_id"] for document in loaded] == [1000 + i for i in range(7)]
    assert margins.margin_for(loaded, margins.LATENCY_P50).limit == pytest.approx(463.7, abs=0.2)


def test_one_bad_file_in_the_directory_fails_the_whole_read(tmp_path: Path) -> None:
    # Five good attestations and one unreadable one is not "five runs": the store the run was handed
    # is not what it claims to be, and a band computed around the bad file would hide that.
    root = write_store(tmp_path / "attestations", p50_runs([100.0] * 5))
    (root / "9999").mkdir()
    (root / "9999" / margins.ATTESTATION_FILENAME).write_text("{", encoding="utf-8")
    with pytest.raises(margins.AttestationError):
        margins.load_attestations(root)


def test_the_same_run_twice_is_a_malformed_store_not_two_observations(tmp_path: Path) -> None:
    # Reviewer finding (Macroscope, #764): copying one attestation.json five times would have made one
    # run look like five and given a zero-width band.
    root = write_store(tmp_path / "attestations", p50_runs([100.0] * 5))
    (root / "copy").mkdir()
    (root / "copy" / margins.ATTESTATION_FILENAME).write_text(
        (root / "1000" / margins.ATTESTATION_FILENAME).read_text(encoding="utf-8"), encoding="utf-8"
    )
    with pytest.raises(margins.AttestationError) as error:
        margins.load_attestations(root)
    assert "1000 appears twice" in str(error.value)


def test_a_malformed_measured_value_is_refused_at_load_not_at_read(tmp_path: Path) -> None:
    # Reviewer finding (Macroscope, #765): a bad value that first surfaced inside a row builder aborted
    # the whole collector. Validating at load means a loaded store can always be read.
    documents = p50_runs([100.0] * 5)
    documents[2]["measured"]["latency_baseline_ms.p95"] = "fast"
    root = write_store(tmp_path / "attestations", documents)
    with pytest.raises(margins.AttestationError) as error:
        margins.load_attestations(root)
    assert "run 1002" in str(error.value) and "measured.latency_baseline_ms.p95" in str(error.value)
    documents[2]["measured"]["latency_baseline_ms.p95"] = None  # null = not measured this run: allowed
    loaded = margins.load_attestations(write_store(tmp_path / "again", documents))
    assert margins.series(loaded, margins.LATENCY_P95) == []


def test_a_root_that_cannot_be_scanned_is_a_refusal_not_a_traceback(tmp_path: Path) -> None:
    # Reviewer finding (Macroscope, #764): PermissionError escaped as a traceback past the gate's
    # structured refusal.
    if os.geteuid() == 0:
        pytest.skip("root ignores directory permissions")
    root = write_store(tmp_path / "attestations", p50_runs([100.0] * 5))
    root.chmod(0o000)
    try:
        with pytest.raises(margins.AttestationError) as error:
            margins.load_attestations(root)
    finally:
        root.chmod(0o755)
    assert "could not be scanned (PermissionError)" in str(error.value)
