"""Pinned behavior for the tuning bench's scoring, which once got this wrong.

An earlier version quoted a run's price from the last concurrency level measured
instead of the cheapest, and checked the floor only at the gate. It stamped a run
`passes_floor` while advertising a 32-stream price whose TTFT and decode both
missed, and that number reached a write-up. These tests pin the corrected rule.

Pure functions only: no GPU box, no proxy, no network.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

MODULE_PATH = Path(__file__).with_name("rtx_tune_bench.py")


@pytest.fixture(scope="module")
def bench():
    spec = importlib.util.spec_from_file_location("rtx_tune_bench", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def level(streams, ttft, decode, usd, agg=None, errors=0):
    return {
        "streams": streams,
        "ttft_p95": ttft,
        "decode_tok_s_p50": decode,
        "usd_per_1m_out": usd,
        "agg_out_tok_s": agg,
        "error_count": errors,
        "errors": [],
    }


# The real 8k-flashattn run: the gate holds, 32 streams misses both thresholds.
REGRESSION = [
    level(1, 0.241, 45.17, 18.71),
    level(8, 0.792, 39.88, 2.69),
    level(32, 2.542, 27.78, 0.98),
]


def test_price_comes_from_the_cheapest_level_that_holds_the_floor(bench):
    best = bench.best_level(REGRESSION)
    assert best["streams"] == 8
    assert best["usd_per_1m_out"] == 2.69


def test_gate_still_passes_when_a_wider_level_misses(bench):
    scored = bench.score(REGRESSION)
    assert scored["passes_floor"] is True
    assert scored["levels_holding_floor"] == [1, 8]
    assert any("32" in reason for reason in scored["levels_breaking_floor"])


def test_no_price_is_quoted_when_no_level_holds_the_floor(bench):
    broken = [level(8, 5.0, 10.0, 0.10), level(32, 9.0, 5.0, 0.01)]
    assert bench.best_level(broken) is None
    assert bench.score(broken)["passes_floor"] is False


def test_cheapest_wins_even_when_it_is_not_the_last_level(bench):
    non_monotonic = [
        level(8, 0.5, 40, 5.0),
        level(16, 0.9, 35, 1.0),
        level(32, 1.1, 33, 3.0),
    ]
    assert bench.best_level(non_monotonic)["streams"] == 16


def test_a_level_below_the_gate_is_never_quoted_as_the_price(bench):
    # Latency only worsens with concurrency, so whenever the gate misses there is
    # almost always a lower level that holds. Quoting it would print a price next
    # to a failing verdict.
    gate_fails = [level(1, 0.2, 50, 20.0), level(8, 2.5, 20, 3.0)]
    assert bench.score(gate_fails)["passes_floor"] is False
    assert bench.best_level(gate_fails) is None


def test_price_is_recomputed_from_throughput_when_a_rate_is_given(bench):
    # A row written under an older pricing rule must not be ranked on its stale
    # number: 300 tok/s at $2.99/GPU-hour is $2.77 per 1M, not the stored $99.
    stale = [level(8, 0.5, 40, 99.0, agg=300.0)]
    best = bench.best_level(stale, gpu_hourly=2.99, num_gpus=1)
    assert best["usd_per_1m_out"] == pytest.approx(2.77, abs=0.01)


def test_every_failure_mode_is_named_once(bench):
    reasons = bench.level_reasons(level(8, 3.0, 12.0, 1.0, errors=20))
    assert len(reasons) == 3
    assert any("20 errors" in r for r in reasons)


def test_error_count_is_not_capped_by_the_sample_slice(bench):
    many = level(8, 0.5, 40, 1.0, errors=20)
    many["errors"] = ["a", "b", "c"]
    assert "20 errors at 8" in " ".join(bench.level_reasons(many))


def test_missing_measurements_fail_the_floor_rather_than_passing(bench):
    assert bench.level_reasons(level(8, None, None, None))


def test_rows_written_before_error_count_existed_are_still_scorable(bench):
    # The log is append-only, so `compare` reads rows from older schemas.
    old = {"streams": 8, "ttft_p95": 0.5, "decode_tok_s_p50": 40, "usd_per_1m_out": 2.0}
    assert bench.level_reasons(old) == []
    assert bench.best_level([old])["streams"] == 8


def test_an_older_row_with_failures_still_counts_them(bench):
    # The fallback has to read the truncated `errors` sample, not assume zero:
    # replacing it with a literal 0 must not leave the suite green.
    old = {
        "streams": 8,
        "ttft_p95": 0.5,
        "decode_tok_s_p50": 40,
        "usd_per_1m_out": 2.0,
        "errors": ["a", "b", "c"],
    }
    assert "3 errors at 8" in " ".join(bench.level_reasons(old))
    assert bench.best_level([old]) is None


# The real 8k-flashinfer run. Decode at 32 clears the floor by 0.02 while TTFT
# misses, so the two thresholds must stay independent.
ASYMMETRIC = [
    level(1, 0.443, 44.82, 19.01),
    level(8, 0.973, 41.12, 2.63),
    level(32, 2.113, 30.02, 0.90),
]


def test_the_two_floor_thresholds_are_judged_independently(bench):
    reasons = bench.level_reasons(ASYMMETRIC[2])
    assert reasons == ["TTFT p95 2.113 >= 2.0 at 32"]
    assert bench.score(ASYMMETRIC)["passes_floor"] is True
    assert bench.best_level(ASYMMETRIC)["streams"] == 8


def test_decode_exactly_on_the_floor_passes(bench):
    assert bench.level_reasons(level(8, 0.5, 30.0, 1.0)) == []
    assert bench.level_reasons(level(8, 0.5, 29.99, 1.0))


def test_ttft_exactly_on_the_ceiling_fails(bench):
    assert bench.level_reasons(level(8, 2.0, 40, 1.0))
    assert bench.level_reasons(level(8, 1.999, 40, 1.0)) == []


def test_one_percentile_definition_for_every_metric(bench):
    values = [float(i) for i in range(1, 25)]
    assert bench.pct(values, 0.5) == 12.0
    assert bench.pct(values, 0.95) == 23.0
    assert bench.pct([], 0.5) is None


def test_workload_key_separates_hardware_and_rate(bench):
    base = {
        "context_lines": 0,
        "max_tokens": 512,
        "model": "m",
        "machine": "GPU-RTXPRO6000",
        "num_gpus": 1,
        "gpu_hourly_usd": 2.99,
    }
    assert bench.workload_key(base) == bench.workload_key(dict(base))
    pricier = {**base, "gpu_hourly_usd": 5.0}
    assert bench.workload_key(base) != bench.workload_key(pricier)
    assert bench.workload_key(base) != bench.workload_key({**base, "machine": "H100"})
