"""Pinned behavior for the tuning bench's scoring, which once got this wrong.

An earlier version quoted a run's price from the last concurrency level measured
instead of the cheapest, and checked the floor only at the gate. It stamped a run
`passes_floor` while advertising a 32-stream price whose TTFT and decode both
missed, and that number reached a write-up. These tests pin the corrected rule.

Pure functions only: no GPU box, no proxy, no network.
"""

from __future__ import annotations

import asyncio
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


def test_cost_per_task_matches_the_wall_clock_form_on_a_healthy_level(bench):
    # 24 tasks at concurrency 8 with a 13 s median is 38.9 s of wall clock, and
    # both ways of charging for it must agree when nothing straggles.
    from_median = bench.usd_per_task(13.0, 8, 2.99, 1)
    from_wall = 2.99 * 1 * (38.9 / 3600) / 24
    assert from_median == pytest.approx(from_wall, rel=0.02)


def test_cost_per_task_does_not_move_when_one_stream_hangs(bench):
    # The motivating bug: one stream timing out at 900 s against a 13 s median
    # dragged the level's wall clock and moved the quoted price about 93x with
    # no change in the engine. Feed both samples through the same median the
    # bench uses, so this exercises the real path rather than asserting on a
    # hand-picked number.
    healthy_s = [13.0] * 24
    with_straggler_s = [13.0] * 23 + [900.0]

    healthy = bench.usd_per_task(bench.pct(healthy_s, 0.5), 8, 2.99, 1)
    strag = bench.usd_per_task(bench.pct(with_straggler_s, 0.5), 8, 2.99, 1)
    assert strag == healthy

    # The wall-clock form this replaced would have moved by more than 20x.
    wall_form_healthy = 2.99 * (sum(healthy_s) / 8 / 3600) / 24
    wall_form_straggler = 2.99 * (900.0 / 3600) / 23
    assert wall_form_straggler / wall_form_healthy > 20


def test_cost_per_task_halves_when_the_gpu_serves_twice_as_many(bench):
    # Independent of the formula: twice the concurrency means each task holds
    # half the GPU, so the same task duration must cost half as much.
    at_8 = bench.usd_per_task(13.0, 8, 2.99, 1)
    at_16 = bench.usd_per_task(13.0, 16, 2.99, 1)
    assert at_16 < at_8
    assert at_8 / at_16 == pytest.approx(2.0, rel=1e-3)


def test_cost_per_task_guards_missing_inputs(bench):
    assert bench.usd_per_task(None, 8, 2.99, 1) is None
    assert bench.usd_per_task(0, 8, 2.99, 1) is None
    assert bench.usd_per_task(13.0, 0, 2.99, 1) is None


def run_level_with(bench, monkeypatch, streams, natural, specs):
    """Drive run_level with canned streams, so run_level's own stats are pinned.

    `specs` is one dict per stream: finish_reason, total_s, out_tokens, tok_s.
    """
    made = iter(specs)

    async def fake_stream(
        http,
        url,
        model,
        index,
        max_tokens,
        context_lines=0,
        natural=False,
        reasoning_effort=None,
    ):
        spec = next(made)
        return {
            "finish_reason": spec["finish"],
            "reasoning_tokens": None,
            "ttft": 0.2,
            "total_s": spec["total_s"],
            "prompt_tokens": 100,
            "cached_tokens": None,
            "out_tokens": spec["out_tokens"],
            "tok_s": spec["tok_s"],
            "error": None,
        }

    monkeypatch.setattr(bench, "one_stream", fake_stream)
    return asyncio.run(
        bench.run_level(None, "u", "m", streams, 512, len(specs), 0, natural, None)
    )


def spec(finish, total_s, out_tokens, tok_s=40.0):
    return {
        "finish": finish,
        "total_s": total_s,
        "out_tokens": out_tokens,
        "tok_s": tok_s,
    }


def test_natural_mode_excludes_tasks_cut_off_at_max_tokens(bench, monkeypatch):
    # Asymmetric on purpose: with a symmetric split the median over the costed
    # subset coincides with the median over every stream, and swapping one for
    # the other would go unnoticed.
    level = run_level_with(
        bench,
        monkeypatch,
        2,
        True,
        [
            spec("stop", 10.0, 50),
            spec("stop", 11.0, 60),
            spec("stop", 12.0, 70),
            spec("length", 90.0, 512),
        ],
    )
    assert level["costed_tasks"] == 3
    assert level["truncated_tasks"] == 1
    assert level["task_s_p50"] == 11.0
    assert level["out_tokens_p50"] == 60.0


def test_a_missing_finish_reason_is_not_counted_as_a_finished_task(bench, monkeypatch):
    # A stream that never said it finished is not evidence of a task's cost.
    level = run_level_with(
        bench,
        monkeypatch,
        2,
        True,
        [spec("stop", 10.0, 50), spec(None, 99.0, 400)],
    )
    assert level["costed_tasks"] == 1
    assert level["truncated_tasks"] == 1
    assert level["task_s_p50"] == 10.0


def test_throughput_spans_every_served_stream_not_just_costed_ones(bench, monkeypatch):
    # The two populations in a level row are deliberate: throughput is asked of
    # everything that served, cost only of tasks that finished.
    level = run_level_with(
        bench,
        monkeypatch,
        2,
        True,
        [
            spec("stop", 10.0, 50, tok_s=10.0),
            spec("length", 90.0, 512, tok_s=90.0),
            spec("length", 90.0, 512, tok_s=90.0),
        ],
    )
    assert level["costed_tasks"] == 1
    # Cost sees only the one finished task.
    assert level["task_s_p50"] == 10.0
    assert level["out_tokens_p50"] == 50.0
    # Throughput sees all three, so its median is 90, not the costed 10.
    assert level["decode_tok_s_p50"] == 90.0
    assert level["agg_out_tok_s"] > 0


def test_straggler_does_not_move_per_task_cost_via_run_level(bench, monkeypatch):
    # The straggler guard has to hold where task_s_p50 is actually produced, not
    # only in the pure function: swapping the median for a mean must fail here.
    healthy = run_level_with(
        bench, monkeypatch, 2, True, [spec("stop", 10.0, 50)] * 4
    )
    strag = run_level_with(
        bench,
        monkeypatch,
        2,
        True,
        [spec("stop", 10.0, 50)] * 3 + [spec("stop", 900.0, 50)],
    )
    assert strag["task_s_p50"] == healthy["task_s_p50"]
    assert bench.usd_per_task(strag["task_s_p50"], 2, 2.99, 1) == bench.usd_per_task(
        healthy["task_s_p50"], 2, 2.99, 1
    )


def test_forced_mode_counts_every_stream_despite_length_finishes(bench, monkeypatch):
    # Forced runs always stop on "length" by construction, so the filter must
    # not touch them or every forced level would report zero costed tasks.
    level = run_level_with(
        bench, monkeypatch, 2, False, [spec("length", 40.0, 512)] * 4
    )
    assert level["costed_tasks"] == 4
    assert level["truncated_tasks"] == 0
    assert level["task_s_p50"] == 40.0


def test_a_fully_truncated_natural_level_reports_no_per_task_cost(bench, monkeypatch):
    level = run_level_with(
        bench, monkeypatch, 2, True, [spec("length", 90.0, 512)] * 2
    )
    assert level["costed_tasks"] == 0
    assert level["task_s_p50"] is None
    assert level["out_tokens_p50"] is None
    assert bench.usd_per_task(level["task_s_p50"], 2, 2.99, 1) is None


def priced_row(ttft, decode, task_s, streams=8):
    lv = level(streams, ttft, decode, 2.0)
    lv["task_s_p50"] = task_s
    return {"levels": [lv], "gpu_hourly_usd": 2.99, "num_gpus": 1}


def test_per_task_price_requires_the_gate_to_hold(bench):
    # The cheapest per task is not a recommendation if the gate misses the
    # floor. This is the same rule best_level applies on the per-token side.
    assert bench.task_price(priced_row(0.5, 40, 13.0)) == bench.usd_per_task(
        13.0, 8, 2.99, 1
    )
    assert bench.task_price(priced_row(2.5, 20, 13.0)) is None


def test_per_task_price_is_recomputed_not_read_from_the_row(bench):
    # A row carrying a price from an older formula must not be ranked on it.
    row = priced_row(0.5, 40, 13.0)
    row["levels"][0]["usd_per_task"] = 99.0
    assert bench.task_price(row) == bench.usd_per_task(13.0, 8, 2.99, 1)


def test_per_task_price_is_none_without_a_gate_or_a_measurement(bench):
    assert bench.task_price(priced_row(0.5, 40, 13.0, streams=32)) is None
    assert bench.task_price(priced_row(0.5, 40, None)) is None
    assert bench.task_price({}) is None


def test_natural_and_reasoning_effort_do_not_rank_against_forced_runs(bench):
    base = {
        "context_lines": 0,
        "max_tokens": 512,
        "model": "m",
        "machine": "GPU-RTXPRO6000",
        "num_gpus": 1,
        "gpu_hourly_usd": 2.99,
    }
    natural = {**base, "natural": True}
    effort = {**base, "reasoning_effort": "low"}
    assert bench.workload_key(base) != bench.workload_key(natural)
    assert bench.workload_key(base) != bench.workload_key(effort)
    assert bench.workload_key(natural) != bench.workload_key(effort)


def test_a_row_from_before_natural_existed_groups_with_forced_runs(bench):
    # Legacy rows have neither key, and they were forced runs, so they must
    # still rank against new forced runs rather than splitting into their own
    # group.
    legacy = {
        "context_lines": 0,
        "max_tokens": 512,
        "model": "m",
        "machine": "GPU-RTXPRO6000",
        "num_gpus": 1,
        "gpu_hourly_usd": 2.99,
    }
    current = {**legacy, "natural": False, "reasoning_effort": None}
    assert bench.workload_key(legacy) == bench.workload_key(current)


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
