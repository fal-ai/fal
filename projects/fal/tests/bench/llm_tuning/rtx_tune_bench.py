"""Coding-agent serving bench through `fal dev app proxy`, for the RTX tuning loop.

Drives http://127.0.0.1:8899/v1/chat/completions (the proxy signs and forwards to
the dev box, so no auth here). Measures TTFT and decode tok/s at a few concurrency
levels on a fixed workload, scores the run against the latency floor, prices it in
USD per 1M output tokens, and appends one row to an append-only JSONL log.

  bench:    uv run --with httpx rtx_tune_bench.py bench --label baseline
  compare:  uv run --with httpx rtx_tune_bench.py compare

Floor (a run fails if either is missed at concurrency 8):
  TTFT p95 < 2.0 s, per-stream decode tok/s p50 >= 30.
Goal: the lowest USD per 1M output tokens that still holds the floor. Every level is
checked against the floor, not only the gate, and the price quoted for a run is the
cheapest level that holds it. A level that is cheaper but misses the floor is not a
price this tool will quote.

Every stream is forced to exactly --max-tokens with ignore_eos, so output volume is
identical across runs and tok/s is comparable. The shared agent prefix is byte-identical
every run and a warmup pass precedes the measurement, so every run is scored with a warm
prefix cache rather than one run getting a cold-cache penalty the next does not.

Two things this does not measure, and you should not read into a row:

- `engine_args` is what the operator says was configured. It is not read back from the
  engine, and it is not the full resolved command line: the SDK's presets add flags of
  their own (`--max-num-seqs` among them). Read the box's
  `[llm-app] starting: vllm serve` line after a restart to confirm a flag actually
  took effect before trusting a run.
- Aggregate throughput divides output tokens by the whole wall clock of a level, and the
  closed loop drains at the end with fewer than `streams` in flight. That biases
  `agg_out_tok_s` slightly low and `$/1M` slightly high, equally for every run.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import time
from pathlib import Path

import httpx

OUT = Path(__file__).parent / "bench-results" / "rtx_tuning.jsonl"
# 8899 is PROXY_PORT in fal.dev.cli.parser, the default for `fal dev app proxy`.
# Spelled out rather than imported: this script has to run from a bare
# `uv run --with httpx` with no fal package installed.
BASE_URL = "http://127.0.0.1:8899/v1/chat/completions"
# RTX PRO 6000, 1 GPU. Overridable with --gpu-hourly.
GPU_HOURLY_USD = 2.99
FLOOR_TTFT_P95_S = 2.0
FLOOR_DECODE_TOK_S = 30.0
FLOOR_LEVEL = 8

SYSTEM = (
    "You are a coding agent working in a Python monorepo. Follow the repo conventions: "
    "type hints on public functions, no bare except, pytest for tests, and no new "
    "dependencies without a note. Answer with the edit you would make and why, "
    "and keep the reasoning tight."
)

# A stand-in for the repo context an agent carries into every turn. Byte-identical
# across runs so the prefix cache is in the same state for every measurement.
REPO_CONTEXT = """Here are the files currently in context.

# billing/invoice.py
def subtotal(lines):
    return sum(line.qty * line.unit_price for line in lines)

def apply_discount(amount, discount_pct):
    return amount - amount * discount_pct

def total(lines, discount_pct, tax_rate):
    net = apply_discount(subtotal(lines), discount_pct)
    return net + net * tax_rate

# billing/models.py
class Line:
    def __init__(self, sku, qty, unit_price):
        self.sku = sku
        self.qty = qty
        self.unit_price = unit_price

# billing/ledger.py
def post(entries, account):
    rows = []
    for e in entries:
        if e.account == account:
            rows.append(e)
    return sorted(rows, key=lambda r: r.posted_at)

def balance(entries, account):
    return sum(e.amount for e in post(entries, account))

# tests/test_invoice.py
def test_ten_percent_discount():
    lines = [Line("a", 2, 100.0)]
    assert total(lines, 10, 0.0) == 180.0

def test_balance_filters_by_account():
    assert balance([], "cash") == 0
"""

# Wording is load-bearing: it sets the prompt token count every recorded row was
# measured at. Reflow these strings, never reword them, or old rows stop being
# comparable with new ones.
TURNS = [
    "test_ten_percent_discount fails. Find the bug and give me the fix.",
    "Rewrite ledger.post so it does one pass and returns a generator. "
    "Keep the ordering.",
    "Add type hints to every public function in billing/invoice.py.",
    "balance() recomputes the sort on every call. Propose a fix that keeps the API.",
    "Write the pytest cases that would have caught the discount bug.",
    "apply_discount takes a percent in one caller and a fraction in another. "
    "Reconcile it.",
    "Review billing/ledger.py for correctness problems and rank them by severity.",
    "Add a currency field to Line and thread it through total() without "
    "breaking callers.",
]


def filler(lines: int) -> str:
    """Deterministic extra repo context, so a bigger prompt is still byte-identical."""
    out = []
    for i in range(lines):
        out.append(
            f"# billing/rules/rule_{i:04d}.py\n"
            f"def rule_{i:04d}(entry, account, rate={i % 97}):\n"
            f"    if entry.account != account:\n"
            f"        return 0.0\n"
            f"    return entry.amount * rate / 100.0\n"
        )
    return "\n".join(out)


def messages(turn_index: int, context_lines: int = 0) -> list[dict[str, str]]:
    context = REPO_CONTEXT
    if context_lines:
        context = f"{REPO_CONTEXT}\n{filler(context_lines)}"
    return [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": f"{context}\n\n{TURNS[turn_index % len(TURNS)]}"},
    ]


def config_sha(engine_args: list[str]) -> str:
    return hashlib.sha256(" ".join(engine_args).encode()).hexdigest()[:12]


async def one_stream(http, url, model, turn_index, max_tokens, context_lines=0):
    body = {
        "model": model,
        "stream": True,
        "max_tokens": max_tokens,
        "ignore_eos": True,
        "stream_options": {"include_usage": True},
        "messages": messages(turn_index, context_lines),
    }
    start = time.monotonic()
    first = None
    usage = {}
    err = None
    try:
        async with http.stream("POST", url, json=body, timeout=900.0) as r:
            if r.status_code != 200:
                return {"error": f"{r.status_code} {(await r.aread())[:300]!r}"}
            async for line in r.aiter_lines():
                if not line.startswith("data:") or line.strip() == "data: [DONE]":
                    continue
                obj = json.loads(line[5:])
                if obj.get("usage"):
                    usage = obj["usage"]
                for ch in obj.get("choices") or []:
                    d = ch.get("delta") or {}
                    if first is None and (
                        d.get("content")
                        or d.get("reasoning_content")
                        or d.get("reasoning")
                        or d.get("tool_calls")
                    ):
                        first = time.monotonic()
    except Exception as exc:
        err = repr(exc)[:300]
    end = time.monotonic()
    out = usage.get("completion_tokens") or 0
    if err is None and not usage:
        # No final usage chunk. Counting this as a successful 0-token stream
        # would deflate aggregate throughput with nothing to show for it.
        err = "stream ended without a usage chunk"
    elif err is None and out and first is None:
        # Tokens were produced but no delta matched the first-token fields. The
        # row would otherwise land in `ok` with ttft and tok_s None, and `pct`
        # drops None, so the percentiles would quietly be computed over a
        # smaller sample with nothing marking it.
        err = "produced tokens but no first-token delta was recognised"
    decode_s = end - first if first else None
    return {
        "ttft": first - start if first else None,
        "total_s": end - start,
        "prompt_tokens": usage.get("prompt_tokens"),
        "cached_tokens": (usage.get("prompt_tokens_details") or {}).get(
            "cached_tokens"
        ),
        "out_tokens": out,
        "tok_s": (out - 1) / decode_s if decode_s and out > 1 else None,
        "error": err,
    }


def pct(values, fraction):
    """Nearest-rank percentile: always one of the measured values.

    One definition for every percentile this tool reports, so two numbers printed
    as p50 in the same row mean the same thing. `statistics.median` interpolates
    on an even sample and would not agree with this at the default 24 samples.

    Duplicates `fal.dev.app.bench.percentile` on purpose. Importing it pulls in
    the whole fal package, and this script has to run from a bare
    `uv run --with httpx` with no fal installed. Keep the two in step by hand.
    """
    kept = sorted(v for v in values if v is not None)
    if not kept:
        return None
    rank = max(1, math.ceil(fraction * len(kept)))
    return round(kept[min(rank, len(kept)) - 1], 3)


async def run_level(
    http, url, model, streams, max_tokens, requests=None, context_lines=0
):
    """Closed loop: `requests` total, never more than `streams` in flight at once.

    More requests than streams so a percentile means something. With one wave of 8,
    "p95" is just the slowest of the 8 and one unlucky stream fails the run.
    """
    total = requests or streams * 3
    gate = asyncio.Semaphore(streams)

    async def guarded(index):
        async with gate:
            return await one_stream(http, url, model, index, max_tokens, context_lines)

    started = time.monotonic()
    rows = await asyncio.gather(*(guarded(i) for i in range(total)))
    wall = time.monotonic() - started
    ok = [r for r in rows if not r.get("error")]
    failed = [r for r in rows if r.get("error")]
    out_tokens = sum(r["out_tokens"] for r in ok)
    return {
        "streams": streams,
        "requests": total,
        "max_tokens": max_tokens,
        "ok": len(ok),
        # The count is uncapped; only the sample messages are truncated, so a
        # mostly-failing level cannot read as "3 errors".
        "error_count": len(failed),
        "errors": [r["error"] for r in failed][:3],
        "ttft_p50": pct([r["ttft"] for r in ok], 0.5),
        "ttft_p95": pct([r["ttft"] for r in ok], 0.95),
        "decode_tok_s_p50": pct([r["tok_s"] for r in ok], 0.5),
        "agg_out_tok_s": round(out_tokens / wall, 1) if wall else None,
        "prompt_tokens": ok[0]["prompt_tokens"] if ok else None,
        "cached_tokens": ok[0]["cached_tokens"] if ok else None,
        "wall_s": round(wall, 1),
    }


def usd_per_1m_out(agg_tok_s, gpu_hourly, num_gpus):
    if not agg_tok_s:
        return None
    tokens_per_hour = agg_tok_s * 3600
    return round(gpu_hourly * num_gpus / tokens_per_hour * 1_000_000, 2)


def level_reasons(level):
    """Why this concurrency level misses the floor. Empty means it holds."""
    reasons = []
    ttft = level.get("ttft_p95")
    decode = level.get("decode_tok_s_p50")
    # Not dead: the log is append-only, and rows written before `error_count`
    # existed carry only the truncated `errors` sample. Scoring those rows is
    # exactly what `compare` does.
    errors = level.get("error_count", len(level.get("errors") or []))
    if ttft is None or ttft >= FLOOR_TTFT_P95_S:
        reasons.append(f"TTFT p95 {ttft} >= {FLOOR_TTFT_P95_S} at {level['streams']}")
    if decode is None or decode < FLOOR_DECODE_TOK_S:
        reasons.append(
            f"decode tok/s p50 {decode} < {FLOOR_DECODE_TOK_S} at {level['streams']}"
        )
    if errors:
        reasons.append(f"{errors} errors at {level['streams']}")
    return reasons


def score(levels):
    """Score every level, not only the gate.

    The gate at FLOOR_LEVEL is what the floor is defined on, but a level that is
    cheaper and breaks the floor must not be reported as the price of the run.
    Scoring the gate alone once stamped a run `passes_floor` while advertising a
    32-stream price whose TTFT and decode both missed.
    """
    gate = next((lv for lv in levels if lv["streams"] == FLOOR_LEVEL), None)
    if gate is None:
        return {
            "passes_floor": None,
            "reasons": [f"no level at concurrency {FLOOR_LEVEL}"],
            "levels_holding_floor": [],
            "levels_breaking_floor": [],
        }
    holding = [lv["streams"] for lv in levels if not level_reasons(lv)]
    broken = [
        r for lv in levels if lv["streams"] != FLOOR_LEVEL for r in level_reasons(lv)
    ]
    return {
        "passes_floor": not level_reasons(gate),
        "reasons": level_reasons(gate),
        "levels_holding_floor": holding,
        "levels_breaking_floor": broken,
    }


def best_level(levels, gpu_hourly=None, num_gpus=None):
    """Cheapest level that actually holds the floor, not the last one measured.

    Only levels at or above FLOOR_LEVEL count. The floor is a promise about
    serving at the gate concurrency, so if the gate itself misses, a lower level
    that trivially holds is not a price for the configuration: latency only gets
    worse as concurrency rises, so there is almost always such a level, and
    quoting it would print a price next to a failing verdict.

    Price is recomputed from the recorded throughput when the rate is supplied,
    so a row written under an older pricing rule is not ranked on a stale number.
    """
    eligible = []
    for lv in levels:
        if lv["streams"] < FLOOR_LEVEL or level_reasons(lv):
            continue
        usd = lv.get("usd_per_1m_out")
        if gpu_hourly is not None and num_gpus is not None:
            usd = usd_per_1m_out(lv.get("agg_out_tok_s"), gpu_hourly, num_gpus)
        if usd is not None:
            eligible.append((usd, lv))
    if not eligible:
        return None
    usd, lv = min(eligible, key=lambda pair: pair[0])
    return {**lv, "usd_per_1m_out": usd}


async def cmd_bench(args):
    url = args.url
    # Deduped: score() takes the first level matching the gate, so a repeated
    # level would be measured twice and silently scored from one of them.
    levels_wanted = sorted({int(x) for x in args.levels.split(",")})
    widest = max(levels_wanted)
    if args.requests_per_level is not None and args.requests_per_level < widest:
        raise SystemExit(
            f"--requests-per-level {args.requests_per_level} is below the widest level "
            f"{widest}: that level would never reach the concurrency its row "
            f"claims, and the row would not be comparable."
        )
    if FLOOR_LEVEL not in levels_wanted:
        raise SystemExit(
            f"--levels must include {FLOOR_LEVEL}: the floor is defined there, and a "
            f"run without it cannot be scored. Got {levels_wanted}."
        )
    row = {
        "label": args.label,
        "note": args.note,
        "t": time.time(),
        "iso": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "model": args.model,
        "machine": args.machine,
        "num_gpus": args.num_gpus,
        "gpu_hourly_usd": args.gpu_hourly,
        "engine_args": args.engine_args,
        "config_sha": config_sha(args.engine_args),
        "max_tokens": args.max_tokens,
        "context_lines": args.context_lines,
    }
    limits = httpx.Limits(max_connections=max(levels_wanted) + 8)
    async with httpx.AsyncClient(limits=limits) as http:
        warm = await one_stream(http, url, args.model, 0, 16, args.context_lines)
        print("warmup", json.dumps(warm), flush=True)
        if warm.get("error"):
            raise SystemExit(f"warmup failed: {warm['error']}")
        row["levels"] = []
        for streams in levels_wanted:
            level = await run_level(
                http, url, args.model, streams, args.max_tokens,
                args.requests_per_level, args.context_lines,
            )
            level["usd_per_1m_out"] = usd_per_1m_out(
                level["agg_out_tok_s"], args.gpu_hourly, args.num_gpus
            )
            print(json.dumps(level), flush=True)
            row["levels"].append(level)
    row.update(score(row["levels"]))
    best = best_level(row["levels"], args.gpu_hourly, args.num_gpus)
    row["usd_per_1m_out_best"] = best["usd_per_1m_out"] if best else None
    row["usd_per_1m_out_best_streams"] = best["streams"] if best else None
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("a") as f:
        f.write(json.dumps(row) + "\n")
    verdict = "PASS" if row["passes_floor"] else "FAIL"
    if best:
        price = f"best ${best['usd_per_1m_out']}/1M out at {best['streams']} streams"
    else:
        price = "no level holds the floor, so no price is quoted"
    print(f"\n{verdict} floor at {FLOOR_LEVEL} | {row['label']} | {price}")
    if row["reasons"]:
        print("  gate: " + "; ".join(row["reasons"]))
    if row.get("levels_breaking_floor"):
        print("  other levels: " + "; ".join(row["levels_breaking_floor"]))
    print(f"appended to {OUT}")


def load_rows():
    if not OUT.exists():
        return []
    rows = []
    for raw in OUT.read_text().splitlines():
        line = raw.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def workload_key(row):
    """Two runs are only comparable if they served the same workload.

    A bigger prompt costs more per output token whatever the engine does, so
    ranking a 400-token run against a 12k-token one rewards the workload, not
    the configuration.
    """
    return (
        row.get("context_lines", 0),
        row.get("max_tokens"),
        row.get("model"),
        # Hardware and price too: a $/1M gap that comes from a different machine
        # or a different assumed rate is not a gap between engine configurations.
        row.get("machine"),
        row.get("num_gpus"),
        row.get("gpu_hourly_usd"),
    )


def cmd_compare(_args):
    rows = load_rows()
    if not rows:
        raise SystemExit(f"no runs in {OUT}")
    rows = [r for r in rows if any(lv["streams"] == FLOOR_LEVEL for lv in r["levels"])]
    if not rows:
        raise SystemExit(f"no runs in {OUT} have a level at concurrency {FLOOR_LEVEL}")

    groups: dict[tuple, list] = {}
    for row in rows:
        groups.setdefault(workload_key(row), []).append(row)

    for key, group in groups.items():
        max_tokens, model, machine = key[1], key[2], key[3]
        prompt = next(
            (
                lv.get("prompt_tokens")
                for lv in group[0]["levels"]
                if lv.get("prompt_tokens")
            ),
            "?",
        )
        print(
            f"\nworkload: {prompt} prompt tokens, {max_tokens} output, "
            f"{model} on {machine}"
        )
        header = (
            f"{'label':<22}{'floor':<7}"
            f"{f'ttft95@{FLOOR_LEVEL}':>9}{f'tok/s@{FLOOR_LEVEL}':>9}"
            f"{'agg@max':>9}{'$/1M':>8}{'  at':>5}"
        )
        print(header)
        print("-" * len(header))
        # Recompute the verdict and the price from the raw levels rather than
        # trusting the stored fields: the log is append-only, so rows written by
        # an older scoring rule would otherwise keep a verdict this tool no
        # longer stands behind.
        for row in group:
            gate = next(lv for lv in row["levels"] if lv["streams"] == FLOOR_LEVEL)
            top = max(row["levels"], key=lambda lv: lv["streams"])
            best = best_level(
                row["levels"], row.get("gpu_hourly_usd"), row.get("num_gpus")
            )
            holds = not level_reasons(gate)
            print(
                f"{row['label']:<22}"
                f"{('pass' if holds else 'FAIL'):<7}"
                f"{str(gate.get('ttft_p95')):>9}"
                f"{str(gate.get('decode_tok_s_p50')):>9}"
                f"{str(top.get('agg_out_tok_s')):>9}"
                f"{str(best['usd_per_1m_out']) if best else '-':>8}"
                f"{str(best['streams']) if best else '-':>5}"
            )
        scored = [
            (r, best_level(r["levels"], r.get("gpu_hourly_usd"), r.get("num_gpus")))
            for r in group
        ]
        passing = [(r, b) for r, b in scored if b is not None]
        if not passing:
            print("  no run holds the floor on this workload yet")
            continue
        best_row, best_lv = min(passing, key=lambda rb: rb[1]["usd_per_1m_out"])
        print(
            f"  best: {best_row['label']} at "
            f"${best_lv['usd_per_1m_out']}/1M output tokens, "
            f"{best_lv['streams']} streams"
        )
        print(f"  engine_args: {' '.join(best_row['engine_args'])}")
        if group[-1]["label"] != best_row["label"]:
            print(f"  last run {group[-1]['label']}: revert, it does not beat the best")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    bench = sub.add_parser("bench", help="run the workload and append a row")
    bench.add_argument(
        "--label", required=True, help="short name for this configuration"
    )
    bench.add_argument("--note", default="", help="what changed since the last run")
    bench.add_argument("--url", default=BASE_URL)
    bench.add_argument("--model", default="qwen3.8-27b")
    bench.add_argument("--machine", default="GPU-RTXPRO6000")
    bench.add_argument("--num-gpus", type=int, default=1)
    bench.add_argument("--gpu-hourly", type=float, default=GPU_HOURLY_USD)
    bench.add_argument("--levels", default="1,8,32")
    bench.add_argument("--max-tokens", type=int, default=512)
    bench.add_argument(
        "--context-lines",
        type=int,
        default=0,
        help="extra deterministic repo-context blocks, for a realistic prompt",
    )
    bench.add_argument(
        "--requests-per-level",
        type=int,
        default=None,
        help="total requests per level (default: 3x the concurrency)",
    )
    bench.add_argument(
        "--engine-args",
        default="",
        help="the flags this run was served with, as one quoted string",
    )
    bench.set_defaults(func=cmd_bench)

    compare = sub.add_parser(
        "compare", help="table of every run, best passing one marked"
    )
    compare.set_defaults(func=cmd_compare)

    args = parser.parse_args()
    if hasattr(args, "engine_args"):
        args.engine_args = args.engine_args.split()
    if asyncio.iscoroutinefunction(args.func):
        asyncio.run(args.func(args))
    else:
        args.func(args)


if __name__ == "__main__":
    main()
