"""Streaming load generator for the LLMApp proxy bench.

Runs ``--streams`` concurrent SSE streams spread over ``--procs`` processes
for ``--duration`` seconds (each finished stream is replaced). Reports TTFT,
per-chunk path latency (client receive time minus the engine's send stamp),
throughput and errors as one JSON object. Event throughput counts every
stamped SSE event (finish and usage included); token throughput counts
generated tokens only, from each stream's usage chunk.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import multiprocessing
import re
import time
from collections import Counter

import uvloop

# Numbers are anchored on the next delimiter so a read that ends inside the
# digits does not match a truncated value now and the full one next read.
TS = re.compile(rb'"ts": ?(\d+\.\d+)[,}]')
CONTENT = re.compile(rb'"delta":\{"content":')
USAGE = re.compile(rb'"completion_tokens": ?(\d+)[,}]')
# Longer than any pattern above, so a match split across reads is still found.
_TAIL = 40


async def one(host, port, body, stats, lat, stop_at, sem):
    t_req = time.time()
    content = 0
    usage_tokens = None
    try:
        r, w = await asyncio.open_connection(host, port)
    except Exception as e:
        stats["err"][f"connect:{type(e).__name__}"] += 1
        return
    try:
        payload = json.dumps(body).encode()
        w.write(
            b"POST /v1/chat/completions HTTP/1.1\r\nHost: x\r\n"
            b"Content-Type: application/json\r\nContent-Length: "
            + str(len(payload)).encode()
            + b"\r\n\r\n"
            + payload
        )
        buf = b""
        while b"\r\n\r\n" not in buf:
            d = await r.read(65536)
            if not d:
                raise ConnectionError("eof before headers")
            buf += d
        head, _, buf = buf.partition(b"\r\n\r\n")
        status = int(head.split(b" ", 2)[1])
        if status != 200:
            stats["err"][f"http{status}"] += 1
            return
        first = True
        tail = b""
        done = False
        data = buf
        while True:
            if data:
                now = time.time()
                if first:
                    stats["ttft"].append(now - t_req)
                    first = False
                buf2 = tail + data
                # Matches that end inside the carried tail were counted last read.
                fresh = len(tail)
                for m in TS.finditer(buf2):
                    if m.end() <= fresh:
                        continue
                    stats["events"] += 1
                    stats["seen"] += 1
                    if stats["seen"] % stats["sample"] == 0:
                        lat.append(now - float(m.group(1)))
                content += sum(1 for m in CONTENT.finditer(buf2) if m.end() > fresh)
                for m in USAGE.finditer(buf2):
                    if m.end() > fresh:
                        usage_tokens = int(m.group(1))
                stats["bytes"] += len(data)
                tail = buf2[-_TAIL:]
                if b"[DONE]" in buf2[-200:]:
                    done = True
                if buf2.endswith(b"0\r\n\r\n"):
                    break
            d = await r.read(65536)
            if not d:
                break
            data = d
        if done:
            stats["ok"] += 1
        else:
            stats["err"]["truncated"] += 1
    except Exception as e:
        stats["err"][type(e).__name__] += 1
    finally:
        w.close()
        # Counted even for a broken stream, like its events.
        stats["chunks"] += content
        if usage_tokens is None:
            # No usage chunk: every content chunk is full except the last.
            usage_tokens = min(content * body["tokens_per_chunk"], body["n_tokens"])
        stats["tokens"] += usage_tokens


async def worker(host, port, body, stats, lat, stop_at):
    while time.time() < stop_at:
        await one(host, port, body, stats, lat, stop_at, None)


async def run(a, n):
    stats = {
        "ttft": [],
        "chunks": 0,
        "events": 0,
        "tokens": 0,
        "bytes": 0,
        "ok": 0,
        "err": Counter(),
        "sample": a.sample,
        "seen": 0,
    }
    lat: list = []
    body = {
        "model": "x",
        "stream": True,
        "messages": [{"role": "user", "content": "hi"}],
        "n_tokens": a.tokens,
        "tok_per_s": a.tok_rate,
        "tokens_per_chunk": a.tpc,
    }
    t0 = time.time()
    stop_at = t0 + a.duration
    await asyncio.gather(
        *[worker(a.host, a.port, body, stats, lat, stop_at) for _ in range(n)]
    )
    stats["wall"] = time.time() - t0
    stats["lat"] = lat
    stats["err"] = dict(stats["err"])
    return stats


def child(a, n, q):
    q.put(uvloop.run(run(a, n)))


def pct(xs, p):
    xs = sorted(xs)
    return xs[min(int(len(xs) * p), len(xs) - 1)] if xs else float("nan")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, required=True)
    p.add_argument("--streams", type=int, default=64)
    p.add_argument("--procs", type=int, default=4)
    p.add_argument("--tok-rate", type=float, default=50.0)
    p.add_argument("--tpc", type=int, default=1)
    p.add_argument("--tokens", type=int, default=500)
    p.add_argument("--duration", type=float, default=15.0)
    p.add_argument("--sample", type=int, default=10)
    p.add_argument("--label", default="")
    a = p.parse_args()
    procs = min(a.procs, a.streams)
    q = multiprocessing.Queue()
    ps = []
    for i in range(procs):
        n = a.streams // procs + (1 if i < a.streams % procs else 0)
        pr = multiprocessing.Process(target=child, args=(a, n, q))
        pr.start()
        ps.append(pr)
    res = [q.get() for _ in ps]
    for pr in ps:
        pr.join()
    ttft = [x for r in res for x in r["ttft"]]
    lat = [x for r in res for x in r["lat"]]
    errs: Counter = Counter()
    for r in res:
        errs.update(r["err"])
    wall = max(r["wall"] for r in res)
    chunks = sum(r["chunks"] for r in res)
    events = sum(r["events"] for r in res)
    tokens = sum(r["tokens"] for r in res)
    ok = sum(r["ok"] for r in res)
    out = {
        "label": a.label,
        "streams": a.streams,
        "ok": ok,
        "errors": dict(errs),
        "chunks_per_s": round(chunks / wall),
        "events_per_s": round(events / wall),
        "tok_per_s": round(tokens / wall),
        "ttft_ms_p50": round(pct(ttft, 0.5) * 1e3, 2),
        "ttft_ms_p99": round(pct(ttft, 0.99) * 1e3, 2),
        "chunk_lat_ms_p50": round(pct(lat, 0.5) * 1e3, 2),
        "chunk_lat_ms_p99": round(pct(lat, 0.99) * 1e3, 2),
        "chunk_lat_ms_max": round(max(lat, default=float("nan")) * 1e3, 1),
    }
    print(json.dumps(out))
