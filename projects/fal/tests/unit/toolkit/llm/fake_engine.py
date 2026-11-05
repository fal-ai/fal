"""Minimal stand-in for the vLLM OpenAI server (CPU only)."""

from __future__ import annotations

import argparse
import asyncio
import json
import os

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, PlainTextResponse, StreamingResponse

app = FastAPI()
ABORT_FILE = os.environ.get("FAKE_ENGINE_ABORT_FILE")
LAST_BODY_FILE = os.environ.get("FAKE_ENGINE_BODY_FILE")


def _mark_abort() -> None:
    if ABORT_FILE:
        with open(ABORT_FILE, "a") as f:
            f.write("abort\n")


@app.get("/health")
async def health():
    return PlainTextResponse("")


@app.get("/metrics")
async def metrics():
    return PlainTextResponse(
        "# HELP x\n"
        'vllm:num_requests_running{engine="0"} 2.0\n'
        'vllm:num_requests_waiting{engine="0"} 1.0\n'
        'vllm:kv_cache_usage_perc{engine="0"} 0.25\n'
        'vllm:spec_decode_num_draft_tokens_total{engine="0"} 40.0\n'
        'vllm:spec_decode_num_accepted_tokens_total{engine="0"} 30.0\n'
        'vllm:spec_decode_num_accepted_tokens_per_pos_total{position="0"} 9.0\n'
        "other_metric 5\n"
    )


@app.get("/v1/models")
async def models():
    return {"object": "list", "data": [{"id": "fake"}]}


async def _complete(request: Request):
    body = await request.json()
    if LAST_BODY_FILE:
        with open(LAST_BODY_FILE, "w") as f:
            json.dump(body, f)
    usage = {"prompt_tokens": 7, "completion_tokens": 5, "total_tokens": 12}
    if body.get("fail"):
        return JSONResponse({"error": "boom"}, status_code=400)
    if body.get("stream"):
        # Hold response headers back, like an engine still queueing the request.
        for _ in range(int(body.get("header_delay_ticks", 0))):
            if await request.is_disconnected():
                _mark_abort()
                return JSONResponse({})
            await asyncio.sleep(0.1)

        async def gen():
            done = False
            try:
                for i in range(int(body.get("n_chunks", 5))):
                    delta = {"choices": [{"delta": {"content": str(i)}}]}
                    event = f"data: {json.dumps(delta)}\n\n"
                    if body.get("split_events"):
                        # Half an event per write, so a cut can land mid-event.
                        yield event[:10]
                        await asyncio.sleep(0.2)
                        yield event[10:]
                        await asyncio.sleep(0.2)
                    else:
                        yield event
                        await asyncio.sleep(0.2)
                if (body.get("stream_options") or {}).get("include_usage"):
                    yield f"data: {json.dumps({'choices': [], 'usage': usage})}\n\n"
                yield "data: [DONE]\n\n"
                done = True
            finally:
                if not done:
                    _mark_abort()

        return StreamingResponse(gen(), media_type="text/event-stream")
    for _ in range(int(body.get("delay_ticks", 0))):
        if await request.is_disconnected():
            _mark_abort()
            return JSONResponse({})
        await asyncio.sleep(0.1)
    return {"choices": [{"message": {"content": "hi"}}], "usage": usage}


app.post("/v1/chat/completions")(_complete)
app.post("/v1/completions")(_complete)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, required=True)
    args, _ = p.parse_known_args()
    uvicorn.run(app, host="127.0.0.1", port=args.port, log_level="warning")
