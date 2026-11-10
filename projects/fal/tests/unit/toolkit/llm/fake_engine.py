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
PORT = 0
# Responses this engine created, like vLLM's in-memory response store.
RESPONSE_IDS: set = set()


def _mark_abort() -> None:
    if ABORT_FILE:
        # Also one file per port, so tests with several engines can tell
        # which one saw the abort.
        for path in (ABORT_FILE, f"{ABORT_FILE}.{PORT}"):
            with open(path, "a") as f:
                f.write("abort\n")


@app.get("/whoami")
async def whoami():
    return {"port": PORT, "gpus": os.environ.get("CUDA_VISIBLE_DEVICES")}


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
        'vllm:prompt_tokens_total{engine="0"} 70.0\n'
        'vllm:generation_tokens_total{engine="0"} 50.0\n'
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
    usage = {
        "prompt_tokens": 7,
        "completion_tokens": 5,
        "total_tokens": 12,
        "prompt_tokens_details": {"cached_tokens": 3},
    }
    if body.get("fail"):
        if body.get("error_body_ticks"):
            # Error headers now, the body later, like a stalled engine.
            async def slow_error():
                try:
                    for _ in range(int(body["error_body_ticks"])):
                        await asyncio.sleep(0.1)
                    yield b'{"error": "boom"}'
                except BaseException:
                    _mark_abort()
                    raise

            return StreamingResponse(
                slow_error(),
                status_code=int(body.get("fail_status", 400)),
                media_type="application/json",
            )
        return JSONResponse(
            {"error": "boom"}, status_code=int(body.get("fail_status", 400))
        )
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


async def _respond(request: Request):
    body = await request.json()
    if LAST_BODY_FILE:
        with open(LAST_BODY_FILE, "w") as f:
            json.dump(body, f)
    usage = {
        "input_tokens": 7,
        "output_tokens": 5,
        "total_tokens": 12,
        "input_tokens_details": {"cached_tokens": 3},
    }
    previous = body.get("previous_response_id")
    if previous is not None and previous not in RESPONSE_IDS:
        return JSONResponse(
            {"error": {"message": f"Response with id '{previous}' not found."}},
            status_code=404,
        )
    response_id = f"resp_{PORT}_{len(RESPONSE_IDS)}"
    RESPONSE_IDS.add(response_id)
    # Padding makes the terminal event larger than the proxy's stream tail.
    text = "x" * int(body.get("pad_chars", 0))
    final = {
        "type": "response.completed",
        "response": {
            "id": response_id,
            "status": "completed",
            "output_text": text,
            "usage": usage,
        },
    }
    if not body.get("stream"):
        for _ in range(int(body.get("delay_ticks", 0))):
            if await request.is_disconnected():
                _mark_abort()
                return JSONResponse({})
            await asyncio.sleep(0.1)
        return final["response"]

    async def gen():
        done = False
        try:
            for i in range(int(body.get("n_chunks", 3))):
                delta = {"type": "response.output_text.delta", "delta": str(i)}
                yield f"event: {delta['type']}\ndata: {json.dumps(delta)}\n\n"
                await asyncio.sleep(0.2)
            yield f"event: response.completed\ndata: {json.dumps(final)}\n\n"
            done = True
        finally:
            if not done:
                _mark_abort()

    return StreamingResponse(gen(), media_type="text/event-stream")


app.post("/v1/responses")(_respond)
app.post("/v1/chat/completions")(_complete)
app.post("/v1/completions")(_complete)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, required=True)
    args, _ = p.parse_known_args()
    PORT = args.port
    uvicorn.run(app, host="127.0.0.1", port=args.port, log_level="warning")
