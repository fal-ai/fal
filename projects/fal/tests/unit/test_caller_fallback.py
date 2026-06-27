"""Unit tests for the universal caller-defined fallback engine (fal._caller_fallback).

Covers the pure pieces that the framework middleware composes: lenient chain
parsing, status -> trigger mapping, output normalization, and the chain walk
(including robustness against bad caller input and the output-shape guard).
"""

import asyncio

import pytest
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.testclient import TestClient
from pydantic import BaseModel, ConfigDict
from starlette.middleware.base import BaseHTTPMiddleware

from fal._caller_fallback import (
    FallbackNode,
    build_fallback_middleware,
    media_field,
    normalize_output,
    parse_fallback_chain,
    primary_timeout_from_body,
    record_fallback,
    run_fallback_chain,
    trigger_for_status,
)

GPT = "fal-ai/gpt-image-2"
SEEDREAM = "fal-ai/bytedance/seedream/v4.5/edit"


# --- parse_fallback_chain -----------------------------------------------------


def test_parse_valid_chain():
    body = {
        "prompt": "x",
        "fal_fallback": [
            {"endpoint": GPT, "input": {"prompt": "x"}, "on": ["error"], "timeout": 30}
        ],
    }
    chain = parse_fallback_chain(body)
    assert len(chain) == 1
    assert chain[0].endpoint == GPT
    assert chain[0].input == {"prompt": "x"}
    assert chain[0].on == ["error"]
    assert chain[0].timeout == 30


@pytest.mark.parametrize("body", [None, "str", 123, {}, {"fal_fallback": "nope"}, {"fal_fallback": {}}])
def test_parse_absent_or_malformed_yields_empty(body):
    assert parse_fallback_chain(body) == []


def test_parse_skips_bad_items_keeps_good():
    body = {"fal_fallback": ["not a dict", {"endpoint": GPT}, 42, {"input": {"a": 1}}]}
    chain = parse_fallback_chain(body)
    # Two dict items survive (one with endpoint, one endpoint-less but parseable).
    assert [n.endpoint for n in chain] == [GPT, None]


def test_parse_lenient_defaults():
    chain = parse_fallback_chain({"fal_fallback": [{"endpoint": GPT}]})
    node = chain[0]
    assert node.input == {}
    assert node.on == ["error", "timeout"]
    assert node.timeout is None


# --- trigger_for_status -------------------------------------------------------


@pytest.mark.parametrize(
    "status,expected",
    [
        (408, "timeout"),
        (504, "timeout"),
        (500, "error"),
        (502, "error"),
        (503, "error"),
        (422, "error"),  # model rejection (content-policy carved out in middleware)
        (429, "error"),  # rate limit -> try another model
    ],
)
def test_trigger_for_failure_statuses(status, expected):
    assert trigger_for_status(status) == expected


@pytest.mark.parametrize("status", [200, 201, 301, 400, 401, 403, 404, 413, 415])
def test_trigger_none_for_non_failure(status):
    assert trigger_for_status(status) is None


# --- normalize_output ---------------------------------------------------------


def test_normalize_single_to_list():
    assert normalize_output({"image": {"url": "a"}}, "images")["images"] == [{"url": "a"}]


def test_normalize_list_to_single():
    assert normalize_output({"images": [{"url": "a"}, {"url": "b"}]}, "image")["image"] == {"url": "a"}


def test_normalize_passthrough_and_non_dict():
    r = {"images": [{"url": "a"}]}
    assert normalize_output(r, "images") is r
    assert normalize_output("x", "images") == "x"
    assert normalize_output({"images": []}, None) == {"images": []}


# --- run_fallback_chain -------------------------------------------------------


def _forward_returning(mapping):
    async def _forward(endpoint, payload):
        value = mapping[endpoint]
        if isinstance(value, Exception):
            raise value
        return value

    return _forward


def test_chain_first_eligible_node_wins():
    chain = parse_fallback_chain({"fal_fallback": [{"endpoint": GPT, "input": {"prompt": "x"}}]})
    forward = _forward_returning({GPT: {"images": [{"url": "ok"}]}})
    result, served = asyncio.run(run_fallback_chain(chain, trigger="error", forward=forward))
    assert served == GPT
    assert result == {"images": [{"url": "ok"}]}


def test_chain_skips_node_when_trigger_not_listed():
    chain = parse_fallback_chain({"fal_fallback": [{"endpoint": GPT, "on": ["error"]}]})
    forward = _forward_returning({GPT: {"images": []}})
    result, served = asyncio.run(run_fallback_chain(chain, trigger="timeout", forward=forward))
    assert (result, served) == (None, None)


def test_chain_advances_past_failing_node():
    chain = parse_fallback_chain(
        {"fal_fallback": [{"endpoint": GPT}, {"endpoint": SEEDREAM}]}
    )
    forward = _forward_returning(
        {GPT: RuntimeError("down"), SEEDREAM: {"images": [{"url": "ok"}]}}
    )
    result, served = asyncio.run(run_fallback_chain(chain, trigger="error", forward=forward))
    assert served == SEEDREAM
    assert result == {"images": [{"url": "ok"}]}


def test_chain_exhausted_returns_none():
    chain = parse_fallback_chain({"fal_fallback": [{"endpoint": GPT}]})
    forward = _forward_returning({GPT: RuntimeError("down")})
    assert asyncio.run(run_fallback_chain(chain, trigger="error", forward=forward)) == (None, None)


def test_chain_endpointless_node_skipped():
    chain = parse_fallback_chain(
        {"fal_fallback": [{"input": {"a": 1}}, {"endpoint": GPT}]}
    )
    forward = _forward_returning({GPT: {"images": [{"url": "ok"}]}})
    result, served = asyncio.run(run_fallback_chain(chain, trigger="error", forward=forward))
    assert served == GPT


def test_chain_output_guard_skips_incompatible_then_serves():
    chain = parse_fallback_chain(
        {"fal_fallback": [{"endpoint": GPT}, {"endpoint": SEEDREAM}]}
    )
    forward = _forward_returning(
        {GPT: {"video": {"url": "v"}}, SEEDREAM: {"images": [{"url": "ok"}]}}
    )
    result, served = asyncio.run(
        run_fallback_chain(chain, trigger="error", forward=forward, output_field="images")
    )
    # GPT returned a video (no `images`) -> rejected; chain advanced to SEEDREAM.
    assert served == SEEDREAM
    assert result["images"] == [{"url": "ok"}]


# --- primary_timeout_from_body ------------------------------------------------


def test_primary_timeout_valid():
    assert primary_timeout_from_body({"fal_fallback_timeout": 30}) == 30.0
    assert primary_timeout_from_body({"fal_fallback_timeout": "12.5"}) == 12.5


@pytest.mark.parametrize(
    "body",
    [None, "x", {}, {"fal_fallback_timeout": 0}, {"fal_fallback_timeout": -5}, {"fal_fallback_timeout": "abc"}],
)
def test_primary_timeout_absent_or_invalid(body):
    assert primary_timeout_from_body(body) is None


# --- media_field --------------------------------------------------------------


def test_media_field_detection():
    class ImagesOut(BaseModel):
        images: list = []

    class ImageOut(BaseModel):
        image: dict = {}

    class TextOut(BaseModel):
        text: str = ""

    class JsonOut(BaseModel):
        data: dict = {}

    assert media_field(ImagesOut) == "images"
    assert media_field(ImageOut) == "image"
    assert media_field(TextOut) == "text"
    assert media_field(JsonOut) is None  # non-text/media outputs stay ungrouped
    assert media_field(None) is None


def test_chain_normalizes_winner_to_output_field():
    chain = parse_fallback_chain({"fal_fallback": [{"endpoint": GPT}]})
    forward = _forward_returning({GPT: {"image": {"url": "single"}}})
    result, served = asyncio.run(
        run_fallback_chain(chain, trigger="error", forward=forward, output_field="images")
    )
    assert result["images"] == [{"url": "single"}]


# --- HTTP integration: the middleware glue via FastAPI TestClient -------------
#
# These exercise the actual middleware behaviour (body buffering + re-read,
# raised-exception vs 5xx-response triggering, primary timeout, response
# building, disable opt-out) without live fal serving. Only fal_client auth and
# real gateway interaction remain out of scope here.


def _recording_forward(result_by_endpoint):
    calls = []

    async def _forward(endpoint, payload):
        calls.append((endpoint, payload))
        return result_by_endpoint.get(endpoint, {"images": [{"url": f"fb:{endpoint}"}]})

    _forward.calls = calls
    return _forward


def _build_app(forward, output_fields=None):
    app = FastAPI()
    app.add_middleware(
        BaseHTTPMiddleware,
        dispatch=build_fallback_middleware(forward=forward, output_fields=output_fields or {}),
    )

    @app.post("/ok")
    async def ok(body: dict):
        return {"images": [{"url": "primary"}], "echo": body.get("prompt")}

    @app.post("/boom")
    async def boom(body: dict):
        raise RuntimeError("primary crashed")

    @app.post("/err")
    async def err(body: dict):
        return JSONResponse({"detail": "downstream"}, status_code=503)

    @app.post("/slow")
    async def slow(body: dict):
        await asyncio.sleep(1.0)
        return {"images": [{"url": "primary"}]}

    class StrictIn(BaseModel):
        model_config = ConfigDict(extra="forbid")
        prompt: str

    @app.post("/strict")
    async def strict(body: StrictIn):
        return {"images": [{"url": "primary"}], "echo": body.prompt}

    @app.post("/sse")
    async def sse(body: dict):
        async def gen():
            yield b"data: hello\n\n"

        return StreamingResponse(gen(), media_type="text/event-stream")

    @app.post("/sse_fail")
    async def sse_fail(body: dict):
        return JSONResponse(
            {"detail": "stream setup failed"},
            status_code=500,
            headers={"content-type": "text/event-stream"},
        )

    @app.post("/raw")
    async def raw(request: Request):
        b = await request.body()
        return {"len": len(b), "ct": request.headers.get("content-type", "")}

    @app.post("/reject422")
    async def reject422(body: dict):
        return JSONResponse(
            {"detail": [{"type": "validation_error", "msg": "model can't handle"}]},
            status_code=422,
        )

    @app.post("/policy422")
    async def policy422(body: dict):
        return JSONResponse(
            {"detail": [{"type": "content_policy_violation", "msg": "blocked"}]},
            status_code=422,
        )

    @app.post("/ratelimit")
    async def ratelimit(body: dict):
        return JSONResponse({"detail": "rate limited"}, status_code=429)

    return app


def _chain(endpoint=GPT):
    return [{"endpoint": endpoint, "input": {"prompt": "fb"}}]


def test_http_success_passthrough_and_body_reread():
    fwd = _recording_forward({})
    client = TestClient(_build_app(fwd))
    r = client.post("/ok", json={"prompt": "hello", "fal_fallback": _chain()})
    assert r.status_code == 200
    # Endpoint still parsed the body after the middleware read it (echo == prompt).
    assert r.json()["echo"] == "hello"
    # Primary succeeded -> no fallback fired.
    assert "x-app-fal-api-fallback" not in r.headers
    assert fwd.calls == []


def test_http_fallback_on_raised_exception():
    fwd = _recording_forward({GPT: {"images": [{"url": "from-gpt"}]}})
    client = TestClient(_build_app(fwd))
    r = client.post("/boom", json={"prompt": "x", "fal_fallback": _chain()})
    assert r.status_code == 200
    assert r.headers["x-app-fal-api-fallback"] == "true"
    assert r.headers["x-app-fal-api-fallback-endpoint"] == GPT
    assert r.headers["x-fal-bill-as"] == GPT
    assert r.json()["images"] == [{"url": "from-gpt"}]
    assert fwd.calls and fwd.calls[0][0] == GPT


def test_http_fallback_on_5xx_response():
    fwd = _recording_forward({GPT: {"images": [{"url": "from-gpt"}]}})
    client = TestClient(_build_app(fwd))
    r = client.post("/err", json={"prompt": "x", "fal_fallback": _chain()})
    assert r.status_code == 200
    assert r.headers["x-app-fal-api-fallback-endpoint"] == GPT


def test_http_disable_header_skips_fallback():
    fwd = _recording_forward({})
    client = TestClient(_build_app(fwd), raise_server_exceptions=False)
    r = client.post(
        "/boom",
        json={"prompt": "x", "fal_fallback": _chain()},
        headers={"x-app-fal-disable-fallback": "true"},
    )
    assert r.status_code == 500  # primary error surfaced, no fallback
    assert fwd.calls == []


def test_http_primary_timeout_falls_back():
    fwd = _recording_forward({GPT: {"images": [{"url": "from-gpt"}]}})
    client = TestClient(_build_app(fwd))
    r = client.post(
        "/slow",
        json={"prompt": "x", "fal_fallback_timeout": 0.1, "fal_fallback": _chain()},
    )
    assert r.status_code == 200
    assert r.headers["x-app-fal-api-fallback-endpoint"] == GPT


def test_http_no_chain_passthrough():
    fwd = _recording_forward({})
    client = TestClient(_build_app(fwd))
    r = client.post("/ok", json={"prompt": "hello"})
    assert r.status_code == 200
    assert "x-app-fal-api-fallback" not in r.headers
    assert fwd.calls == []


def test_http_output_normalization_single_to_list():
    # Primary route declares images:list; fallback returns a single image.
    fwd = _recording_forward({GPT: {"image": {"url": "single"}}})
    client = TestClient(_build_app(fwd, output_fields={"/err": "images"}))
    r = client.post("/err", json={"prompt": "x", "fal_fallback": _chain()})
    assert r.status_code == 200
    assert r.json()["images"] == [{"url": "single"}]


# --- metrics: record_fallback -------------------------------------------------


def test_record_fallback_increments_when_available():
    pytest.importorskip("prometheus_client")
    from prometheus_client import REGISTRY

    labels = {"trigger": "error", "outcome": "served", "endpoint": GPT}
    before = REGISTRY.get_sample_value("fal_caller_fallback_total", labels) or 0.0
    assert record_fallback(trigger="error", outcome="served", endpoint=GPT) is True
    after = REGISTRY.get_sample_value("fal_caller_fallback_total", labels) or 0.0
    assert after == before + 1


def test_record_fallback_handles_missing_endpoint():
    # endpoint=None must not crash; it is labelled "none".
    assert record_fallback(trigger="timeout", outcome="exhausted", endpoint=None) in (
        True,
        False,
    )


# --- Must-fixes: extra="forbid" stripping + streaming skip ---------------------


def test_http_strip_fields_avoid_422_on_extra_forbid():
    # /strict input forbids extras; without stripping, `fal_fallback` in the body
    # would 422 the whole request. The middleware must strip it so the primary runs.
    fwd = _recording_forward({})
    client = TestClient(_build_app(fwd), raise_server_exceptions=False)
    r = client.post("/strict", json={"prompt": "hi", "fal_fallback": _chain()})
    assert r.status_code == 200
    assert r.json()["echo"] == "hi"
    assert "x-app-fal-api-fallback" not in r.headers
    assert fwd.calls == []


def test_http_strip_also_applies_when_disabled():
    # Even with fallback disabled, the reserved fields must be stripped so an
    # extra="forbid" app isn't 422'd.
    fwd = _recording_forward({})
    client = TestClient(_build_app(fwd), raise_server_exceptions=False)
    r = client.post(
        "/strict",
        json={"prompt": "hi", "fal_fallback_timeout": 5, "fal_fallback": _chain()},
        headers={"x-app-fal-disable-fallback": "true"},
    )
    assert r.status_code == 200
    assert r.json()["echo"] == "hi"


def test_http_sse_success_passthrough():
    fwd = _recording_forward({})
    client = TestClient(_build_app(fwd))
    r = client.post("/sse", json={"prompt": "x", "fal_fallback": _chain()})
    assert r.status_code == 200
    assert "text/event-stream" in r.headers.get("content-type", "")
    assert "x-app-fal-api-fallback" not in r.headers
    assert fwd.calls == []


def test_http_sse_failure_not_replaced_by_fallback():
    # A failed streaming response must NOT be swapped for a JSON fallback.
    fwd = _recording_forward({})
    client = TestClient(_build_app(fwd), raise_server_exceptions=False)
    r = client.post("/sse_fail", json={"prompt": "x", "fal_fallback": _chain()})
    assert r.status_code == 500
    assert fwd.calls == []


def test_chain_text_output_guard_skips_non_text():
    # Primary returns `text`; a node lacking `text` is rejected, chain advances.
    chain = parse_fallback_chain(
        {"fal_fallback": [{"endpoint": GPT}, {"endpoint": SEEDREAM}]}
    )
    forward = _forward_returning(
        {GPT: {"images": [{"url": "x"}]}, SEEDREAM: {"text": "hello world"}}
    )
    result, served = asyncio.run(
        run_fallback_chain(chain, trigger="error", forward=forward, output_field="text")
    )
    assert served == SEEDREAM
    assert result["text"] == "hello world"


# --- Fail-safe: our code must never lose the primary's outcome ----------------


def _raising_forward():
    async def _f(endpoint, payload):
        raise RuntimeError("forward unavailable (e.g. fal_client missing)")

    _f.calls = []
    return _f


def test_http_keeps_primary_5xx_when_fallback_unavailable():
    # Primary returns 503 and every fallback forward fails -> the primary 5xx is
    # kept (we never crash or lose it).
    client = TestClient(_build_app(_raising_forward()), raise_server_exceptions=False)
    r = client.post("/err", json={"prompt": "x", "fal_fallback": _chain()})
    assert r.status_code == 503


def test_http_reraises_primary_error_when_fallback_unavailable():
    # Primary raises and the fallback forward also fails -> the original primary
    # error is surfaced (a 500), not swallowed.
    client = TestClient(_build_app(_raising_forward()), raise_server_exceptions=False)
    r = client.post("/boom", json={"prompt": "x", "fal_fallback": _chain()})
    assert r.status_code == 500


def test_http_non_json_post_passes_through_untouched():
    # Non-JSON POST (e.g. upload/binary) must skip the middleware entirely: no
    # body buffering, no _body touch, no fallback attempt.
    fwd = _recording_forward({})
    client = TestClient(_build_app(fwd))
    payload = b"\x00\x01 raw upload bytes \xff"
    r = client.post(
        "/raw", content=payload, headers={"content-type": "application/octet-stream"}
    )
    assert r.status_code == 200
    assert r.json()["len"] == len(payload)  # endpoint received the body intact
    assert fwd.calls == []


# --- 4xx triggers: 422 model rejection + 429 rate limit, content-policy carved out


def test_http_422_model_rejection_falls_back():
    fwd = _recording_forward({GPT: {"images": [{"url": "fb"}]}})
    client = TestClient(_build_app(fwd))
    r = client.post("/reject422", json={"prompt": "x", "fal_fallback": _chain()})
    assert r.status_code == 200
    assert r.headers["x-app-fal-api-fallback-endpoint"] == GPT
    assert fwd.calls


def test_http_422_content_policy_is_not_fallen_back():
    # A content-policy 422 must be surfaced, never fall back (no moderation bypass).
    fwd = _recording_forward({GPT: {"images": [{"url": "fb"}]}})
    client = TestClient(_build_app(fwd))
    r = client.post("/policy422", json={"prompt": "x", "fal_fallback": _chain()})
    assert r.status_code == 422
    assert "x-app-fal-api-fallback" not in r.headers
    assert "content_policy_violation" in r.text  # body preserved
    assert fwd.calls == []


def test_http_429_rate_limit_falls_back():
    fwd = _recording_forward({GPT: {"images": [{"url": "fb"}]}})
    client = TestClient(_build_app(fwd))
    r = client.post("/ratelimit", json={"prompt": "x", "fal_fallback": _chain()})
    assert r.status_code == 200
    assert r.headers["x-app-fal-api-fallback-endpoint"] == GPT
