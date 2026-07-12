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
    parse_fallback_chain,
    primary_timeout_from_body,
    record_fallback,
    resolve_input_refs,
    run_fallback_chain,
    trigger_for_status,
    validate_chain_shapes,
)

GPT = "fal-ai/gpt-image-2"
SEEDREAM = "fal-ai/bytedance/seedream/v4.5/edit"


# --- parse_fallback_chain -----------------------------------------------------


def test_parse_valid_chain():
    body = {
        "prompt": "x",
        "fal_fallback": [
            {"endpoint": GPT, "input": {"prompt": "x"}, "on": ["server_error"], "timeout": 30}
        ],
    }
    chain = parse_fallback_chain(body)
    assert len(chain) == 1
    assert chain[0].endpoint == GPT
    assert chain[0].input == {"prompt": "x"}
    assert chain[0].on == ["server_error"]
    assert chain[0].timeout == 30


@pytest.mark.parametrize("body", [None, "str", 123, {}, {"fal_fallback": "nope"}, {"fal_fallback": {}}])
def test_parse_absent_or_malformed_yields_empty(body):
    assert parse_fallback_chain(body) == []


def test_parse_skips_bad_items_keeps_good():
    body = {"fal_fallback": ["not a dict", {"endpoint": GPT}, 42, {"input": {"a": 1}}]}
    chain = parse_fallback_chain(body)
    # Non-dict items are skipped; the first parseable node fills the single slot.
    assert [n.endpoint for n in chain] == [GPT]


def test_parse_lenient_defaults():
    chain = parse_fallback_chain({"fal_fallback": [{"endpoint": GPT}]})
    node = chain[0]
    assert node.input == {}
    assert node.on == ["server_error", "timeout"]
    assert node.timeout is None
    assert node.retry is False


def test_parse_retry_node():
    chain = parse_fallback_chain({"fal_fallback": [{"retry": True}]})
    assert chain[0].retry is True
    assert chain[0].endpoint is None
    normal = parse_fallback_chain({"fal_fallback": [{"endpoint": GPT}]})
    assert normal[0].retry is False


def test_parse_chain_capped_at_one_node():
    # 2 total attempts incl. the primary -> at most ONE chain node (retry OR
    # fallback); extras are leniently ignored (never an error).
    body = {
        "fal_fallback": [
            {"endpoint": "a/a"},
            {"endpoint": "b/b"},
            {"endpoint": "c/c"},
        ]
    }
    assert [n.endpoint for n in parse_fallback_chain(body)] == ["a/a"]


# --- trigger_for_status -------------------------------------------------------


@pytest.mark.parametrize(
    "status,expected",
    [
        (408, "timeout"),
        (500, "server_error"),
        (502, "server_error"),
        (503, "server_error"),
        (504, "server_error"),  # 504 counts as a server error, not a timeout
        (422, "policy"),  # opt-in: only nodes listing "policy" in `on` match
    ],
)
def test_trigger_for_failure_statuses(status, expected):
    assert trigger_for_status(status) == expected


# Other 4xx are surfaced and never fall back; 429 stays out of scope.
@pytest.mark.parametrize("status", [200, 201, 301, 400, 401, 403, 404, 413, 415, 429])
def test_trigger_none_for_non_failure(status):
    assert trigger_for_status(status) is None


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
    result, served = asyncio.run(run_fallback_chain(chain, trigger="server_error", forward=forward))
    assert served == GPT
    assert result == {"images": [{"url": "ok"}]}


def test_chain_skips_node_when_trigger_not_listed():
    chain = parse_fallback_chain({"fal_fallback": [{"endpoint": GPT, "on": ["server_error"]}]})
    forward = _forward_returning({GPT: {"images": []}})
    result, served = asyncio.run(run_fallback_chain(chain, trigger="timeout", forward=forward))
    assert (result, served) == (None, None)


def test_chain_advances_past_failing_node():
    # Engine-level: the walk advances past failures (the 1-node cap is a parse
    # concern; the engine itself handles any list it is given).
    chain = [FallbackNode(endpoint=GPT), FallbackNode(endpoint=SEEDREAM)]
    forward = _forward_returning(
        {GPT: RuntimeError("down"), SEEDREAM: {"images": [{"url": "ok"}]}}
    )
    result, served = asyncio.run(run_fallback_chain(chain, trigger="server_error", forward=forward))
    assert served == SEEDREAM
    assert result == {"images": [{"url": "ok"}]}


def test_chain_exhausted_returns_none():
    chain = parse_fallback_chain({"fal_fallback": [{"endpoint": GPT}]})
    forward = _forward_returning({GPT: RuntimeError("down")})
    assert asyncio.run(run_fallback_chain(chain, trigger="server_error", forward=forward)) == (None, None)


def test_chain_endpointless_node_skipped():
    chain = [FallbackNode(input={"a": 1}), FallbackNode(endpoint=GPT)]
    forward = _forward_returning({GPT: {"images": [{"url": "ok"}]}})
    result, served = asyncio.run(run_fallback_chain(chain, trigger="server_error", forward=forward))
    assert served == GPT


def test_chain_output_guard_skips_incompatible_then_serves():
    chain = [FallbackNode(endpoint=GPT), FallbackNode(endpoint=SEEDREAM)]
    forward = _forward_returning(
        {GPT: {"video": {"url": "v"}}, SEEDREAM: {"images": [{"url": "ok"}]}}
    )
    result, served = asyncio.run(
        run_fallback_chain(chain, trigger="server_error", forward=forward, output_field="images")
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


def test_chain_rejects_mismatched_shape_no_single_list_bridge():
    # Grouping is by EXACT output shape: a primary declaring `images` (list) is
    # NOT satisfied by a fallback returning a single `image`. The single<->list
    # bridge was removed, so this node is rejected and (with no other node) the
    # chain is exhausted -- the caller keeps the primary's failure.
    chain = parse_fallback_chain({"fal_fallback": [{"endpoint": GPT}]})
    forward = _forward_returning({GPT: {"image": {"url": "single"}}})
    result, served = asyncio.run(
        run_fallback_chain(chain, trigger="server_error", forward=forward, output_field="images")
    )
    assert (result, served) == (None, None)


# --- retry nodes: re-attempt the previous attempt ------------------------------


PRIMARY = "fal-ai/primary"


def test_chain_retry_first_reattempts_primary():
    # [retry] as the first node re-attempts the primary with its own payload.
    calls = []

    async def forward(endpoint, payload):
        calls.append((endpoint, payload))
        return {"images": [{"url": "retry-win"}]}

    chain = parse_fallback_chain({"fal_fallback": [{"retry": True}]})
    result, served = asyncio.run(
        run_fallback_chain(
            chain,
            trigger="server_error",
            forward=forward,
            primary_endpoint=PRIMARY,
            primary_input={"prompt": "orig"},
        )
    )
    assert served == PRIMARY
    assert calls == [(PRIMARY, {"prompt": "orig"})]
    assert result["images"] == [{"url": "retry-win"}]


def test_chain_retry_reattempts_last_attempt_not_primary():
    # Engine-level: a retry node re-attempts the LAST attempt (the fallback
    # that just failed), not the primary. (With the 1-node parse cap this
    # combination is not reachable via a request body; the engine keeps the
    # general semantics for future/direct callers.)
    calls = []

    async def forward(endpoint, payload):
        calls.append((endpoint, payload))
        if len(calls) == 1:
            raise RuntimeError("first attempt down")
        return {"images": [{"url": "second-try"}]}

    chain = [FallbackNode(endpoint=GPT, input={"prompt": "fb"}), FallbackNode(retry=True)]
    result, served = asyncio.run(
        run_fallback_chain(
            chain,
            trigger="server_error",
            forward=forward,
            primary_endpoint=PRIMARY,
            primary_input={"prompt": "orig"},
        )
    )
    assert served == GPT
    assert calls == [(GPT, {"prompt": "fb"}), (GPT, {"prompt": "fb"})]
    assert result["images"] == [{"url": "second-try"}]


def test_chain_retry_without_primary_endpoint_skipped():
    # A retry node with no known prior attempt (no primary endpoint available)
    # is unusable -> skipped, chain exhausted, primary failure kept.
    chain = parse_fallback_chain({"fal_fallback": [{"retry": True}]})
    forward = _forward_returning({})
    assert asyncio.run(run_fallback_chain(chain, trigger="server_error", forward=forward)) == (
        None,
        None,
    )


def test_chain_retry_node_ignores_its_own_endpoint():
    # When `retry: true`, any endpoint/input set on the node are ignored.
    calls = []

    async def forward(endpoint, payload):
        calls.append((endpoint, payload))
        return {"images": [{"url": "ok"}]}

    chain = parse_fallback_chain(
        {"fal_fallback": [{"retry": True, "endpoint": SEEDREAM, "input": {"x": 1}}]}
    )
    result, served = asyncio.run(
        run_fallback_chain(
            chain,
            trigger="server_error",
            forward=forward,
            primary_endpoint=PRIMARY,
            primary_input={"prompt": "orig"},
        )
    )
    assert served == PRIMARY
    assert calls == [(PRIMARY, {"prompt": "orig"})]


def test_chain_retry_respects_on_filter():
    # A retry node only fires for the outcomes listed in its `on`.
    chain = parse_fallback_chain({"fal_fallback": [{"retry": True, "on": ["timeout"]}]})
    forward = _forward_returning({})
    result, served = asyncio.run(
        run_fallback_chain(
            chain,
            trigger="server_error",
            forward=forward,
            primary_endpoint=PRIMARY,
            primary_input={},
        )
    )
    assert (result, served) == (None, None)


def test_chain_serves_exact_shape_match():
    # Same field on both sides (`images` -> `images`) is served untouched.
    chain = parse_fallback_chain({"fal_fallback": [{"endpoint": GPT}]})
    forward = _forward_returning({GPT: {"images": [{"url": "ok"}], "seed": 7}})
    result, served = asyncio.run(
        run_fallback_chain(chain, trigger="server_error", forward=forward, output_field="images")
    )
    assert served == GPT
    assert result == {"images": [{"url": "ok"}], "seed": 7}


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


async def _unknown_resolver(endpoint):
    # Default for tests: shape unknown -> pre-flight fails open, no network.
    return False, None


def _build_app(forward, output_fields=None, shape_resolver=None):
    app = FastAPI()
    app.add_middleware(
        BaseHTTPMiddleware,
        dispatch=build_fallback_middleware(
            forward=forward,
            output_fields=output_fields or {},
            shape_resolver=shape_resolver or _unknown_resolver,
        ),
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

    @app.post("/err400_downstream")
    async def err400_downstream(body: dict):
        return JSONResponse(
            {"detail": [{"type": "downstream_service_error", "msg": "partner down"}]},
            status_code=400,
        )

    @app.post("/err400_plain")
    async def err400_plain(body: dict):
        return JSONResponse({"detail": "bad request"}, status_code=400)

    @app.post("/haystatic")
    async def haystatic(request: Request):
        # Simulates an app carrying its own @fallback_to: when fallback is
        # disabled for the request it surfaces its primary error (mirroring the
        # real static handler, which raises DownstreamServiceUnavailableError
        # when disabled); otherwise its curated static fallback serves.
        disabled = request.headers.get(
            "x-app-fal-disable-fallback", ""
        ).lower() in ("true", "1", "yes", "y")
        if disabled:
            raise RuntimeError("primary failed; app static fallback suppressed")
        return {"images": [{"url": "app-static"}], "via": "app-static"}

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


def test_http_mismatched_shape_rejected_keeps_primary():
    # Primary route declares images:list; fallback returns a single `image`.
    # With the single<->list bridge removed, the guard rejects the fallback and
    # the primary's 503 is kept (no shape the caller didn't ask for).
    fwd = _recording_forward({GPT: {"image": {"url": "single"}}})
    client = TestClient(_build_app(fwd, output_fields={"/err": "images"}))
    r = client.post("/err", json={"prompt": "x", "fal_fallback": _chain()})
    assert r.status_code == 503
    assert "x-app-fal-api-fallback" not in r.headers


# --- metrics: record_fallback -------------------------------------------------


def test_record_fallback_increments_when_available():
    pytest.importorskip("prometheus_client")
    from prometheus_client import REGISTRY

    labels = {"trigger": "server_error", "outcome": "served", "endpoint": GPT}
    before = REGISTRY.get_sample_value("fal_caller_fallback_total", labels) or 0.0
    assert record_fallback(trigger="server_error", outcome="served", endpoint=GPT) is True
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
    chain = [FallbackNode(endpoint=GPT), FallbackNode(endpoint=SEEDREAM)]
    forward = _forward_returning(
        {GPT: {"images": [{"url": "x"}]}, SEEDREAM: {"text": "hello world"}}
    )
    result, served = asyncio.run(
        run_fallback_chain(chain, trigger="server_error", forward=forward, output_field="text")
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


# --- 422 -> "policy" (opt-in); other 4xx surfaced, never fall back -------------


def test_http_422_surfaced_without_policy_opt_in():
    # Default `on` is ["server_error", "timeout"]: a 422 walks the chain but matches no
    # node -> surfaced to the caller, no fallback.
    fwd = _recording_forward({GPT: {"images": [{"url": "fb"}]}})
    client = TestClient(_build_app(fwd))
    r = client.post("/reject422", json={"prompt": "x", "fal_fallback": _chain()})
    assert r.status_code == 422
    assert "x-app-fal-api-fallback" not in r.headers
    assert fwd.calls == []


def test_http_422_falls_back_with_policy_opt_in():
    # A node explicitly listing "policy" opts into falling back on a 422.
    fwd = _recording_forward({GPT: {"images": [{"url": "fb"}]}})
    client = TestClient(_build_app(fwd))
    r = client.post(
        "/reject422",
        json={
            "prompt": "x",
            "fal_fallback": [{"endpoint": GPT, "on": ["policy"], "input": {"prompt": "fb"}}],
        },
    )
    assert r.status_code == 200
    assert r.headers["x-app-fal-api-fallback-endpoint"] == GPT
    assert fwd.calls == [(GPT, {"prompt": "fb"})]


def test_http_content_policy_422_surfaced_without_opt_in():
    # A content-policy 422 is surfaced with its body intact unless the caller
    # explicitly opted into "policy" -- no silent moderation bypass.
    fwd = _recording_forward({GPT: {"images": [{"url": "fb"}]}})
    client = TestClient(_build_app(fwd))
    r = client.post("/policy422", json={"prompt": "x", "fal_fallback": _chain()})
    assert r.status_code == 422
    assert "x-app-fal-api-fallback" not in r.headers
    assert "content_policy_violation" in r.text  # body preserved
    assert fwd.calls == []


def test_http_content_policy_422_falls_back_with_explicit_opt_in():
    # Opting into "policy" is a conscious caller choice and covers content-policy
    # 422s too (trying another model on a rejection is explicitly requested).
    fwd = _recording_forward({GPT: {"images": [{"url": "fb"}]}})
    client = TestClient(_build_app(fwd))
    r = client.post(
        "/policy422",
        json={"prompt": "x", "fal_fallback": [{"endpoint": GPT, "on": ["policy"]}]},
    )
    assert r.status_code == 200
    assert r.headers["x-app-fal-api-fallback-endpoint"] == GPT


def test_http_429_rate_limit_surfaced_no_fallback():
    # 429 (rate limit) stays out of scope -> surfaced, no fallback.
    fwd = _recording_forward({GPT: {"images": [{"url": "fb"}]}})
    client = TestClient(_build_app(fwd))
    r = client.post("/ratelimit", json={"prompt": "x", "fal_fallback": _chain()})
    assert r.status_code == 429
    assert "x-app-fal-api-fallback" not in r.headers
    assert fwd.calls == []


def test_chain_policy_trigger_matches_only_opted_in_nodes():
    # With trigger="policy", default-`on` nodes are skipped; the first node that
    # explicitly lists "policy" serves.
    chain = [FallbackNode(endpoint=GPT), FallbackNode(endpoint=SEEDREAM, on=["policy"])]
    forward = _forward_returning(
        {GPT: {"images": [{"url": "skipped"}]}, SEEDREAM: {"images": [{"url": "ok"}]}}
    )
    result, served = asyncio.run(
        run_fallback_chain(chain, trigger="policy", forward=forward)
    )
    assert served == SEEDREAM
    assert result["images"] == [{"url": "ok"}]


# --- Scenario 4: caller chain suppresses the app's own static @fallback_to ----


def test_http_caller_chain_suppresses_app_static_fallback():
    # Option B: when the caller supplies a chain, the app's static @fallback_to
    # is suppressed so the CALLER'S choice wins. The middleware injects the
    # disable header, the app surfaces its primary error, and our chain serves
    # the caller's endpoint (not the app author's curated fallback).
    fwd = _recording_forward({GPT: {"images": [{"url": "caller-choice"}]}})
    client = TestClient(_build_app(fwd))
    r = client.post("/haystatic", json={"prompt": "x", "fal_fallback": _chain()})
    assert r.status_code == 200
    assert r.headers["x-app-fal-api-fallback-endpoint"] == GPT
    assert r.json()["images"] == [{"url": "caller-choice"}]
    assert fwd.calls and fwd.calls[0][0] == GPT


def test_http_app_static_runs_without_caller_chain():
    # No caller chain -> nothing injected -> the app's own static fallback serves
    # as usual (unchanged behaviour for the existing @fallback_to endpoints).
    fwd = _recording_forward({})
    client = TestClient(_build_app(fwd))
    r = client.post("/haystatic", json={"prompt": "x"})
    assert r.status_code == 200
    assert r.json()["via"] == "app-static"
    assert fwd.calls == []


# --- Retry shortcut + chain cap over HTTP --------------------------------------


def test_http_retry_node_reattempts_primary():
    # `{"retry": true}` resolves to the primary endpoint (x-fal-endpoint header)
    # and the STRIPPED primary payload (no fal_fallback in the forward).
    fwd = _recording_forward({"fal-ai/my-app": {"images": [{"url": "retry"}]}})
    client = TestClient(_build_app(fwd))
    r = client.post(
        "/err",
        json={"prompt": "x", "fal_fallback": [{"retry": True}]},
        headers={"x-fal-endpoint": "fal-ai/my-app"},
    )
    assert r.status_code == 200
    assert r.headers["x-fal-bill-as"] == "fal-ai/my-app"
    assert fwd.calls == [("fal-ai/my-app", {"prompt": "x"})]


def test_http_retry_without_endpoint_header_keeps_primary_failure():
    # No x-fal-endpoint header -> the retry node can't resolve a target -> it is
    # skipped and the primary's failure is kept (fail-safe, no crash).
    fwd = _recording_forward({})
    client = TestClient(_build_app(fwd), raise_server_exceptions=False)
    r = client.post("/err", json={"prompt": "x", "fal_fallback": [{"retry": True}]})
    assert r.status_code == 503
    assert fwd.calls == []


def _static_resolver(mapping):
    async def _resolve(endpoint):
        if endpoint in mapping:
            return True, mapping[endpoint]
        return False, None

    return _resolve


def test_validate_chain_reports_mismatched_node_index():
    chain = [FallbackNode(retry=True), FallbackNode(endpoint=SEEDREAM)]
    err = asyncio.run(
        validate_chain_shapes(chain, "images", _static_resolver({SEEDREAM: "video"}))
    )
    assert err["detail"][0]["loc"] == ["body", "fal_fallback", 1, "endpoint"]
    assert err["detail"][0]["type"] == "fallback_output_shape_mismatch"
    assert SEEDREAM in err["detail"][0]["msg"]


def test_validate_chain_passes_without_primary_field_or_resolution():
    chain = parse_fallback_chain({"fal_fallback": [{"endpoint": GPT}]})
    # Non-media primary: nothing to validate against.
    assert asyncio.run(
        validate_chain_shapes(chain, None, _static_resolver({GPT: "video"}))
    ) is None
    # Unresolvable target: fail open.
    assert asyncio.run(
        validate_chain_shapes(chain, "images", _static_resolver({}))
    ) is None


def test_http_shape_mismatch_rejected_up_front():
    # Primary produces images; the chain targets a video model -> a typed 400
    # BEFORE the primary (or any billable forward) runs.
    fwd = _recording_forward({})
    client = TestClient(
        _build_app(
            fwd,
            output_fields={"/ok": "images"},
            shape_resolver=_static_resolver({GPT: "video"}),
        )
    )
    r = client.post("/ok", json={"prompt": "x", "fal_fallback": _chain()})
    assert r.status_code == 400
    detail = r.json()["detail"][0]
    assert detail["type"] == "fallback_output_shape_mismatch"
    assert detail["loc"] == ["body", "fal_fallback", 0, "endpoint"]
    assert fwd.calls == []


def test_http_shape_match_passes_preflight_and_serves():
    fwd = _recording_forward({GPT: {"images": [{"url": "fb"}]}})
    client = TestClient(
        _build_app(
            fwd,
            output_fields={"/err": "images"},
            shape_resolver=_static_resolver({GPT: "images"}),
        )
    )
    r = client.post("/err", json={"prompt": "x", "fal_fallback": _chain()})
    assert r.status_code == 200
    assert r.headers["x-app-fal-api-fallback-endpoint"] == GPT


def test_http_unknown_shape_fails_open_runtime_guard_still_protects():
    # Resolver can't determine the target's shape -> the request proceeds;
    # the runtime output-shape guard still rejects the mismatched result and
    # the primary's failure is kept.
    fwd = _recording_forward({GPT: {"video": {"url": "v"}}})
    client = TestClient(
        _build_app(fwd, output_fields={"/err": "images"}),
        raise_server_exceptions=False,
    )
    r = client.post("/err", json={"prompt": "x", "fal_fallback": _chain()})
    assert r.status_code == 503


def test_http_retry_node_skips_preflight_validation():
    calls = []

    async def recording_resolver(endpoint):
        calls.append(endpoint)
        return True, "video"

    fwd = _recording_forward({"fal-ai/my-app": {"images": [{"url": "r"}]}})
    client = TestClient(
        _build_app(fwd, output_fields={"/err": "images"}, shape_resolver=recording_resolver)
    )
    r = client.post(
        "/err",
        json={"prompt": "x", "fal_fallback": [{"retry": True}]},
        headers={"x-fal-endpoint": "fal-ai/my-app"},
    )
    assert r.status_code == 200  # retry served; resolver never consulted
    assert calls == []


# --- 400 `downstream_service_error` upgrade (partner-backend failure) ----------


def test_http_400_downstream_service_error_falls_back():
    # A partner failure surfaces as 400 downstream_service_error -> that is a
    # server-side failure of the target, so the chain rescues it.
    fwd = _recording_forward({GPT: {"images": [{"url": "fb"}]}})
    client = TestClient(_build_app(fwd))
    r = client.post("/err400_downstream", json={"prompt": "x", "fal_fallback": _chain()})
    assert r.status_code == 200
    assert r.headers["x-app-fal-api-fallback-endpoint"] == GPT


def test_http_plain_400_surfaced_intact():
    # An ordinary 400 (caller mistake) never falls back; body preserved even
    # though it was buffered for inspection.
    fwd = _recording_forward({GPT: {"images": [{"url": "fb"}]}})
    client = TestClient(_build_app(fwd))
    r = client.post("/err400_plain", json={"prompt": "x", "fal_fallback": _chain()})
    assert r.status_code == 400
    assert r.json() == {"detail": "bad request"}
    assert fwd.calls == []


def test_http_chain_capped_at_two_total_attempts():
    # 3 nodes supplied but the cap keeps only the FIRST (primary + 1 = 2 total
    # attempts): the others are never tried; primary failure is kept.
    calls = []

    async def failing_forward(endpoint, payload):
        calls.append(endpoint)
        raise RuntimeError("down")

    client = TestClient(_build_app(failing_forward), raise_server_exceptions=False)
    r = client.post(
        "/err",
        json={
            "prompt": "x",
            "fal_fallback": [
                {"endpoint": "a/a"},
                {"endpoint": "b/b"},
                {"endpoint": "c/c"},
            ],
        },
    )
    assert r.status_code == 503
    assert calls == ["a/a"]


# --- $ref: referencing the primary input from a fallback step ------------------


def test_resolve_refs_top_level_and_dot_path():
    primary = {"prompt": "a very long prompt", "image_urls": ["u1", "u2"]}
    step_input = {
        "prompt": {"$ref": "prompt"},
        "image_url": {"$ref": "image_urls.0"},
        "quality": "high",
    }
    assert resolve_input_refs(step_input, primary) == {
        "prompt": "a very long prompt",
        "image_url": "u1",
        "quality": "high",
    }


def test_resolve_refs_inside_lists():
    # A single primary value can be wrapped into a list for a list-shaped target.
    primary = {"image_url": "u1"}
    assert resolve_input_refs({"image_urls": [{"$ref": "image_url"}]}, primary) == {
        "image_urls": ["u1"]
    }


def test_unresolvable_ref_drops_the_field():
    # Fail-open: the target model must never receive a stray ref object.
    primary = {"prompt": "p"}
    assert resolve_input_refs(
        {"prompt": {"$ref": "does.not.exist"}, "seed": 7}, primary
    ) == {"seed": 7}


def test_ref_like_dict_with_extra_keys_is_literal():
    primary = {"prompt": "p"}
    literal = {"meta": {"$ref": "prompt", "other": 1}}
    assert resolve_input_refs(literal, primary) == literal


def test_chain_resolves_refs_before_forward():
    calls = []

    async def forward(endpoint, payload):
        calls.append((endpoint, payload))
        return {"images": [{"url": "ok"}]}

    chain = parse_fallback_chain(
        {
            "fal_fallback": [
                {"endpoint": GPT, "input": {"prompt": {"$ref": "prompt"}, "quality": "high"}}
            ]
        }
    )
    result, served = asyncio.run(
        run_fallback_chain(
            chain,
            trigger="server_error",
            forward=forward,
            primary_input={"prompt": "big prompt", "image_size": "1024x1024"},
        )
    )
    assert served == GPT
    assert calls == [(GPT, {"prompt": "big prompt", "quality": "high"})]


def test_http_ref_resolves_against_stripped_primary_body():
    # Over HTTP the refs resolve against the STRIPPED primary body (no
    # reserved fields), so `{"$ref": "prompt"}` picks up the caller's prompt.
    fwd = _recording_forward({GPT: {"images": [{"url": "fb"}]}})
    client = TestClient(_build_app(fwd))
    r = client.post(
        "/err",
        json={
            "prompt": "the real prompt",
            "fal_fallback": [
                {"endpoint": GPT, "input": {"prompt": {"$ref": "prompt"}}}
            ],
        },
    )
    assert r.status_code == 200
    assert fwd.calls == [(GPT, {"prompt": "the real prompt"})]


def test_chain_strips_nested_fallback_from_forwarded_input():
    # A nested `fal_fallback` inside node.input must NOT reach the target (its
    # middleware would walk it -> chains-of-chains amplification). Reserved
    # fields are stripped from the forwarded input.
    calls = []

    async def forward(endpoint, payload):
        calls.append((endpoint, payload))
        return {"images": [{"url": "ok"}]}

    chain = parse_fallback_chain(
        {
            "fal_fallback": [
                {
                    "endpoint": GPT,
                    "input": {
                        "prompt": "x",
                        "fal_fallback": [{"endpoint": SEEDREAM}],
                        "fal_fallback_timeout": 5,
                    },
                }
            ]
        }
    )
    result, served = asyncio.run(
        run_fallback_chain(chain, trigger="server_error", forward=forward)
    )
    assert served == GPT
    assert calls == [(GPT, {"prompt": "x"})]
