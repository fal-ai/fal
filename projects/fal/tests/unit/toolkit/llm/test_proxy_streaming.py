from __future__ import annotations

import asyncio
import json
import os
import socket
import threading
import time
from pathlib import Path

import httpx
import pytest
from llm_helpers import _run, _wait_for, _wait_inflight

from fal.toolkit.llm import LLMApp, _app, _stream, _upstream
from fal.toolkit.llm._app import _deadline_budget, _wait_limit
from fal.toolkit.llm._stream import _event_cut, _RelayResponse, _Stream


def _wait_idle(app, timeout: float = 5.0) -> bool:
    end = time.time() + timeout
    while time.time() < end:
        if not app._inflight:
            return True
        time.sleep(0.05)
    return False


def test_unary_usage_model_rewrite_and_hints(served):
    app, url, _, body_file = served
    r = httpx.post(
        f"{url}/v1/chat/completions",
        json={"model": "whatever", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert r.status_code == 200
    assert json.loads(body_file.read_text())["model"] == "served-name"
    assert app._usage_totals == {
        "prompt_tokens": 7,
        "completion_tokens": 5,
        "cached_tokens": 3,
        "requests": 1,
    }
    # No session id sent: nothing advertised.
    assert r.headers["x-fal-runner-hints"] == ""
    assert httpx.post(f"{url}/v1/completions", json={"prompt": "x"}).status_code == 200
    assert httpx.get(f"{url}/v1/models").json()["data"][0]["id"] == "fake"


def test_hints_follow_client_session_ids(served):
    app, url, _, _ = served
    msgs = [{"role": "user", "content": "hi"}]
    post = httpx.post
    r = post(
        f"{url}/v1/chat/completions",
        json={"messages": msgs},
        headers={"X-Fal-Runner-Hint": "sess-a"},
    )
    assert r.headers["x-fal-runner-hints"] == "sess-a"
    # prompt_cache_key and user work too; the header wins over both.
    post(f"{url}/v1/chat/completions", json={"messages": msgs, "user": "u-1"})
    r = post(
        f"{url}/v1/responses",
        json={"input": "x", "prompt_cache_key": "k-2", "user": "ignored"},
    )
    assert r.headers["x-fal-runner-hints"] == "sess-a,u-1,k-2"
    # A repeat moves the id to the most recent slot instead of duplicating it.
    r = post(
        f"{url}/v1/chat/completions",
        json={"messages": msgs},
        headers={"X-Fal-Runner-Hint": "sess-a"},
    )
    assert r.headers["x-fal-runner-hints"] == "u-1,k-2,sess-a"
    # Ids the header cannot carry (or that could never match) are not advertised.
    before = r.headers["x-fal-runner-hints"]
    for bad in ("a,b c", "a\r\nX: y", "z" * 65, "naïve"):
        r = post(f"{url}/v1/chat/completions", json={"user": bad})
        assert r.status_code == 200
        assert r.headers["x-fal-runner-hints"] == before
    # A non-string first choice does not hide a valid later one.
    r = post(f"{url}/v1/chat/completions", json={"prompt_cache_key": 5, "user": "u-ok"})
    assert r.headers["x-fal-runner-hints"].split(",")[-1] == "u-ok"


@pytest.mark.parametrize("served", [{"prefix_hint_max": 3}], indirect=True)
def test_hints_are_bounded(served):
    _, url, _, _ = served
    for i in range(5):
        r = httpx.post(f"{url}/v1/chat/completions", json={"user": f"s{i}"})
    assert r.headers["x-fal-runner-hints"] == "s2,s3,s4"


@pytest.mark.parametrize("served", [{"prefix_hints": False}], indirect=True)
def test_hints_off_by_default(served):
    _, url, _, _ = served
    r = httpx.post(f"{url}/v1/chat/completions", json={"user": "s"})
    assert "x-fal-runner-hints" not in r.headers


def test_responses_unary_passthrough_and_usage(served):
    app, url, _, body_file = served
    r = httpx.post(f"{url}/v1/responses", json={"model": "x", "input": "hi"})
    assert r.status_code == 200
    assert json.loads(body_file.read_text())["model"] == "served-name"
    assert app._usage_totals["prompt_tokens"] == 7
    assert app._usage_totals["completion_tokens"] == 5
    assert app._usage_totals["cached_tokens"] == 3


@pytest.mark.parametrize("pad", [0, 20000])
def test_responses_stream_passthrough_usage(served, pad):
    app, url, _, body_file = served
    with httpx.stream(
        "POST", f"{url}/v1/responses", json={"stream": True, "pad_chars": pad}
    ) as r:
        assert r.headers["content-type"].startswith("text/event-stream")
        text = "".join(r.iter_text())
    events = [ln[7:] for ln in text.splitlines() if ln.startswith("event: ")]
    assert events == ["response.output_text.delta"] * 3 + ["response.completed"]
    assert "[DONE]" not in text
    # No chat-only stream_options is injected for the Responses API.
    assert "stream_options" not in json.loads(body_file.read_text())
    assert app._usage_totals["completion_tokens"] == 5
    assert app._usage_totals["cached_tokens"] == 3


def test_responses_stream_deadline_ends_with_incomplete_event(served):
    _, url, abort_file, _ = served
    with httpx.stream(
        "POST",
        f"{url}/v1/responses",
        json={"stream": True},
        headers={"X-Fal-Request-Deadline": "0.3"},
    ) as r:
        text = "".join(r.iter_text())
    assert "response.incomplete" in text
    assert '"reason": "deadline"' in text
    assert _wait_for(abort_file)


def test_stream_usage_parsing():
    chat = b'data: {"choices": [], "usage": {"prompt_tokens": 2}}\n\ndata: [DONE]\n\n'
    assert _stream._parse_stream_usage(chat) == {"prompt_tokens": 2}
    cut = (
        b'..."output_text": "xx"}, "usage": {"input_tokens": 9, '
        b'"input_tokens_details": {"cached_tokens": 4}, "output_tokens": 1}}}\n\n'
        b"event: response.completed\n"
    )
    # The start of the data line fell out of the tail.
    usage = _stream._parse_stream_usage(cut)
    assert usage["input_tokens"] == 9
    assert _stream._cached_tokens(usage) == 4
    assert _stream._cached_tokens({"prompt_tokens_details": None}) == 0


def test_upstream_error_and_bad_body(served):
    _, url, _, _ = served
    r = httpx.post(f"{url}/v1/chat/completions", json={"fail": True})
    assert r.status_code == 400
    assert httpx.post(f"{url}/v1/chat/completions", content=b"nope").status_code == 400


def test_stalled_stream_error_body_is_bounded(served, monkeypatch):
    app, url, _, _ = served
    closed = []

    class Stalled:
        status_code = 500

        async def aread(self):
            await asyncio.sleep(3600)

        async def aclose(self):
            closed.append(1)

    async def open_upstream(path, payload):
        return Stalled()

    monkeypatch.setattr(_app, "_ERROR_BODY_TIMEOUT_S", 0.2)
    monkeypatch.setattr(app, "_open_upstream", open_upstream)
    r = httpx.post(f"{url}/v1/chat/completions", json={"stream": True}, timeout=10)
    assert r.status_code == 500
    assert "timed out" in r.text
    assert closed == [1]


@pytest.mark.parametrize(
    "error",
    [
        ConnectionResetError("reset"),
        httpx.ReadError("reset"),
        httpx.RemoteProtocolError("cut"),
    ],
)
def test_stream_error_body_connection_failure_is_a_counted_503(
    served, monkeypatch, error
):
    app, url, _, _ = served
    closed = []

    class Reset:
        status_code = 500

        async def aread(self):
            raise error

        async def aclose(self):
            closed.append(1)

    async def open_upstream(path, payload):
        return Reset()

    monkeypatch.setattr(app, "_open_upstream", open_upstream)
    r = httpx.post(f"{url}/v1/chat/completions", json={"stream": True}, timeout=10)
    assert r.status_code == 503
    assert r.json()["error"]["type"] == "engine_unavailable"
    assert r.headers["x-fal-runner-health-check"] == "true"
    assert app._replicas[0]._engine_errors == 1
    assert closed == [1]


def _eof_upstream(chunks):
    class Upstream:
        status_code = 200

        async def aiter_raw(self):
            for chunk in chunks:
                yield chunk

        async def aread(self):
            return b""

        async def aclose(self):
            pass

    return Upstream()


@pytest.mark.parametrize(
    "path,body,chunks,complete",
    [
        # Cut inside an event, before any finish.
        ("/v1/chat/completions", {}, [b'data: {"choices":'], False),
        # Whole events, but no finish and no [DONE].
        (
            "/v1/chat/completions",
            {},
            [b'data: {"choices":[{"index":0,"delta":{"content":"a"}}]}\n\n'],
            False,
        ),
        # Finished, only the terminator missing.
        (
            "/v1/chat/completions",
            {},
            [b'data: {"choices":[{"index":0,"finish_reason":"stop"}]}\n\n'],
            True,
        ),
        # Finished, then the include_usage chunk, then EOF without [DONE].
        (
            "/v1/chat/completions",
            {},
            [
                b'data: {"choices":[{"index":0,"finish_reason":"stop"}]}\n\n',
                b'data: {"choices":[],"usage":{"completion_tokens":1}}\n\n',
            ],
            True,
        ),
        # Terminator without its blank line.
        (
            "/v1/chat/completions",
            {"n": 2},
            [
                b'data: {"choices":[{"index":0,"finish_reason":"stop"}]}\n\n',
                b"data: [DONE]",
            ],
            True,
        ),
        # With n > 1 another choice may still be open.
        (
            "/v1/chat/completions",
            {"n": 2},
            [b'data: {"choices":[{"index":0,"finish_reason":"stop"}]}\n\n'],
            False,
        ),
        (
            "/v1/completions",
            {},
            [b'data: {"choices":[{"index":0,"text":"a"}]}\n\n'],
            False,
        ),
        # Responses stream without its terminal event.
        (
            "/v1/responses",
            {},
            [b'event: response.created\ndata: {"type":"response.created"}\n\n'],
            False,
        ),
        # Responses stream that reached its terminal event.
        (
            "/v1/responses",
            {},
            [
                b"event: response.completed\n"
                b'data: {"type":"response.completed","response":{}}\n\n'
            ],
            True,
        ),
    ],
)
def test_stream_clean_eof_needs_terminal_event(
    served, monkeypatch, path, body, chunks, complete
):
    app, url, _, _ = served

    async def open_upstream(p, payload):
        return _eof_upstream(chunks)

    monkeypatch.setattr(app, "_open_upstream", open_upstream)
    replica = app._replicas[0]
    replica._engine_errors, replica._engine_error_at = 3, time.monotonic()
    r = httpx.post(f"{url}{path}", json={"stream": True, **body}, timeout=10)
    assert r.status_code == 200
    events = [e for e in r.text.split("\n\n") if e]
    # A dropped partial event is not glued onto the next one.
    assert all(e.count("data:") == 1 for e in events)
    if path == "/v1/responses":
        assert ("response.incomplete" in events[-1]) is not complete
    else:
        assert events[-1] == "data: [DONE]"
        error_finish = '"finish_reason": "error"' in events[-2]
        assert error_finish is not complete
    # A clean reply resets the consecutive count; a cut one adds to it.
    assert app._replicas[0]._engine_errors == (0 if complete else 4)


_TERMINAL_EVENT = (
    b'event: response.completed\ndata: {"response":{"output":[]},'
    b'"type":"response.completed"}\n\n'
)


@pytest.mark.parametrize(
    "split_after",
    [b'"ty', b'"type":', b'"type":"response.comp', b'"type":"response.complete'],
)
def test_stream_terminal_event_split_across_reads(served, monkeypatch, split_after):
    app, url, _, _ = served
    # Forces the oversized-event path: each read is forwarded as it comes.
    monkeypatch.setattr(_app, "_MAX_LINE_BYTES", 16)
    at = _TERMINAL_EVENT.index(split_after) + len(split_after)
    chunks = [_TERMINAL_EVENT[:at], _TERMINAL_EVENT[at:]]

    async def open_upstream(p, payload):
        return _eof_upstream(chunks)

    monkeypatch.setattr(app, "_open_upstream", open_upstream)
    replica = app._replicas[0]
    replica._engine_errors, replica._engine_error_at = 3, time.monotonic()
    r = httpx.post(f"{url}/v1/responses", json={"stream": True}, timeout=10)
    assert r.status_code == 200
    assert "response.incomplete" not in r.text
    assert app._replicas[0]._engine_errors == 0


def test_stream_passthrough_usage(served):
    app, url, _, _ = served
    chunks = []
    with httpx.stream(
        "POST", f"{url}/v1/chat/completions", json={"stream": True, "n_chunks": 3}
    ) as r:
        assert r.headers["content-type"].startswith("text/event-stream")
        chunks = [c for c in r.iter_lines() if c.startswith("data:")]
    assert chunks[-1] == "data: [DONE]"
    assert any('"usage"' in c for c in chunks)
    assert app._usage_totals["completion_tokens"] == 5


def test_record_usage_error_does_not_fail_unary_or_stream(served, capsys):
    app, url, _, _ = served
    calls = []

    def boom(path, usage):
        calls.append(path)
        raise RuntimeError("billing down")

    app.record_usage = boom
    r = httpx.post(
        f"{url}/v1/chat/completions",
        json={"model": "m", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert r.status_code == 200
    assert r.json()["usage"]["completion_tokens"] == 5
    with httpx.stream(
        "POST", f"{url}/v1/chat/completions", json={"stream": True, "n_chunks": 3}
    ) as sr:
        assert sr.status_code == 200
        lines = [c for c in sr.iter_lines() if c.startswith("data:")]
    assert lines[-1] == "data: [DONE]"
    assert len(calls) == 2
    assert capsys.readouterr().out.count("record_usage failed") == 2


def test_stream_disconnect_aborts_engine(served):
    _, url, abort_file, _ = served
    with httpx.stream(
        "POST", f"{url}/v1/chat/completions", json={"stream": True, "n_chunks": 100}
    ) as r:
        next(r.iter_lines())
    assert _wait_for(abort_file)


def test_unary_disconnect_aborts_engine(served):
    _, url, abort_file, _ = served
    with pytest.raises(httpx.ReadTimeout):
        httpx.post(f"{url}/v1/chat/completions", json={"delay_ticks": 300}, timeout=1.0)
    assert _wait_for(abort_file, timeout=8.0)


def _disconnect_once_engine_has_it(url: str, body: dict, body_file: Path) -> None:
    # Raw socket so the client leaves only after the engine got the request:
    # a short client timeout could leave before that on a slow runner.
    host, port = url.removeprefix("http://").split(":")
    payload = json.dumps(body).encode()
    with socket.create_connection((host, int(port))) as s:
        s.sendall(
            b"POST /v1/chat/completions HTTP/1.1\r\nHost: x\r\n"
            b"Content-Type: application/json\r\n"
            b"Content-Length: " + str(len(payload)).encode() + b"\r\n\r\n" + payload
        )
        assert _wait_for(body_file)


@pytest.mark.parametrize(
    "served", [{}, {"raw_stream_client": False}], indirect=True, ids=["raw", "httpx"]
)
def test_disconnect_before_stream_headers_aborts_engine(served):
    # The client leaves while the engine still holds headers back.
    app, url, abort_file, body_file = served
    _disconnect_once_engine_has_it(
        url, {"stream": True, "header_delay_ticks": 300}, body_file
    )
    assert _wait_for(abort_file)
    assert _wait_idle(app)


def test_disconnect_during_error_body_aborts_engine(served):
    app, url, abort_file, body_file = served
    _disconnect_once_engine_has_it(
        url, {"stream": True, "fail": True, "error_body_ticks": 300}, body_file
    )
    assert _wait_for(abort_file)
    assert _wait_idle(app)


def test_llm_app_opts_into_gateway_cancel():
    assert LLMApp.cancel_on_gateway_cancel is True


def _gateway_cancel(url: str, path: str, request_id: str) -> dict:
    r = httpx.post(f"{url}{path}/cancel", headers={"x-fal-request-id": request_id})
    assert r.status_code == 200
    assert r.headers["x-fal-billable-units"] == "0"
    return r.json()


@pytest.mark.parametrize("path", ["/v1/chat/completions", "/v1/responses"])
def test_gateway_cancel_aborts_unary_engine_request(served, path):
    app, url, abort_file, _ = served
    result: dict = {}

    def call():
        result["r"] = httpx.post(
            f"{url}{path}",
            json={"delay_ticks": 300},
            headers={"x-fal-request-id": "req-unary"},
            timeout=20.0,
        )

    t = threading.Thread(target=call)
    t.start()
    assert _wait_inflight(app)
    assert _gateway_cancel(url, path, "req-unary") == {"cancelled": True}
    t.join(timeout=10)
    assert result["r"].status_code == 499
    assert result["r"].headers["x-fal-billable-units"] == "0"
    assert _wait_for(abort_file, timeout=8.0)
    assert app._usage_totals["cancel_aborts"] == 1
    assert _wait_idle(app)


@pytest.mark.parametrize(
    "served", [{}, {"raw_stream_client": False}], indirect=True, ids=["raw", "httpx"]
)
@pytest.mark.parametrize("path", ["/v1/chat/completions", "/v1/responses"])
def test_gateway_cancel_aborts_stream_engine_request(served, path):
    app, url, abort_file, _ = served
    with httpx.stream(
        "POST",
        f"{url}{path}",
        json={"stream": True, "n_chunks": 300},
        headers={"x-fal-request-id": "req-stream"},
        timeout=20.0,
    ) as r:
        lines = r.iter_lines()
        next(lines)
        assert _gateway_cancel(url, path, "req-stream") == {"cancelled": True}
        # Mid-stream the connection is cut rather than ended cleanly.
        with pytest.raises(httpx.HTTPError):
            for _ in lines:
                pass
    assert _wait_for(abort_file)
    assert app._usage_totals["cancel_aborts"] == 1
    assert _wait_idle(app)


@pytest.mark.parametrize(
    "served", [{}, {"raw_stream_client": False}], indirect=True, ids=["raw", "httpx"]
)
def test_gateway_cancel_before_stream_headers_aborts_engine_request(served):
    app, url, abort_file, _ = served
    result: dict = {}

    def call():
        result["r"] = httpx.post(
            f"{url}/v1/chat/completions",
            json={"stream": True, "header_delay_ticks": 300},
            headers={"x-fal-request-id": "req-headers"},
            timeout=20.0,
        )

    t = threading.Thread(target=call)
    t.start()
    assert _wait_inflight(app)
    assert _gateway_cancel(url, "/v1/chat/completions", "req-headers") == {
        "cancelled": True
    }
    t.join(timeout=10)
    assert result["r"].status_code == 499
    assert _wait_for(abort_file, timeout=8.0)
    assert app._usage_totals["cancel_aborts"] == 1
    assert _wait_idle(app)


def test_gateway_cancel_of_unknown_or_finished_request_is_a_no_op(served):
    app, url, abort_file, _ = served
    r = httpx.post(
        f"{url}/v1/chat/completions",
        json={},
        headers={"x-fal-request-id": "req-done"},
    )
    assert r.status_code == 200
    assert _gateway_cancel(url, "/v1/chat/completions", "req-done") == {
        "cancelled": False
    }
    assert not abort_file.exists()
    assert "cancel_aborts" not in app._usage_totals


def _sse(r: httpx.Response) -> list[str]:
    return [c for c in r.iter_lines() if c.startswith("data:")]


def test_stream_deadline_ends_with_deadline_chunk_and_spares_siblings(served):
    app, url, abort_file, _ = served
    body = {"stream": True, "n_chunks": 100}
    sibling: dict = {}

    def run_sibling():
        with httpx.stream(
            "POST",
            f"{url}/v1/chat/completions",
            json={"stream": True, "n_chunks": 6},
            timeout=30,
        ) as r:
            sibling["lines"] = _sse(r)

    t = threading.Thread(target=run_sibling)
    t.start()
    with httpx.stream(
        "POST",
        f"{url}/v1/chat/completions",
        json=body,
        headers={"X-Fal-Request-Deadline": "0.5"},
        timeout=30,
    ) as r:
        lines = _sse(r)
    t.join(timeout=30)

    assert lines[-1] == "data: [DONE]"
    last = json.loads(lines[-2][5:])
    assert last["object"] == "chat.completion.chunk"
    assert last["choices"] == [{"index": 0, "delta": {}, "finish_reason": "deadline"}]
    assert _wait_for(abort_file)
    assert sibling["lines"][-1] == "data: [DONE]"
    assert any('"usage"' in c for c in sibling["lines"])
    assert not any("deadline" in c for c in sibling["lines"])
    assert app._usage_totals["deadline_aborts"] == 1
    assert app._usage_totals["requests"] == 1


def test_completions_stream_deadline_chunk_shape(served):
    _, url, _, _ = served
    with httpx.stream(
        "POST",
        f"{url}/v1/completions",
        json={"stream": True, "n_chunks": 100},
        headers={"X-Fal-Request-Deadline": "0.5"},
        timeout=30,
    ) as r:
        lines = _sse(r)
    assert lines[-1] == "data: [DONE]"
    last = json.loads(lines[-2][5:])
    assert last["object"] == "text_completion"
    assert last["choices"] == [{"index": 0, "text": "", "finish_reason": "deadline"}]


def test_stream_deadline_mid_event_keeps_events_whole(served):
    _, url, _, _ = served
    # Half events arrive every 0.2s (first halves at 0.0, 0.4, ...), so 0.5s
    # expires 0.1s after a first half and 0.1s before its second half.
    with httpx.stream(
        "POST",
        f"{url}/v1/chat/completions",
        json={"stream": True, "n_chunks": 100, "split_events": True},
        headers={"X-Fal-Request-Deadline": "0.5"},
        timeout=30,
    ) as r:
        body = r.read().decode()
    events = [e for e in body.split("\n\n") if e]
    assert events[-1] == "data: [DONE]"
    for event in events[:-1]:
        assert event.startswith("data: ")
        json.loads(event[6:])
    assert json.loads(events[-2][6:])["choices"][0]["finish_reason"] == "deadline"


def test_stream_deadline_before_headers(served):
    app, url, abort_file, _ = served
    with httpx.stream(
        "POST",
        f"{url}/v1/chat/completions",
        json={"stream": True, "header_delay_ticks": 300},
        headers={"X-Fal-Request-Deadline": "0.5"},
        timeout=30,
    ) as r:
        assert r.status_code == 200
        lines = _sse(r)
    assert lines[-1] == "data: [DONE]"
    assert json.loads(lines[-2][5:])["choices"][0]["finish_reason"] == "deadline"
    assert app._usage_totals["deadline_aborts"] == 1
    assert _wait_for(abort_file, timeout=8.0)


def test_stream_error_body_read_honors_deadline(served):
    app, url, abort_file, _ = served
    start = time.monotonic()
    r = httpx.post(
        f"{url}/v1/chat/completions",
        json={
            "stream": True,
            "fail": True,
            "fail_status": 500,
            "error_body_ticks": 300,
        },
        headers={"X-Fal-Request-Deadline": "0.5"},
        timeout=30,
    )
    assert time.monotonic() - start < 5
    assert r.status_code == 200
    lines = [c for c in r.text.splitlines() if c.startswith("data:")]
    assert json.loads(lines[-2][5:])["choices"][0]["finish_reason"] == "deadline"
    assert app._usage_totals["deadline_aborts"] == 1
    # The 500 still counts toward engine_error_threshold.
    assert app._replicas[0]._engine_errors == 1
    assert _wait_for(abort_file, timeout=8.0)
    assert _wait_idle(app)


def test_event_cut():
    assert _event_cut(b"") == 0
    assert _event_cut(b"data: a") == 0
    assert _event_cut(b"data: a\n\ndata: b") == 9
    assert _event_cut(b"data: a\r\n\r\ndata: b\r\n\r\n") == 22


def test_unary_deadline_returns_504_and_aborts_engine(served):
    app, url, abort_file, _ = served
    r = httpx.post(
        f"{url}/v1/chat/completions",
        json={"delay_ticks": 300},
        headers={"X-Fal-Request-Deadline": "0.5"},
        timeout=30,
    )
    assert r.status_code == 504
    assert r.headers["x-fal-needs-retry"] == "false"
    # The engine is fine: the gateway should probe health, not count a 5xx.
    assert r.headers["x-fal-runner-health-check"] == "true"
    assert r.json() == {
        "error": {
            "message": "request deadline of 0.5s exceeded",
            "type": "deadline_exceeded",
        }
    }
    assert _wait_for(abort_file, timeout=8.0)
    assert app._usage_totals["deadline_aborts"] == 1


def test_gateway_header_alone_triggers_deadline(served):
    app, url, abort_file, _ = served
    # remaining 1.5s minus the 1s margin leaves a 0.5s budget.
    with httpx.stream(
        "POST",
        f"{url}/v1/chat/completions",
        json={"stream": True, "n_chunks": 100},
        headers={"X-Fal-Request-Timeout-Remaining": "1.500"},
        timeout=30,
    ) as r:
        lines = _sse(r)
    assert json.loads(lines[-2][5:])["choices"][0]["finish_reason"] == "deadline"
    assert _wait_for(abort_file)


@pytest.mark.parametrize("served", [{"request_deadline_s": 0.5}], indirect=True)
def test_class_deadline_applies_without_header(served):
    _, url, _, _ = served
    r = httpx.post(f"{url}/v1/chat/completions", json={"delay_ticks": 300}, timeout=30)
    assert r.status_code == 504


@pytest.mark.parametrize("served", [{"request_deadline_s": 0.5}], indirect=True)
def test_longer_deadline_header_is_clamped_to_class_default(served):
    _, url, _, _ = served
    r = httpx.post(
        f"{url}/v1/chat/completions",
        json={"delay_ticks": 300},
        headers={"X-Fal-Request-Deadline": "60"},
        timeout=30,
    )
    assert r.status_code == 504


def test_invalid_deadline_header_is_400(served):
    _, url, _, _ = served
    for bad in ("abc", "0", "-1", "nan"):
        r = httpx.post(
            f"{url}/v1/chat/completions",
            json={},
            headers={"X-Fal-Request-Deadline": bad},
        )
        assert r.status_code == 400
        assert "error" in r.json()


def test_deadline_budget_precedence():
    h = {}
    assert _deadline_budget(h, None, 1.0) is None
    assert _deadline_budget(h, 30.0, 1.0) == 30.0
    # Override shortens the class default but is clamped to it, not rejected.
    assert _deadline_budget({"x-fal-request-deadline": "10"}, 30.0, 1.0) == 10.0
    assert _deadline_budget({"x-fal-request-deadline": "60"}, 30.0, 1.0) == 30.0
    # Without a class default the override stands alone.
    assert _deadline_budget({"x-fal-request-deadline": "60"}, None, 1.0) == 60.0
    gw = {"x-fal-request-timeout-remaining": "297.350"}
    assert _deadline_budget(gw, None, 1.0) == pytest.approx(296.35)
    # The smaller of override/default and gateway wins.
    assert _deadline_budget(gw, 10.0, 1.0) == 10.0
    assert _deadline_budget({**gw, "x-fal-request-deadline": "500"}, None, 1.0) == (
        pytest.approx(296.35)
    )
    # Unparseable gateway values are ignored.
    assert (
        _deadline_budget({"x-fal-request-timeout-remaining": "soon"}, 5.0, 1.0) == 5.0
    )
    with pytest.raises(ValueError):
        _deadline_budget({"x-fal-request-deadline": "0"}, None, 1.0)


class _FakeUpstream:
    def __init__(self, chunks, hang=False):
        self._chunks = chunks
        self._hang = hang
        self.closed = 0

    async def aiter_raw(self):
        for c in self._chunks:
            yield c
        if self._hang:
            await asyncio.Event().wait()

    async def aclose(self):
        self.closed += 1


def test_no_shutdown_chunk_after_done_forwarded(served):
    app, _, _, _ = served

    async def go():
        up = _FakeUpstream([b"data: {}\n\n", b"data: [DONE]\n\n"], hang=True)
        abort = asyncio.Event()
        stream = _Stream(app._settle_stream, up, "/v1/chat/completions", abort)
        out = []
        async for c in app._relay(stream, "/v1/chat/completions"):
            out.append(c)
            if c.startswith(b"data: [DONE]"):
                abort.set()
        return b"".join(out)

    body = _run(app, go())
    assert b"shutdown" not in body
    assert "shutdown_aborts" not in app._usage_totals


def test_abort_not_starved_by_buffered_chunks(served):
    app, _, _, _ = served

    async def go():
        # Always-ready chunks: a read wins every race unless abort is checked
        # first.
        up = _FakeUpstream([b"data: {}\n\n"] * 1000)
        abort = asyncio.Event()
        stream = _Stream(app._settle_stream, up, "/v1/chat/completions", abort)
        out = []
        async for c in app._relay(stream, "/v1/chat/completions"):
            out.append(c)
            if len(out) == 3:
                abort.set()
        return out

    out = _run(app, go())
    # A read already finished when the abort lands is forwarded first.
    assert len(out) in (4, 5) and b"shutdown" in out[-1]
    assert app._usage_totals["shutdown_aborts"] == 1


def test_read_finishing_with_abort_is_forwarded(served):
    app, _, _, _ = served
    finish = b'data: {"choices": [{"finish_reason": "stop"}]}\n\n'

    async def go():
        abort = asyncio.Event()

        class Up(_FakeUpstream):
            async def aiter_raw(self):
                # The abort lands while this read completes.
                abort.set()
                yield finish
                await asyncio.Event().wait()

        stream = _Stream(app._settle_stream, Up([]), "/v1/chat/completions", abort)
        return [c async for c in app._relay(stream, "/v1/chat/completions")]

    out = _run(app, go())
    # The real finish event survives and only the terminator is added.
    assert out == [finish, b"data: [DONE]\n\n"]


def test_expired_deadline_wins_over_abort(served):
    app, _, _, _ = served

    async def go():
        up = _FakeUpstream([b"data: {}\n\n"], hang=True)
        abort = asyncio.Event()
        stream = _Stream(app._settle_stream, up, "/v1/chat/completions", abort)
        deadline = time.monotonic() + 0.2
        out = []
        async for c in app._relay(stream, "/v1/chat/completions", True, deadline):
            out.append(c)
            if len(out) == 1:
                await asyncio.sleep(0.3)
                abort.set()
                # Let the relay's abort waiter finish, so both are set.
                await asyncio.sleep(0)
        return out

    out = _run(app, go())
    assert (
        json.loads(out[-1].split(b"\n\n")[0][6:])["choices"][0]["finish_reason"]
        == "deadline"
    )
    assert app._usage_totals["deadline_aborts"] == 1
    assert "shutdown_aborts" not in app._usage_totals


def test_stream_idle_timeout_ends_stream_and_closes_upstream(served, capsys):
    app, _, _, _ = served
    app.stream_idle_timeout_s = 0.2

    async def go():
        up = _FakeUpstream([b"data: {}\n\n"], hang=True)
        stream = _Stream(
            app._settle_stream, up, "/v1/chat/completions", asyncio.Event()
        )
        # A far deadline: the idle limit is what ends this stream.
        deadline = time.monotonic() + 30
        out = [
            c async for c in app._relay(stream, "/v1/chat/completions", True, deadline)
        ]
        return out, up.closed

    out, closed = _run(app, go())
    assert out[0] == b"data: {}\n\n"
    final = json.loads(out[-1].split(b"\n\n")[0][6:])
    assert final["choices"][0]["finish_reason"] == "idle_timeout"
    assert out[-1].endswith(b"data: [DONE]\n\n")
    assert closed == 1  # the connection close aborts the engine request
    assert app._usage_totals["idle_aborts"] == 1
    assert "deadline_aborts" not in app._usage_totals
    assert "nothing from" in capsys.readouterr().out


def test_stream_idle_timeout_after_done_adds_nothing(served):
    app, _, _, _ = served
    app.stream_idle_timeout_s = 0.2

    async def go():
        up = _FakeUpstream([b"data: {}\n\n", b"data: [DONE]\n\n"], hang=True)
        stream = _Stream(
            app._settle_stream, up, "/v1/chat/completions", asyncio.Event()
        )
        return [c async for c in app._relay(stream, "/v1/chat/completions")]

    assert _run(app, go()) == [b"data: {}\n\n", b"data: [DONE]\n\n"]
    assert "idle_aborts" not in app._usage_totals


@pytest.mark.parametrize("served", [{"stream_idle_timeout_s": 0.5}], indirect=True)
def test_stream_idle_timeout_before_headers(served):
    app, url, abort_file, _ = served
    with httpx.stream(
        "POST",
        f"{url}/v1/chat/completions",
        json={"stream": True, "header_delay_ticks": 300},
        timeout=30,
    ) as r:
        assert r.status_code == 200
        lines = _sse(r)
    assert lines[-1] == "data: [DONE]"
    assert json.loads(lines[-2][5:])["choices"][0]["finish_reason"] == "idle_timeout"
    assert app._usage_totals["idle_aborts"] == 1
    assert _wait_for(abort_file, timeout=8.0)


def test_wait_limit_picks_the_sooner_bound():
    assert _wait_limit(None, None) == (None, False)
    assert _wait_limit(None, 300.0) == (300.0, True)
    soon = time.monotonic() + 1.0
    limit, idle = _wait_limit(soon, 300.0)
    assert 0.0 < limit <= 1.0 and not idle
    later = time.monotonic() + 600.0
    assert _wait_limit(later, 300.0) == (300.0, True)
    limit, idle = _wait_limit(soon, None)
    assert limit <= 1.0 and not idle


def test_unary_expired_deadline_wins_over_abort(served):
    app, _, abort_file, _ = served

    class Req:
        async def receive(self):
            await asyncio.Event().wait()

    async def go():
        abort = asyncio.Event()
        abort.set()
        return await app._proxy_unary(
            Req(),
            "/v1/chat/completions",
            {"delay_ticks": 300},
            abort,
            time.monotonic() - 1,
            1.0,
        )

    r = _run(app, go())
    assert r.status_code == 504
    assert r.headers["x-fal-needs-retry"] == "false"
    assert app._usage_totals["deadline_aborts"] == 1
    assert "shutdown_aborts" not in app._usage_totals


def test_unframed_body_flushed_past_line_limit(served, monkeypatch):
    app, _, _, _ = served
    monkeypatch.setattr(_app, "_MAX_LINE_BYTES", 10)

    async def go():
        up = _FakeUpstream([b"x" * 20], hang=True)
        stream = _Stream(
            app._settle_stream, up, "/v1/chat/completions", asyncio.Event()
        )
        relay = app._relay(stream, "/v1/chat/completions", True, time.monotonic() + 0.3)
        return [c async for c in relay]

    out = _run(app, go())
    assert out[0] == b"x" * 20
    assert b"deadline" in out[-1]


@pytest.mark.parametrize("single_choice", [True, False])
def test_abort_after_finish_reason_only_sends_done(served, single_choice):
    app, _, _, _ = served

    async def go():
        up = _FakeUpstream(
            [b'data: {"choices": [{"finish_reason": "stop"}]}\n\n'], hang=True
        )
        abort = asyncio.Event()
        stream = _Stream(app._settle_stream, up, "/v1/chat/completions", abort)
        out = []
        relay = app._relay(stream, "/v1/chat/completions", single_choice)
        async for c in relay:
            out.append(c)
            abort.set()
        return out

    out = _run(app, go())
    assert out[-1].endswith(b"data: [DONE]\n\n")
    # With n > 1 other choices may still be open, so they get the reason.
    assert any(b"shutdown" in c for c in out) is not single_choice


def test_upstream_close_retried_after_cancelled_disconnect():
    # Starlette cancels the body task on disconnect, so the close runs in a
    # cancelled scope. Like httpx, the fake marks itself closed before its
    # await, so an interrupted close is never retried: only a close that is
    # not interrupted reaches the connection.
    class Up:
        is_closed = False
        closed = False

        async def aiter_raw(self):
            while True:
                await asyncio.sleep(0.05)
                yield b'data: {"a": 1}\n\n'

        async def aclose(self):
            if self.is_closed:
                return
            self.is_closed = True
            await asyncio.sleep(0.01)
            self.closed = True

    class Probe(LLMApp):
        model = "m"

    app = Probe()
    app._inflight = set()
    app._usage_totals = {}

    async def go():
        up = Up()
        abort = asyncio.Event()
        app._inflight.add(abort)
        stream = _Stream(app._settle_stream, up, "/v1/chat/completions", abort)
        resp = _RelayResponse(
            app._relay(stream, "/v1/chat/completions"),
            stream.close,
            media_type="text/event-stream",
        )

        async def receive():
            await asyncio.sleep(0.3)
            return {"type": "http.disconnect"}

        async def send(message):
            pass

        scope = {"type": "http", "asgi": {"spec_version": "2.3"}}
        await resp(scope, receive, send)
        return up.closed, abort in app._inflight

    closed, still = asyncio.run(go())
    assert closed and not still


def test_response_cleanup_runs_once_when_generator_never_starts():
    calls = []
    started = []

    async def body():
        started.append(1)
        yield b"x"

    async def cleanup():
        calls.append(1)

    async def send(message):
        raise OSError("client gone")

    async def receive():
        return {"type": "http.disconnect"}

    async def go():
        resp = _RelayResponse(body(), cleanup)
        with pytest.raises(Exception):
            await resp({"type": "http", "asgi": {"spec_version": "2.4"}}, receive, send)

    asyncio.run(go())
    assert calls == [1]
    assert not started


def test_response_cleanup_runs_once_after_full_iteration():
    calls = []
    sent = []

    async def body():
        yield b"x"

    async def cleanup():
        calls.append(1)

    async def send(message):
        sent.append(message)

    async def receive():
        await asyncio.Event().wait()

    async def go():
        resp = _RelayResponse(body(), cleanup)
        await resp({"type": "http", "asgi": {"spec_version": "2.4"}}, receive, send)

    asyncio.run(go())
    assert calls == [1]
    assert sent[1]["body"] == b"x"


def test_stream_close_is_idempotent_and_records_usage_once(served):
    app, _, _, _ = served

    async def go():
        up = _FakeUpstream([])
        abort = asyncio.Event()
        app._inflight.add(abort)
        stream = _Stream(app._settle_stream, up, "/v1/chat/completions", abort)
        stream.tail = (
            b'data: {"usage": {"prompt_tokens": 1, "completion_tokens": 2}}\n\n'
        )
        await stream.close()
        await stream.close()
        return up.closed, abort in app._inflight

    closed, still = _run(app, go())
    assert closed == 1 and not still
    assert app._usage_totals["requests"] == 1


def test_stream_tail_keeps_recent_bytes_and_usage_survives_many_chunks(served):
    app, _, _, _ = served
    stream = _Stream(app._settle_stream, None, "/v1/chat/completions", asyncio.Event())
    for _ in range(5000):
        stream.feed(b'data: {"choices": [{"delta": {"content": "x"}}]}\n\n')
    usage = b'data: {"choices": [], "usage": {"completion_tokens": 3}}\n\n'
    stream.feed(usage)
    stream.feed(b"data: [DONE]\n\n")
    tail = stream.tail
    assert len(tail) == _stream._STREAM_TAIL_BYTES
    # Eviction keeps memory bounded: about one tail plus one chunk.
    assert stream._recent_len < _stream._STREAM_TAIL_BYTES + 200
    assert len(stream._recent) < 200
    assert tail.endswith(b"data: [DONE]\n\n")
    assert _stream._parse_stream_usage(tail) == {"completion_tokens": 3}


def test_engine_pool_is_sized_for_many_streams(served):
    app, _, _, _ = served
    assert type(app).engine_max_connections >= 1024


@pytest.mark.skipif(os.name == "nt", reason="resource is POSIX-only")
def test_pool_fits_the_fd_limit(monkeypatch):
    import resource  # noqa: PLC0415

    calls = []
    monkeypatch.setattr(resource, "getrlimit", lambda _: (1024, 1 << 20))
    monkeypatch.setattr(resource, "setrlimit", lambda *a: calls.append(a))
    assert _app._fit_pool_to_fd_limit(4096) == 4096
    assert calls and calls[0][1][0] == 4 * 4096 + _app._FD_RESERVE
    monkeypatch.setattr(resource, "getrlimit", lambda _: (1024, 1024))
    assert _app._fit_pool_to_fd_limit(4096) == 64


@pytest.mark.parametrize("served", [{"raw_stream_client": False}], indirect=True)
def test_httpx_stream_client_still_works_when_selected(served):
    app, url, abort_file, _ = served
    with httpx.stream(
        "POST", f"{url}/v1/chat/completions", json={"stream": True, "n_chunks": 3}
    ) as r:
        chunks = [c for c in r.iter_lines() if c.startswith("data:")]
    assert chunks[-1] == "data: [DONE]"
    assert app._usage_totals["completion_tokens"] == 5
    with httpx.stream(
        "POST", f"{url}/v1/chat/completions", json={"stream": True, "n_chunks": 100}
    ) as r:
        next(r.iter_lines())
    assert _wait_for(abort_file)


def test_bad_requests_use_openai_error_shape(served):
    _, url, _, _ = served
    r = httpx.post(
        f"{url}/v1/chat/completions",
        content=b"not json",
        headers={"content-type": "application/json"},
    )
    assert r.status_code == 400
    assert r.json()["error"]["type"] == "invalid_request_error"
    r = httpx.post(f"{url}/v1/chat/completions", json=[1])
    assert r.status_code == 400
    assert r.json()["error"]["message"] == "body must be an object"


@pytest.mark.parametrize("flag", ["true", 1, "false"])
def test_non_boolean_stream_is_rejected(served, flag):
    _, url, _, body_file = served
    r = httpx.post(f"{url}/v1/chat/completions", json={"stream": flag})
    assert r.status_code == 400
    assert "`stream`" in r.json()["error"]["message"]
    assert not body_file.exists()


@pytest.mark.parametrize("stream", [False, True])
def test_unreachable_engine_is_a_counted_503(served, stream):
    app, url, _, _ = served
    app._proc.kill()
    app._proc.wait()
    r = httpx.post(f"{url}/v1/chat/completions", json={"stream": stream})
    assert r.status_code == 503
    assert r.json()["error"]["type"] == "engine_unavailable"
    assert r.headers["retry-after"] == "1"
    assert r.headers["x-fal-runner-health-check"] == "true"
    assert app._replicas[0]._engine_errors == 1
    assert not app._inflight
    r = httpx.get(f"{url}/v1/models")
    assert r.status_code == 503


def test_mid_stream_engine_failure_ends_in_band(served):
    app, url, _, _ = served
    r = httpx.post(
        f"{url}/v1/chat/completions", json={"stream": True, "die_after_chunks": 2}
    )
    assert r.status_code == 200
    events = [e for e in r.text.split("\n\n") if e]
    assert events[-1] == "data: [DONE]"
    assert json.loads(events[-2][len("data: ") :])["choices"][0]["finish_reason"] == (
        "error"
    )
    assert app._replicas[0]._engine_errors == 1
    # Response headers alone do not reset the count, so repeats add up.
    httpx.post(
        f"{url}/v1/chat/completions", json={"stream": True, "die_after_chunks": 1}
    )
    assert app._replicas[0]._engine_errors == 2
    # A stream that then ends cleanly resets the count.
    assert httpx.post(f"{url}/v1/chat/completions", json={"stream": True}).is_success
    assert app._replicas[0]._engine_errors == 0


def test_stream_that_breaks_after_finishing_gets_only_done(served):
    _, url, _, _ = served
    r = httpx.post(
        f"{url}/v1/chat/completions",
        json={"stream": True, "n_chunks": 1, "finish_then_die": True},
    )
    assert r.status_code == 200
    events = [e for e in r.text.split("\n\n") if e]
    assert events[-1] == "data: [DONE]"
    reasons = [
        c.get("finish_reason")
        for e in events[:-1]
        for c in json.loads(e[len("data: ") :])["choices"]
    ]
    assert [x for x in reasons if x] == ["stop"]


def test_busy_engine_is_an_uncounted_503(served):
    app, _, _, _ = served
    replica = app._replicas[0]
    for exc in (_upstream.UpstreamBusy("full"), httpx.PoolTimeout("full")):
        r = app._engine_unavailable(replica, exc)
        assert r.status_code == 503
        assert r.headers["retry-after"] == "1"
        assert "x-fal-runner-health-check" not in r.headers
        assert r.headers["x-fal-stop-runner"] == "false"
    assert replica._engine_errors == 0


def test_stream_usage_ignores_non_object_events():
    assert _stream._parse_stream_usage(b'data: ["usage"]\n\n') is None
    assert _stream._parse_stream_usage(b'data: "usage"\n\n') is None
