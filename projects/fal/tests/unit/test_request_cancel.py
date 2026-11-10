import asyncio
import time

import pytest
from fastapi.testclient import TestClient
from starlette.applications import Starlette
from starlette.responses import JSONResponse
from starlette.routing import Route

import fal
from fal._request_cancel import RequestCancellation, parse_timeout_remaining
from fal.app import RequestContext


class Harness:
    """Drives an ASGI app by hand so disconnects can be injected."""

    def __init__(self, app, path="/run", method="POST", request_id="req-1"):
        headers = [(b"x-fal-request-id", request_id.encode())] if request_id else []
        self.scope = {
            "type": "http",
            "method": method,
            "path": path,
            "headers": headers,
            "app": app,
        }
        self.app = app
        self.inbox: asyncio.Queue = asyncio.Queue()
        self.sent: list = []

    async def receive(self):
        return await self.inbox.get()

    async def send(self, message):
        self.sent.append(message)

    def start(self):
        self.inbox.put_nowait(
            {"type": "http.request", "body": b"{}", "more_body": False}
        )
        return asyncio.ensure_future(
            self.app(dict(self.scope), self.receive, self.send)
        )

    @property
    def status(self):
        return next(
            m["status"] for m in self.sent if m["type"] == "http.response.start"
        )


def make_app(cancel_on_disconnect, events, extra_routes=(), gateway_cancel=True):
    async def run(request):
        await request.body()
        events.append("started")
        try:
            await asyncio.sleep(30)
        except asyncio.CancelledError:
            events.append("cancelled")
            raise
        return JSONResponse({"ok": True})

    starlette = Starlette(routes=[Route("/run", run, methods=["POST"]), *extra_routes])
    starlette.add_middleware(
        RequestCancellation,
        cancel_on_gateway_cancel=gateway_cancel,
        cancel_on_disconnect=cancel_on_disconnect,
    )
    return starlette


async def settle():
    for _ in range(5):
        await asyncio.sleep(0)


async def test_disconnect_cancels_handler_when_opted_in():
    events: list = []
    app = make_app(True, events)
    h = Harness(app)
    task = h.start()
    await settle()
    h.inbox.put_nowait({"type": "http.disconnect"})
    await asyncio.wait_for(task, 2)
    assert events == ["started", "cancelled"]
    assert h.status == 499  # discarded by the server, but never an empty return


async def test_disconnect_ignored_by_default():
    events: list = []
    app = make_app(False, events)
    h = Harness(app)
    task = h.start()
    await settle()
    h.inbox.put_nowait({"type": "http.disconnect"})
    await settle()
    assert events == ["started"]
    assert not task.done()
    task.cancel()
    # Awaited so no cancelling task outlives the test's event loop.
    with pytest.raises(asyncio.CancelledError):
        await task


async def test_gateway_cancel_stops_handler_by_request_id():
    events: list = []
    app = make_app(False, events)
    run = Harness(app)
    run_task = run.start()
    await settle()

    cancel = Harness(app, path="/run/cancel")
    # A Starlette app wraps middleware; call through the built stack.
    await app(dict(cancel.scope), cancel.receive, cancel.send)
    assert cancel.status == 200
    assert b'"cancelled": true' in cancel.sent[-1]["body"]

    await asyncio.wait_for(run_task, 2)
    assert events == ["started", "cancelled"]
    assert run.status == 499
    start = next(m for m in run.sent if m["type"] == "http.response.start")
    assert (b"x-fal-billable-units", b"0") in start["headers"]


async def test_gateway_cancel_reaches_every_flight_with_the_id():
    # A retry can reuse the request id while the first attempt still runs.
    events: list = []
    app = make_app(False, events)
    first, second = Harness(app), Harness(app)
    tasks = [first.start(), second.start()]
    await settle()
    cancel = Harness(app, path="/run/cancel")
    await app(dict(cancel.scope), cancel.receive, cancel.send)
    assert b'"cancelled": true' in cancel.sent[-1]["body"]
    await asyncio.wait_for(asyncio.gather(*tasks), 2)
    assert events == ["started", "started", "cancelled", "cancelled"]
    assert first.status == second.status == 499


async def test_repeat_cancel_does_not_interrupt_handler_cleanup():
    events: list = []

    async def run(request):
        await request.body()
        try:
            await asyncio.sleep(30)
        except asyncio.CancelledError:
            try:
                await asyncio.sleep(0.2)  # e.g. closing an upstream stream
                events.append("cleanup done")
            except asyncio.CancelledError:
                events.append("cleanup interrupted")
            raise
        return JSONResponse({"ok": True})

    app = Starlette(routes=[Route("/run", run, methods=["POST"])])
    app.add_middleware(RequestCancellation, cancel_on_gateway_cancel=True)
    h = Harness(app)
    task = h.start()
    await settle()
    for _ in range(2):
        cancel = Harness(app, path="/run/cancel")
        await app(dict(cancel.scope), cancel.receive, cancel.send)
        assert b'"cancelled": true' in cancel.sent[-1]["body"]
        await settle()
    await asyncio.wait_for(task, 2)
    assert events == ["cleanup done"]
    assert h.status == 499


async def test_client_forwarded_cancel_path_does_not_cancel_by_request_id():
    # A queue retry lets a client pick x-fal-request-id, but every request the
    # gateway forwards from a client carries x-fal-endpoint; its own cancel
    # call never does.
    events: list = []
    app = make_app(False, events)
    run = Harness(app)
    run_task = run.start()
    await settle()

    spoof = Harness(app, path="/run/cancel")
    spoof.scope["headers"].append((b"x-fal-endpoint", b"someone/app/run/cancel"))
    spoof.inbox.put_nowait({"type": "http.request", "body": b"", "more_body": False})
    await asyncio.wait_for(app(dict(spoof.scope), spoof.receive, spoof.send), 2)
    assert spoof.status == 404  # routed to the app, which has no such route
    await settle()
    assert events == ["started"]
    assert not run_task.done()

    run_task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await run_task


async def test_gateway_cancel_is_ignored_unless_opted_in():
    events: list = []
    app = make_app(False, events, gateway_cancel=False)
    run = Harness(app)
    run_task = run.start()
    await settle()

    cancel = Harness(app, path="/run/cancel")
    cancel.inbox.put_nowait({"type": "http.request", "body": b"", "more_body": False})
    await asyncio.wait_for(app(dict(cancel.scope), cancel.receive, cancel.send), 2)
    assert cancel.status == 404  # passed to the app, which has no such route
    await settle()
    assert events == ["started"]
    assert not run_task.done()

    run_task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await run_task


def test_plain_app_does_not_opt_into_gateway_cancel():
    def options(app_cls):
        built = app_cls(_allow_init=True)._build_app()
        mw = next(m for m in built.user_middleware if m.cls is RequestCancellation)
        return mw.kwargs

    class Plain(fal.App):
        @fal.endpoint("/")
        def run(self) -> dict:
            return {}

    class OptedIn(Plain):
        cancel_on_gateway_cancel = True

    assert options(Plain)["cancel_on_gateway_cancel"] is False
    assert options(OptedIn)["cancel_on_gateway_cancel"] is True


async def test_gateway_cancel_unknown_id_is_idempotent_noop():
    app = make_app(False, [])
    cancel = Harness(app, path="/run/cancel", request_id="nope")
    await app(dict(cancel.scope), cancel.receive, cancel.send)
    assert cancel.status == 200
    assert b'"cancelled": false' in cancel.sent[-1]["body"]


async def test_user_defined_cancel_route_wins():
    async def own_cancel(request):
        return JSONResponse({"mine": True})

    app = make_app(
        False, [], extra_routes=[Route("/run/cancel", own_cancel, methods=["POST"])]
    )
    cancel = Harness(app, path="/run/cancel")
    task = asyncio.ensure_future(app(dict(cancel.scope), cancel.receive, cancel.send))
    cancel.inbox.put_nowait({"type": "http.request", "body": b"", "more_body": False})
    await asyncio.wait_for(task, 2)
    assert b"mine" in cancel.sent[-1]["body"]


async def test_completed_request_is_not_cancelled():
    async def quick(request):
        return JSONResponse({"ok": True})

    starlette = Starlette(routes=[Route("/run", quick, methods=["POST"])])
    starlette.add_middleware(RequestCancellation, cancel_on_disconnect=True)
    h = Harness(starlette)
    task = h.start()
    await asyncio.wait_for(task, 2)
    assert h.status == 200


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("297.350", 297.35),
        ("0", 0.0),
        (None, None),
        ("abc", None),
        ("-1", None),
        ("nan", None),
        ("inf", None),
    ],
)
def test_parse_timeout_remaining(raw, expected):
    assert parse_timeout_remaining(raw) == expected


def test_request_context_time_remaining():
    ctx = RequestContext(None, None, None, {}, deadline_monotonic=time.monotonic() + 10)
    assert 9 < ctx.time_remaining() <= 10
    assert RequestContext(None, None, None, {}).time_remaining() is None
    assert RequestContext(None, None, None, {}).cancelled is False


async def test_websocket_scope_passes_through():
    seen = []

    async def inner(scope, receive, send):
        seen.append(scope["type"])

    mw = RequestCancellation(inner, cancel_on_disconnect=True)
    await mw({"type": "websocket", "headers": []}, None, None)
    assert seen == ["websocket"]


async def test_outer_cancellation_propagates_and_clears_registry():
    events: list = []
    mw = RequestCancellation(make_app(False, events), cancel_on_gateway_cancel=True)
    h = Harness(mw)
    h.scope["app"] = None
    task = h.start()
    await settle()
    assert "req-1" in mw._flights
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert mw._flights == {}


async def test_disconnect_after_complete_response_does_not_cancel():
    async def inner(scope, receive, send):
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b"x"})
        await asyncio.sleep(0.05)

    mw = RequestCancellation(inner, cancel_on_disconnect=True)
    h = Harness(mw)
    h.scope["app"] = None
    task = h.start()
    await settle()
    h.inbox.put_nowait({"type": "http.disconnect"})
    await asyncio.wait_for(task, 2)
    assert h.status == 200


async def test_mid_stream_gateway_cancel_aborts_without_499():
    async def inner(scope, receive, send):
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b"a", "more_body": True})
        await asyncio.sleep(30)

    mw = RequestCancellation(inner, cancel_on_gateway_cancel=True)
    h = Harness(mw)
    h.scope["app"] = None
    task = h.start()
    await settle()
    cancel = Harness(mw, path="/x/cancel")
    cancel.scope["app"] = None
    await mw(dict(cancel.scope), cancel.receive, cancel.send)
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(task, 2)
    assert h.status == 200 and len(h.sent) == 2


async def test_app_wires_middleware_and_context():
    class A(fal.App):
        cancel_on_gateway_cancel = True
        cancel_on_disconnect = True

        @fal.endpoint("/")
        def run(self) -> dict:
            ctx = self.current_request
            return {
                "cancelled": ctx.cancelled,
                "remaining": ctx.time_remaining(),
            }

    app = A(_allow_init=True)
    with TestClient(app._build_app()) as client:
        _check(client)


def _check(client):
    resp = client.post(
        "/",
        headers={
            "x-fal-request-id": "abc",
            "x-fal-request-timeout-remaining": "12.5",
        },
    )
    assert resp.status_code == 200, resp.text
    assert resp.json()["cancelled"] is False
    assert 11 < resp.json()["remaining"] <= 12.5
    assert client.post("/cancel", headers={"x-fal-request-id": "zzz"}).json() == {
        "cancelled": False
    }
