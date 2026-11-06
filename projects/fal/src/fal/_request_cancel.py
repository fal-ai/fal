"""Request cancellation for served apps.

Two signals can end an in-flight handler, each opt-in per app:

* the gateway's ``POST <endpoint>/cancel`` (carrying the original
  ``x-fal-request-id``), sent for queue cancels and request timeouts, with
  ``cancel_on_gateway_cancel``. An app's own ``<endpoint>/cancel`` route wins.
* client disconnect, with ``cancel_on_disconnect``.

The handler runs as a child task, so cancelling it raises ``CancelledError`` in
the handler (an ``except``/``finally`` around the engine call can abort it).
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

from starlette.routing import Match
from starlette.types import ASGIApp, Message, Receive, Scope, Send

REQUEST_ID_HEADER = b"x-fal-request-id"
# The gateway sets x-fal-endpoint on every request it forwards from a client
# (client values are stripped) but never on its own cancel call. A queue retry
# lets a client choose x-fal-request-id, so a /cancel carrying this header came
# from a client and must not cancel by that id.
ENDPOINT_HEADER = b"x-fal-endpoint"
TIMEOUT_REMAINING_HEADER = b"x-fal-request-timeout-remaining"
FLIGHT_SCOPE_KEY = "fal.request_flight"

# Read-ahead cap, so a streamed upload keeps some backpressure.
_MAX_BUFFERED_MESSAGES = 16
CANCELLED_STATUS = 499


def header_value(scope: Scope, name: bytes) -> str | None:
    for key, value in scope.get("headers", ()):
        if key.lower() == name:
            return value.decode("latin-1")
    return None


def parse_timeout_remaining(raw: str | None) -> float | None:
    """Seconds left on the gateway's request timeout, or ``None``."""
    if raw is None:
        return None
    try:
        value = float(raw)
    except ValueError:
        return None
    if value != value or value < 0 or value == float("inf"):
        return None
    return value


class RequestFlight:
    """One in-flight HTTP request that can be cancelled."""

    def __init__(self, request_id: str | None) -> None:
        self.request_id = request_id
        self.task: asyncio.Task[None] | None = None
        self.cancelled = False
        self.disconnected = False
        self.response_started = False
        self.response_complete = False

    def cancel(self) -> bool:
        if self.task is None or self.task.done() or self.response_complete:
            return False
        if self.cancelled:
            # A second cancel (a queue cancel and a timeout can overlap) must
            # not interrupt the handler's cleanup from the first.
            return True
        self.cancelled = True
        self.task.cancel()
        return True


class RequestCancellation:
    """Raw ASGI middleware, outermost, so it sees receive() before the app."""

    def __init__(
        self,
        app: ASGIApp,
        cancel_on_gateway_cancel: bool = False,
        cancel_on_disconnect: bool = False,
    ) -> None:
        self.app = app
        self.cancel_on_gateway_cancel = cancel_on_gateway_cancel
        self.cancel_on_disconnect = cancel_on_disconnect
        # A list per id: a retry can reuse the id while the first attempt runs,
        # and a cancel must reach both.
        self._flights: dict[str, list[RequestFlight]] = {}

    def _has_user_route(self, scope: Scope) -> bool:
        app = scope.get("app")
        router = getattr(app, "router", None)
        if router is None:
            return False
        return any(route.matches(scope)[0] == Match.FULL for route in router.routes)

    def _is_gateway_cancel(self, scope: Scope, request_id: str | None) -> bool:
        return (
            self.cancel_on_gateway_cancel
            and request_id is not None
            and scope["method"] == "POST"
            and scope["path"].rstrip("/").endswith("/cancel")
            and header_value(scope, ENDPOINT_HEADER) is None
            and not self._has_user_route(scope)
        )

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        request_id = header_value(scope, REQUEST_ID_HEADER)
        if self._is_gateway_cancel(scope, request_id):
            await self._answer_cancel(request_id, send)  # type: ignore[arg-type]
            return

        gateway_cancellable = self.cancel_on_gateway_cancel and request_id is not None
        if not gateway_cancellable and not self.cancel_on_disconnect:
            await self.app(scope, receive, send)
            return

        await self._run(scope, receive, send, request_id)

    async def _answer_cancel(self, request_id: str, send: Send) -> None:
        results = [flight.cancel() for flight in self._flights.get(request_id, ())]
        cancelled = any(results)
        body = json.dumps({"cancelled": cancelled}).encode()
        await send(
            {
                "type": "http.response.start",
                "status": 200,
                "headers": [
                    (b"content-type", b"application/json"),
                    (b"content-length", str(len(body)).encode()),
                    (b"x-fal-billable-units", b"0"),
                ],
            }
        )
        await send({"type": "http.response.body", "body": body})

    async def _run(
        self, scope: Scope, receive: Receive, send: Send, request_id: str | None
    ) -> None:
        flight = RequestFlight(request_id)
        scope[FLIGHT_SCOPE_KEY] = flight
        queue: asyncio.Queue[Message] = asyncio.Queue()
        space = asyncio.Event()
        space.set()
        disconnect_message: Message | None = None

        async def pump() -> None:
            # Sole reader of the server's receive(): after the body ends, the
            # next receive() only returns on client disconnect.
            while True:
                while queue.qsize() >= _MAX_BUFFERED_MESSAGES:
                    space.clear()
                    await space.wait()
                try:
                    message = await receive()
                except Exception:
                    message = {"type": "http.disconnect"}
                queue.put_nowait(message)
                if message["type"] == "http.disconnect":
                    flight.disconnected = True
                    if self.cancel_on_disconnect:
                        flight.cancel()
                    return

        async def app_receive() -> Message:
            nonlocal disconnect_message
            if disconnect_message is not None:
                return disconnect_message
            message = await queue.get()
            space.set()
            if message["type"] == "http.disconnect":
                disconnect_message = message
            return message

        async def app_send(message: Message) -> None:
            if message["type"] == "http.response.start":
                flight.response_started = True
            elif message["type"] == "http.response.body" and not message.get(
                "more_body", False
            ):
                flight.response_complete = True
            await send(message)

        if request_id is not None:
            self._flights.setdefault(request_id, []).append(flight)
        pump_task = asyncio.ensure_future(pump())
        flight.task = asyncio.ensure_future(self.app(scope, app_receive, app_send))
        try:
            await flight.task
        except asyncio.CancelledError:
            task = asyncio.current_task()
            cancelling = getattr(task, "cancelling", None)  # 3.11+
            outer_cancelled = cancelling is not None and cancelling() > 0
            if not flight.cancelled or outer_cancelled:
                flight.task.cancel()
                raise
            if not flight.response_started:
                # Sent even after a disconnect (discarded) so an outer
                # BaseHTTPMiddleware never sees an app that returned nothing.
                await _send_cancelled(send)
            elif not flight.disconnected:
                # Mid-stream: abort the connection instead of ending cleanly.
                raise
        finally:
            pump_task.cancel()
            if request_id is not None:
                flights = self._flights.get(request_id, [])
                if flight in flights:
                    flights.remove(flight)
                if not flights:
                    self._flights.pop(request_id, None)


async def _send_cancelled(send: Send) -> None:
    body = json.dumps({"detail": "Request cancelled"}).encode()
    await send(
        {
            "type": "http.response.start",
            "status": CANCELLED_STATUS,
            "headers": [
                (b"content-type", b"application/json"),
                (b"content-length", str(len(body)).encode()),
                # Without it the platform charges the endpoint's default.
                (b"x-fal-billable-units", b"0"),
            ],
        }
    )
    await send({"type": "http.response.body", "body": body})


def flight_from_scope(scope: Any) -> RequestFlight | None:
    return scope.get(FLIGHT_SCOPE_KEY)
