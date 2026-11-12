"""Ownership of one upstream streaming response.

A ``_Stream`` owns the upstream response and closes it exactly once. Its
``on_settle`` callback (the app's in-flight and usage bookkeeping) also runs
once, and the upstream close is shielded from cancellation so a client
disconnect cannot leave the engine request running.
"""

import asyncio
import json
import re
from collections import deque
from typing import TYPE_CHECKING, Any, Awaitable, Callable, Deque, Dict, Optional

import anyio
from fastapi import Request
from fastapi.responses import StreamingResponse

if TYPE_CHECKING:
    from fal.toolkit.llm._app import _Replica

_STREAM_TAIL_BYTES = 8192
# A forwarded chunk that already carries a non-null finish reason.
_FINISH_REASON_RE = re.compile(rb'"finish_reason":\s*"')


# ``"usage": {...}`` with one level of nesting, for a Responses API terminal
# event whose start fell out of the stream tail (it carries the whole output).
_USAGE_TAIL_RE = re.compile(r'"usage":\s*(\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\})')
# Responses API stream events that end a response (no ``[DONE]`` follows).
_RESPONSES_TERMINAL_RE = re.compile(
    rb'"type":\s*"response\.(?:completed|incomplete|failed)"'
)


def _parse_stream_usage(tail: bytes) -> Optional[Dict[str, Any]]:
    text = tail.decode("utf-8", errors="replace")
    for raw in reversed(text.splitlines()):
        line = raw.strip()
        if line.startswith("data:") and '"usage"' in line:
            try:
                event = json.loads(line[5:])
            except ValueError:
                break
            if not isinstance(event, dict):
                break
            # Chat and completions: top level. Responses: under ``response``.
            usage = event.get("usage")
            if usage is None and isinstance(event.get("response"), dict):
                usage = event["response"].get("usage")
            return usage if isinstance(usage, dict) else None
    # The terminal Responses event can outgrow the tail and lose its start.
    match = None
    for match in _USAGE_TAIL_RE.finditer(text):
        pass
    if match:
        try:
            usage = json.loads(match.group(1))
        except ValueError:
            return None
        return usage if isinstance(usage, dict) else None
    return None


def _cached_tokens(usage: Dict[str, Any]) -> int:
    """Prompt tokens served from the prefix cache (chat or Responses shape)."""
    for key in ("prompt_tokens_details", "input_tokens_details"):
        details = usage.get(key)
        if isinstance(details, dict):
            return int(details.get("cached_tokens") or 0)
    return 0


def _event_cut(buf: bytes) -> int:
    """Length of the leading run of whole SSE events in ``buf`` (0 if none)."""
    lf, crlf = buf.rfind(b"\n\n"), buf.rfind(b"\r\n\r\n")
    return max(lf + 2 if lf >= 0 else 0, crlf + 4 if crlf >= 0 else 0)


def _sse_response_id(events: bytes) -> Optional[str]:
    """The ``response.id`` of the first Responses API event in ``events``
    that has one."""
    for line in events.splitlines():
        if not line.startswith(b"data:"):
            continue
        try:
            event = json.loads(line[5:])
        except ValueError:
            continue
        response = event.get("response") if isinstance(event, dict) else None
        if isinstance(response, dict) and isinstance(response.get("id"), str):
            return response["id"]
    return None


class _RelayResponse(StreamingResponse):
    """Streaming response that always runs ``cleanup``, even if never iterated.

    Starlette skips the body generator when the client disconnects before the
    first iteration, so cleanup cannot live in the generator's ``finally``.
    """

    def __init__(self, content: Any, cleanup: Callable[[], Awaitable[None]], **kw):
        super().__init__(content, **kw)
        self._cleanup = cleanup

    async def __call__(self, scope, receive, send) -> None:
        try:
            await super().__call__(scope, receive, send)
        finally:
            # Unwinds a suspended generator; a no-op if it finished or never ran.
            try:
                aclose = getattr(self.body_iterator, "aclose", None)
                if aclose is not None:
                    await aclose()
            finally:
                await self._cleanup()


class _Stream:
    """An upstream streaming response and its once-only cleanup."""

    def __init__(
        self,
        on_settle: "Callable[[_Stream], None]",
        upstream: Any,
        path: str,
        abort: asyncio.Event,
        replica: Optional["_Replica"] = None,
    ) -> None:
        self._on_settle = on_settle
        self.upstream = upstream
        self.path = path
        self.abort = abort
        self.replica = replica
        self._recent: Deque[bytes] = deque()
        self._recent_len = 0
        self._settled = False
        self._upstream_closed = False

    @property
    def tail(self) -> bytes:
        """The last ``_STREAM_TAIL_BYTES`` bytes relayed."""
        return b"".join(self._recent)[-_STREAM_TAIL_BYTES:]

    @tail.setter
    def tail(self, value: bytes) -> None:
        self._recent = deque([value])
        self._recent_len = len(value)

    def feed(self, chunk: bytes) -> None:
        # Keeps whole chunks instead of re-slicing one tail buffer per chunk,
        # which copied the full tail for every few-hundred-byte event.
        self._recent.append(chunk)
        self._recent_len += len(chunk)
        while len(self._recent) > 1 and (
            self._recent_len - len(self._recent[0]) >= _STREAM_TAIL_BYTES
        ):
            self._recent_len -= len(self._recent.popleft())

    async def close(self) -> None:
        # Bookkeeping runs once; the upstream close is retried by the next
        # owner if a cancelled scope (client disconnect) interrupted it.
        try:
            if not self._settled:
                self._settled = True
                self._on_settle(self)
        finally:
            if not self._upstream_closed:
                # Closing the upstream connection makes the engine abort the
                # request, which also covers client disconnects and shutdown.
                # Shielded: on a client disconnect this runs in a cancelled scope,
                # and an interrupted httpx close cannot be retried (the response
                # is marked closed first), which with httpcore < 1.0 left the
                # engine connection open. The shield covers anyio scope
                # cancellation (Starlette's task group on ASGI spec 2.3), not a
                # second native task cancel.
                with anyio.CancelScope(shield=True):
                    await self.upstream.aclose()
                self._upstream_closed = True


async def _drop_upstream(send: "asyncio.Future[Any]") -> None:
    """Cancel a pending upstream open and close the response if it won the
    race. Cancelling closes the connection, which aborts the engine request."""
    send.cancel()
    with anyio.CancelScope(shield=True):
        await asyncio.wait({send})
        if not send.cancelled() and send.exception() is None:
            await send.result().aclose()


async def _wait_for_disconnect(request: Request) -> None:
    # Blocking receive instead of Request.is_disconnected(): the latter polls
    # with a cancelled scope, which can miss the event.
    while True:
        message = await request.receive()
        if message["type"] == "http.disconnect":
            return
