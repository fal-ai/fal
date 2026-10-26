"""Per-second duration billing for ``@fal.realtime`` WebSocket endpoints.

Why this exists
---------------
A ``@fal.realtime`` handler never sees a FastAPI ``Response``, so it cannot set
``x-fal-billable-units``. With no unit count from the app the gateway records
the **default of one billable unit per session** — for an endpoint priced per
seconds, that means every session is billed as one second. Endpoint pricing
sets the price per unit; the app must supply the unit count.

The gateway sums ``{"type": "x-fal-message", "action": "timings", "timing": <s>}``
text frames into the request's ``billable_duration`` (never ``billable_units``); the
reporter charges it only if the endpoint's billing row has ``use_compute_seconds`` on.

This module wraps the SDK-generated WebSocket handler so a session emits those
frames for its wall-clock duration:

* ``accept`` is patched to emit a **zero** frame right after the handshake, so
  the gateway locks onto the sum-of-timings path and a session that ends before
  any output bills ≈ 0 rather than the default unit or the whole timeout window.
* Billing starts when the app calls :func:`mark_billing_started` (e.g. when the
  partner's SDP ``answer`` arrives), not on connect, so signaling / connect time
  is free and a session that never negotiates bills nothing.
* Once started, the wall-clock delta since the last emission is flushed before
  every outgoing event, every ``update_interval`` seconds (bounds the loss when a
  client vanishes without a close handshake), and inside ``close`` — the SDK
  closes the socket before the handler returns, so ``close`` is the last moment
  the residual slice can reach the gateway. A post-handler flush is a
  best-effort backstop for paths where the SDK never reaches its own ``close``.
* Nothing is emitted until ``grace_period`` seconds after the clock starts: frames
  can't be retracted and the gateway rounds any positive sub-second sum up to 0.25s,
  so a session the partner rejects right after the answer must sum to exactly 0.
* :func:`mark_session_failed` freezes billing (no further frames): a session that
  fails before its first flush bills 0; a later failure bills what was already flushed.
* The SDK's own per-yield timing frames (``emit_timings`` for unary /
  server-streaming routes) are dropped so they cannot double-bill.

All billing state is closure-local to one WebSocket invocation — never stored
on ``self`` — so overlapping sessions under runtime multiplexing cannot race.
Watermarks advance only *after* a successful send, so a transient send failure
never permanently drops a slice.

Usage::

    class MyRealtimeApp(fal.App):
        @fal.realtime("/realtime", buffering=8, session_timeout=300)
        async def realtime(self, inputs):
            async for out in relay.run(inputs):
                if out.type == "answer":
                    mark_billing_started()
                yield out

    install_duration_billing(MyRealtimeApp, "realtime", label="my-app")

"""

import asyncio
import json
import math
import time
from contextvars import ContextVar
from typing import Any, Callable, Union

from starlette.websockets import WebSocketDisconnect

import fal

TIMING_MESSAGE_TYPE = "x-fal-message"
TIMING_ACTION = "timings"
DEFAULT_UPDATE_INTERVAL_SECONDS = 5.0
# Sessions shorter than this after the clock starts bill 0; longer ones are billed
# from the clock start. Covers a partner rejection right after the answer (~0.5s).
DEFAULT_GRACE_PERIOD_SECONDS = 2.0

_Clock = Callable[[], float]
# Resolved at call time (not at import / install time) so tests can swap in a
# fake clock for handlers that were wrapped at module import.
_MONOTONIC: _Clock = time.monotonic


def timing_frame(seconds: float) -> str:
    """Build the JSON text frame the fal gateway consumes for billing."""
    return json.dumps(
        {"type": TIMING_MESSAGE_TYPE, "action": TIMING_ACTION, "timing": seconds},
        separators=(",", ":"),
    )


def is_timing_frame(data: Any) -> bool:
    """True when ``data`` is a gateway timing frame (ours or the SDK's)."""
    if not isinstance(data, str) or f'"{TIMING_MESSAGE_TYPE}"' not in data:
        return False
    try:
        payload = json.loads(data)
    except ValueError:
        return False
    return (
        isinstance(payload, dict)
        and payload.get("type") == TIMING_MESSAGE_TYPE
        and payload.get("action") == TIMING_ACTION
    )


# Created lazily: module-level ContextVar instances are not picklable and fal
# serializes app-module globals when shipping work to runners. ``@fal.cached``
# makes it a lock-protected singleton.
@fal.cached
def _billing_start_callback_var() -> "ContextVar[Callable[[], None] | None]":
    return ContextVar("realtime_duration_billing_start", default=None)


@fal.cached
def _billing_fail_callback_var() -> "ContextVar[Callable[[], None] | None]":
    return ContextVar("realtime_duration_billing_fail", default=None)


def mark_billing_started() -> None:
    """Start the billing clock for the current realtime session.

    Safe to call more than once and safe to call outside a billed handler
    (a no-op). The wrapper installed by :func:`install_duration_billing`
    binds the callback per WebSocket invocation; asyncio copies the context
    into the tasks the SDK spawns, so the app's generator sees it.
    """
    callback = _billing_start_callback_var().get()
    if callback is not None:
        callback()


def mark_session_failed() -> None:
    """Freeze billing for the current realtime session (no further timing frames).

    Idempotent; a no-op outside a billed handler. Frames already sent stand.
    """
    callback = _billing_fail_callback_var().get()
    if callback is not None:
        callback()


def make_duration_billed_handler(  # type: ignore[no-untyped-def]
    inner_handler,
    *,
    label: str,
    update_interval: float = DEFAULT_UPDATE_INTERVAL_SECONDS,
    grace_period: float = DEFAULT_GRACE_PERIOD_SECONDS,
    clock: Union[_Clock, None] = None,
    log: Union[Callable[[str], None], None] = None,
):
    """Wrap an ``async (self, websocket)`` handler with duration billing.

    ``grace_period``: seconds after the clock starts during which nothing is
    emitted (see the module docstring). ``0`` disables the hold.

    Factored out (rather than inlined in :func:`install_duration_billing`) so
    unit tests can drive the full lifecycle against a fake WebSocket and a
    fake inner handler.
    """
    for name, value, allow_zero in (
        ("update_interval", update_interval, False),
        ("grace_period", grace_period, True),
    ):
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value < 0
            or (value == 0 and not allow_zero)
        ):
            bound = "non-negative" if allow_zero else "positive"
            raise ValueError(f"{name} must be finite and {bound}")
    _log = log or (lambda msg: print(msg, flush=True))

    async def _handler(self, websocket):  # type: ignore[no-untyped-def]
        now: _Clock = clock if clock is not None else _MONOTONIC
        original_accept = websocket.accept
        original_send_text = websocket.send_text
        original_send_bytes = websocket.send_bytes
        original_close = websocket.close
        lock = asyncio.Lock()

        session_start: Union[float, None] = None
        # Wall-clock seconds already reported to the gateway. Emitted deltas
        # sum to this so the gateway's running total tracks real duration.
        last_billed_elapsed = 0.0
        periodic_task: Union[asyncio.Task, None] = None
        socket_gone_logged = False
        # Set by ``mark_session_failed``: no further frames, see module docstring.
        failed = False

        def _start_billing() -> None:
            nonlocal session_start, periodic_task
            if failed or session_start is not None:
                return
            session_start = now()
            periodic_task = asyncio.create_task(_periodic_updates())

        def _fail_session() -> None:
            nonlocal failed
            if failed:
                return
            failed = True
            unbilled = 0.0
            if session_start is not None:
                unbilled = max(0.0, (now() - session_start) - last_billed_elapsed)
            _log(
                f"[{label}] billing: session failed; freezing at "
                f"{last_billed_elapsed:.2f}s billed, {unbilled:.2f}s not billed"
            )

        start_var = _billing_start_callback_var()
        start_token = start_var.set(_start_billing)
        fail_var = _billing_fail_callback_var()
        fail_token = fail_var.set(_fail_session)

        async def _emit_current_timing() -> None:
            """Emit a timing frame covering wall-clock since the last emission."""
            nonlocal last_billed_elapsed, socket_gone_logged
            delta = 0.0
            try:
                async with lock:
                    if session_start is None or failed:
                        return
                    now_elapsed = now() - session_start
                    if now_elapsed < grace_period:
                        return  # hold: keep a rejected session's sum at exactly 0
                    delta = now_elapsed - last_billed_elapsed
                    if delta <= 0:
                        return
                    await original_send_text(timing_frame(delta))
                    # Advance the watermark only after the send succeeded, and
                    # under the same lock, so concurrent flushes can neither
                    # overlap nor double-count a delta.
                    last_billed_elapsed = now_elapsed
            except (WebSocketDisconnect, RuntimeError):
                # The client (via the gateway) already hung up, so the residual
                # slice since the last tick cannot be billed. Expected on every
                # abrupt disconnect (tab close); bounded by ``update_interval``.
                # Log once — the close flush and the post-handler backstop both
                # land here.
                if not socket_gone_logged:
                    socket_gone_logged = True
                    _log(
                        f"[{label}] billing: socket already closed; "
                        f"{delta:.2f}s residual unbilled"
                    )
            except Exception as exc:
                _log(f"[{label}] billing: timing emit failed: {exc!r}")

        async def _periodic_updates() -> None:
            while True:
                await asyncio.sleep(update_interval)
                await _emit_current_timing()

        async def _stop_periodic() -> None:
            if periodic_task is not None:
                periodic_task.cancel()
                try:
                    await periodic_task
                except (asyncio.CancelledError, Exception):
                    pass

        async def _accept_and_emit_zero(*args, **kwargs):  # type: ignore[no-untyped-def]
            await original_accept(*args, **kwargs)
            try:
                async with lock:
                    await original_send_text(timing_frame(0.0))
            except Exception as exc:
                _log(f"[{label}] billing: initial zero-timing emit failed: {exc!r}")

        async def _send_bytes(data):  # type: ignore[no-untyped-def]
            # Bill up to this event first so the gateway's total advances even
            # if the session ends right after this frame.
            await _emit_current_timing()
            try:
                async with lock:
                    await original_send_bytes(data)
            except (WebSocketDisconnect, RuntimeError):
                # fal 1.79.1's background emitter acks ``queue.task_done()`` only after
                # send_bytes returns, so a raising send wedges ``close_emitter()`` in
                # ``queue.join()``; swallow the dead-socket errors the SDK suppresses.
                pass

        async def _send_text(data):  # type: ignore[no-untyped-def]
            # Drop the SDK's auto timings — ours cover the session; both would sum.
            if is_timing_frame(data):
                return
            # JSON control/signaling events are billable boundaries too.
            await _emit_current_timing()
            try:
                async with lock:
                    await original_send_text(data)
            except (WebSocketDisconnect, RuntimeError):
                pass  # same emitter-wedge hazard as ``_send_bytes``

        async def _flush_and_close(*args, **kwargs):  # type: ignore[no-untyped-def]
            try:
                await _emit_current_timing()
            finally:
                await original_close(*args, **kwargs)

        websocket.accept = _accept_and_emit_zero
        websocket.send_bytes = _send_bytes
        websocket.send_text = _send_text
        websocket.close = _flush_and_close

        try:
            await inner_handler(self, websocket)
        finally:
            await _stop_periodic()
            # Backstop for paths where the SDK never reaches ``close``. If the
            # socket is gone the send fails quietly; if ``close`` already
            # flushed there is no residual delta left to emit.
            await _emit_current_timing()
            fail_var.reset(fail_token)
            start_var.reset(start_token)

    return _handler


def install_duration_billing(
    app_cls: type,
    attr: str,
    *,
    label: str,
    update_interval: float = DEFAULT_UPDATE_INTERVAL_SECONDS,
    grace_period: float = DEFAULT_GRACE_PERIOD_SECONDS,
) -> None:
    """Replace ``app_cls.<attr>`` (a ``@fal.realtime`` handler) with a billed one.

    Preserves the SDK's introspection metadata (``route_signature`` with
    ``emit_timings=True``, ``original_func``, annotations) so route collection
    and OpenAPI generation keep working. Call at module scope right after the
    class body.
    """
    original = getattr(app_cls, attr)
    route_signature = getattr(original, "route_signature", None)
    if route_signature is None:
        raise TypeError(
            f"{app_cls.__name__}.{attr} is not a fal realtime handler "
            "(missing route_signature)"
        )
    wrapped = make_duration_billed_handler(
        original,
        label=label,
        update_interval=update_interval,
        grace_period=grace_period,
    )
    wrapped.route_signature = route_signature._replace(emit_timings=True)  # type: ignore[attr-defined]
    wrapped.original_func = getattr(original, "original_func", original)  # type: ignore[attr-defined]
    wrapped.__annotations__ = dict(getattr(original, "__annotations__", {}))
    wrapped.__name__ = getattr(original, "__name__", attr)
    wrapped.__qualname__ = getattr(original, "__qualname__", attr)
    wrapped.__doc__ = getattr(original, "__doc__", None)
    setattr(app_cls, attr, wrapped)


__all__ = [
    "DEFAULT_GRACE_PERIOD_SECONDS",
    "DEFAULT_UPDATE_INTERVAL_SECONDS",
    "install_duration_billing",
    "is_timing_frame",
    "make_duration_billed_handler",
    "mark_billing_started",
    "mark_session_failed",
    "timing_frame",
]
