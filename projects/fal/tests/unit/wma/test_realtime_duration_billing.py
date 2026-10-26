"""Unit tests for ``fal.wma.duration_billing``.

Background: a ``@fal.realtime`` handler cannot set ``x-fal-billable-units``, so
without timing frames the gateway bills the default single unit — one second
per session for a per-second catalog price, however long the session ran.
These tests drive the billing wrapper against a fake
WebSocket and fake inner handlers and assert the frame stream the gateway sums.
"""

import asyncio
import json
from typing import Any, List, Tuple

import pytest

from fal.wma.duration_billing import (
    DEFAULT_GRACE_PERIOD_SECONDS,
    install_duration_billing,
    is_timing_frame,
    make_duration_billed_handler,
    mark_billing_started,
    mark_session_failed,
    timing_frame,
)


class FakeWebSocket:
    """Records every accept/send/close in order, like the SDK's socket.

    After ``close`` — exactly as with a real WebSocket — sends raise, so a
    timing frame can only appear in ``timeline`` before the close.
    """

    def __init__(self) -> None:
        self.timeline: List[Tuple[str, Any]] = []
        self.closed = False

    async def accept(self, *args: Any, **kwargs: Any) -> None:
        self.timeline.append(("accept", None))

    async def send_text(self, data: str) -> None:
        if self.closed:
            raise RuntimeError("send_text after close")
        self.timeline.append(("text", data))

    async def send_bytes(self, data: bytes) -> None:
        if self.closed:
            raise RuntimeError("send_bytes after close")
        self.timeline.append(("bytes", data))

    async def close(self, *args: Any, **kwargs: Any) -> None:
        self.closed = True
        self.timeline.append(("close", None))


def _timings(ws: FakeWebSocket) -> List[Tuple[int, float]]:
    """``(timeline_index, seconds)`` for every emitted timing frame."""
    out: List[Tuple[int, float]] = []
    for i, (kind, data) in enumerate(ws.timeline):
        if kind == "text" and is_timing_frame(data):
            out.append((i, json.loads(data)["timing"]))
    return out


def _index_of(ws: FakeWebSocket, kind: str) -> int:
    return [i for i, (k, _) in enumerate(ws.timeline) if k == kind][0]


class FakeClock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def _make(handler: Any, clock: FakeClock, **kwargs: Any) -> Any:
    # ``grace_period`` is left at its default so the existing lifecycle tests
    # also prove that real sessions (all longer than the grace period) are
    # billed from the clock start, not from the end of the grace period.
    return make_duration_billed_handler(
        handler, label="test", clock=clock, update_interval=3600, **kwargs
    )


def test_timing_frame_shape() -> None:
    assert json.loads(timing_frame(1.5)) == {
        "type": "x-fal-message",
        "action": "timings",
        "timing": 1.5,
    }
    assert is_timing_frame(timing_frame(0.0))
    assert not is_timing_frame('{"type":"x-fal-message","action":"other"}')
    assert not is_timing_frame('{"type":"answer","sdp":"v=0"}')
    assert not is_timing_frame(b"binary")


@pytest.mark.asyncio
async def test_zero_frame_emitted_immediately_after_accept() -> None:
    # Locks the gateway onto sum-of-timings billing so an instant disconnect
    # bills ≈ 0 instead of the default unit / the whole timeout window.
    async def _instant_close(self: Any, ws: Any) -> None:
        await ws.accept()
        await ws.close()

    ws = FakeWebSocket()
    await asyncio.wait_for(_make(_instant_close, FakeClock())(None, ws), timeout=5)

    assert ws.timeline[0] == ("accept", None)
    assert _timings(ws) == [(1, 0.0)]
    assert ws.timeline[-1] == ("close", None)


@pytest.mark.asyncio
async def test_no_billing_before_mark_billing_started() -> None:
    # Connect + signaling time is free: only the zero frame is emitted when
    # the session never negotiates (no ``answer`` -> no clock start).
    clock = FakeClock()

    async def _never_negotiates(self: Any, ws: Any) -> None:
        await ws.accept()
        await ws.send_bytes(b"ready")
        clock.advance(120.0)
        await ws.send_bytes(b"iceServers")
        await ws.close()

    ws = FakeWebSocket()
    await asyncio.wait_for(_make(_never_negotiates, clock)(None, ws), timeout=5)

    assert [t for _, t in _timings(ws)] == [0.0]


@pytest.mark.asyncio
async def test_session_duration_is_billed_and_residual_flushed_before_close() -> None:
    clock = FakeClock()

    async def _sdk_like(self: Any, ws: Any) -> None:
        await ws.accept()
        clock.advance(2.0)  # signaling — free
        mark_billing_started()
        clock.advance(4.0)
        await ws.send_bytes(b"icecandidate")  # flushes 4s before the event
        clock.advance(6.0)  # no outgoing events
        await ws.close()  # flushes the 6s residual before the close handshake

    ws = FakeWebSocket()
    await asyncio.wait_for(_make(_sdk_like, clock)(None, ws), timeout=5)

    timings = _timings(ws)
    close_idx = _index_of(ws, "close")
    assert all(idx < close_idx for idx, _ in timings)
    assert ws.timeline[-1] == ("close", None)
    assert [t for _, t in timings] == pytest.approx([0.0, 4.0, 6.0])
    # Total billed == post-answer wall clock; connect/signaling time excluded.
    assert sum(t for _, t in timings) == pytest.approx(10.0)
    # The 4s frame precedes the event it bills up to.
    assert timings[1][0] < _index_of(ws, "bytes")


@pytest.mark.asyncio
async def test_mark_billing_started_is_idempotent() -> None:
    clock = FakeClock()

    async def _renegotiates(self: Any, ws: Any) -> None:
        await ws.accept()
        mark_billing_started()
        clock.advance(5.0)
        mark_billing_started()  # second ``answer`` must not reset the clock
        clock.advance(5.0)
        await ws.close()

    ws = FakeWebSocket()
    await asyncio.wait_for(_make(_renegotiates, clock)(None, ws), timeout=5)
    assert sum(t for _, t in _timings(ws)) == pytest.approx(10.0)


@pytest.mark.asyncio
async def test_backstop_flushes_when_sdk_never_closes() -> None:
    clock = FakeClock()

    async def _crashes(self: Any, ws: Any) -> None:
        await ws.accept()
        mark_billing_started()
        clock.advance(7.0)
        raise RuntimeError("boom")

    ws = FakeWebSocket()
    with pytest.raises(RuntimeError, match="boom"):
        await asyncio.wait_for(_make(_crashes, clock)(None, ws), timeout=5)

    assert not ws.closed
    assert [t for _, t in _timings(ws)] == pytest.approx([0.0, 7.0])


@pytest.mark.asyncio
async def test_periodic_tick_bounds_loss_when_client_vanishes() -> None:
    # A client that drops without a close handshake still gets billed up to
    # the last periodic tick.
    clock = FakeClock()
    ticked = asyncio.Event()

    async def _idle_forever(self: Any, ws: Any) -> None:
        await ws.accept()
        mark_billing_started()
        clock.advance(9.0)
        await ticked.wait()
        await ws.close()

    class TickingWebSocket(FakeWebSocket):
        async def send_text(self, data: str) -> None:
            await super().send_text(data)
            if is_timing_frame(data) and json.loads(data)["timing"] > 0:
                ticked.set()

    ws = TickingWebSocket()
    handler = make_duration_billed_handler(
        _idle_forever, label="test", clock=clock, update_interval=0.01
    )
    await asyncio.wait_for(handler(None, ws), timeout=5)

    timings = [t for _, t in _timings(ws)]
    assert timings[0] == 0.0
    assert timings[1] == pytest.approx(9.0)  # emitted by the ticker, not close
    assert sum(timings) == pytest.approx(9.0)  # close found nothing left


@pytest.mark.asyncio
async def test_sdk_auto_timings_dropped_but_other_text_relayed() -> None:
    clock = FakeClock()

    async def _emits_sdk_timing(self: Any, ws: Any) -> None:
        await ws.accept()
        mark_billing_started()
        clock.advance(3.0)
        await ws.send_text(timing_frame(123.0))  # SDK per-yield timing
        await ws.send_text('{"type":"x-fal-error","error":"TIMEOUT"}')
        await ws.close()

    ws = FakeWebSocket()
    await asyncio.wait_for(_make(_emits_sdk_timing, clock)(None, ws), timeout=5)

    assert sum(t for _, t in _timings(ws)) == pytest.approx(3.0)
    assert ("text", '{"type":"x-fal-error","error":"TIMEOUT"}') in ws.timeline


@pytest.mark.asyncio
async def test_billing_state_is_isolated_between_concurrent_sessions() -> None:
    clock = FakeClock()
    a_started = asyncio.Event()
    b_done = asyncio.Event()

    async def _session_a(self: Any, ws: Any) -> None:
        await ws.accept()
        mark_billing_started()
        a_started.set()
        await b_done.wait()
        clock.advance(2.0)
        await ws.close()

    async def _session_b(self: Any, ws: Any) -> None:
        await a_started.wait()
        await ws.accept()
        # Never negotiates: must not inherit A's running clock.
        clock.advance(50.0)
        await ws.close()
        b_done.set()

    ws_a, ws_b = FakeWebSocket(), FakeWebSocket()
    await asyncio.wait_for(
        asyncio.gather(
            _make(_session_a, clock)(None, ws_a),
            _make(_session_b, clock)(None, ws_b),
        ),
        timeout=5,
    )

    assert sum(t for _, t in _timings(ws_a)) == pytest.approx(52.0)
    assert [t for _, t in _timings(ws_b)] == [0.0]


@pytest.mark.asyncio
async def test_watermark_advances_only_after_successful_send() -> None:
    clock = FakeClock()

    class FlakyWebSocket(FakeWebSocket):
        fail_next = False

        async def send_text(self, data: str) -> None:
            if self.fail_next and is_timing_frame(data):
                self.fail_next = False
                raise ConnectionError("transient")
            await super().send_text(data)

    async def _handler(self: Any, ws: Any) -> None:
        await ws.accept()
        mark_billing_started()
        clock.advance(4.0)
        ws.fail_next = True
        await ws.send_bytes(b"evt")  # timing send fails; slice must survive
        clock.advance(1.0)
        await ws.close()  # flush covers the full 5s

    ws = FlakyWebSocket()
    await asyncio.wait_for(_make(_handler, clock)(None, ws), timeout=5)
    assert [t for _, t in _timings(ws)] == pytest.approx([0.0, 5.0])


def test_mark_billing_started_outside_handler_is_noop() -> None:
    mark_billing_started()  # no ContextVar binding -> nothing to do


def test_mark_session_failed_outside_handler_is_noop() -> None:
    mark_session_failed()  # no ContextVar binding -> nothing to do


# Failed / too-short sessions bill exactly 0: the gateway rounds any positive
# sub-second sum up to 0.25s and frames can't be retracted, so a rejected session
# must never emit a positive frame (prod 01a0a0d2-e5d4-7a02-a79a-40c2095cd87d).


@pytest.mark.asyncio
async def test_partner_rejection_right_after_answer_bills_zero() -> None:
    clock = FakeClock()

    async def _rejected(self: Any, ws: Any) -> None:
        await ws.accept()
        clock.advance(1.2)  # connect + offer: free
        mark_billing_started()  # Decart's answer
        clock.advance(0.01)
        await ws.send_bytes(b"answer")  # pre-send flush must NOT emit ~10ms
        clock.advance(0.34)
        mark_session_failed()  # upstream closed 1013 "Session Limit Reached"
        await ws.send_bytes(b"error-event")  # nor before the error event
        clock.advance(6.8)  # relay/SDK teardown lag before the close
        await ws.close()  # nor the residual at close

    ws = FakeWebSocket()
    await asyncio.wait_for(_make(_rejected, clock)(None, ws), timeout=5)

    # Only the zero frame: the gateway's sum is exactly 0 -> 0 units, not 0.25.
    assert [t for _, t in _timings(ws)] == [0.0]
    # Application traffic still relayed.
    assert ("bytes", b"answer") in ws.timeline
    assert ("bytes", b"error-event") in ws.timeline
    assert ws.timeline[-1] == ("close", None)


@pytest.mark.asyncio
async def test_session_shorter_than_grace_period_bills_zero() -> None:
    # No failure signal at all (e.g. the browser leaves immediately): the grace
    # hold alone keeps the sum at 0 instead of 0.25 for a blink of a session.
    clock = FakeClock()

    async def _blink(self: Any, ws: Any) -> None:
        await ws.accept()
        mark_billing_started()
        clock.advance(DEFAULT_GRACE_PERIOD_SECONDS / 2)
        await ws.send_bytes(b"icecandidate")
        clock.advance(DEFAULT_GRACE_PERIOD_SECONDS / 4)
        await ws.close()

    ws = FakeWebSocket()
    await asyncio.wait_for(_make(_blink, clock)(None, ws), timeout=5)
    assert [t for _, t in _timings(ws)] == [0.0]


@pytest.mark.asyncio
async def test_first_flush_after_grace_covers_time_since_clock_start() -> None:
    # A real session loses nothing to the hold: the first frame past the grace
    # period bills everything since the answer, including the held part.
    clock = FakeClock()
    grace = DEFAULT_GRACE_PERIOD_SECONDS

    async def _real(self: Any, ws: Any) -> None:
        await ws.accept()
        mark_billing_started()
        clock.advance(grace * 0.9)
        await ws.send_bytes(b"icecandidate")  # inside grace: held
        clock.advance(grace * 0.6)
        await ws.send_bytes(b"icecandidate")  # past grace: flushes 1.5 * grace
        clock.advance(3.0)
        await ws.close()

    ws = FakeWebSocket()
    await asyncio.wait_for(_make(_real, clock)(None, ws), timeout=5)

    timings = [t for _, t in _timings(ws)]
    assert timings == pytest.approx([0.0, grace * 1.5, 3.0])
    assert sum(timings) == pytest.approx(grace * 1.5 + 3.0)


@pytest.mark.asyncio
async def test_grace_period_zero_disables_hold() -> None:
    clock = FakeClock()

    async def _short(self: Any, ws: Any) -> None:
        await ws.accept()
        mark_billing_started()
        clock.advance(0.3)
        await ws.close()

    ws = FakeWebSocket()
    await asyncio.wait_for(_make(_short, clock, grace_period=0.0)(None, ws), timeout=5)
    assert [t for _, t in _timings(ws)] == pytest.approx([0.0, 0.3])


@pytest.mark.asyncio
async def test_failure_after_billing_began_freezes_at_last_flush() -> None:
    # Mid-session partner failure: what already reached the gateway stands,
    # nothing more is added — not at the error event, the close, or the backstop.
    clock = FakeClock()
    logs: List[str] = []

    async def _drops_midway(self: Any, ws: Any) -> None:
        await ws.accept()
        mark_billing_started()
        clock.advance(4.0)
        await ws.send_bytes(b"icecandidate")  # flushes 4s
        clock.advance(2.5)
        mark_session_failed()  # upstream error; 2.5s accrued since last flush
        await ws.send_bytes(b"error-event")
        clock.advance(1.0)
        await ws.close()

    ws = FakeWebSocket()
    handler = make_duration_billed_handler(
        _drops_midway,
        label="test",
        clock=clock,
        update_interval=3600,
        log=logs.append,
    )
    await asyncio.wait_for(handler(None, ws), timeout=5)

    assert [t for _, t in _timings(ws)] == pytest.approx([0.0, 4.0])
    assert any("session failed" in line and "4.00s billed" in line for line in logs)


@pytest.mark.asyncio
async def test_failure_before_answer_prevents_clock_from_starting() -> None:
    # Connect/init failure surfaces before any answer; a later (stray)
    # mark_billing_started must not start a ticker or bill anything.
    clock = FakeClock()

    async def _connect_failed(self: Any, ws: Any) -> None:
        await ws.accept()
        mark_session_failed()  # upstream failed before any answer
        await ws.send_bytes(b"error-event")
        mark_billing_started()
        clock.advance(30.0)
        await ws.close()

    ws = FakeWebSocket()
    await asyncio.wait_for(_make(_connect_failed, clock)(None, ws), timeout=5)
    assert [t for _, t in _timings(ws)] == [0.0]


@pytest.mark.asyncio
async def test_mark_session_failed_is_idempotent_and_logged_once() -> None:
    clock = FakeClock()
    logs: List[str] = []

    async def _double_fail(self: Any, ws: Any) -> None:
        await ws.accept()
        mark_billing_started()
        clock.advance(0.5)
        mark_session_failed()
        mark_session_failed()
        await ws.close()

    ws = FakeWebSocket()
    handler = make_duration_billed_handler(
        _double_fail, label="test", clock=clock, update_interval=3600, log=logs.append
    )
    await asyncio.wait_for(handler(None, ws), timeout=5)
    assert [t for _, t in _timings(ws)] == [0.0]
    assert sum("session failed" in line for line in logs) == 1


@pytest.mark.asyncio
async def test_failure_state_is_isolated_between_concurrent_sessions() -> None:
    clock = FakeClock()
    a_failed = asyncio.Event()
    b_done = asyncio.Event()

    async def _session_a(self: Any, ws: Any) -> None:
        await ws.accept()
        mark_billing_started()
        clock.advance(0.2)
        mark_session_failed()
        a_failed.set()
        await b_done.wait()
        await ws.close()

    async def _session_b(self: Any, ws: Any) -> None:
        await a_failed.wait()
        await ws.accept()
        mark_billing_started()  # must not see A's failure
        clock.advance(10.0)
        await ws.close()
        b_done.set()

    ws_a, ws_b = FakeWebSocket(), FakeWebSocket()
    await asyncio.wait_for(
        asyncio.gather(
            _make(_session_a, clock)(None, ws_a),
            _make(_session_b, clock)(None, ws_b),
        ),
        timeout=5,
    )
    assert [t for _, t in _timings(ws_a)] == [0.0]
    assert sum(t for _, t in _timings(ws_b)) == pytest.approx(10.0)


class _FakeRouteSignature:
    def __init__(self) -> None:
        self.emit_timings = False
        self.path = "/realtime"

    def _replace(self, **kwargs: Any) -> "_FakeRouteSignature":
        clone = _FakeRouteSignature()
        clone.__dict__.update(self.__dict__)
        clone.__dict__.update(kwargs)
        return clone


def test_install_duration_billing_preserves_sdk_metadata() -> None:
    async def realtime(self: Any, websocket: Any) -> None:  # pragma: no cover
        pass

    realtime.route_signature = _FakeRouteSignature()  # type: ignore[attr-defined]
    realtime.original_func = object()  # type: ignore[attr-defined]
    realtime.__annotations__ = {"websocket": object, "return": None}

    class App:
        pass

    App.realtime = realtime  # type: ignore[attr-defined]
    install_duration_billing(App, "realtime", label="test")

    wrapped = App.realtime  # type: ignore[attr-defined]
    assert wrapped is not realtime
    assert wrapped.route_signature.emit_timings is True
    assert wrapped.__name__ == "realtime"
    assert wrapped.route_signature.path == "/realtime"
    assert wrapped.original_func is realtime.original_func  # type: ignore[attr-defined]
    assert wrapped.__annotations__ == {"websocket": object, "return": None}


def test_install_duration_billing_rejects_non_realtime_handler() -> None:
    class App:
        async def plain(self, websocket: Any) -> None:  # pragma: no cover
            pass

    with pytest.raises(TypeError, match="route_signature"):
        install_duration_billing(App, "plain", label="test")


@pytest.mark.asyncio
async def test_text_event_flushes_billing_before_abrupt_disconnect() -> None:
    clock = FakeClock()
    payload = '{"type":"answer","sdp":"v=0"}'

    async def handler(self: Any, ws: Any) -> None:
        await ws.accept()
        mark_billing_started()
        clock.advance(7.0)
        await ws.send_text(payload)
        ws.closed = True  # no final frame can reach the gateway

    ws = FakeWebSocket()
    await asyncio.wait_for(_make(handler, clock)(None, ws), timeout=5)
    assert [t for _, t in _timings(ws)] == pytest.approx([0.0, 7.0])
    assert ws.timeline[-1] == ("text", payload)


@pytest.mark.parametrize(
    "name,values",
    [
        ("update_interval", [0, -1, float("nan"), float("inf"), -float("inf"), True]),
        ("grace_period", [-1, float("nan"), float("inf"), -float("inf"), True]),
    ],
)
def test_invalid_timing_configuration_rejected_before_wrapping(name, values):
    async def inner(self, websocket):
        pass

    for value in values:
        with pytest.raises(ValueError, match=name):
            make_duration_billed_handler(inner, label="test", **{name: value})
