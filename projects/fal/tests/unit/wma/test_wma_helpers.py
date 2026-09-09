"""Unit tests for the raw-path WMA helpers (``fal/wma/_raw.py``).

Everything under test is importable in the repo venv: the aiortc/av-dependent
pieces (``make_video_queue_track``) are exercised behind ``importorskip`` —
aiortc/av are dev dependencies here, runner-only packages in production.
"""

from __future__ import annotations

import asyncio
import json
import time

import pytest

from fal.wma import (
    CONNECTION_REPORT_VERSION,
    INITIAL_CONNECT_TIMEOUT_SECONDS,
    SSE_KEEPALIVE,
    SessionSlot,
    close_peer_connection,
    filter_sdp_ice_candidates,
    make_video_queue_track,
    queue_put_drop_oldest,
    sse_event,
    wait_for_initial_connect,
    watch_connection_state,
    wma_session_stream,
)


class TestSse:
    def test_event_format(self):
        event = sse_event({"type": "answer", "sdp": "v=0"})
        assert event.startswith("data: ")
        assert event.endswith("\n\n")
        assert json.loads(event[len("data: ") :]) == {"type": "answer", "sdp": "v=0"}

    def test_named_event_format_and_name_validation(self):
        event = sse_event({"version": 1}, event="connection_report")
        assert event == 'event: connection_report\ndata: {"version": 1}\n\n'
        with pytest.raises(ValueError, match="newlines"):
            sse_event({}, event="connection_report\nevent: injected")

    def test_keepalive_is_a_comment(self):
        assert SSE_KEEPALIVE.startswith(":")


class TestFilterSdpIceCandidates:
    def test_strips_private_loopback_and_link_local_candidates(self):
        sdp = (
            "v=0\r\n"
            "a=candidate:1 1 UDP 2130706431 192.168.1.5 51000 typ host\r\n"
            "a=candidate:2 1 UDP 2130706431 169.254.169.254 51000 typ host\r\n"
            "a=candidate:3 1 UDP 2130706431 127.0.0.1 51000 typ host\r\n"
            "a=candidate:4 1 UDP 2130706431 224.0.0.1 51000 typ host\r\n"
        )
        filtered = filter_sdp_ice_candidates(sdp)
        assert "a=candidate:" not in filtered
        assert filtered.startswith("v=0\r\n")

    def test_keeps_public_candidates_and_non_candidate_lines(self):
        sdp = (
            "v=0\r\n"
            "a=candidate:1 1 UDP 1694498815 8.8.8.8 51000 typ srflx\r\n"
            "a=mid:0\r\n"
        )
        assert filter_sdp_ice_candidates(sdp) == sdp

    def test_passes_through_mdns_hostname_candidates(self):
        # mDNS hostnames (RFC 8445 6.2) are the only legitimate non-IP candidate
        # shape — left alone rather than dropped.
        sdp = "a=candidate:1 1 UDP 2130706431 abcd1234.local 51000 typ host\r\n"
        assert filter_sdp_ice_candidates(sdp) == sdp

    def test_strips_non_mdns_hostname_candidates(self):
        # Any other hostname could be attacker-controlled DNS crafted to
        # resolve to an internal address (DNS rebinding) — reject it.
        sdp = (
            "v=0\r\n"
            "a=candidate:1 1 UDP 2130706431 169-254-169-254.attacker.com"
            " 51000 typ host\r\n"
        )
        filtered = filter_sdp_ice_candidates(sdp)
        assert "a=candidate:" not in filtered
        assert filtered == "v=0\r\n"

    def test_strips_cgnat_candidates(self):
        # RFC 6598 shared-address/CGNAT range (100.64.0.0/10): plain
        # ipaddress predicates (is_private/is_reserved/...) don't flag it,
        # but it's still an internal-network SSRF target — the shared
        # is_globally_routable_ip helper this filter delegates to does
        # reject it.
        sdp = "a=candidate:1 1 UDP 2130706431 100.64.1.1 51000 typ host\r\n"
        assert "a=candidate:" not in filter_sdp_ice_candidates(sdp)


class _FakePeerConnection:
    """Minimal stand-in for aiortc's ``@pc.on("connectionstatechange")`` API."""

    def __init__(self) -> None:
        self.connectionState = "new"
        self._handlers: dict[str, list] = {}

    def on(self, event: str):
        def register(handler):
            self._handlers.setdefault(event, []).append(handler)
            return handler

        return register

    def set_state(self, state: str) -> None:
        self.connectionState = state
        for handler in self._handlers.get("connectionstatechange", []):
            handler()


class TestWatchConnectionState:
    def test_disconnected_is_not_terminal(self):
        pc = _FakePeerConnection()
        closed = asyncio.Event()
        watch_connection_state(pc, closed)

        pc.set_state("disconnected")
        assert not closed.is_set()  # transient ICE blip, may still recover

    def test_failed_and_closed_are_terminal(self):
        for terminal_state in ("failed", "closed"):
            pc = _FakePeerConnection()
            closed = asyncio.Event()
            watch_connection_state(pc, closed)

            pc.set_state(terminal_state)
            assert closed.is_set()

    def test_connected_sets_the_connected_event(self):
        pc = _FakePeerConnection()
        closed = asyncio.Event()
        connected = asyncio.Event()
        watch_connection_state(pc, closed, connected)

        pc.set_state("connected")
        assert connected.is_set()
        assert not closed.is_set()

    @pytest.mark.allow_real_sleep
    def test_disconnected_grace_recovers_without_closing(self):
        """A ``disconnected`` blip that recovers to ``connected`` before the grace
        elapses must NOT be treated as terminal (no ``closed``)."""

        async def scenario() -> bool:
            pc = _FakePeerConnection()
            closed = asyncio.Event()
            connected = asyncio.Event()
            watch_connection_state(pc, closed, connected, disconnected_grace=0.15)
            pc.set_state("disconnected")
            await asyncio.sleep(0.03)  # well under the grace
            pc.set_state("connected")  # recovered
            await asyncio.sleep(0.25)  # past when the grace would have fired
            return closed.is_set()

        assert asyncio.run(scenario()) is False

    @pytest.mark.allow_real_sleep
    def test_disconnected_grace_closes_when_sustained(self):
        """A ``disconnected`` that persists past the grace window becomes
        terminal so the runner slot is eventually freed."""

        async def scenario() -> bool:
            pc = _FakePeerConnection()
            closed = asyncio.Event()
            connected = asyncio.Event()
            watch_connection_state(pc, closed, connected, disconnected_grace=0.1)
            pc.set_state("disconnected")
            await asyncio.sleep(0.25)  # never recovers
            return closed.is_set()

        assert asyncio.run(scenario()) is True

    @pytest.mark.allow_real_sleep
    @pytest.mark.parametrize("recovery_state", ["connected", "connecting"])
    def test_disconnected_grace_restarts_after_recovery(self, recovery_state):
        """A later disconnect gets a fresh grace timer after recovery."""

        async def scenario() -> bool:
            pc = _FakePeerConnection()
            closed = asyncio.Event()
            connected = asyncio.Event()
            watch_connection_state(pc, closed, connected, disconnected_grace=0.05)
            pc.set_state("disconnected")
            await asyncio.sleep(0.01)
            pc.set_state(recovery_state)
            await asyncio.sleep(0.08)
            pc.set_state("disconnected")
            await asyncio.sleep(0.08)
            return closed.is_set()

        assert asyncio.run(scenario()) is True

    @pytest.mark.allow_real_sleep
    def test_disconnected_grace_none_is_never_terminal(self):
        """Default (no grace) keeps the legacy contract: ``disconnected`` is a
        blip and never sets ``closed`` on its own."""

        async def scenario() -> bool:
            pc = _FakePeerConnection()
            closed = asyncio.Event()
            watch_connection_state(pc, closed, disconnected_grace=None)
            pc.set_state("disconnected")
            await asyncio.sleep(0.2)
            return closed.is_set()

        assert asyncio.run(scenario()) is False

    def test_terminal_before_grace_still_closes(self):
        """disconnected -> failed cancels the grace timer but still closes."""

        async def scenario() -> bool:
            pc = _FakePeerConnection()
            closed = asyncio.Event()
            connected = asyncio.Event()
            watch_connection_state(pc, closed, connected, disconnected_grace=5.0)
            pc.set_state("disconnected")
            await asyncio.sleep(0.02)
            pc.set_state("failed")
            return closed.is_set()

        assert asyncio.run(scenario()) is True


class TestWaitForInitialConnect:
    """The bounded initial-ICE/connect gate that frees a failed session's slot
    instead of holding it for the full ``MAX_SESSION_SECONDS`` (the reported
    root cause: a client on a restrictive network never completes ICE, so every
    reconnect gets a 503 for up to 10 minutes)."""

    def test_returns_true_when_connected(self):
        async def scenario() -> tuple[bool, float]:
            connected = asyncio.Event()
            closed = asyncio.Event()
            asyncio.get_event_loop().call_later(0.02, connected.set)
            t0 = time.monotonic()
            ok = await wait_for_initial_connect(connected, closed, timeout=1.0)
            return ok, time.monotonic() - t0

        ok, dt = asyncio.run(scenario())
        assert ok is True
        assert dt < 0.5  # returned promptly on connect, not at the timeout

    def test_returns_false_fast_when_closed_first(self):
        """An ICE ``failed`` sets ``closed``; the gate must return immediately
        rather than sitting out the whole timeout (the connected-only bug)."""

        async def scenario() -> tuple[bool, float]:
            connected = asyncio.Event()
            closed = asyncio.Event()
            asyncio.get_event_loop().call_later(0.02, closed.set)
            t0 = time.monotonic()
            ok = await wait_for_initial_connect(connected, closed, timeout=2.0)
            return ok, time.monotonic() - t0

        ok, dt = asyncio.run(scenario())
        assert ok is False
        assert dt < 0.5  # freed via `closed`, not after the 2s timeout

    def test_returns_false_on_timeout_when_never_connects(self):
        async def scenario() -> tuple[bool, float]:
            connected = asyncio.Event()
            closed = asyncio.Event()
            t0 = time.monotonic()
            ok = await wait_for_initial_connect(connected, closed, timeout=0.2)
            return ok, time.monotonic() - t0

        ok, dt = asyncio.run(scenario())
        assert ok is False
        assert 0.15 < dt < 1.0  # bounded by the timeout, not MAX_SESSION_SECONDS

    def test_default_timeout_is_the_shared_bound(self):
        # Allows a cold first-time TURN relay establishment on a restrictive
        # network while still bounding resource retention on failed handshakes.
        assert INITIAL_CONNECT_TIMEOUT_SECONDS == 35.0


class TestSlotReleaseOnFailedConnect:
    """End-to-end proof of the fix at the producer level: a session whose ICE
    never connects must free its ``SessionSlot`` within the initial-connect
    bound so the next offer is admitted (not 503'd for MAX_SESSION_SECONDS)."""

    def test_slot_freed_within_initial_connect_bound(self):
        async def scenario() -> tuple[float, bool]:
            slot = SessionSlot()
            assert slot.try_acquire() is True  # session 1 holds the slot

            pc = _FakePeerConnection()
            closed = asyncio.Event()
            connected = asyncio.Event()
            watch_connection_state(pc, closed, connected, disconnected_grace=0.5)

            # Mirror produce_frames' pre-connect gate + finally teardown with a
            # short bound standing in for INITIAL_CONNECT_TIMEOUT_SECONDS. The PC
            # is left stuck in "checking" (never connected, never failed).
            async def producer() -> None:
                try:
                    if not await wait_for_initial_connect(
                        connected, closed, timeout=0.2
                    ):
                        return
                finally:
                    closed.set()
                    slot.release()

            t0 = time.monotonic()
            await producer()
            freed_at = time.monotonic() - t0
            # A second offer is now admitted (the 503 window is bounded).
            readmitted = slot.try_acquire()
            return freed_at, readmitted

        freed_at, readmitted = asyncio.run(scenario())
        assert freed_at < 1.0
        assert readmitted is True


class TestSessionSlot:
    def test_single_acquire(self):
        slot = SessionSlot()
        assert slot.try_acquire() is True
        assert slot.try_acquire() is False
        slot.release()
        assert slot.try_acquire() is True


class TestQueuePutDropOldest:
    def test_drops_oldest_when_full(self):
        async def scenario() -> list[int]:
            queue: asyncio.Queue[int] = asyncio.Queue(maxsize=2)
            for item in (1, 2, 3):
                queue_put_drop_oldest(queue, item)
            return [queue.get_nowait(), queue.get_nowait()]

        assert asyncio.run(scenario()) == [2, 3]


class TestMakeVideoQueueTrack:
    """Latency instrumentation of the outbound track (aiortc/av required)."""

    def _frame(self):
        import numpy as np

        return np.zeros((4, 4, 3), dtype=np.uint8)

    def test_bare_and_tuple_items_both_emit_frames(self):
        pytest.importorskip("aiortc")

        async def scenario():
            queue: asyncio.Queue = asyncio.Queue(maxsize=4)
            track = make_video_queue_track(queue, fps=1000.0)
            queue.put_nowait(self._frame())
            queue.put_nowait((time.monotonic(), self._frame()))
            first = await track.recv()
            second = await track.recv()
            return first, second

        first, second = asyncio.run(scenario())
        assert (first.width, first.height) == (4, 4)
        assert (second.width, second.height) == (4, 4)
        assert second.pts > first.pts

    def test_stats_record_queue_age_and_pace_sleep(self):
        pytest.importorskip("aiortc")

        async def scenario():
            queue: asyncio.Queue = asyncio.Queue(maxsize=4)
            stats: dict = {}
            track = make_video_queue_track(queue, fps=1000.0, stats=stats)
            # A tuple item records its push->pop queue age; a bare item does
            # not (there is no timestamp to age against).
            queue.put_nowait((time.monotonic() - 0.05, self._frame()))
            queue.put_nowait(self._frame())
            await track.recv()
            await track.recv()
            return stats

        stats = asyncio.run(scenario())
        assert len(stats["queue_age_ms"]) == 1
        assert stats["queue_age_ms"][0] >= 50.0
        # The second frame goes through the pacing branch (first frame sets
        # the clock origin); an entry is appended whether or not it slept.
        assert len(stats["pace_sleep_ms"]) == 1
        assert stats["pace_sleep_ms"][0] >= 0.0


class TestWmaSessionStream:
    def test_answer_first_then_cleanup_on_close(self):
        async def scenario() -> tuple[list[str], bool]:
            closed = asyncio.Event()
            cleaned = False

            async def cleanup() -> None:
                nonlocal cleaned
                cleaned = True

            stream = wma_session_stream(
                {"type": "answer", "sdp": "v=0"}, closed, cleanup, 0.01
            )
            events = [await stream.__anext__()]
            closed.set()
            async for chunk in stream:
                events.append(chunk)
            return events, cleaned

        events, cleaned = asyncio.run(scenario())
        assert json.loads(events[0][len("data: ") :])["type"] == "answer"
        assert cleaned is True

    def test_keepalives_flow_until_closed(self):
        async def scenario() -> list[str]:
            closed = asyncio.Event()

            async def cleanup() -> None:
                pass

            stream = wma_session_stream({"sdp": "v=0"}, closed, cleanup, 0.01)
            events = [await stream.__anext__(), await stream.__anext__()]
            closed.set()
            async for chunk in stream:
                events.append(chunk)
            return events

        events = asyncio.run(scenario())
        assert SSE_KEEPALIVE in events[1:]

    def test_advertises_and_emits_one_connection_report(self):
        async def scenario() -> tuple[list[str], bool]:
            closed = asyncio.Event()
            report_ready = asyncio.Event()
            cleaned = False

            async def connection_report():
                await report_ready.wait()
                return {
                    "version": CONNECTION_REPORT_VERSION,
                    "runner_candidate": "srflx",
                    "browser_candidate": "relay",
                    "ice_protocol": "udp",
                    "setup_ms": 842,
                }

            async def cleanup() -> None:
                nonlocal cleaned
                cleaned = True

            stream = wma_session_stream(
                {"type": "answer", "sdp": "v=0"},
                closed,
                cleanup,
                10,
                connection_report=connection_report,
            )
            events = [await stream.__anext__()]
            report_ready.set()
            events.append(await asyncio.wait_for(stream.__anext__(), timeout=0.1))
            closed.set()
            async for chunk in stream:
                events.append(chunk)
            return events, cleaned

        events, cleaned = asyncio.run(scenario())
        answer = json.loads(events[0][len("data: ") :])
        assert answer["connection_report_version"] == CONNECTION_REPORT_VERSION
        assert events[1].startswith("event: connection_report\ndata: ")
        assert json.loads(events[1].split("data: ", 1)[1])["setup_ms"] == 842
        assert events.count(events[1]) == 1
        assert cleaned is True

    def test_connection_report_failure_does_not_end_the_stream(self):
        async def scenario() -> str:
            closed = asyncio.Event()

            async def connection_report():
                raise RuntimeError("telemetry unavailable")

            async def cleanup() -> None:
                pass

            stream = wma_session_stream(
                {"sdp": "v=0"},
                closed,
                cleanup,
                0.01,
                connection_report=connection_report,
            )
            await stream.__anext__()
            next_event = await stream.__anext__()
            await stream.aclose()
            return next_event

        assert asyncio.run(scenario()) == SSE_KEEPALIVE

    def test_invalid_connection_report_does_not_end_the_stream(self):
        async def scenario() -> str:
            closed = asyncio.Event()

            async def connection_report():
                return {"not_json": object()}

            async def cleanup() -> None:
                pass

            stream = wma_session_stream(
                {"sdp": "v=0"},
                closed,
                cleanup,
                0.01,
                connection_report=connection_report,
            )
            await stream.__anext__()
            next_event = await stream.__anext__()
            await stream.aclose()
            return next_event

        assert asyncio.run(scenario()) == SSE_KEEPALIVE

    @pytest.mark.parametrize("mode", ["raises", "not_awaitable"])
    def test_connection_report_startup_failure_still_cleans_up(self, mode):
        async def scenario() -> tuple[dict, bool]:
            closed = asyncio.Event()
            cleaned = False

            def connection_report():
                if mode == "raises":
                    raise RuntimeError("telemetry unavailable")
                return {}

            async def cleanup() -> None:
                nonlocal cleaned
                cleaned = True

            stream = wma_session_stream(
                {"sdp": "v=0"},
                closed,
                cleanup,
                0.01,
                connection_report=connection_report,
            )
            first = json.loads((await stream.__anext__())[len("data: ") :])
            await stream.aclose()
            return first, cleaned

        answer, cleaned = asyncio.run(scenario())
        assert "connection_report_version" not in answer
        assert cleaned is True


class _CloseCountingPC:
    """Stand-in aiortc peer connection recording how many times close() ran and
    optionally raising to prove the helper swallows a teardown-time error."""

    def __init__(self, raise_on_close: bool = False) -> None:
        self.close_calls = 0
        self._raise = raise_on_close

    async def close(self) -> None:
        self.close_calls += 1
        if self._raise:
            raise RuntimeError("simulated aiortc teardown error")


class TestClosePeerConnection:
    """The shared idempotent teardown used by raw-path WMA apps."""

    def test_awaits_close(self):
        pc = _CloseCountingPC()
        asyncio.run(close_peer_connection(pc))
        assert pc.close_calls == 1

    def test_double_close_is_safe(self):
        pc = _CloseCountingPC()

        async def scenario() -> None:
            await close_peer_connection(pc)
            await close_peer_connection(pc)  # negotiation-fail path + cleanup both fire

        asyncio.run(scenario())
        assert pc.close_calls == 2  # both ran, neither raised

    def test_teardown_error_is_swallowed(self):
        """On the negotiation-failure path an unguarded raise here would mask the
        mapped client-facing error, so the helper must never propagate."""
        pc = _CloseCountingPC(raise_on_close=True)
        # Must not raise.
        asyncio.run(close_peer_connection(pc))
        assert pc.close_calls == 1


class TestAioiceOrderlyTeardown:
    """Pins both the aioice STUN teardown race and the orderly-teardown shim
    (:mod:`fal.wma._aioice_teardown`) that fixes it — against the *real*
    ``aioice.ice.Connection``, not a stand-in.

    The race: stock ``Connection.close()`` closes the datagram transports without
    cancelling in-flight ICE check tasks, so a still-scheduled
    ``stun.Transaction.__retry`` timer fires ``send_stun`` -> ``transport.sendto``
    on a transport whose ``_sock`` is already ``None`` (host-candidate transports
    have no ``remote_addr``, so asyncio's ``_conn_lost`` fast-return is skipped),
    which asyncio logs as ``Exception in callback Transaction.__retry()``.

    The shim cancels and awaits those check tasks before ``close()`` closes the
    transports, so every retransmit timer is cancelled first.

    If a future aioice teaches ``close()`` to cancel check tasks itself, the
    unpatched test below flips (canary for an upstream fix) and the shim's source
    guard makes ``install_orderly_ice_teardown`` decline to wrap.
    """

    @staticmethod
    async def _build_leaked_connection(loop, stun):
        """A real Connection with one StunProtocol and one in-flight check task
        whose transaction has a live retransmit timer — the exact state stock
        ``close()`` leaves untouched when a triggered check outlived connect()'s
        cancel sweep."""
        pytest.importorskip("aioice")
        from aioice import ice

        conn = ice.Connection(ice_controlling=True)
        # Host-candidate shape: local_addr only, NO remote_addr.
        _transport, protocol = await loop.create_datagram_endpoint(
            lambda: ice.StunProtocol(conn), local_addr=("127.0.0.1", 0)
        )
        conn._protocols.append(protocol)

        msg = stun.Message(
            message_method=stun.Method.BINDING, message_class=stun.Class.REQUEST
        )
        # Target an unrouted TEST-NET-1 address so no response ever arrives.
        task = asyncio.ensure_future(
            protocol.request(msg, ("192.0.2.1", 3478), retransmissions=6)
        )
        cand = ice.Candidate(
            foundation="x",
            component=1,
            transport="udp",
            priority=1,
            host="192.0.2.1",
            port=3478,
            type="host",
        )
        pair = ice.CandidatePair(protocol, cand)
        pair.task = task
        conn._check_list.append(pair)

        await asyncio.sleep(0.02)  # first send happens, retry timer scheduled
        return conn

    def _run_close_scenario(self, install_shim: bool) -> list[str]:
        pytest.importorskip("aioice")
        pytest.importorskip("aioice")
        from aioice import ice, stun

        original_close = ice.Connection.close
        original_rto = stun.RETRY_RTO
        stun.RETRY_RTO = 0.05  # fire the retry quickly

        async def scenario() -> list[str]:
            loop = asyncio.get_running_loop()
            errors: list[str] = []
            loop.set_exception_handler(
                lambda _l, ctx: errors.append(type(ctx.get("exception")).__name__)
            )
            if install_shim:
                from fal.wma._aioice_teardown import (
                    install_orderly_ice_teardown,
                )

                assert install_orderly_ice_teardown() is True

            conn = await self._build_leaked_connection(loop, stun)
            await conn.close()
            # Let any orphaned retry timer fire (well past the 0.05s RTO).
            await asyncio.sleep(0.1)
            return errors

        try:
            return asyncio.run(scenario())
        finally:
            ice.Connection.close = original_close
            stun.RETRY_RTO = original_rto

    def test_stock_close_emits_the_retry_race(self):
        """Canary: unpatched aioice still leaks the orphaned-retry error."""
        errors = self._run_close_scenario(install_shim=False)
        assert errors == ["AttributeError"], (
            "aioice teardown behaviour changed — a possible upstream fix; "
            "revisit the orderly-teardown shim and its source guard. "
            f"got: {errors}"
        )

    def test_shim_makes_close_orderly(self):
        """With the shim installed, orderly teardown emits no loop errors."""
        errors = self._run_close_scenario(install_shim=True)
        assert errors == [], f"orderly teardown should be silent; got: {errors}"

    def test_install_is_idempotent(self):
        pytest.importorskip("aioice")
        pytest.importorskip("aioice")
        from aioice import ice

        from fal.wma._aioice_teardown import install_orderly_ice_teardown

        original_close = ice.Connection.close
        try:
            assert install_orderly_ice_teardown() is True
            wrapped = ice.Connection.close
            # A second install must not re-wrap (no stacking).
            assert install_orderly_ice_teardown() is True
            assert ice.Connection.close is wrapped
        finally:
            ice.Connection.close = original_close

    def test_guard_skips_when_close_already_cancels_tasks(self):
        """Self-disabling: if a future close() already cancels check tasks, the
        source guard declines to wrap it."""
        pytest.importorskip("aioice")
        pytest.importorskip("aioice")
        from aioice import ice

        from fal.wma._aioice_teardown import install_orderly_ice_teardown

        original_close = ice.Connection.close

        async def _future_close(self) -> None:  # mentions .task + cancel
            for pair in list(self._check_list):
                if pair.task:
                    pair.task.cancel()
            await original_close(self)

        try:
            ice.Connection.close = _future_close
            assert install_orderly_ice_teardown() is False
            assert ice.Connection.close is _future_close  # untouched
        finally:
            ice.Connection.close = original_close
