"""Unit tests for the REST session helpers in ``fal.wma.app``.

Covers the pieces future WebRTC apps build on: BatchedFnTrack batching
semantics, the heartbeat-driven SessionStore, and event dispatch. The
HTTP endpoints themselves are exercised through app integration tests.
"""

import asyncio
from typing import List

import pytest
from pydantic import ValidationError

# REST session helpers are available in ``fal.wma.app``; the package root
# exposes the connection-oriented SDK API.
from fal.wma.app import (
    BatchedFnTrack,
    RealtimeApp,
    SessionEventHandler,
    SessionStore,
    StartSessionRequest,
    TrackEnded,
)


class FakeTrack:
    """Async source track yielding a preset sequence of frames."""

    def __init__(self, frames):
        self._frames = list(frames)
        self.stopped = False

    async def recv(self):
        return self._frames.pop(0)

    def stop(self):
        self.stopped = True


def run(coro):
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


class TestBatchedFnTrack:
    def test_batches_frames_and_replays_fn_output(self):
        seen_batches = []

        def fn(frames):
            seen_batches.append(list(frames))
            return [f * 10 for f in frames]

        track = BatchedFnTrack(FakeTrack([1, 2, 3, 4]), batch_size=2, fn=fn)

        async def collect():
            return [await track.recv() for _ in range(4)]

        assert run(collect()) == [10, 20, 30, 40]
        assert seen_batches == [[1, 2], [3, 4]]

    def test_supports_async_fn_and_single_frame_result(self):
        async def fn(frames):
            return sum(frames)

        track = BatchedFnTrack(FakeTrack([1, 2, 3, 4]), batch_size=4, fn=fn)
        assert run(track.recv()) == 10

    def test_none_result_consumes_batch_without_output(self):
        results = iter([None, [99]])
        track = BatchedFnTrack(
            FakeTrack([1, 2]), batch_size=1, fn=lambda frames: next(results)
        )
        # First batch is dropped (fn returned None); second yields output.
        assert run(track.recv()) == 99

    def test_stop_propagates_to_source_and_ends_recv(self):
        source = FakeTrack([])
        track = BatchedFnTrack(source, batch_size=1, fn=lambda frames: frames)
        track.stop()
        assert source.stopped
        with pytest.raises(TrackEnded):
            run(track.recv())

    def test_exposes_media_stream_track_contract(self):
        # RTCPeerConnection.addTrack / RTCRtpSender need a real
        # MediaStreamTrack surface (id, readyState), not just kind/recv/stop.
        # The fallback base used without aiortc must uphold the same contract.
        track = BatchedFnTrack(
            FakeTrack([]), batch_size=1, fn=lambda frames: frames, kind="video"
        )
        assert isinstance(track.id, str) and track.id
        assert track.kind == "video"
        assert track.readyState == "live"
        track.stop()
        assert track.readyState == "ended"

    def test_stop_discards_buffered_output(self):
        # fn returned two frames for one batch; one was consumed, one is still
        # buffered. Stopping must end recv() instead of emitting the leftover.
        track = BatchedFnTrack(FakeTrack([1]), batch_size=1, fn=lambda frames: [10, 20])
        assert run(track.recv()) == 10
        track.stop()
        with pytest.raises(TrackEnded):
            run(track.recv())

    def test_rejects_invalid_batch_size(self):
        with pytest.raises(ValueError):
            BatchedFnTrack(FakeTrack([]), batch_size=0, fn=lambda frames: frames)

    def test_expands_generator_results(self):
        track = BatchedFnTrack(
            FakeTrack([1, 2]),
            batch_size=2,
            fn=lambda frames: (f * 10 for f in frames),
        )

        async def collect():
            return [await track.recv() for _ in range(2)]

        assert run(collect()) == [10, 20]


class TestStartSessionRequest:
    def test_accepts_offer_type_only(self):
        request = StartSessionRequest(offer={"sdp": "v=0", "type": "offer"})
        assert request.offer is not None and request.offer.type == "offer"
        with pytest.raises(ValidationError):
            StartSessionRequest(offer={"sdp": "v=0", "type": "answer"})


class TestSessionStore:
    def make_store(self, timeout=10.0):
        clock = {"now": 100.0}
        store = SessionStore(timeout_sec=timeout, clock=lambda: clock["now"])
        return store, clock

    def test_create_and_get(self):
        store, _ = self.make_store()
        session = store.create({"world": "w1"})
        assert session.session_id.startswith("sess_")
        assert store.get(session.session_id) is session
        assert store.get("sess_unknown") is None

    def test_session_expires_without_heartbeat(self):
        store, clock = self.make_store(timeout=10.0)
        session = store.create({})
        clock["now"] += 11
        assert store.get(session.session_id) is None
        assert store.heartbeat(session.session_id) is None

    def test_heartbeat_extends_lifetime(self):
        store, clock = self.make_store(timeout=10.0)
        session = store.create({})
        clock["now"] += 8
        assert store.heartbeat(session.session_id) is session
        clock["now"] += 8
        # 16s since creation but only 8s since the heartbeat: still alive.
        assert store.get(session.session_id) is session

    def test_pop_expired_drains_only_stale_sessions(self):
        store, clock = self.make_store(timeout=10.0)
        stale = store.create({})
        clock["now"] += 6
        fresh = store.create({})
        clock["now"] += 6
        expired = store.pop_expired()
        assert [s.session_id for s in expired] == [stale.session_id]
        assert store.get(fresh.session_id) is fresh
        assert len(store) == 1


class TestSessionEventHandler:
    def test_sync_handlers_run_inline(self):
        handler = SessionEventHandler()
        calls = []

        @handler.on("track")
        def on_track(track):
            calls.append(track)
            handler.add_track(track)

        handler.dispatch("track", "t1")
        assert calls == ["t1"]
        assert handler.tracks == ["t1"]

    def test_handler_exception_does_not_break_dispatch(self):
        handler = SessionEventHandler()
        calls = []

        @handler.on("close")
        def boom(reason):
            raise RuntimeError("boom")

        @handler.on("close")
        def record(reason):
            calls.append(reason)

        handler.dispatch("close", "expired")
        assert calls == ["expired"]

    def test_async_handlers_are_scheduled(self):
        handler = SessionEventHandler()
        calls = []

        @handler.on("session_params")
        async def on_params(params):
            calls.append(params)

        async def scenario():
            handler.dispatch("session_params", {"a": 1})
            await asyncio.sleep(0)

        run(scenario())
        assert calls == [{"a": 1}]


class _TwoArgConnectStub:
    _invoke_on_connect = RealtimeApp._invoke_on_connect
    _on_connect_request_mode = RealtimeApp._on_connect_request_mode

    def __init__(self):
        self.calls = 0

    async def on_connect(self, event_handler, session_params):
        self.calls += 1
        return {"arity": 2}


class _ThreeArgConnectStub(_TwoArgConnectStub):
    async def on_connect(self, event_handler, session_params, request=None):
        self.calls += 1
        return {"arity": 3, "request": request}


class _TypeErrorConnectStub(_TwoArgConnectStub):
    async def on_connect(self, event_handler, session_params):
        self.calls += 1
        raise TypeError("raised inside override")


class TestOnConnectCompatibility:
    def test_preserves_existing_two_argument_override(self):
        stub = _TwoArgConnectStub()
        result = run(stub._invoke_on_connect(object(), {"world": "w1"}, object()))
        assert result == {"arity": 2}
        assert stub.calls == 1
        assert stub._wma_on_connect_request_mode == "none"

    def test_delivers_request_to_three_argument_override(self):
        stub = _ThreeArgConnectStub()
        request = object()
        result = run(stub._invoke_on_connect(object(), {}, request))
        assert result == {"arity": 3, "request": request}
        assert stub.calls == 1
        assert stub._wma_on_connect_request_mode == "keyword"

    def test_internal_type_error_is_not_retried(self):
        stub = _TypeErrorConnectStub()
        with pytest.raises(TypeError, match="raised inside override"):
            run(stub._invoke_on_connect(object(), {}, object()))
        assert stub.calls == 1


class _ExpiryFinalizeStub:
    _finalize_expired_sessions = RealtimeApp._finalize_expired_sessions

    def __init__(self):
        self.calls: List[str] = []

    async def _finalize_session(self, session, *, reason):
        self.calls.append(session.session_id)
        if len(self.calls) == 1:
            raise RuntimeError("one teardown failed")


class TestReaperIsolation:
    def test_one_teardown_failure_does_not_skip_later_sessions(self):
        store = SessionStore(timeout_sec=10)
        first = store.create({})
        second = store.create({})
        stub = _ExpiryFinalizeStub()
        run(stub._finalize_expired_sessions([first, second]))
        assert stub.calls == [first.session_id, second.session_id]


def test_legacy_negotiation_filters_internal_ice_candidates(monkeypatch):
    import sys
    from types import SimpleNamespace

    from fal.wma.app import SDPOffer

    offers = []

    class Peer:
        localDescription = SimpleNamespace(sdp="answer", type="answer")

        def on(self, event):
            return lambda handler: handler

        async def setRemoteDescription(self, description):
            offers.append(description.sdp)

        async def createAnswer(self):
            return self.localDescription

        async def setLocalDescription(self, answer):
            pass

    monkeypatch.setitem(
        sys.modules,
        "aiortc",
        SimpleNamespace(
            RTCPeerConnection=Peer,
            RTCSessionDescription=lambda sdp, type: SimpleNamespace(sdp=sdp, type=type),
        ),
    )
    sdp = "\r\n".join(
        ["v=0", "m=application 9 UDP/DTLS/SCTP webrtc-datachannel"]
        + [
            f"a=candidate:1 1 UDP 100 {ip} 1234 typ host"
            for ip in ("127.0.0.1", "10.0.0.1", "169.254.169.254", "8.8.8.8")
        ]
    )
    session = SimpleNamespace(state={}, handler=SessionEventHandler())
    answer = asyncio.run(
        RealtimeApp._negotiate_webrtc(None, session, SDPOffer(sdp=sdp, type="offer"))
    )
    assert answer.sdp == "answer"
    assert "8.8.8.8" in offers[0]
    assert all(
        ip not in offers[0] for ip in ("127.0.0.1", "10.0.0.1", "169.254.169.254")
    )


@pytest.mark.parametrize("shutdown_during_connect", [False, True])
def test_pending_session_does_not_expire_and_shutdown_releases_resources(
    shutdown_during_connect,
):
    from fastapi import Response

    async def scenario():
        entered = asyncio.Event()
        release = asyncio.Event()
        disconnected = []
        track = FakeTrack([])
        closed_peers = []

        class Peer:
            async def close(self):
                closed_peers.append(True)

        class App(RealtimeApp):
            async def on_connect(self, event_handler, session_params):
                event_handler.tracks.append(track)
                entered.set()
                await release.wait()
                return {"ready": True}

            async def on_disconnect(self, session, reason):
                disconnected.append(reason)

        app = App(_allow_init=True)
        await app.setup()
        clock = [0.0]
        app._wma_sessions = SessionStore(timeout_sec=30, clock=lambda: clock[0])
        response = Response()
        connecting = asyncio.create_task(
            app.start_session(StartSessionRequest(), response, None)
        )
        await entered.wait()
        session = next(iter(app._wma_sessions._sessions.values()))
        session.state["_pc"] = Peer()
        clock[0] = 100.0
        assert app._wma_sessions.pop_expired() == []
        if shutdown_during_connect:
            await asyncio.wait_for(app.teardown(), timeout=1)
            assert connecting.cancelled()
            assert response.headers["x-fal-billable-units"] == "0"
            assert disconnected == ["connect-failed"]
        else:
            release.set()
            result = await connecting
            assert app._wma_sessions.get(result.session_id) is session
            assert session.last_seen_at == 100.0
            assert not session.pending
            assert response.headers["x-fal-billable-units"] == "1"
            await asyncio.wait_for(app.teardown(), timeout=1)
            assert disconnected == ["shutdown"]
        await app.teardown()
        assert app._wma_reaper.done()
        assert len(app._wma_sessions) == 0
        assert track.stopped
        assert closed_peers == [True]
        with pytest.raises(Exception) as exc:
            await app.start_session(StartSessionRequest(), Response(), None)
        assert exc.value.status_code == 503

    run(scenario())


@pytest.mark.parametrize("blocked_phase", ["close_handler", "peer", "disconnect"])
def test_shutdown_awaits_finalization_after_close_request_is_cancelled(blocked_phase):
    from fastapi import Response

    from fal.wma.app import SessionRef

    async def scenario():
        entered = asyncio.Event()
        release = asyncio.Event()
        calls = []
        track = FakeTrack([])

        async def step(phase):
            calls.append(phase)
            if blocked_phase == phase:
                entered.set()
                await release.wait()

        class App(RealtimeApp):
            async def on_disconnect(self, session, reason):
                await step("disconnect")

        class Peer:
            async def close(self):
                await step("peer")

        app = App(_allow_init=True)
        await app.setup()
        session = app._wma_sessions.create({})
        session.handler.add_track(track)
        session.state["_pc"] = Peer()

        @session.handler.on("close")
        async def on_close(reason):
            await step("close_handler")

        closing = asyncio.create_task(
            app.close_session(SessionRef(session_id=session.session_id), Response())
        )
        await asyncio.wait_for(entered.wait(), timeout=1)
        assert len(app._wma_sessions) == 0
        closing.cancel()
        with pytest.raises(asyncio.CancelledError):
            await closing
        shutdown = asyncio.create_task(app.teardown())
        await asyncio.sleep(0.01)
        assert not shutdown.done()
        release.set()
        await asyncio.wait_for(shutdown, timeout=1)
        assert sorted(calls) == ["close_handler", "disconnect", "peer"]
        assert track.stopped
        assert not app._wma_finalizers

    run(scenario())
