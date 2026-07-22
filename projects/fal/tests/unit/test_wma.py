from __future__ import annotations

import asyncio
import json
import sys
import threading
import types
from fractions import Fraction

import pytest

import fal.wma
from fal.wma import (
    WMA_APP_REQUIREMENTS,
    BatchedFnTrack,
    EventHandler,
    RealtimeApp,
    SessionParams,
    StartSessionRequest,
)


class FakeEmitter:
    """Minimal pyee-style event emitter, as used by aiortc objects."""

    def __init__(self):
        self._handlers: dict[str, list] = {}

    def on(self, event, handler=None):
        def register(fn):
            self._handlers.setdefault(event, []).append(fn)
            return fn

        if handler is not None:
            return register(handler)
        return register

    def emit(self, event, *args):
        for fn in self._handlers.get(event, []):
            fn(*args)


class FakeChannel(FakeEmitter):
    def __init__(self, ready_state="connecting", label="fal"):
        super().__init__()
        self.readyState = ready_state
        self.label = label
        self.sent: list[str] = []
        self.sender_threads: list[int] = []

    def send(self, data):
        self.sent.append(data)
        self.sender_threads.append(threading.get_ident())

    def open(self):
        self.readyState = "open"
        self.emit("open")


class FakePeerConnection(FakeEmitter):
    def __init__(self):
        super().__init__()
        self.tracks: list = []

    def addTrack(self, track):  # noqa: N802
        self.tracks.append(track)
        return track


class FakeFrame:
    def __init__(self, pts, time_base=Fraction(1, 90000)):
        self.pts = pts
        self.time_base = time_base


class FakeSourceTrack:
    kind = "video"
    id = "source-track"

    def __init__(self, frames):
        self._frames = list(frames)
        self.stopped = False

    async def recv(self):
        return self._frames.pop(0)

    def stop(self):
        self.stopped = True


def make_handler():
    pc = FakePeerConnection()
    params = SessionParams()
    handler = EventHandler(pc, params, session_id="session-1")
    channel = FakeChannel()
    handler._register_channel(channel, primary=True)
    return pc, params, handler, channel


def test_session_params_pushed_to_client_on_mutation():
    _, params, _, channel = make_handler()
    channel.open()
    channel.sent.clear()

    params["prompt"] = "a forest"

    assert len(channel.sent) == 1
    message = json.loads(channel.sent[0])
    assert message == {
        "type": "session_params",
        "params": {"prompt": "a forest"},
    }


def test_session_params_synced_when_channel_opens():
    _, params, _, channel = make_handler()
    params["prompt"] = "before open"
    assert channel.sent == []

    channel.open()

    message = json.loads(channel.sent[-1])
    assert message["params"] == {"prompt": "before open"}


def test_client_session_params_merged_without_echo():
    _, params, _, channel = make_handler()
    channel.open()
    channel.sent.clear()

    channel.emit(
        "message",
        json.dumps({"type": "session_params", "params": {"prompt": "from client"}}),
    )

    assert params["prompt"] == "from client"
    assert channel.sent == []


def test_ping_answered_with_pong():
    _, _, _, channel = make_handler()
    channel.open()
    channel.sent.clear()

    channel.emit("message", json.dumps({"type": "ping", "ts": 123}))

    assert json.loads(channel.sent[0]) == {"type": "pong", "client_ts": 123}


def test_non_json_messages_ignored():
    _, params, _, channel = make_handler()
    channel.open()

    channel.emit("message", "not json")
    channel.emit("message", b"\xff\xfe")
    channel.emit("message", json.dumps(["a", "list"]))

    assert params == {}


def test_client_created_channel_used_when_no_primary():
    pc = FakePeerConnection()
    params = SessionParams()
    handler = EventHandler(pc, params)

    client_channel = FakeChannel(ready_state="open")
    handler._register_channel(client_channel)

    assert handler.send({"type": "hello"})
    assert json.loads(client_channel.sent[0]) == {"type": "hello"}


def test_send_returns_false_without_open_channel():
    _, _, handler, _ = make_handler()
    assert handler.send({"type": "hello"}) is False


def test_event_handler_on_and_add_track_delegate_to_peer_connection():
    pc, _, handler, _ = make_handler()
    seen = []

    @handler.on("track")
    def _on_track(track):
        seen.append(track)

    pc.emit("track", "the-track")
    assert seen == ["the-track"]

    handler.add_track("outbound")
    assert pc.tracks == ["outbound"]


def test_batched_fn_track_batches_frames():
    inputs = [FakeFrame(pts) for pts in (0, 3000, 6000, 9000)]
    batches = []

    def fn(frames):
        batches.append(list(frames))
        return [FakeFrame(None) for _ in frames]

    track = BatchedFnTrack(FakeSourceTrack(inputs), batch_size=2, fn=fn)

    outputs = [asyncio.run(track.recv()) for _ in range(4)]

    assert [len(batch) for batch in batches] == [2, 2]
    assert [frame.pts for frame in outputs] == [0, 3000, 6000, 9000]
    assert all(frame.time_base == inputs[0].time_base for frame in outputs)


def test_batched_fn_track_supports_async_fn():
    inputs = [FakeFrame(pts) for pts in (0, 3000)]

    async def fn(frames):
        return [FakeFrame(None) for _ in frames]

    track = BatchedFnTrack(FakeSourceTrack(inputs), batch_size=2, fn=fn)

    async def collect():
        return [await track.recv(), await track.recv()]

    outputs = asyncio.run(collect())
    assert [frame.pts for frame in outputs] == [0, 3000]


def test_batched_fn_track_keeps_existing_pts():
    inputs = [FakeFrame(1000)]

    track = BatchedFnTrack(
        FakeSourceTrack(inputs),
        batch_size=1,
        fn=lambda frames: [FakeFrame(42)],
    )

    frame = asyncio.run(track.recv())
    assert frame.pts == 42


def test_batched_fn_track_stop_propagates_to_source():
    source = FakeSourceTrack([])
    track = BatchedFnTrack(source, batch_size=1, fn=lambda frames: frames)
    track.stop()
    assert source.stopped


def test_realtime_app_registers_start_session_endpoint():
    class MyWorldModel(RealtimeApp):
        async def on_connect(self, event_handler, session_params):
            pass

    assert "/start-session" in MyWorldModel.get_endpoints()


def test_realtime_app_default_request_timeout_covers_session_max_age():
    class MyWorldModel(RealtimeApp):
        async def on_connect(self, event_handler, session_params):
            pass

    assert MyWorldModel.host_kwargs["request_timeout"] == 3660


def test_realtime_app_subclass_can_override_request_timeout():
    class ShortSessions(RealtimeApp):
        request_timeout = 600

        async def on_connect(self, event_handler, session_params):
            pass

    assert ShortSessions.host_kwargs["request_timeout"] == 600


def test_base_on_connect_raises():
    app = RealtimeApp(_allow_init=True)
    with pytest.raises(NotImplementedError):
        asyncio.run(app.on_connect(None, None))


def test_session_params_popitem_and_ior_sync():
    _, params, _, channel = make_handler()
    channel.open()
    params["a"] = 1
    channel.sent.clear()

    params |= {"b": 2}
    assert json.loads(channel.sent[-1])["params"] == {"a": 1, "b": 2}

    params.popitem()
    assert json.loads(channel.sent[-1])["params"] == {"a": 1}


def test_send_from_worker_thread_is_scheduled_on_loop():
    async def scenario():
        pc = FakePeerConnection()
        handler = EventHandler(pc, SessionParams())
        channel = FakeChannel(ready_state="open")
        handler._register_channel(channel, primary=True)

        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(None, handler.send, {"n": 1})
        assert result is True

        await asyncio.sleep(0)
        assert json.loads(channel.sent[0]) == {"n": 1}
        # The send must have been marshalled back onto the loop's thread.
        assert channel.sender_threads == [threading.get_ident()]

    asyncio.run(scenario())


class FakePC(FakeEmitter):
    instances: list = []

    def __init__(self, *args, **kwargs):
        super().__init__()
        FakePC.instances.append(self)
        self.connectionState = "new"
        self.closed = False
        self.localDescription = None
        self.remoteDescription = None

    def createDataChannel(self, label):
        return FakeChannel(label=label)

    def addTrack(self, track):
        return track

    async def setRemoteDescription(self, description):
        self.remoteDescription = description

    async def createAnswer(self):
        return types.SimpleNamespace(sdp="v=0 fake answer", type="answer")

    async def setLocalDescription(self, description):
        self.localDescription = description

    async def close(self):
        self.closed = True
        self.connectionState = "closed"


@pytest.fixture
def fake_aiortc(monkeypatch):
    module = types.ModuleType("aiortc")
    module.RTCPeerConnection = FakePC
    module.RTCSessionDescription = lambda sdp, type: types.SimpleNamespace(
        sdp=sdp, type=type
    )
    monkeypatch.setitem(sys.modules, "aiortc", module)
    FakePC.instances = []
    return module


class StreamingEcho(RealtimeApp):
    connect_calls = 0

    async def on_connect(self, event_handler, session_params):
        type(self).connect_calls += 1


def test_start_session_sse_contract(fake_aiortc, monkeypatch):
    monkeypatch.setattr(fal.wma, "SSE_KEEPALIVE_INTERVAL", 0.01)
    StreamingEcho.connect_calls = 0
    app = StreamingEcho(_allow_init=True)

    async def scenario():
        response = await app.start_session(
            StartSessionRequest(sdp="v=0 offer", type="offer", session_id="s-1")
        )
        assert response.media_type == "text/event-stream"

        pc = FakePC.instances[-1]
        assert pc.remoteDescription.sdp == "v=0 offer"

        stream = response.body_iterator
        first = await stream.__anext__()
        assert first.startswith("data: ")
        payload = json.loads(first[len("data: ") :])
        assert payload == {
            "sdp": "v=0 fake answer",
            "type": "answer",
            "session_id": "s-1",
        }

        keepalive = await stream.__anext__()
        assert keepalive == ": keepalive\n\n"

        pc.connectionState = "closed"
        pc.emit("connectionstatechange")
        remaining = [chunk async for chunk in stream]
        assert all(chunk == ": keepalive\n\n" for chunk in remaining)
        assert pc.closed

    asyncio.run(scenario())
    assert StreamingEcho.connect_calls == 1


def test_start_session_closes_pc_when_bridge_drops(fake_aiortc):
    app = StreamingEcho(_allow_init=True)

    async def scenario():
        response = await app.start_session(
            StartSessionRequest(sdp="v=0 offer", type="offer", session_id="s-2")
        )
        stream = response.body_iterator
        await stream.__anext__()

        # The bridge dropping the request cancels the generator.
        await stream.aclose()
        assert FakePC.instances[-1].closed

    asyncio.run(scenario())


def test_start_session_closes_pc_on_setup_error(fake_aiortc):
    class Exploding(RealtimeApp):
        async def on_connect(self, event_handler, session_params):
            raise RuntimeError("boom")

    app = Exploding(_allow_init=True)

    async def scenario():
        with pytest.raises(RuntimeError, match="boom"):
            await app.start_session(StartSessionRequest(sdp="v=0 offer", type="offer"))
        assert FakePC.instances[-1].closed

    asyncio.run(scenario())


def test_wrap_app_adds_wma_requirements(monkeypatch):
    monkeypatch.setenv("IS_ISOLATE_AGENT", "1")
    from fal.app import wrap_app

    class MyWorldModel(RealtimeApp):
        requirements = ["numpy"]

        async def on_connect(self, event_handler, session_params):
            pass

    fn = wrap_app(MyWorldModel)
    requirements = fn.options.environment["requirements"]
    for requirement in WMA_APP_REQUIREMENTS:
        assert requirement in requirements
    assert "numpy" in requirements
