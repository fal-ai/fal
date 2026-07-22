from __future__ import annotations

import asyncio
import json
import sys
import threading
import types

import pytest

import fal.wma
from fal.wma import (
    AiortcPeer,
    App,
    Session,
    SessionAnswer,
    StartSessionRequest,
)


class FakeEmitter:
    def __init__(self) -> None:
        self.handlers: dict[str, list] = {}

    def on(self, event, handler=None):
        def register(fn):
            self.handlers.setdefault(event, []).append(fn)
            return fn

        return register(handler) if handler is not None else register

    def emit(self, event, *args) -> None:
        for handler in self.handlers.get(event, []):
            handler(*args)


class FakeChannel(FakeEmitter):
    def __init__(self, ready_state="connecting", label="fal") -> None:
        super().__init__()
        self.readyState = ready_state
        self.label = label
        self.sent: list[str] = []

    def send(self, data) -> None:
        self.sent.append(data)

    def open(self) -> None:
        self.readyState = "open"
        self.emit("open")


class FakePC(FakeEmitter):
    instances: list[FakePC] = []

    def __init__(self, configuration=None) -> None:
        super().__init__()
        type(self).instances.append(self)
        self.configuration = configuration
        self.connectionState = "new"
        self.closed = False
        self.localDescription = None
        self.remoteDescription = None
        self.channel = None

    def createDataChannel(self, label):
        self.channel = FakeChannel(label=label)
        return self.channel

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


class FakeBackend:
    def __init__(self) -> None:
        self.closed = asyncio.Event()
        self.close_calls = 0

    async def negotiate(self, offer):
        return SessionAnswer(
            sdp="v=0 native answer",
            metadata={"transport": "native"},
        )

    async def wait_closed(self):
        await self.closed.wait()

    async def close(self):
        self.close_calls += 1
        self.closed.set()


class NativeApp(App):
    backend = None

    async def create_backend(self, session):
        type(self).backend = FakeBackend()
        return type(self).backend


def test_session_params_sync_and_client_merge_without_echo():
    async def scenario():
        session = Session(StartSessionRequest(sdp="offer"))
        sent = []
        session.bind_sender(lambda message: sent.append(message) is None)

        session.params["prompt"] = "server"
        assert sent[-1] == {
            "type": "session_params",
            "params": {"prompt": "server"},
        }

        sent.clear()
        session.receive(
            {
                "type": "session_params",
                "params": {"prompt": "client"},
            }
        )
        assert session.params == {"prompt": "client"}
        assert sent == []

    asyncio.run(scenario())


def test_session_channel_open_handlers_run_once_and_late_handlers_run_immediately():
    async def scenario():
        session = Session(StartSessionRequest(sdp="offer"))
        called = []

        session.on_channel_open(lambda: called.append("early"))
        session.channel_opened()
        session.channel_opened()
        session.on_channel_open(lambda: called.append("late"))

        assert called == ["early", "late"]

    asyncio.run(scenario())


def test_session_allows_custom_ping_handler():
    async def scenario():
        session = Session(StartSessionRequest(sdp="offer"))
        loop_thread = threading.get_ident()
        handler_threads = []
        sent = []
        session.bind_sender(lambda message: sent.append(message) is None)

        def handle_ping(message):
            handler_threads.append(threading.get_ident())
            session.send({"type": "pong", "ts": message["ts"]})

        session.on_message(
            "ping",
            handle_ping,
        )

        await asyncio.get_running_loop().run_in_executor(
            None,
            session.receive,
            {"type": "ping", "ts": 42},
        )
        while not sent:
            await asyncio.sleep(0)

        assert sent == [{"type": "pong", "ts": 42}]
        assert handler_threads == [loop_thread]

    asyncio.run(scenario())


def test_session_dispatches_messages_ping_and_cleanup_in_reverse_order():
    async def scenario():
        session = Session(StartSessionRequest(sdp="offer"))
        received = []
        sent = []
        cleaned = []
        session.bind_sender(lambda message: sent.append(message) is None)
        session.on_message("input", received.append)
        session.defer(lambda: cleaned.append("first"))
        session.defer(lambda: cleaned.append("second"))

        session.receive({"type": "input", "value": 1})
        session.receive({"type": "ping", "client_ts": 7})
        await session.close()
        await session.close()

        assert received == [{"type": "input", "value": 1}]
        assert sent == [{"type": "pong", "client_ts": 7}]
        assert cleaned == ["second", "first"]

    asyncio.run(scenario())


def test_session_inline_handler_avoids_event_loop_hop_from_worker():
    async def scenario():
        session = Session(StartSessionRequest(sdp="offer"))
        loop_thread = threading.get_ident()
        inline_threads = []
        regular_threads = []
        regular_called = asyncio.Event()

        session.on_message(
            "input",
            lambda _message: inline_threads.append(threading.get_ident()),
            inline=True,
        )

        def regular(_message):
            regular_threads.append(threading.get_ident())
            regular_called.set()

        session.on_message("input", regular)
        await asyncio.get_running_loop().run_in_executor(
            None,
            session.receive,
            {"type": "input"},
        )
        await asyncio.wait_for(regular_called.wait(), timeout=1)

        assert inline_threads[0] != loop_thread
        assert regular_threads == [loop_thread]

    asyncio.run(scenario())


def test_session_close_waits_for_inline_handler_and_rejects_late_messages():
    async def scenario():
        session = Session(StartSessionRequest(sdp="offer"))
        handler_started = threading.Event()
        release_handler = threading.Event()
        received = []
        cleaned = []

        def handler(message):
            handler_started.set()
            release_handler.wait(timeout=1)
            received.append(message)

        session.on_message("input", handler, inline=True)
        session.defer(lambda: cleaned.append(True))
        loop = asyncio.get_running_loop()
        worker = loop.run_in_executor(
            None,
            session.receive,
            {"type": "input", "seq": 1},
        )
        await asyncio.to_thread(handler_started.wait, 1)

        close_task = asyncio.create_task(session.close())
        await asyncio.sleep(0)
        assert cleaned == []
        release_handler.set()
        await worker
        await close_task

        session.receive({"type": "input", "seq": 2})
        assert session.create_task(asyncio.sleep(0)) is None
        assert received == [{"type": "input", "seq": 1}]
        assert cleaned == [True]

    asyncio.run(scenario())


def test_session_isolates_synchronous_handler_failure(caplog):
    async def scenario():
        session = Session(StartSessionRequest(sdp="offer"))
        received = []

        def fail(_message):
            raise ValueError("bad input")

        session.on_message("input", fail)
        session.on_message("input", received.append)
        session.receive({"type": "input", "seq": 1})

        assert received == [{"type": "input", "seq": 1}]

    asyncio.run(scenario())
    assert "WMA message handler failed" in caplog.text


def test_session_closes_backend_and_owned_tasks():
    async def scenario():
        session = Session(StartSessionRequest(sdp="offer"))
        backend = FakeBackend()
        session.bind_backend(backend)
        cancelled = asyncio.Event()

        async def work():
            try:
                await asyncio.Future()
            finally:
                cancelled.set()

        session.create_task(work())
        await asyncio.sleep(0)
        await session.close()

        assert backend.close_calls == 1
        assert cancelled.is_set()
        assert session.closed.is_set()

    asyncio.run(scenario())


def test_transport_neutral_app_streams_answer_and_closes_backend():
    app = NativeApp(_allow_init=True)

    async def scenario():
        response = await app.start_session(
            StartSessionRequest(sdp="v=0 offer", session_id="native-1")
        )
        assert response.media_type == "text/event-stream"
        stream = response.body_iterator
        first = await stream.__anext__()
        assert json.loads(first.removeprefix("data: ")) == {
            "sdp": "v=0 native answer",
            "type": "answer",
            "session_id": "native-1",
            "transport": "native",
        }
        await stream.aclose()
        assert NativeApp.backend.close_calls == 1

    asyncio.run(scenario())


def test_app_streams_session_metadata_and_response_headers():
    class ConfiguredApp(App):
        async def create_backend(self, session):
            session.answer_metadata["model"] = "source"
            session.set_response_header("x-fal-billable-units", "0")
            return FakeBackend()

    app = ConfiguredApp(_allow_init=True)

    async def scenario():
        response = await app.start_session(
            StartSessionRequest(sdp="offer", session_id="configured")
        )
        assert response.headers["x-fal-billable-units"] == "0"
        first = await response.body_iterator.__anext__()
        assert json.loads(first.removeprefix("data: ")) == {
            "sdp": "v=0 native answer",
            "type": "answer",
            "session_id": "configured",
            "transport": "native",
            "model": "source",
        }
        await response.body_iterator.aclose()

    asyncio.run(scenario())


def test_app_closes_session_when_backend_setup_fails():
    backend = FakeBackend()

    class BrokenApp(App):
        async def create_backend(self, session):
            session.bind_backend(backend)
            raise RuntimeError("boom")

    app = BrokenApp(_allow_init=True)

    async def scenario():
        with pytest.raises(RuntimeError, match="boom"):
            await app.start_session(StartSessionRequest(sdp="offer"))
        assert backend.close_calls == 1

    asyncio.run(scenario())


def test_aiortc_peer_negotiates_and_routes_data_channel(fake_aiortc):
    async def scenario():
        request = StartSessionRequest(
            sdp="v=0 offer",
            type="offer",
            session_id="session-1",
        )
        session = Session(request)
        received = []
        session.on_message("input", received.append)
        configured = []

        async def configure(peer_connection):
            configured.append(peer_connection)

        backend = AiortcPeer(session, configure)
        session.bind_backend(backend)
        answer = await backend.negotiate(request)
        pc = FakePC.instances[-1]

        assert configured == [pc]
        assert pc.remoteDescription.sdp == "v=0 offer"
        assert answer.sdp == "v=0 fake answer"
        assert session.send({"type": "before-open"}) is False

        pc.channel.open()
        assert session.send({"type": "ready"}) is True
        assert json.loads(pc.channel.sent[-1]) == {"type": "ready"}

        pc.channel.emit("message", json.dumps({"type": "input", "seq": 1}))
        assert received == [{"type": "input", "seq": 1}]

        await session.close()
        assert pc.closed

    asyncio.run(scenario())


def test_aiortc_peer_uses_client_channel_without_default(fake_aiortc):
    async def scenario():
        request = StartSessionRequest(sdp="offer")
        session = Session(request)
        backend = AiortcPeer(
            session,
            lambda _peer_connection: None,
            create_default_channel=False,
        )
        session.bind_backend(backend)
        await backend.negotiate(request)

        pc = FakePC.instances[-1]
        channel = FakeChannel(ready_state="open", label="input")
        pc.emit("datachannel", channel)
        assert session.send({"type": "ready"})
        assert json.loads(channel.sent[-1]) == {"type": "ready"}

        pc.connectionState = "disconnected"
        pc.emit("connectionstatechange")
        await asyncio.wait_for(backend.wait_closed(), timeout=1)
        await session.close()

    asyncio.run(scenario())


def test_aiortc_peer_accepts_rtc_configuration(fake_aiortc):
    async def scenario():
        request = StartSessionRequest(sdp="offer")
        session = Session(request)
        configuration = object()
        backend = AiortcPeer(
            session,
            lambda _peer_connection: None,
            rtc_configuration=configuration,
        )
        session.bind_backend(backend)
        await backend.negotiate(request)

        assert FakePC.instances[-1].configuration is configuration
        await session.close()

    asyncio.run(scenario())


def test_aiortc_peer_accepts_async_peer_connection_factory(fake_aiortc):
    async def scenario():
        request = StartSessionRequest(sdp="offer")
        session = Session(request)
        custom_peer = FakePC()

        async def create_peer():
            return custom_peer

        backend = AiortcPeer(
            session,
            lambda _peer_connection: None,
            peer_connection_factory=create_peer,
        )
        session.bind_backend(backend)
        await backend.negotiate(request)

        assert backend._pc is custom_peer
        await session.close()

    asyncio.run(scenario())


def test_aiortc_peer_rejects_conflicting_peer_configuration():
    async def scenario():
        session = Session(StartSessionRequest(sdp="offer"))
        with pytest.raises(ValueError, match="mutually exclusive"):
            AiortcPeer(
                session,
                lambda _peer_connection: None,
                rtc_configuration=object(),
                peer_connection_factory=FakePC,
            )
        with pytest.raises(ValueError, match="cannot be negative"):
            AiortcPeer(
                session,
                lambda _peer_connection: None,
                disconnected_grace_seconds=-1,
            )

    asyncio.run(scenario())


def test_aiortc_peer_allows_temporary_disconnect_with_grace(fake_aiortc):
    async def scenario():
        request = StartSessionRequest(sdp="offer")
        session = Session(request)
        backend = AiortcPeer(
            session,
            lambda _peer_connection: None,
            disconnected_grace_seconds=0.03,
        )
        session.bind_backend(backend)
        await backend.negotiate(request)
        pc = FakePC.instances[-1]

        pc.connectionState = "disconnected"
        pc.emit("connectionstatechange")
        await asyncio.sleep(0.01)
        assert not backend._closed.is_set()

        pc.connectionState = "connected"
        pc.emit("connectionstatechange")
        await asyncio.sleep(0.03)
        assert not backend._closed.is_set()

        pc.connectionState = "disconnected"
        pc.emit("connectionstatechange")
        await asyncio.wait_for(backend.wait_closed(), timeout=0.1)
        await session.close()

    asyncio.run(scenario())


def test_aiortc_peer_closes_when_data_channel_closes(fake_aiortc):
    async def scenario():
        request = StartSessionRequest(sdp="offer")
        session = Session(request)
        backend = AiortcPeer(session, lambda _peer_connection: None)
        session.bind_backend(backend)
        await backend.negotiate(request)

        channel = FakePC.instances[-1].channel
        channel.open()
        channel.emit("close")
        await asyncio.wait_for(backend.wait_closed(), timeout=1)
        await session.close()

    asyncio.run(scenario())


def test_aiortc_peer_ignores_auxiliary_channel_close(fake_aiortc):
    async def scenario():
        request = StartSessionRequest(sdp="offer")
        session = Session(request)
        backend = AiortcPeer(session, lambda _peer_connection: None)
        session.bind_backend(backend)
        await backend.negotiate(request)

        pc = FakePC.instances[-1]
        pc.channel.open()
        auxiliary = FakeChannel(ready_state="open", label="aux")
        pc.emit("datachannel", auxiliary)
        auxiliary.emit("close")

        assert not backend._closed.is_set()
        pc.channel.emit("close")
        assert backend._closed.is_set()
        await session.close()

    asyncio.run(scenario())


def test_public_api_has_no_legacy_wma_classes():
    for name in (
        "BatchedFnTrack",
        "EventHandler",
        "RealtimeApp",
        "WMA_APP_REQUIREMENTS",
    ):
        assert not hasattr(fal.wma, name)
