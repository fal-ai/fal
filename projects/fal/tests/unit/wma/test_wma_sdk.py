from __future__ import annotations

import asyncio
import json
import threading
from types import SimpleNamespace

import pytest
from fastapi import Request
from pydantic import ValidationError

import fal.wma.sdk as wma_sdk
from fal.compat import run_in_thread
from fal.wma import (
    CONNECTION_REPORT_VERSION,
    AiortcPeer,
    App,
    ClientOfferError,
    Session,
    SessionAnswer,
    StartSessionRequest,
)
from fal.wma._errors import InputValueError, InternalServerError


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
    def __init__(self, ready_state="connecting", label="control") -> None:
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
        self.channel = None

    def createDataChannel(self, label):
        self.channel = FakeChannel(label=label)
        return self.channel

    async def close(self):
        self.closed = True
        self.connectionState = "closed"


@pytest.fixture
def fake_aiortc(monkeypatch):
    module = SimpleNamespace(RTCPeerConnection=FakePC)
    monkeypatch.setitem(__import__("sys").modules, "aiortc", module)
    FakePC.instances = []

    async def negotiate(_pc, sdp, sdp_type):
        assert sdp == "v=0 offer"
        assert sdp_type == "offer"
        return "v=0 fake answer"

    monkeypatch.setattr(wma_sdk, "negotiate_answer", negotiate)


class FakeBackend:
    # ``asyncio.Event()`` binds the current loop on Python <=3.9, so tests must
    # construct backends inside a running loop (their ``scenario()`` coroutine
    # or ``create_backend``), never at test-function top level.
    def __init__(self) -> None:
        self.closed = asyncio.Event()
        self.close_calls = 0

    async def negotiate(self, _offer):
        return SessionAnswer(
            sdp="v=0 native answer",
            metadata={"transport": "native"},
        )

    async def wait_closed(self):
        await self.closed.wait()

    async def close(self):
        self.close_calls += 1
        self.closed.set()


class FakeReportingBackend(FakeBackend):
    connection_report_version = CONNECTION_REPORT_VERSION

    def __init__(self) -> None:
        super().__init__()
        self.report = asyncio.get_running_loop().create_future()

    async def wait_connection_report(self):
        return await self.report


class NativeApp(App):
    backend = None
    session = None

    async def create_backend(self, session):
        type(self).session = session
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
        session.receive({"type": "session_params", "params": {"prompt": "client"}})
        assert session.params == {"prompt": "client"}
        assert sent == []

    asyncio.run(scenario())


def test_session_dispatch_ping_channel_open_and_cleanup_in_reverse_order():
    async def scenario():
        session = Session(StartSessionRequest(sdp="offer"))
        received = []
        opened = []
        sent = []
        cleaned = []
        session.bind_sender(lambda message: sent.append(message) is None)
        session.on_message("input", received.append)
        session.on_channel_open(lambda: opened.append("early"))
        session.defer(lambda: cleaned.append("first"))
        session.defer(lambda: cleaned.append("second"))

        session.channel_opened()
        session.channel_opened()
        session.on_channel_open(lambda: opened.append("late"))
        session.receive({"type": "input", "value": 1})
        session.receive({"type": "ping", "client_ts": 7})
        await session.close()
        await session.close()

        assert received == [{"type": "input", "value": 1}]
        assert sent == [{"type": "pong", "client_ts": 7}]
        assert opened == ["early", "late"]
        assert cleaned == ["second", "first"]

    asyncio.run(scenario())


def test_session_wildcard_receives_unknown_messages_but_not_builtin_ping():
    async def scenario():
        session = Session(StartSessionRequest(sdp="offer"))
        received = []
        sent = []
        session.bind_sender(lambda message: sent.append(message) is None)
        session.on_message("*", received.append)

        session.receive({"type": "custom", "value": 1})
        session.receive({"type": "ping", "ts": 7})

        assert received == [{"type": "custom", "value": 1}]
        assert sent == [{"type": "pong", "client_ts": 7}]

    asyncio.run(scenario())


def test_session_inline_handler_waits_during_close_and_rejects_late_work():
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
            None, session.receive, {"type": "input", "seq": 1}
        )
        await run_in_thread(handler_started.wait, 1)

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


def test_wma_app_base_declares_the_billing_credential():
    assert App.secrets == ["FAL_KEY"]


def test_app_streams_answer_metadata_headers_keepalive_and_closes(monkeypatch):
    monkeypatch.setattr(wma_sdk, "SSE_KEEPALIVE_INTERVAL", 0.01)

    class ConfiguredApp(NativeApp):
        async def create_backend(self, session):
            backend = await super().create_backend(session)
            session.answer_metadata["model"] = "source"
            session.set_response_header("x-fal-billable-units", "0")
            return backend

    app = ConfiguredApp(_allow_init=True)

    async def scenario():
        response = await app.start_session(
            StartSessionRequest(sdp="v=0 offer", session_id="native-1"),
            x_fal_caller_user_id="user-1",
        )
        assert response.media_type == "text/event-stream"
        assert response.headers["x-fal-billable-units"] == "0"
        first = await response.body_iterator.__anext__()
        assert json.loads(first[len("data: ") :]) == {
            "sdp": "v=0 native answer",
            "type": "answer",
            "session_id": "native-1",
            "transport": "native",
            "model": "source",
        }
        assert await response.body_iterator.__anext__() == ": keepalive\n\n"
        assert ConfiguredApp.session.caller_user_id == "user-1"
        await response.body_iterator.aclose()
        assert ConfiguredApp.backend.close_calls == 1

    asyncio.run(scenario())


def test_app_advertises_and_streams_backend_connection_report(monkeypatch):
    monkeypatch.setattr(wma_sdk, "SSE_KEEPALIVE_INTERVAL", 10)

    class ReportingApp(App):
        backend = None

        async def create_backend(self, _session):
            type(self).backend = FakeReportingBackend()
            return type(self).backend

    app = ReportingApp(_allow_init=True)

    async def scenario():
        response = await app.start_session(StartSessionRequest(sdp="offer"))
        first = await response.body_iterator.__anext__()
        answer = json.loads(first[len("data: ") :])
        assert answer["connection_report_version"] == CONNECTION_REPORT_VERSION

        ReportingApp.backend.report.set_result(
            {
                "version": CONNECTION_REPORT_VERSION,
                "runner_candidate": "host",
                "browser_candidate": "relay",
                "ice_protocol": "udp",
                "setup_ms": 125,
            }
        )
        report = await asyncio.wait_for(response.body_iterator.__anext__(), timeout=0.1)
        assert report.startswith("event: connection_report\ndata: ")
        assert json.loads(report.split("data: ", 1)[1])["setup_ms"] == 125
        await response.body_iterator.aclose()
        assert ReportingApp.backend.close_calls == 1

    asyncio.run(scenario())


def test_app_ignores_invalid_backend_connection_report(monkeypatch):
    monkeypatch.setattr(wma_sdk, "SSE_KEEPALIVE_INTERVAL", 0.01)

    class ReportingApp(App):
        backend = None

        async def create_backend(self, _session):
            type(self).backend = FakeReportingBackend()
            return type(self).backend

    app = ReportingApp(_allow_init=True)

    async def scenario():
        response = await app.start_session(StartSessionRequest(sdp="offer"))
        await response.body_iterator.__anext__()
        ReportingApp.backend.report.set_result({"not_json": object()})
        assert await response.body_iterator.__anext__() == ": keepalive\n\n"
        await response.body_iterator.aclose()
        assert ReportingApp.backend.close_calls == 1

    asyncio.run(scenario())


def test_app_background_cleanup_is_idempotent():
    app = NativeApp(_allow_init=True)

    async def scenario():
        response = await app.start_session(StartSessionRequest(sdp="offer"))
        assert response.background is not None
        await response.background()
        await response.background()
        assert NativeApp.backend.close_calls == 1

    asyncio.run(scenario())


def test_app_closes_session_when_backend_wait_fails():
    class FailingWaitBackend(FakeBackend):
        async def wait_closed(self):
            raise RuntimeError("backend wait failed")

    class FailingWaitApp(App):
        backend = None

        async def create_backend(self, _session):
            type(self).backend = FailingWaitBackend()
            return type(self).backend

    app = FailingWaitApp(_allow_init=True)

    async def scenario():
        response = await app.start_session(StartSessionRequest(sdp="offer"))
        await response.body_iterator.__anext__()
        with pytest.raises(RuntimeError, match="backend wait failed"):
            await response.body_iterator.__anext__()
        assert FailingWaitApp.backend.close_calls == 1

    asyncio.run(scenario())


def test_app_watchdog_closes_session_when_stream_never_starts(monkeypatch):
    monkeypatch.setattr(wma_sdk, "STREAM_START_TIMEOUT_SECONDS", 0.01)
    app = NativeApp(_allow_init=True)

    async def scenario():
        response = await app.start_session(StartSessionRequest(sdp="offer"))
        await asyncio.wait_for(NativeApp.backend.closed.wait(), timeout=1)
        assert NativeApp.backend.close_calls == 1
        await response.body_iterator.aclose()

    asyncio.run(scenario())


def test_app_closes_bound_backend_when_setup_fails():
    backends = []

    class BrokenApp(App):
        async def create_backend(self, session):
            backend = FakeBackend()
            backends.append(backend)
            session.bind_backend(backend)
            session.set_response_header("x-fal-billable-units", "0")
            raise RuntimeError("boom")

    app = BrokenApp(_allow_init=True)

    async def scenario():
        # An unexpected server-side setup fault is translated into a 500 that
        # carries the zero-billing header (the streaming response that would
        # have carried ``session.response_headers`` never exists on this
        # path) instead of leaking the raw exception.
        with pytest.raises(InternalServerError) as exc_info:
            await app.start_session(StartSessionRequest(sdp="offer"))
        assert backends[0].close_calls == 1
        assert isinstance(exc_info.value.__cause__, RuntimeError)
        assert exc_info.value.headers["x-fal-billable-units"] == "0"

    asyncio.run(scenario())


def test_app_error_during_setup_keeps_session_billing_header():
    backends = []

    class RejectingApp(App):
        async def create_backend(self, session):
            backend = FakeBackend()
            backends.append(backend)
            session.bind_backend(backend)
            session.set_response_header("x-fal-billable-units", "0")
            raise InputValueError.from_generic_error(
                "bad session params", input=None, billing_units=None
            )

    app = RejectingApp(_allow_init=True)

    async def scenario():
        # A platform-shaped AppError raised before the streaming response exists
        # keeps its own headers but inherits the session's billing header
        # when it did not set one itself.
        with pytest.raises(InputValueError) as exc_info:
            await app.start_session(StartSessionRequest(sdp="offer"))
        assert backends[0].close_calls == 1
        assert exc_info.value.headers["x-fal-billable-units"] == "0"
        assert exc_info.value.headers["X-Fal-needs-retry"] == "false"

    asyncio.run(scenario())


def test_client_offer_error_is_a_422_located_at_sdp():
    backends = []

    async def rejecting_negotiate(_offer):
        raise ClientOfferError("could not apply remote description")

    class OfferRejectingApp(App):
        async def create_backend(self, session):
            backend = FakeBackend()
            backend.negotiate = rejecting_negotiate  # type: ignore[method-assign]
            backends.append(backend)
            return backend

    app = OfferRejectingApp(_allow_init=True)

    async def scenario():
        # A malformed offer fails when the SDP is applied (``type`` is already
        # constrained by validation), so the 422 must point clients at the
        # ``sdp`` field, not the whole body.
        with pytest.raises(InputValueError) as exc_info:
            await app.start_session(StartSessionRequest(sdp="not an sdp"))
        assert backends[0].close_calls == 1
        assert exc_info.value.status_code == 422
        assert exc_info.value.detail[0]["loc"] == ["body", "sdp"]
        assert exc_info.value.headers["x-fal-billable-units"] == "0"

    asyncio.run(scenario())


def test_start_session_request_rejects_non_offer_type():
    with pytest.raises(ValidationError):
        StartSessionRequest(sdp="v=0", type="answer")


def test_start_session_request_carries_bridge_provisioned_ice():
    async def scenario():
        request = StartSessionRequest(
            sdp="v=0",
            ice_servers=[
                {
                    "urls": "turn:global.relay.metered.ca:443",
                    "username": "u",
                    "credential": "p",
                }
            ],
            ice_status="turn",
            credential_age_seconds=31.5,
        )
        session = Session(request)
        assert session.offer.ice_servers == request.ice_servers
        assert session.offer.ice_status == "turn"
        assert session.offer.credential_age_seconds == 31.5
        await session.close()

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

        backend = AiortcPeer(session, configure, create_default_channel=True)
        session.bind_backend(backend)
        answer = await backend.negotiate(request)
        pc = FakePC.instances[-1]

        assert configured == [pc]
        assert answer.sdp == "v=0 fake answer"
        assert backend.connection_report_version == CONNECTION_REPORT_VERSION
        assert session.send({"type": "before-open"}) is False

        pc.connectionState = "connected"
        pc.emit("connectionstatechange")
        report = await backend.wait_connection_report()
        assert report["version"] == CONNECTION_REPORT_VERSION
        assert report["runner_candidate"] == "unknown"
        assert report["browser_candidate"] == "unknown"
        assert report["ice_protocol"] == "unknown"
        assert report["setup_ms"] >= 0

        assert pc.channel.label == wma_sdk.DATA_CHANNEL_LABEL == "control"
        pc.channel.open()
        assert session.send({"type": "ready"}) is True
        assert json.loads(pc.channel.sent[-1]) == {"type": "ready"}

        pc.channel.emit("message", json.dumps({"type": "input", "seq": 1}))
        assert received == [{"type": "input", "seq": 1}]

        await session.close()
        assert pc.closed

    asyncio.run(scenario())


def test_aiortc_peer_uses_client_channel_and_disconnect_grace(fake_aiortc):
    async def scenario():
        request = StartSessionRequest(sdp="v=0 offer")
        session = Session(request)
        backend = AiortcPeer(
            session,
            lambda _pc: None,
            create_default_channel=False,
            disconnected_grace_seconds=0.02,
        )
        session.bind_backend(backend)
        await backend.negotiate(request)
        pc = FakePC.instances[-1]
        channel = FakeChannel(ready_state="open", label="control")
        pc.emit("datachannel", channel)
        assert session.send({"type": "ready"})

        pc.connectionState = "disconnected"
        pc.emit("connectionstatechange")
        await asyncio.sleep(0.005)
        pc.connectionState = "connected"
        pc.emit("connectionstatechange")
        await asyncio.sleep(0.03)
        assert not backend._closed.is_set()

        channel.emit("close")
        await asyncio.wait_for(backend.wait_closed(), timeout=1)
        await session.close()

    asyncio.run(scenario())


def test_aiortc_peer_defaults_the_initial_connect_backstop(fake_aiortc):
    # A stalled ICE negotiation must not hold the session forever by default;
    # the documented ``AiortcPeer(session, on_connect)`` usage inherits the
    # raw path's 35s bound, and only an explicit ``None`` disables it.
    from fal.wma import INITIAL_CONNECT_TIMEOUT_SECONDS

    async def scenario():
        request = StartSessionRequest(sdp="v=0 offer")
        session = Session(request)
        backend = AiortcPeer(session, lambda _pc: None)
        assert (
            backend._initial_connect_timeout_seconds == INITIAL_CONNECT_TIMEOUT_SECONDS
        )
        disabled = AiortcPeer(
            session, lambda _pc: None, initial_connect_timeout_seconds=None
        )
        assert disabled._initial_connect_timeout_seconds is None
        await session.close()

    asyncio.run(scenario())


def test_aiortc_peer_closes_when_initial_connection_never_completes(fake_aiortc):
    async def scenario():
        request = StartSessionRequest(sdp="v=0 offer")
        session = Session(request)
        backend = AiortcPeer(
            session,
            lambda _pc: None,
            initial_connect_timeout_seconds=0.01,
        )
        session.bind_backend(backend)
        await backend.negotiate(request)

        await asyncio.wait_for(backend.wait_closed(), timeout=1)
        await session.close()
        assert FakePC.instances[-1].closed

    asyncio.run(scenario())


@pytest.mark.allow_real_sleep
def test_aiortc_peer_connects_to_real_client_data_channel(monkeypatch):
    pytest.importorskip("aiortc")
    from aioice import ice
    from aiortc import RTCConfiguration, RTCPeerConnection, RTCSessionDescription

    from fal.wma import _raw

    async def scenario():
        monkeypatch.setattr(ice.Connection, "close", ice.Connection.close)
        # Zero-network setup: ``iceServers=[]`` (not None) disables aiortc's
        # default Google STUN, host discovery is pinned to loopback (aioice
        # excludes 127.0.0.1, and CI runners have no globally routable
        # interface, so nothing else can pair), and the SSRF candidate filter
        # is bypassed for this test only — with real classification the
        # loopback offer would rightly be stripped. The filter's strictness is
        # pinned separately (TestFilterSdpIceCandidates, the security pins).
        monkeypatch.setattr(
            ice, "get_host_addresses", lambda use_ipv4, use_ipv6: ["127.0.0.1"]
        )
        monkeypatch.setattr(_raw, "is_globally_routable_ip", lambda ip: True)
        client = RTCPeerConnection(configuration=RTCConfiguration(iceServers=[]))
        channel = client.createDataChannel("control")
        channel_open = asyncio.Event()
        hello_received = asyncio.Event()
        messages = []

        @channel.on("open")
        def on_open():
            channel_open.set()

        @channel.on("message")
        def on_message(raw):
            messages.append(json.loads(raw))
            hello_received.set()

        await asyncio.wait_for(
            client.setLocalDescription(await client.createOffer()), timeout=10
        )
        request = StartSessionRequest(
            sdp=client.localDescription.sdp,
            type=client.localDescription.type,
            session_id="real-aiortc",
        )
        session = Session(request)
        session.on_channel_open(lambda: session.send({"type": "hello"}))
        session.on_message("echo", session.send)
        backend = AiortcPeer(
            session,
            lambda _pc: None,
            disconnected_grace_seconds=0,
            rtc_configuration=RTCConfiguration(iceServers=[]),
        )
        session.bind_backend(backend)

        try:
            answer = await asyncio.wait_for(backend.negotiate(request), timeout=10)
            await client.setRemoteDescription(
                RTCSessionDescription(sdp=answer.sdp, type=answer.type)
            )
            await asyncio.wait_for(channel_open.wait(), timeout=5)
            await asyncio.wait_for(hello_received.wait(), timeout=5)
            assert messages == [{"type": "hello"}]
            hello_received.clear()
            channel.send(json.dumps({"type": "echo", "command": "stop"}))
            await asyncio.wait_for(hello_received.wait(), timeout=5)
            assert messages[-1] == {"type": "echo", "command": "stop"}
            report = await asyncio.wait_for(backend.wait_connection_report(), timeout=5)
            assert report["version"] == CONNECTION_REPORT_VERSION
            assert report["runner_candidate"] in {"host", "srflx", "prflx", "relay"}
            assert report["browser_candidate"] in {
                "host",
                "srflx",
                "prflx",
                "relay",
            }
            assert report["ice_protocol"] in {"udp", "tcp"}
            assert report["setup_ms"] >= 0
            assert not ({"ip", "port", "address", "candidate"} & set(report))
        finally:
            # Close both peers together and bound cleanup as well as setup:
            # a transport hang must produce an actionable traceback rather
            # than killing the pytest worker at its outer timeout.
            await asyncio.wait_for(
                asyncio.gather(client.close(), session.close()), timeout=10
            )

    asyncio.run(scenario())


def test_public_api_has_no_rest_lifecycle_shim():
    # The REST-lifecycle shim predating the connection-oriented protocol must
    # never surface here.
    import fal.wma

    for name in (
        "BatchedFnTrack",
        "RealtimeApp",
        "SessionEventHandler",
        "SessionStore",
    ):
        assert not hasattr(fal.wma, name)


BILLING_REQUEST_ID = "2f9c8f6a-0d1e-4b7a-9c3d-5e6f7a8b9c0d"


@pytest.fixture
def billing_reports(monkeypatch):
    calls: list[tuple[str, float]] = []

    async def fake_report(rest_client, request_id, units, *, log_prefix, timeout=None):
        calls.append((request_id, units))

    from fal.wma import _billing

    monkeypatch.setattr(_billing, "report_stream_billing_units", fake_report)
    monkeypatch.setattr(wma_sdk, "_billing_rest_client", lambda: object())
    return calls


def test_session_billable_units_accumulate_thread_safe_and_validate():
    async def scenario():
        session = Session(
            StartSessionRequest(sdp="offer"), request_id=BILLING_REQUEST_ID
        )
        session.add_billable_units()
        session.add_billable_units(2.5)
        session.add_billable_units(0)
        assert session.billable_units == 3.5
        with pytest.raises(ValueError):
            session.add_billable_units(-1)
        with pytest.raises(ValueError):
            session.add_billable_units(float("nan"))
        with pytest.raises(ValueError):
            session.add_billable_units(float("inf"))
        # A finite increment that would overflow the running total is refused,
        # keeping everything accumulated so far billable at close.
        session.add_billable_units(1.7e308)
        with pytest.raises(ValueError, match="overflowed"):
            session.add_billable_units(1.7e308)
        assert session.billable_units == 3.5 + 1.7e308
        await session.close()

    asyncio.run(scenario())


def test_session_request_id_is_canonicalized_or_dropped():
    async def scenario():
        # The header is caller-controlled; only a canonical UUID may ever be
        # interpolated into the billing REST path.
        upper = Session(
            StartSessionRequest(sdp="offer"),
            request_id=BILLING_REQUEST_ID.upper(),
        )
        assert upper.request_id == BILLING_REQUEST_ID
        bogus = Session(
            StartSessionRequest(sdp="offer"), request_id="../evil?injected=1"
        )
        assert bogus.request_id is None
        assert not bogus._activate_deferred_billing()
        await upper.close()
        await bogus.close()

    asyncio.run(scenario())


def test_app_defers_billing_and_reports_total_once_on_close(billing_reports):
    app = NativeApp(_allow_init=True)

    async def scenario():
        response = await app.start_session(
            StartSessionRequest(sdp="offer"),
            x_fal_request_id=BILLING_REQUEST_ID,
        )
        assert response.headers["x-fal-billable-units-webhook"] == "1"
        NativeApp.session.add_billable_units(3)
        await response.body_iterator.__anext__()
        assert billing_reports == []
        await response.body_iterator.aclose()
        assert billing_reports == [(BILLING_REQUEST_ID, 3.0)]
        # A second close (reaper racing the stream teardown) must not
        # double-report.
        await NativeApp.session.close()
        assert billing_reports == [(BILLING_REQUEST_ID, 3.0)]

    asyncio.run(scenario())


def test_app_reports_zero_units_for_unmetered_session(billing_reports):
    app = NativeApp(_allow_init=True)

    async def scenario():
        # Zero must still be reported: once the webhook header went out, the
        # gateway request sits WAITING until a report settles it.
        response = await app.start_session(
            StartSessionRequest(sdp="offer"),
            x_fal_request_id=BILLING_REQUEST_ID,
        )
        await response.body_iterator.__anext__()
        await response.body_iterator.aclose()
        assert billing_reports == [(BILLING_REQUEST_ID, 0.0)]

    asyncio.run(scenario())


def test_app_without_request_id_keeps_immediate_billing(billing_reports):
    app = NativeApp(_allow_init=True)

    async def scenario():
        # A direct call that bypassed the gateway has no request to report
        # against: no webhook header, no report, billing stays on the
        # immediate response headers.
        response = await app.start_session(StartSessionRequest(sdp="offer"))
        assert "x-fal-billable-units-webhook" not in response.headers
        NativeApp.session.add_billable_units(5)
        await response.body_iterator.__anext__()
        await response.body_iterator.aclose()
        assert billing_reports == []

    asyncio.run(scenario())


def test_app_setup_failure_never_defers_billing(billing_reports):
    backends = []

    class BrokenApp(App):
        async def create_backend(self, session):
            backend = FakeBackend()
            backends.append(backend)
            session.bind_backend(backend)
            session.set_response_header("x-fal-billable-units", "0")
            raise RuntimeError("boom")

    app = BrokenApp(_allow_init=True)

    async def scenario():
        # Failed session starts bill zero through the immediate header and
        # must never park the gateway request in WAITING.
        with pytest.raises(InternalServerError) as exc_info:
            await app.start_session(
                StartSessionRequest(sdp="offer"),
                x_fal_request_id=BILLING_REQUEST_ID,
            )
        assert "x-fal-billable-units-webhook" not in (exc_info.value.headers or {})
        assert exc_info.value.headers["x-fal-billable-units"] == "0"
        assert billing_reports == []

    asyncio.run(scenario())


@pytest.mark.allow_real_sleep
def test_offer_without_media_for_server_tracks_is_a_client_offer_error(monkeypatch):
    """An offer that omits a track the server streams is a 422, not a 500.

    Director's ``on_connect`` adds audio and video tracks before negotiation; a
    client offer with only a data channel leaves those transceivers without an
    ``m=`` section and aiortc's ``setLocalDescription`` raises ``None is not in
    list``. ``negotiate_answer`` must map that to ``ClientOfferError`` (422 at ``sdp``).
    """
    pytest.importorskip("aiortc")
    from aioice import ice
    from aiortc import (
        AudioStreamTrack,
        RTCConfiguration,
        RTCPeerConnection,
        VideoStreamTrack,
    )

    from fal.wma._raw import negotiate_answer

    # Negotiation installs a process-wide shim; restore it after this test.
    monkeypatch.setattr(ice.Connection, "close", ice.Connection.close)

    monkeypatch.setattr(
        ice, "get_host_addresses", lambda use_ipv4, use_ipv6: ["127.0.0.1"]
    )

    async def scenario():
        client = RTCPeerConnection(configuration=RTCConfiguration(iceServers=[]))
        client.createDataChannel("control")
        await client.setLocalDescription(await client.createOffer())
        offer_sdp = client.localDescription.sdp
        offer_type = client.localDescription.type
        await client.close()

        server = RTCPeerConnection(configuration=RTCConfiguration(iceServers=[]))
        server.addTrack(AudioStreamTrack())
        server.addTrack(VideoStreamTrack())
        try:
            with pytest.raises(ClientOfferError):
                await negotiate_answer(server, offer_sdp, offer_type)
        finally:
            await server.close()

    asyncio.run(scenario())


def test_session_accepts_units_from_in_flight_task_during_close(billing_reports):
    async def scenario():
        session = Session(
            StartSessionRequest(sdp="offer"), request_id=BILLING_REQUEST_ID
        )
        session._activate_deferred_billing()
        task_started = asyncio.Event()

        async def meter_when_cancelled():
            task_started.set()
            try:
                await asyncio.Future()
            finally:
                session.add_billable_units(2)

        session.create_task(meter_when_cancelled())
        await task_started.wait()
        await session.close()

        assert billing_reports == [(BILLING_REQUEST_ID, 2.0)]

    asyncio.run(scenario())


def test_session_minimum_billable_units_floors_the_close_report(billing_reports):
    async def scenario():
        # Below the floor: the close report bills the minimum, not the usage.
        floored = Session(
            StartSessionRequest(sdp="offer"),
            request_id=BILLING_REQUEST_ID,
            minimum_billable_units=60,
        )
        floored._activate_deferred_billing()
        floored.add_billable_units(18.625)
        await floored.close()
        assert floored.billable_units == 60.0

        # At or above the floor: usage bills unchanged.
        above = Session(
            StartSessionRequest(sdp="offer"),
            request_id=BILLING_REQUEST_ID,
            minimum_billable_units=60,
        )
        above._activate_deferred_billing()
        above.add_billable_units(100)
        await above.close()

        # A misconfigured floor fails session construction loudly.
        with pytest.raises(ValueError, match="minimum billable units"):
            Session(StartSessionRequest(sdp="offer"), minimum_billable_units=-1)
        with pytest.raises(ValueError, match="minimum billable units"):
            Session(
                StartSessionRequest(sdp="offer"),
                minimum_billable_units=float("inf"),
            )

    asyncio.run(scenario())
    assert billing_reports == [
        (BILLING_REQUEST_ID, 60.0),
        (BILLING_REQUEST_ID, 100.0),
    ]


def test_session_minimum_does_not_apply_without_deferred_billing(billing_reports):
    async def scenario():
        # No gateway request id: billing stays on the immediate response
        # headers; the floor must not fabricate a report or inflate the
        # accumulated total.
        session = Session(StartSessionRequest(sdp="offer"), minimum_billable_units=60)
        session.add_billable_units(2)
        await session.close()
        assert session.billable_units == 2.0

    asyncio.run(scenario())
    assert billing_reports == []


def test_session_caps_undelivered_usage_before_finalization(billing_reports):
    async def scenario():
        session = Session(
            StartSessionRequest(sdp="offer"), request_id=BILLING_REQUEST_ID
        )
        session._activate_deferred_billing()
        session.add_billable_units(10)
        session.cap_billable_units(3)
        session.cap_billable_units(8)
        assert session.billable_units == 3
        for invalid in (-1, float("inf"), float("nan")):
            with pytest.raises(ValueError):
                session.cap_billable_units(invalid)
        await session.close()
        assert billing_reports == [(BILLING_REQUEST_ID, 3.0)]
        with pytest.raises(RuntimeError, match="finalized"):
            session.cap_billable_units(0)
        with pytest.raises(RuntimeError, match="finalized"):
            session.add_billable_units(1)

    asyncio.run(scenario())


def test_app_minimum_billable_units_floors_short_sessions(billing_reports):
    class MinimumApp(NativeApp):
        minimum_billable_units = 60

    app = MinimumApp(_allow_init=True)

    async def scenario():
        response = await app.start_session(
            StartSessionRequest(sdp="offer"),
            x_fal_request_id=BILLING_REQUEST_ID,
        )
        MinimumApp.session.add_billable_units(18.625)
        await response.body_iterator.__anext__()
        await response.body_iterator.aclose()

    asyncio.run(scenario())
    assert billing_reports == [(BILLING_REQUEST_ID, 60.0)]


def test_gpu_failure_during_setup_preserves_recycle_signal_and_closes_backend():
    from fal.exceptions import GPUException

    backend = FakeBackend()
    failure = GPUException(message="Resident worker group is retired")

    class RetiredApp(App):
        async def create_backend(self, session):
            session.bind_backend(backend)
            raise failure

    async def scenario():
        with pytest.raises(GPUException) as exc:
            await RetiredApp(_allow_init=True).start_session(
                StartSessionRequest(sdp="offer")
            )
        assert exc.value is failure
        assert backend.close_calls == 1

    asyncio.run(scenario())


def test_start_session_fastapi_injects_original_request_and_trusted_headers():
    import httpx
    from fastapi import Depends, FastAPI

    async def scenario():
        captured = []
        request_id = "12345678-1234-4234-8234-123456789abc"
        body = {
            "sdp": "offer",
            "session_id": "http-session",
            "user": "end-user",
            "caller_user_id": "spoof",
            "request_id": "spoof",
        }

        async def capture(request: Request):
            assert await request.json() == body
            captured.append(request)

        class FiniteApp(NativeApp):
            async def create_backend(self, session):
                assert session.http_request is captured[0]
                assert session.http_request._json == body
                assert session.caller_user_id == "trusted-caller"
                assert session.request_id == request_id
                assert (
                    session.http_request.headers["x-fal-endpoint"]
                    == "example/world/start-session"
                )
                backend = await super().create_backend(session)
                backend.closed.set()
                return backend

        app = FiniteApp(_allow_init=True)
        api = FastAPI()
        api.add_api_route(
            "/start-session",
            app.start_session,
            methods=["POST"],
            dependencies=[Depends(capture)],
        )
        operation = api.openapi()["paths"]["/start-session"]["post"]
        assert not any(
            parameter["in"] == "query" or parameter.get("required", False)
            for parameter in operation.get("parameters", [])
        )
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=api), base_url="http://test"
        ) as client:
            response = await client.post(
                "/start-session",
                json=body,
                headers={
                    "x-fal-caller-user-id": "trusted-caller",
                    "x-fal-request-id": request_id,
                    "x-fal-endpoint": "example/world/start-session",
                },
            )
        assert response.status_code == 200, response.text
        assert response.headers["content-type"].startswith("text/event-stream")
        assert json.loads(response.text.strip()[len("data: ") :]) == {
            "sdp": "v=0 native answer",
            "type": "answer",
            "session_id": "http-session",
            "transport": "native",
        }
        assert FiniteApp.backend.close_calls == 1
        assert FiniteApp.session.closed.is_set()

    asyncio.run(scenario())


def test_aiortc_peer_ignores_noncontrol_client_channels(fake_aiortc):
    async def scenario():
        session = Session(StartSessionRequest(sdp="v=0 offer"))
        backend = AiortcPeer(session, lambda pc: None)
        session.bind_backend(backend)
        received = []
        session.on_message("input", received.append)
        await backend.negotiate(session.offer)
        pc = FakePC.instances[-1]
        other = FakeChannel(ready_state="open", label="telemetry")
        pc.emit("datachannel", other)
        other.emit("message", '{"type":"input"}')
        assert not session.send({"type": "ready"})
        control = FakeChannel(ready_state="open", label="control")
        pc.emit("datachannel", control)
        assert session.send({"type": "ready"})
        assert json.loads(control.sent[-1]) == {"type": "ready"}
        assert other.sent == [] and received == []
        other.emit("close")
        assert not backend._closed.is_set()
        control.emit("message", '{"type":"input"}')
        assert received == [{"type": "input"}]
        control.emit("close")
        assert backend._closed.is_set()
        await session.close()

    asyncio.run(scenario())


def test_live_billing_floor_rejects_invalid_updates_without_losing_settlement(
    billing_reports,
):
    async def scenario():
        session = Session(
            StartSessionRequest(sdp="offer"), request_id=BILLING_REQUEST_ID
        )
        session._activate_deferred_billing()
        session.add_billable_units(2)
        session.minimum_billable_units = 5
        for invalid in (float("inf"), float("nan"), -1, "invalid", None):
            with pytest.raises(ValueError, match="minimum billable units"):
                session.minimum_billable_units = invalid
            assert session.minimum_billable_units == 5
        await session.close()
        assert billing_reports == [(BILLING_REQUEST_ID, 5.0)]
        with pytest.raises(RuntimeError, match="finalized"):
            session.minimum_billable_units = 10

    asyncio.run(scenario())


def test_recursive_close_does_not_deadlock_or_release_concurrent_waiters_early(
    billing_reports,
):
    async def scenario():
        session = Session(
            StartSessionRequest(sdp="offer"), request_id=BILLING_REQUEST_ID
        )
        session._activate_deferred_billing()
        cleanup_started = asyncio.Event()
        finish_cleanup = asyncio.Event()
        worker_started = asyncio.Event()

        async def worker():
            worker_started.set()
            try:
                await asyncio.Future()
            finally:
                await session.close()
                session.add_billable_units(1)

        async def cleanup():
            cleanup_started.set()
            await finish_cleanup.wait()
            # wait_for creates a child task; direct and child recursive calls
            # must both recognize the active teardown context.
            await session.close()
            await asyncio.wait_for(session.close(), timeout=0.1)
            session.add_billable_units(2)

        session.create_task(worker())
        await worker_started.wait()
        session.defer(cleanup)
        closing = asyncio.create_task(session.close())
        await asyncio.wait_for(cleanup_started.wait(), timeout=1)
        other = asyncio.create_task(session.close())
        await asyncio.sleep(0)
        assert not other.done()
        assert billing_reports == []
        finish_cleanup.set()
        await asyncio.wait_for(asyncio.gather(closing, other), timeout=1)
        assert billing_reports == [(BILLING_REQUEST_ID, 3.0)]

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "join",
    ["direct", "gather", "shield", "wait_for", "wait", "as_completed", "task_group"],
)
@pytest.mark.parametrize("yield_before_join", [False, True])
@pytest.mark.parametrize("phase", ["backend", "cleanup"])
def test_backend_owned_cancelled_task_can_close_session(
    join, yield_before_join, phase, billing_reports
):
    async def scenario():
        session = Session(
            StartSessionRequest(sdp="offer"), request_id=BILLING_REQUEST_ID
        )
        session._activate_deferred_billing()
        started = asyncio.Event()

        async def worker():
            started.set()
            try:
                await asyncio.Future()
            finally:
                await session.close()
                session.add_billable_units(2)

        if join == "task_group":
            if not hasattr(asyncio, "TaskGroup"):
                pytest.skip("TaskGroup requires Python 3.11")
            group = asyncio.TaskGroup()
            await group.__aenter__()
            task = group.create_task(worker())
        else:
            task = asyncio.create_task(worker())
        await started.wait()

        class Backend:
            async def close(self):
                task.cancel()
                if yield_before_join:
                    await asyncio.sleep(0)
                if join == "direct":
                    await task
                elif join == "gather":
                    await asyncio.gather(task, return_exceptions=True)
                elif join == "shield":
                    await asyncio.shield(task)
                elif join == "wait_for":
                    await asyncio.wait_for(task, timeout=0.5)
                elif join == "as_completed":
                    for completed in asyncio.as_completed([task]):
                        await completed
                elif join == "task_group":
                    await group.__aexit__(None, None, None)
                else:
                    await asyncio.wait([task])

        if phase == "backend":
            session._backend = Backend()
        else:
            session.defer(Backend().close)
        await asyncio.wait_for(session.close(), timeout=1)
        assert task.done()
        assert billing_reports == [(BILLING_REQUEST_ID, 2.0)]

    asyncio.run(scenario())


@pytest.mark.parametrize("stage", ["backend", "cleanup", "billing"])
def test_cancelled_close_caller_does_not_interrupt_settlement(stage, billing_reports):
    async def scenario():
        session = Session(
            StartSessionRequest(sdp="offer"), request_id=BILLING_REQUEST_ID
        )
        session._activate_deferred_billing()
        entered = asyncio.Event()
        release = asyncio.Event()
        calls = []

        async def pause(name):
            calls.append(name)
            if stage == name:
                entered.set()
                await release.wait()

        class Backend:
            async def close(self):
                await pause("backend")

        async def cleanup():
            await pause("cleanup")
            session.add_billable_units(3)

        report = session._report_billable_units

        async def billing():
            await pause("billing")
            await report()

        session._backend = Backend()
        session.defer(cleanup)
        session._report_billable_units = billing
        first = asyncio.create_task(session.close())
        await asyncio.wait_for(entered.wait(), timeout=1)
        first.cancel()
        with pytest.raises(asyncio.CancelledError):
            await first
        second = asyncio.create_task(session.close())
        await asyncio.sleep(0)
        assert not second.done()
        release.set()
        await asyncio.wait_for(second, timeout=1)
        await session.close()
        assert calls == ["backend", "cleanup", "billing"]
        assert billing_reports == [(BILLING_REQUEST_ID, 3.0)]
        assert session._backend is None

    asyncio.run(scenario())


def test_recovered_cancelled_caller_still_waits_for_settlement(billing_reports):
    async def scenario():
        session = Session(
            StartSessionRequest(sdp="offer"), request_id=BILLING_REQUEST_ID
        )
        session._activate_deferred_billing()
        backend_started = asyncio.Event()
        finish_backend = asyncio.Event()
        recovered = asyncio.Event()
        waiting = asyncio.Event()
        joined = asyncio.Event()

        class Backend:
            async def close(self):
                backend_started.set()
                await finish_backend.wait()

        session._backend = Backend()
        session.add_billable_units(1)

        async def caller():
            waiting.set()
            try:
                await asyncio.Future()
            except asyncio.CancelledError:
                pass
            recovered.set()
            await session.close()
            joined.set()

        first = asyncio.create_task(session.close())
        await backend_started.wait()
        second = asyncio.create_task(caller())
        await waiting.wait()
        second.cancel()
        await recovered.wait()
        await asyncio.sleep(0)
        assert not joined.is_set()
        assert billing_reports == []
        finish_backend.set()
        await asyncio.wait_for(asyncio.gather(first, second), timeout=1)
        assert joined.is_set()
        assert billing_reports == [(BILLING_REQUEST_ID, 1.0)]

    asyncio.run(scenario())


def test_session_task_initiating_close_continues_after_settlement(billing_reports):
    async def scenario():
        session = Session(
            StartSessionRequest(sdp="offer"), request_id=BILLING_REQUEST_ID
        )
        session._activate_deferred_billing()
        continued = []

        async def handler():
            session.add_billable_units(2)
            await session.close()
            assert billing_reports == [(BILLING_REQUEST_ID, 2.0)]
            continued.append(True)

        task = session.create_task(handler())
        await asyncio.wait_for(task, timeout=1)
        assert continued == [True]

    asyncio.run(scenario())


@pytest.mark.parametrize("source", ["backend", "cleanup"])
def test_cancelled_worker_does_not_abort_remaining_teardown(source, billing_reports):
    async def scenario():
        session = Session(
            StartSessionRequest(sdp="offer"), request_id=BILLING_REQUEST_ID
        )
        session._activate_deferred_billing()
        started = asyncio.Event()
        cleanups = []

        async def worker():
            started.set()
            await asyncio.Future()

        task = asyncio.create_task(worker())
        await started.wait()

        async def join_cancelled_worker():
            task.cancel()
            await task

        class Backend:
            async def close(self):
                await join_cancelled_worker()

        def finish():
            cleanups.append(True)
            session.add_billable_units(3)

        session.defer(finish)
        if source == "backend":
            session._backend = Backend()
        else:
            session.defer(join_cancelled_worker)
        await asyncio.wait_for(session.close(), timeout=1)
        await session.close()
        assert cleanups == [True]
        assert billing_reports == [(BILLING_REQUEST_ID, 3.0)]
        assert session._backend is None

    asyncio.run(scenario())


@pytest.mark.parametrize("session_owned", [False, True])
@pytest.mark.parametrize("inside_cancellation", [False, True])
def test_concurrent_close_waiters_join_settlement(
    session_owned, inside_cancellation, billing_reports
):
    async def scenario():
        session = Session(
            StartSessionRequest(sdp="offer"), request_id=BILLING_REQUEST_ID
        )
        session._activate_deferred_billing()
        started = asyncio.Event()
        release = asyncio.Event()
        caller_entered = asyncio.Event()
        caller_waiting = asyncio.Event()
        finished = []

        class Backend:
            async def close(self):
                started.set()
                await release.wait()

        session._backend = Backend()
        session.add_billable_units(2)

        async def caller():
            caller_entered.set()
            if inside_cancellation:
                try:
                    await asyncio.Future()
                except asyncio.CancelledError:
                    caller_waiting.set()
                    await session.close()
            else:
                await started.wait()
                caller_waiting.set()
                await session.close()
            assert billing_reports == [(BILLING_REQUEST_ID, 2.0)]
            finished.append(True)

        second = (
            session.create_task(caller())
            if session_owned
            else asyncio.create_task(caller())
        )
        await caller_entered.wait()
        first = asyncio.create_task(session.close())
        await started.wait()
        if inside_cancellation:
            second.cancel()
        await caller_waiting.wait()
        await asyncio.sleep(0)
        assert not second.done()
        assert billing_reports == []
        release.set()
        await asyncio.wait_for(asyncio.gather(first, second), timeout=1)
        assert finished == [True]

    asyncio.run(scenario())


def test_independent_close_waiter_spawned_by_cleanup_waits_for_settlement(
    billing_reports,
):
    async def scenario():
        session = Session(
            StartSessionRequest(sdp="offer"), request_id=BILLING_REQUEST_ID
        )
        session._activate_deferred_billing()
        session.add_billable_units(1)
        entered = asyncio.Event()
        release = asyncio.Event()
        waiters = []

        async def waiter():
            await session.close()
            assert billing_reports == [(BILLING_REQUEST_ID, 1.0)]

        def spawn_waiter():
            waiters.append(asyncio.create_task(waiter()))

        async def blocked_cleanup():
            entered.set()
            await release.wait()

        session.defer(blocked_cleanup)
        session.defer(spawn_waiter)
        closing = asyncio.create_task(session.close())
        await entered.wait()
        await asyncio.sleep(0)
        assert not waiters[0].done()
        release.set()
        await asyncio.wait_for(asyncio.gather(closing, *waiters), timeout=1)

    asyncio.run(scenario())


@pytest.mark.parametrize("phase", ["backend", "cleanup", "finished"])
def test_defer_rejects_registration_after_close_starts(phase):
    async def scenario():
        session = Session(StartSessionRequest(sdp="offer"))
        calls = []

        async def reject():
            with pytest.raises(RuntimeError, match="closing starts"):
                session.defer(lambda: calls.append("late"))

        if phase == "backend":

            class Backend:
                close = staticmethod(reject)

            session._backend = Backend()
        elif phase == "cleanup":
            session.defer(reject)
        session.defer(lambda: calls.append("registered"))
        await session.close()
        if phase == "finished":
            await reject()
        assert calls == ["registered"]

    asyncio.run(scenario())


def test_error_echo_accepts_non_json_mapping_keys():
    from starlette.responses import JSONResponse

    error = InputValueError.from_generic_error(
        "bad input",
        input={
            ("nested", 1): {frozenset({2}): b"bytes"},
            float("nan"): "nan",
            "normal": 3,
        },
    )
    response = JSONResponse(
        status_code=error.status_code, content={"detail": error.detail}
    )
    assert response.status_code == 422
    assert b'"normal":3' in response.body
    assert b"<tuple>" in response.body
