from __future__ import annotations

import asyncio
import json
import threading
from types import SimpleNamespace

import pytest
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


def test_wma_app_base_inherits_the_platform_secrets_default():
    # An explicit ``[]`` here would be inherited by every subclass as an
    # opt-in to zero secrets, silently stripping dashboard/pyproject secrets
    # from ported apps; the base must stay unset like ``fal.App``.
    assert App.secrets is None
    assert "secrets" not in App.host_kwargs


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

        backend = AiortcPeer(session, configure)
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
def test_aiortc_peer_connects_to_real_client_data_channel():
    pytest.importorskip("aiortc")
    from aioice import ice
    from aiortc import RTCConfiguration, RTCPeerConnection, RTCSessionDescription

    async def scenario():
        original_ice_close = ice.Connection.close
        # ``iceServers=[]`` (not None) disables aiortc's default Google STUN:
        # both peers gather host candidates only, so the negotiation stays on
        # this machine and the unit suite makes no external network calls.
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

        await client.setLocalDescription(await client.createOffer())
        request = StartSessionRequest(
            sdp=client.localDescription.sdp,
            type=client.localDescription.type,
            session_id="real-aiortc",
        )
        session = Session(request)
        session.on_channel_open(lambda: session.send({"type": "hello"}))
        backend = AiortcPeer(
            session,
            lambda _pc: None,
            create_default_channel=False,
            disconnected_grace_seconds=0,
            rtc_configuration=RTCConfiguration(iceServers=[]),
        )
        session.bind_backend(backend)

        try:
            answer = await backend.negotiate(request)
            await client.setRemoteDescription(
                RTCSessionDescription(sdp=answer.sdp, type=answer.type)
            )
            await asyncio.wait_for(channel_open.wait(), timeout=5)
            await asyncio.wait_for(hello_received.wait(), timeout=5)
            assert messages == [{"type": "hello"}]
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
            await client.close()
            await session.close()
            ice.Connection.close = original_ice_close

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
