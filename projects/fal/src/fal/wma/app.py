"""REST session lifecycle helpers for WebRTC applications.

.. deprecated::
    Superseded by the connection-oriented surface in :mod:`fal.wma.sdk`;
    not re-exported from the package root. New apps should use
    ``fal.wma.App``.

``RealtimeApp`` and ``BatchedFnTrack`` provide REST session management
and media processing on top of ``fal.App``.

Protocol (mirrors the WMA bridge contract at ``wma.fal.run``):

- ``POST /session``            create a session. An optional SDP ``offer``
                               is answered server-side (requires ``aiortc``
                               in the app's requirements) after the subclass
                               attaches media tracks in ``on_connect``.
- ``POST /session/heartbeat``  keep the session alive; clients should call
                               it every ``heartbeat_interval_sec`` (~5s).
                               WMA sessions are not resumable: an expired or
                               unknown session answers ``alive=false`` and
                               the client must create a fresh session.
- ``POST /session/close``      end the session and release resources.

Two kinds of subclasses are supported:

1. Media apps: ``on_connect`` registers ``event_handler.on("track")`` and
   attaches processed tracks (e.g. ``BatchedFnTrack``); the base class then
   negotiates the WebRTC answer from the client's SDP offer.
2. Control-plane apps whose media is carried by a partner RTC network
   use ``on_connect`` to return connection material (tickets,
   tokens, RTC config) in the ``connection`` field and no SDP leg is used.
"""

import asyncio
import inspect
import json
import logging
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable, ClassVar, Dict, Iterator, List, Literal, Union

import fastapi
from fastapi import Response
from pydantic import BaseModel, Field

import fal
from fal.wma._errors import InputValueError
from fal.wma._raw import filter_sdp_ice_candidates

logger = logging.getLogger(__name__)

# Mutable per-session parameter mapping shared between client and runner;
# data-channel payloads sent by the client are merged into it. It is
# CLIENT-WRITABLE by contract: a connected peer can set or overwrite any key
# at any time, so apps must never store server-trusted values (entitlements,
# limits, feature gates) here — keep those in ``Session.state``, which the
# client can never touch.
SessionParams = Dict[str, Any]


class _FallbackTrackBase:
    """Minimal ``aiortc.MediaStreamTrack`` look-alike (aiortc-less envs only).

    Mirrors the pieces of the real base class that ``BatchedFnTrack`` and its
    unit tests rely on: ``id``, ``readyState``, and an idempotent ``stop``.
    """

    kind = "unknown"

    def __init__(self) -> None:
        self._fallback_ended = False
        self._fallback_id = str(uuid.uuid4())

    @property
    def id(self) -> str:
        return self._fallback_id

    @property
    def readyState(self) -> str:
        return "ended" if self._fallback_ended else "live"

    def stop(self) -> None:
        self._fallback_ended = True


# aiortc is a runner-only package, but inheriting from its MediaStreamTrack
# cannot be deferred to method scope: RTCPeerConnection.addTrack requires a
# real MediaStreamTrack instance (id, readyState, `ended` event), not a
# duck-type. The guarded import keeps this module importable — and the track
# unit-testable — without aiortc.
try:  # pragma: no cover - exercised on runners that install aiortc
    from aiortc.mediastreams import MediaStreamError as _MediaStreamError
    from aiortc.mediastreams import MediaStreamTrack as _TrackBase
except ImportError:
    _MediaStreamError = Exception  # type: ignore[assignment, misc]
    _TrackBase = _FallbackTrackBase  # type: ignore[assignment, misc]


class TrackEnded(_MediaStreamError):  # type: ignore[valid-type, misc]
    """Raised by ``BatchedFnTrack.recv`` once the track has been stopped.

    Subclasses aiortc's ``MediaStreamError`` when available, so the RTP
    sender loop treats a stopped track as a normal end of media rather than
    an unexpected error.
    """


class BatchedFnTrack(_TrackBase):  # type: ignore[valid-type, misc]
    """Buffer frames from ``source``, run ``fn`` per batch, re-emit results.

    Decouples the processing cadence from the input frame rate by grouping
    ``batch_size`` frames per
    inference call. ``fn`` receives the frame batch (a list) and may be sync
    or async; it may return a single frame, an iterable of frames (list,
    tuple, generator, or other iterator), or ``None`` (batch consumed
    without output).

    A real ``aiortc.MediaStreamTrack`` subclass when aiortc is installed
    (required by ``RTCPeerConnection.addTrack``); a minimal look-alike
    otherwise, so the module stays importable without aiortc.
    """

    def __init__(
        self,
        source: Any,
        *,
        batch_size: int,
        fn: Callable[[List[Any]], Any],
        kind: str = "video",
    ) -> None:
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        super().__init__()
        self.source = source
        self.batch_size = batch_size
        self.fn = fn
        self.kind = kind
        self._output: deque[Any] = deque()

    async def recv(self) -> Any:
        # Checked before the buffer too: a track stopped at teardown must not
        # emit frames it had already produced.
        if self.readyState != "live":
            raise TrackEnded("BatchedFnTrack has been stopped")
        while not self._output:
            if self.readyState != "live":
                raise TrackEnded("BatchedFnTrack has been stopped")
            frames = [await self.source.recv() for _ in range(self.batch_size)]
            result = self.fn(frames)
            if inspect.isawaitable(result):
                result = await result
            if result is None:
                continue
            # Expand containers and lazy iterables (generators, iterators)
            # into individual frames. Deliberately NOT generic iterable
            # detection: single frames (e.g. numpy arrays) are themselves
            # iterable and must be emitted whole.
            if isinstance(result, (list, tuple, Iterator)):
                self._output.extend(result)
            else:
                self._output.append(result)
        return self._output.popleft()

    def as_media_stream_track(self) -> Any:
        """Return an aiortc track, including after shipping without local aiortc.

        A class serialized with the fallback base keeps that base on a runner.
        Construct the adapter against the runner's installed aiortc at use time.
        """
        from aiortc import MediaStreamTrack
        from aiortc.mediastreams import MediaStreamError

        if isinstance(self, MediaStreamTrack):
            return self
        source = self

        class RuntimeTrack(MediaStreamTrack):
            kind = source.kind

            async def recv(self) -> Any:
                if self.readyState != "live":
                    raise MediaStreamError
                try:
                    return await source.recv()
                except TrackEnded as exc:
                    raise MediaStreamError from exc

            def stop(self) -> None:
                super().stop()
                source.stop()

        return RuntimeTrack()

    def stop(self) -> None:
        # Marks the track ended (and, with aiortc, emits the `ended` event).
        super().stop()
        stop = getattr(self.source, "stop", None)
        if callable(stop):
            stop()


class SessionEventHandler:
    """The ``event_handler`` object passed to ``RealtimeApp.on_connect``.

    ``on(event)`` registers a callback (sync or async) and ``add_track``
    queues a track for the outgoing WebRTC answer. Known events:

    - ``"track"``           a remote media track arrived (media apps only)
    - ``"session_params"``  the client updated session params via the data
                            channel (receives the updated mapping)
    - ``"close"``           the session is being torn down (receives the
                            close reason string)
    """

    def __init__(self) -> None:
        self._handlers: Dict[str, List[Callable[..., Any]]] = {}
        self.tracks: List[Any] = []

    def on(self, event: str) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        def register(fn: Callable[..., Any]) -> Callable[..., Any]:
            self._handlers.setdefault(event, []).append(fn)
            return fn

        return register

    def add_track(self, track: Any) -> None:
        self.tracks.append(track)

    def dispatch(self, event: str, *args: Any) -> None:
        """Run sync handlers inline and schedule async ones.

        Sync-inline matters for the ``"track"`` event: handlers typically
        call ``add_track`` and must do so before the SDP answer is created.
        """
        for handler in self._handlers.get(event, []):
            try:
                result = handler(*args)
            except Exception:
                logger.exception("wma: %r handler failed", event)
                continue
            if inspect.isawaitable(result):
                task = asyncio.ensure_future(result)
                task.add_done_callback(_log_handler_task_error(event))


def _log_handler_task_error(event: str) -> Callable[[asyncio.Task], None]:
    def callback(task: asyncio.Task) -> None:
        if not task.cancelled() and task.exception() is not None:
            logger.error(
                "wma: async %r handler failed", event, exc_info=task.exception()
            )

    return callback


@dataclass
class Session:
    """One realtime session tracked by a ``RealtimeApp`` runner."""

    session_id: str
    params: SessionParams
    handler: SessionEventHandler
    created_at: float
    last_seen_at: float
    # Server-only scratch space for app/base-class state (e.g. the peer
    # connection, upstream resource ids to release on close). Unlike
    # ``params``, the client can never write here — server-trusted values
    # (entitlements, limits, feature gates) belong in this dict.
    state: Dict[str, Any] = field(default_factory=dict)
    closed: bool = False


class SessionStore:
    """In-memory session registry with heartbeat-based expiry.

    Sessions live in runner memory, so heartbeats only work while requests
    from one client keep hitting the same runner. Keep concurrency low for
    apps that rely on server-side session state, or treat the state as
    advisory (clients must handle ``alive=false`` by reconnecting — WMA
    sessions are not resumable either way).
    """

    def __init__(
        self,
        *,
        timeout_sec: float,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._timeout_sec = timeout_sec
        self._clock = clock
        self._sessions: Dict[str, Session] = {}

    def __len__(self) -> int:
        return len(self._sessions)

    def create(self, params: SessionParams) -> Session:
        now = self._clock()
        session = Session(
            session_id=f"sess_{uuid.uuid4().hex}",
            params=params,
            handler=SessionEventHandler(),
            created_at=now,
            last_seen_at=now,
        )
        self._sessions[session.session_id] = session
        return session

    def _is_expired(self, session: Session) -> bool:
        return self._clock() - session.last_seen_at > self._timeout_sec

    def get(self, session_id: str) -> Union[Session, None]:
        session = self._sessions.get(session_id)
        if session is None or self._is_expired(session):
            return None
        return session

    def heartbeat(self, session_id: str) -> Union[Session, None]:
        session = self.get(session_id)
        if session is not None:
            session.last_seen_at = self._clock()
        return session

    def pop(self, session_id: str) -> Union[Session, None]:
        return self._sessions.pop(session_id, None)

    def pop_expired(self) -> List[Session]:
        expired = [s for s in self._sessions.values() if self._is_expired(s)]
        for session in expired:
            self._sessions.pop(session.session_id, None)
        return expired


class SDPMessage(BaseModel):
    sdp: str = Field(description="Session description protocol payload.")
    type: str = Field(description="SDP message type: `offer` or `answer`.")


class SDPOffer(SDPMessage):
    """Request-side SDP: only `offer` is negotiable, so anything else (e.g.
    an `answer`) is rejected as a field-level 422 instead of reaching aiortc
    and failing mid-negotiation after `on_connect` has already run."""

    type: Literal["offer"] = Field(
        default="offer", description="SDP message type; must be `offer`."
    )


class StartSessionRequest(BaseModel):
    session_params: SessionParams = Field(
        default_factory=dict,
        description=(
            "Mutable session parameters shared between client and runner. "
            "The app validates and consumes app-specific keys."
        ),
    )
    offer: Union[SDPOffer, None] = Field(
        default=None,
        description=(
            "Optional WebRTC SDP offer. Only used by apps that terminate "
            "media on the runner; control-plane apps ignore it."
        ),
    )


class StartSessionResponse(BaseModel):
    session_id: str = Field(description="Server-issued session identifier.")
    session_timeout_sec: float = Field(
        description="Seconds of heartbeat silence after which the session expires."
    )
    heartbeat_interval_sec: float = Field(
        description="Recommended interval between heartbeat calls, in seconds."
    )
    answer: Union[SDPMessage, None] = Field(
        default=None,
        description="WebRTC SDP answer; present only when an offer was sent.",
    )
    connection: Dict[str, Any] = Field(
        default_factory=dict,
        description="App-specific connection material returned by on_connect.",
    )


class SessionRef(BaseModel):
    session_id: str = Field(description="Session identifier from POST /session.")


class HeartbeatResponse(BaseModel):
    session_id: str
    alive: bool = Field(
        description=(
            "False when the session is unknown or expired. Sessions are not "
            "resumable: create a fresh session instead of retrying."
        )
    )
    heartbeat_interval_sec: float


class CloseSessionResponse(BaseModel):
    session_id: str
    closed: bool = Field(
        description="False when the session was already gone (call is idempotent)."
    )


class RealtimeApp(fal.App):
    """``fal.wma.RealtimeApp`` look-alike built on a naked ``fal.App``.

    Subclasses implement ``on_connect`` (and optionally ``on_disconnect``)
    and inherit the ``/session`` lifecycle endpoints. Regular
    ``@fal.endpoint`` routes can be added alongside as usual.
    """

    # Heartbeat silence tolerated before a session is reaped. The WMA bridge
    # expects a ~5s heartbeat cadence; 30s tolerates a few missed beats.
    session_timeout_sec: ClassVar[float] = 30.0
    heartbeat_interval_sec: ClassVar[float] = 5.0
    # Billable units reported for a successful POST /session.
    session_billable_units: ClassVar[int] = 1

    async def setup(self) -> None:
        self._wma_sessions = SessionStore(timeout_sec=self.session_timeout_sec)
        self._wma_reaper = asyncio.create_task(self._reap_expired_sessions())

    async def on_connect(
        self,
        event_handler: SessionEventHandler,
        session_params: SessionParams,
        request: Union[fastapi.Request, None] = None,
    ) -> Union[Dict[str, Any], None]:
        """Handle a new session; return the ``connection`` payload (if any).

        ``request`` is the raw HTTP request behind ``POST /session`` (when
        available), so apps can read server-trusted headers such as the
        fal caller identity. It defaults to ``None`` to stay signature-
        compatible with the documented ``fal.wma`` contract.
        """
        raise NotImplementedError

    async def on_disconnect(self, session: Session, reason: str) -> None:
        """Release app-held resources for ``session``; default no-op."""

    def _on_connect_request_mode(self) -> Literal["none", "positional", "keyword"]:
        """Inspect an override once to preserve the original two-arg contract."""
        cached = getattr(self, "_wma_on_connect_request_mode", None)
        if cached is not None:
            return cached

        parameters = tuple(inspect.signature(self.on_connect).parameters.values())
        request_parameter = next(
            (parameter for parameter in parameters if parameter.name == "request"),
            None,
        )
        if request_parameter is not None:
            if request_parameter.kind is inspect.Parameter.POSITIONAL_ONLY:
                mode: Literal["none", "positional", "keyword"] = "positional"
            else:
                mode = "keyword"
        elif any(
            parameter.kind is inspect.Parameter.VAR_KEYWORD for parameter in parameters
        ):
            mode = "keyword"
        elif (
            any(
                parameter.kind is inspect.Parameter.VAR_POSITIONAL
                for parameter in parameters
            )
            or len(
                [
                    parameter
                    for parameter in parameters
                    if parameter.kind
                    in (
                        inspect.Parameter.POSITIONAL_ONLY,
                        inspect.Parameter.POSITIONAL_OR_KEYWORD,
                    )
                ]
            )
            >= 3
        ):
            mode = "positional"
        else:
            mode = "none"
        self._wma_on_connect_request_mode = mode
        return mode

    async def _invoke_on_connect(
        self,
        event_handler: SessionEventHandler,
        session_params: SessionParams,
        request: fastapi.Request,
    ) -> Union[Dict[str, Any], None]:
        """Call a two- or three-argument override without TypeError retries."""
        mode = self._on_connect_request_mode()
        if mode == "keyword":
            return await self.on_connect(event_handler, session_params, request=request)
        if mode == "positional":
            return await self.on_connect(event_handler, session_params, request)
        return await self.on_connect(event_handler, session_params)

    @fal.endpoint("/session")
    async def start_session(
        self,
        request: StartSessionRequest,
        response: Response,
        http_request: fastapi.Request,
    ) -> StartSessionResponse:
        # Zero by default so a failed session start (bad offer, on_connect
        # error) is never billed; overridden on the success path below.
        response.headers["x-fal-billable-units"] = "0"
        session = self._wma_sessions.create(dict(request.session_params))
        try:
            connection = (
                await self._invoke_on_connect(
                    session.handler, session.params, http_request
                )
                or {}
            )
            answer = None
            if request.offer is not None:
                answer = await self._negotiate_webrtc(session, request.offer)
        except BaseException:
            self._wma_sessions.pop(session.session_id)
            await self._finalize_session(session, reason="connect-failed")
            raise
        response.headers["x-fal-billable-units"] = str(self.session_billable_units)
        return StartSessionResponse(
            session_id=session.session_id,
            session_timeout_sec=self.session_timeout_sec,
            heartbeat_interval_sec=self.heartbeat_interval_sec,
            answer=answer,
            connection=connection,
        )

    @fal.endpoint("/session/heartbeat")
    async def session_heartbeat(
        self, request: SessionRef, response: Response
    ) -> HeartbeatResponse:
        session = self._wma_sessions.heartbeat(request.session_id)
        response.headers["x-fal-billable-units"] = "0"
        return HeartbeatResponse(
            session_id=request.session_id,
            alive=session is not None,
            heartbeat_interval_sec=self.heartbeat_interval_sec,
        )

    @fal.endpoint("/session/close")
    async def close_session(
        self, request: SessionRef, response: Response
    ) -> CloseSessionResponse:
        session = self._wma_sessions.pop(request.session_id)
        if session is not None:
            await self._finalize_session(session, reason="client-close")
        response.headers["x-fal-billable-units"] = "0"
        return CloseSessionResponse(
            session_id=request.session_id, closed=session is not None
        )

    async def _reap_expired_sessions(self) -> None:
        interval = max(1.0, min(self.session_timeout_sec / 2, 5.0))
        while True:
            await asyncio.sleep(interval)
            await self._finalize_expired_sessions(self._wma_sessions.pop_expired())

    async def _finalize_expired_sessions(self, sessions: List[Session]) -> None:
        """Finalize one expiry batch without one session killing the reaper."""
        for session in sessions:
            try:
                await self._finalize_session(session, reason="expired")
            except Exception:
                # Subclasses may override finalization. Keep later sessions
                # and future batches alive while still propagating
                # cancellation/system exits.
                logger.exception(
                    "wma: failed to finalize expired session %s",
                    session.session_id,
                )

    async def _finalize_session(self, session: Session, *, reason: str) -> None:
        if session.closed:
            return
        session.closed = True
        session.handler.dispatch("close", reason)
        for track in session.handler.tracks:
            try:
                track.stop()
            except Exception:
                logger.exception("wma: failed to stop track for %s", session.session_id)
        pc = session.state.get("_pc")
        if pc is not None:
            try:
                await pc.close()
            except Exception:
                logger.exception(
                    "wma: failed to close peer connection for %s", session.session_id
                )
        try:
            await self.on_disconnect(session, reason)
        except Exception:
            logger.exception(
                "wma: on_disconnect failed for %s (%s)", session.session_id, reason
            )

    async def _negotiate_webrtc(self, session: Session, offer: SDPOffer) -> SDPMessage:
        """Answer the client's SDP offer with aiortc (media apps only)."""
        try:
            from aiortc import RTCPeerConnection, RTCSessionDescription
        except ImportError:
            raise InputValueError.from_field_error(
                field="offer",
                msg=(
                    "This app does not terminate WebRTC media on the runner; "
                    "omit `offer` and use the `connection` payload instead."
                ),
                input={"offer": offer.type},
            )

        pc = RTCPeerConnection()
        session.state["_pc"] = pc

        @pc.on("track")
        def _on_track(track: Any) -> None:
            session.handler.dispatch("track", track)

        @pc.on("datachannel")
        def _on_datachannel(channel: Any) -> None:
            @channel.on("message")
            def _on_message(message: Any) -> None:
                try:
                    update = json.loads(message)
                except (TypeError, ValueError):
                    logger.warning("wma: ignoring non-JSON data-channel payload")
                    return
                if isinstance(update, dict):
                    # session.params is client-writable by contract (any key,
                    # any time) — trusted server-side values live in
                    # session.state, which this path can never reach.
                    session.params.update(update)
                    session.handler.dispatch("session_params", session.params)

        await pc.setRemoteDescription(
            RTCSessionDescription(filter_sdp_ice_candidates(offer.sdp), offer.type)
        )
        for track in session.handler.tracks:
            media_track = (
                track.as_media_stream_track()
                if isinstance(track, BatchedFnTrack)
                else track
            )
            pc.addTrack(media_track)
        answer = await pc.createAnswer()
        await pc.setLocalDescription(answer)
        local = pc.localDescription
        return SDPMessage(sdp=local.sdp, type=local.type)
