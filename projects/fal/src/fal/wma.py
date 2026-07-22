"""World Model Accelerator (WMA) support."""

import asyncio
import inspect
import json
import uuid
from collections import deque
from typing import Any, Callable, ClassVar, List, Optional, Tuple

from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from fal.app import App, endpoint

WMA_APP_REQUIREMENTS = ["aiortc"]

# The bridge times a session out after 60 seconds without a chunk from the
# runner, so keepalive comments must be emitted well within that window.
SSE_KEEPALIVE_INTERVAL = 15
DATA_CHANNEL_LABEL = "fal"


class StartSessionRequest(BaseModel):
    sdp: str
    type: str = "offer"
    session_id: Optional[str] = None


class SessionParams(dict):
    """Session-scoped parameters kept in sync with the client.

    Mutations made on the server are pushed to the client over the session
    data channel as ``{"type": "session_params", "params": {...}}`` messages.
    Messages of the same shape sent by the client are merged in without being
    echoed back.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._push: Optional[Callable[[dict], Any]] = None

    def _bind(self, push: Callable[[dict], Any]) -> None:
        self._push = push

    def _sync(self) -> None:
        if self._push is not None:
            self._push(dict(self))

    def _merge_from_client(self, params: dict) -> None:
        dict.update(self, params)

    def __setitem__(self, key: Any, value: Any) -> None:
        dict.__setitem__(self, key, value)
        self._sync()

    def __delitem__(self, key: Any) -> None:
        dict.__delitem__(self, key)
        self._sync()

    def update(self, *args: Any, **kwargs: Any) -> None:
        dict.update(self, *args, **kwargs)
        self._sync()

    def pop(self, *args: Any) -> Any:
        result = dict.pop(self, *args)
        self._sync()
        return result

    def clear(self) -> None:
        dict.clear(self)
        self._sync()

    def setdefault(self, key: Any, default: Any = None) -> Any:
        result = dict.setdefault(self, key, default)
        self._sync()
        return result

    def popitem(self) -> Tuple[Any, Any]:
        result = dict.popitem(self)
        self._sync()
        return result

    def __ior__(self, other: Any) -> "SessionParams":  # type: ignore[misc, override]
        dict.update(self, other)
        self._sync()
        return self


def _send_if_open(channel: Any, payload: str) -> None:
    if channel.readyState == "open":
        channel.send(payload)


class EventHandler:
    """Per-session helper handed to :meth:`RealtimeApp.on_connect`.

    Registers callbacks on the session's ``RTCPeerConnection``, attaches
    outbound tracks, and exchanges JSON messages with the client over the
    session data channel.
    """

    def __init__(
        self,
        peer_connection: Any,
        session_params: SessionParams,
        session_id: Optional[str] = None,
    ) -> None:
        self.peer_connection = peer_connection
        self.session_id = session_id
        self._session_params = session_params
        self._channel: Optional[Any] = None
        try:
            self._loop: Optional[asyncio.AbstractEventLoop] = asyncio.get_running_loop()
        except RuntimeError:
            self._loop = None
        session_params._bind(
            lambda params: self.send({"type": "session_params", "params": params})
        )

    def on(self, event: str) -> Callable:
        """Register a callback for a peer connection event (e.g. ``"track"``)."""

        def decorator(fn: Callable) -> Callable:
            self.peer_connection.on(event, fn)
            return fn

        return decorator

    def add_track(self, track: Any) -> Any:
        """Attach an outbound media track to the peer connection."""
        return self.peer_connection.addTrack(track)

    def send(self, message: dict) -> bool:
        """Send a JSON message to the client over the session data channel.

        Returns False if no data channel is open yet; session_params are
        re-synchronized automatically once one opens. Safe to call from
        worker threads (e.g. a synchronous ``BatchedFnTrack`` function):
        the send is then scheduled onto the session's event loop.
        """
        channel = self._channel
        if channel is None or channel.readyState != "open":
            return False
        payload = json.dumps(message)
        try:
            running_loop: Optional[asyncio.AbstractEventLoop] = (
                asyncio.get_running_loop()
            )
        except RuntimeError:
            running_loop = None
        if self._loop is not None and running_loop is not self._loop:
            # aiortc data channels are not thread-safe.
            self._loop.call_soon_threadsafe(_send_if_open, channel, payload)
        else:
            channel.send(payload)
        return True

    def _register_channel(self, channel: Any, primary: bool = False) -> None:
        @channel.on("message")
        def _on_message(raw: Any) -> None:
            self._handle_message(channel, raw)

        def make_current() -> None:
            if primary or self._channel is None:
                self._channel = channel
                if self._session_params:
                    self._session_params._sync()

        if channel.readyState == "open":
            make_current()
        else:
            channel.on("open", make_current)

    def _handle_message(self, channel: Any, raw: Any) -> None:
        if isinstance(raw, bytes):
            try:
                raw = raw.decode()
            except UnicodeDecodeError:
                return
        try:
            message = json.loads(raw)
        except (TypeError, ValueError):
            return
        if not isinstance(message, dict):
            return

        kind = message.get("type")
        if kind == "ping":
            _send_if_open(
                channel,
                json.dumps({"type": "pong", "client_ts": message.get("ts")}),
            )
        elif kind == "session_params":
            params = message.get("params")
            if isinstance(params, dict):
                self._session_params._merge_from_client(params)


def _media_stream_track_base() -> type:
    # aiortc requires outbound tracks to be MediaStreamTrack instances, but it
    # is only installed on runners; fall back to a plain class so this module
    # stays importable (e.g. during `fal deploy`) without it.
    try:
        from aiortc.mediastreams import MediaStreamTrack  # noqa: PLC0415
    except ImportError:
        return object
    return MediaStreamTrack


_MEDIA_STREAM_TRACK_BASE = _media_stream_track_base()


class BatchedFnTrack(_MEDIA_STREAM_TRACK_BASE):  # type: ignore[misc, valid-type]
    """A video track that runs batches of incoming frames through a function.

    Buffers ``batch_size`` frames from ``source``, invokes ``fn`` with the
    batch (in a thread pool executor when ``fn`` is synchronous, so inference
    does not block the event loop), and yields the returned frames one by one.

    ``fn`` may return ``av.VideoFrame`` objects or numpy arrays; arrays are
    converted with ``av.VideoFrame.from_ndarray(..., format=format)``. Output
    frames without a ``pts`` inherit the timing of the input frame at the same
    batch position.
    """

    kind = "video"

    def __init__(
        self,
        source: Any,
        *,
        batch_size: int = 1,
        fn: Callable[[list], Any],
        format: str = "bgr24",
    ) -> None:
        if _MEDIA_STREAM_TRACK_BASE is not object:
            super().__init__()
        else:
            self.id = str(uuid.uuid4())
        self._source = source
        self.batch_size = batch_size
        self.fn = fn
        self.format = format
        self._pending: list = []
        self._ready: deque = deque()

    async def recv(self) -> Any:
        while not self._ready:
            frame = await self._source.recv()
            self._pending.append(frame)
            if len(self._pending) >= self.batch_size:
                batch, self._pending = self._pending, []
                self._ready.extend(await self._process(batch))
        return self._ready.popleft()

    async def _process(self, batch: list) -> list:
        if inspect.iscoroutinefunction(self.fn):
            outputs = await self.fn(batch)
        else:
            loop = asyncio.get_running_loop()
            outputs = await loop.run_in_executor(None, self.fn, batch)

        frames = []
        for index, output in enumerate(outputs):
            frame = output
            if not hasattr(frame, "pts"):
                import av  # noqa: PLC0415

                frame = av.VideoFrame.from_ndarray(output, format=self.format)
            if frame.pts is None and index < len(batch):
                frame.pts = batch[index].pts
                frame.time_base = batch[index].time_base
            frames.append(frame)
        return frames

    def stop(self) -> None:
        self._source.stop()
        if _MEDIA_STREAM_TRACK_BASE is not object:
            super().stop()


class RealtimeApp(App):
    """A fal application serving interactive WebRTC sessions through WMA.

    Subclasses implement :meth:`on_connect`, which is invoked once per
    incoming session with an :class:`EventHandler` for wiring up tracks and
    data channels, and a :class:`SessionParams` dict that stays synchronized
    with the client for the session's lifetime.

    Each active session holds its ``/start-session`` request open until the
    client disconnects, so ``max_multiplexing`` controls how many concurrent
    sessions a single runner serves.
    """

    # The WMA bridge caps sessions at one hour (SESSION_MAX_AGE); the
    # /start-session request stays open for the whole session, so the platform
    # request timeout must outlast it. The extra minute ensures the bridge's
    # clean teardown fires before the platform timeout does.
    request_timeout: ClassVar[Optional[int]] = 3660

    _extra_serve_requirements: ClassVar[List[str]] = WMA_APP_REQUIREMENTS

    async def on_connect(
        self, event_handler: EventHandler, session_params: SessionParams
    ) -> None:
        raise NotImplementedError("RealtimeApp subclasses must implement on_connect()")

    @endpoint("/start-session")
    async def start_session(self, request: StartSessionRequest) -> StreamingResponse:
        from aiortc import (  # noqa: PLC0415
            RTCPeerConnection,
            RTCSessionDescription,
        )

        pc = RTCPeerConnection()
        closed = asyncio.Event()

        try:
            session_params = SessionParams()
            handler = EventHandler(pc, session_params, session_id=request.session_id)

            handler._register_channel(
                pc.createDataChannel(DATA_CHANNEL_LABEL), primary=True
            )

            @pc.on("datachannel")
            def _on_datachannel(channel: Any) -> None:
                handler._register_channel(channel)

            @pc.on("connectionstatechange")
            def _on_connection_state_change() -> None:
                if pc.connectionState in ("closed", "failed", "disconnected"):
                    closed.set()
                    if pc.connectionState == "failed":
                        # Safety net in case the SSE generator never ran
                        # (request dropped before the first chunk).
                        asyncio.ensure_future(pc.close())

            result = self.on_connect(handler, session_params)
            if inspect.isawaitable(result):
                await result

            await pc.setRemoteDescription(
                RTCSessionDescription(sdp=request.sdp, type=request.type)
            )
            answer = await pc.createAnswer()
            # aiortc completes ICE gathering here, so the answer SDP below
            # already contains all candidates (WMA does not use trickle ICE).
            await pc.setLocalDescription(answer)
        except BaseException:
            await pc.close()
            raise

        async def event_stream():
            try:
                yield (
                    "data: "
                    + json.dumps(
                        {
                            "sdp": pc.localDescription.sdp,
                            "type": pc.localDescription.type,
                            "session_id": request.session_id,
                        }
                    )
                    + "\n\n"
                )

                # The bridge holds this stream open for the session's lifetime
                # and tears the session down when it ends. Closing the peer
                # connection ends the stream; the bridge dropping the request
                # cancels the generator and closes the peer connection.
                while not closed.is_set():
                    try:
                        await asyncio.wait_for(
                            closed.wait(), timeout=SSE_KEEPALIVE_INTERVAL
                        )
                    except asyncio.TimeoutError:
                        yield ": keepalive\n\n"
            finally:
                # When the bridge drops the request, anyio keeps re-cancelling
                # this task on every checkpoint, which would abort a plain
                # `await pc.close()` partway through. Run it as a separate
                # task so teardown completes even while we are cancelled.
                close_task = asyncio.ensure_future(pc.close())
                try:
                    await asyncio.shield(close_task)
                except asyncio.CancelledError:
                    pass

        return StreamingResponse(event_stream(), media_type="text/event-stream")
