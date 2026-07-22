"""World Model Accelerator (WMA) support."""

import asyncio
import inspect
import json
import logging
import threading
from contextlib import suppress
from typing import (
    TYPE_CHECKING,
    Any,
    Awaitable,
    Callable,
    ClassVar,
    Dict,
    List,
    Optional,
    Protocol,
    Tuple,
)

from fastapi import Body
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from starlette.background import BackgroundTask

from fal.app import App as FalApp
from fal.app import endpoint

# The bridge times a session out after 60 seconds without a chunk from the
# runner, so keepalive comments must be emitted well within that window.
SSE_KEEPALIVE_INTERVAL = 15
DATA_CHANNEL_LABEL = "fal"

# Strong references to fire-and-forget peer-connection close tasks.
_CLOSE_TASKS: set = set()
_SESSION_TASKS: set = set()
logger = logging.getLogger(__name__)

__all__ = [
    "AiortcPeer",
    "App",
    "DATA_CHANNEL_LABEL",
    "GStreamerPeer",
    "PeerBackend",
    "PipelineSpec",
    "Session",
    "SessionAnswer",
    "SessionParams",
    "StartSessionRequest",
    "VideoProcessorBinding",
    "VideoProcessorPeer",
    "VideoProcessorPolicy",
    "VideoProcessorStats",
    "VideoProcessorTrack",
    "VideoSourcePeer",
    "VideoSourcePolicy",
    "VideoSourceStats",
    "VideoSourceTrack",
    "attach_video_processor",
]

if TYPE_CHECKING:
    from fal.wma_gstreamer import GStreamerPeer, PipelineSpec
    from fal.wma_models import (
        VideoProcessorBinding,
        VideoProcessorPeer,
        VideoProcessorPolicy,
        VideoProcessorStats,
        VideoProcessorTrack,
        VideoSourcePeer,
        VideoSourcePolicy,
        VideoSourceStats,
        VideoSourceTrack,
        attach_video_processor,
    )


class StartSessionRequest(BaseModel):
    sdp: str
    type: str = "offer"
    session_id: Optional[str] = None


class SessionAnswer(BaseModel):
    sdp: str
    type: str = "answer"
    metadata: Dict[str, Any] = Field(default_factory=dict)


class PeerBackend(Protocol):
    async def negotiate(self, offer: StartSessionRequest) -> SessionAnswer: ...

    async def wait_closed(self) -> None: ...

    async def close(self) -> None: ...


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


class Session:
    """Transport-neutral state and lifecycle for one WMA connection."""

    def __init__(self, request: StartSessionRequest) -> None:
        self.id = request.session_id
        self.offer = request
        self.params = SessionParams()
        self.answer_metadata: Dict[str, Any] = {}
        self.response_headers: Dict[str, str] = {}
        self._loop = asyncio.get_running_loop()
        self._handlers: Dict[str, List[Tuple[Callable[[dict], Any], bool]]] = {}
        self._channel_open_handlers: List[Callable[[], Any]] = []
        self._channel_is_open = False
        self._sender: Optional[Callable[[dict], bool]] = None
        self._sender_thread_safe = False
        self._backend: Optional[PeerBackend] = None
        self._cleanup: List[Callable[[], Any]] = []
        self._tasks: set = set()
        self._close_lock = asyncio.Lock()
        self._inline_condition = threading.Condition()
        self._inline_active = 0
        self._closed = asyncio.Event()
        self._is_closed = False
        self.params._bind(
            lambda params: self.send({"type": "session_params", "params": params})
        )

    @property
    def closed(self) -> asyncio.Event:
        return self._closed

    def bind_backend(self, backend: PeerBackend) -> None:
        if self._backend is not None:
            raise RuntimeError("WMA session already has a peer backend")
        self._backend = backend

    def bind_sender(
        self,
        sender: Callable[[dict], bool],
        *,
        thread_safe: bool = False,
    ) -> None:
        self._sender = sender
        self._sender_thread_safe = thread_safe

    def channel_opened(self) -> None:
        if self._channel_is_open:
            return
        self._channel_is_open = True
        if self.params:
            self.params._sync()
        for handler in self._channel_open_handlers:
            try:
                result = handler()
            except Exception:
                logger.exception("WMA channel-open handler failed")
                continue
            self._handle_result(result)

    def on_channel_open(self, handler: Callable[[], Any]) -> Callable[[], Any]:
        self._channel_open_handlers.append(handler)
        if self._channel_is_open:
            try:
                result = handler()
            except Exception:
                logger.exception("WMA channel-open handler failed")
            else:
                self._handle_result(result)
        return handler

    def on_message(
        self,
        kind: str,
        handler: Callable[[dict], Any],
        *,
        inline: bool = False,
    ) -> Callable[[dict], Any]:
        self._handlers.setdefault(kind, []).append((handler, inline))
        return handler

    def receive(self, message: dict) -> None:
        if self._is_closed or not isinstance(message, dict):
            return
        try:
            running_loop = asyncio.get_running_loop()
        except RuntimeError:
            running_loop = None
        if running_loop is self._loop:
            self._dispatch(message)
        else:
            kind = message.get("type")
            if kind == "ping" and kind not in self._handlers:
                self._dispatch(message)
            elif kind == "session_params":
                self._loop.call_soon_threadsafe(self._dispatch, message)
            elif self._dispatch_inline(message):
                self._loop.call_soon_threadsafe(self._dispatch, message, True)

    def _dispatch_inline(self, message: dict) -> bool:
        kind = message.get("type")
        if not isinstance(kind, str):
            return False
        handlers = self._handlers.get(kind, [])
        with self._inline_condition:
            if self._is_closed:
                return False
            self._inline_active += 1
        try:
            for handler, inline in handlers:
                if inline:
                    self._invoke_handler(handler, message)
        finally:
            with self._inline_condition:
                self._inline_active -= 1
                if self._inline_active == 0:
                    self._inline_condition.notify_all()
        return any(not inline for _handler, inline in handlers)

    def _dispatch(self, message: dict, skip_inline: bool = False) -> None:
        if self._is_closed:
            return
        kind = message.get("type")
        if kind == "ping" and kind not in self._handlers:
            timestamp = message.get("client_ts", message.get("ts"))
            self.send({"type": "pong", "client_ts": timestamp})
            return
        if kind == "session_params":
            params = message.get("params")
            if isinstance(params, dict):
                self.params._merge_from_client(params)
            return
        if not isinstance(kind, str):
            return
        for handler, inline in self._handlers.get(kind, []):
            if not (skip_inline and inline):
                self._invoke_handler(handler, message)

    def _invoke_handler(
        self,
        handler: Callable[[dict], Any],
        message: dict,
    ) -> None:
        try:
            result = handler(message)
        except Exception:
            logger.exception("WMA message handler failed for %r", message.get("type"))
            return
        self._handle_result(result)

    def _handle_result(self, result: Any) -> None:
        if inspect.isawaitable(result):
            try:
                running_loop = asyncio.get_running_loop()
            except RuntimeError:
                running_loop = None
            if running_loop is self._loop:
                self.create_task(result)
            else:
                self._loop.call_soon_threadsafe(self.create_task, result)

    def send(self, message: dict) -> bool:
        sender = self._sender
        if sender is None or self._is_closed:
            return False
        try:
            running_loop = asyncio.get_running_loop()
        except RuntimeError:
            running_loop = None
        if not self._sender_thread_safe and running_loop is not self._loop:
            self._loop.call_soon_threadsafe(sender, message)
            return True
        return sender(message)

    def defer(self, cleanup: Callable[[], Any]) -> None:
        self._cleanup.append(cleanup)

    def set_response_header(self, name: str, value: str) -> None:
        self.response_headers[name] = value

    def create_task(self, awaitable: Awaitable[Any]) -> Optional[asyncio.Task]:
        if self._is_closed:
            if inspect.iscoroutine(awaitable):
                awaitable.close()
            elif isinstance(awaitable, asyncio.Future):
                awaitable.cancel()
            return None

        async def run() -> Any:
            return await awaitable

        task: asyncio.Task = self._loop.create_task(run())
        self._tasks.add(task)
        task.add_done_callback(self._task_done)
        return task

    def _task_done(self, task: asyncio.Task) -> None:
        self._tasks.discard(task)
        if task.cancelled():
            return
        error = task.exception()
        if error is not None:
            logger.error(
                "WMA session task failed",
                exc_info=(type(error), error, error.__traceback__),
            )

    async def wait_closed(self) -> None:
        await self._closed.wait()

    async def close(self) -> None:
        async with self._close_lock:
            if self._is_closed:
                return
            with self._inline_condition:
                self._is_closed = True
            self._closed.set()

            backend = self._backend
            if backend is not None:
                with suppress(Exception):
                    await backend.close()

            await asyncio.to_thread(self._wait_for_inline_handlers)

            current = asyncio.current_task()
            tasks = [task for task in self._tasks if task is not current]
            for task in tasks:
                task.cancel()
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)

            for cleanup in reversed(self._cleanup):
                try:
                    result = cleanup()
                    if inspect.isawaitable(result):
                        await result
                except Exception:
                    pass
            self._cleanup.clear()

    def _wait_for_inline_handlers(self) -> None:
        with self._inline_condition:
            while self._inline_active:
                self._inline_condition.wait()


class App(FalApp):
    """A WMA application whose media transport is selected per session."""

    request_timeout: ClassVar[Optional[int]] = 3660

    async def create_backend(self, session: Session) -> PeerBackend:
        raise NotImplementedError("WMA App subclasses must implement create_backend()")

    @endpoint("/start-session")
    async def start_session(
        self, request: StartSessionRequest = Body(...)
    ) -> StreamingResponse:
        session = Session(request)
        try:
            backend = await self.create_backend(session)
            session.bind_backend(backend)
            answer = await backend.negotiate(request)
        except BaseException:
            await session.close()
            raise

        stream_started = asyncio.Event()

        async def close_session() -> None:
            close_task = asyncio.ensure_future(session.close())
            _CLOSE_TASKS.add(close_task)
            close_task.add_done_callback(_CLOSE_TASKS.discard)
            with suppress(asyncio.CancelledError):
                await asyncio.shield(close_task)

        async def close_if_stream_never_starts() -> None:
            try:
                await asyncio.wait_for(stream_started.wait(), timeout=5)
            except asyncio.TimeoutError:
                await close_session()

        async def event_stream():
            stream_started.set()
            backend_closed = asyncio.ensure_future(backend.wait_closed())
            try:
                payload = {
                    **session.answer_metadata,
                    **answer.metadata,
                    "sdp": answer.sdp,
                    "type": answer.type,
                    "session_id": request.session_id,
                }
                yield "data: " + json.dumps(payload) + "\n\n"

                while not backend_closed.done():
                    done, _ = await asyncio.wait(
                        {backend_closed},
                        timeout=SSE_KEEPALIVE_INTERVAL,
                    )
                    if not done:
                        yield ": keepalive\n\n"
            finally:
                if not backend_closed.done():
                    backend_closed.cancel()
                with suppress(asyncio.CancelledError):
                    await backend_closed
                await close_session()

        watchdog = asyncio.ensure_future(close_if_stream_never_starts())
        _SESSION_TASKS.add(watchdog)
        watchdog.add_done_callback(_SESSION_TASKS.discard)
        return StreamingResponse(
            event_stream(),
            media_type="text/event-stream",
            headers=session.response_headers,
            background=BackgroundTask(close_session),
        )


class AiortcPeer:
    """aiortc implementation of the WMA peer backend contract."""

    def __init__(
        self,
        session: Session,
        on_connect: Callable[[Any], Any],
        *,
        create_default_channel: bool = True,
        rtc_configuration: Any = None,
        peer_connection_factory: Optional[Callable[[], Any]] = None,
        disconnected_grace_seconds: Optional[float] = 0,
    ) -> None:
        if rtc_configuration is not None and peer_connection_factory is not None:
            raise ValueError(
                "rtc_configuration and peer_connection_factory are mutually exclusive"
            )
        if disconnected_grace_seconds is not None and disconnected_grace_seconds < 0:
            raise ValueError("disconnected_grace_seconds cannot be negative")
        self._session = session
        self._on_connect = on_connect
        self._create_default_channel = create_default_channel
        self._rtc_configuration = rtc_configuration
        self._peer_connection_factory = peer_connection_factory
        self._disconnected_grace_seconds = disconnected_grace_seconds
        self._pc: Any = None
        self._channel: Any = None
        self._closed = asyncio.Event()
        self._disconnect_task: Optional[asyncio.Task] = None

    async def negotiate(self, offer: StartSessionRequest) -> SessionAnswer:
        from aiortc import RTCPeerConnection, RTCSessionDescription  # noqa: PLC0415

        if self._peer_connection_factory is not None:
            pc = self._peer_connection_factory()
            if inspect.isawaitable(pc):
                pc = await pc
        elif self._rtc_configuration is not None:
            pc = RTCPeerConnection(configuration=self._rtc_configuration)
        else:
            pc = RTCPeerConnection()
        self._pc = pc
        if self._create_default_channel:
            self._register_channel(
                pc.createDataChannel(DATA_CHANNEL_LABEL),
                primary=True,
            )
        self._session.bind_sender(self.send)

        @pc.on("datachannel")
        def _on_datachannel(channel: Any) -> None:
            self._register_channel(channel)

        @pc.on("connectionstatechange")
        def _on_connection_state_change() -> None:
            if pc.connectionState == "connected":
                self._cancel_disconnect_timer()
            elif pc.connectionState == "disconnected":
                self._start_disconnect_timer(pc)
            elif pc.connectionState in ("closed", "failed"):
                self._cancel_disconnect_timer()
                self._closed.set()
                if pc.connectionState == "failed":
                    task = asyncio.ensure_future(pc.close())
                    _CLOSE_TASKS.add(task)
                    task.add_done_callback(_CLOSE_TASKS.discard)

        try:
            result = self._on_connect(pc)
            if inspect.isawaitable(result):
                await result
            await pc.setRemoteDescription(
                RTCSessionDescription(sdp=offer.sdp, type=offer.type)
            )
            answer = await pc.createAnswer()
            await pc.setLocalDescription(answer)
        except BaseException:
            await self.close()
            raise

        return SessionAnswer(
            sdp=pc.localDescription.sdp,
            type=pc.localDescription.type,
        )

    def _start_disconnect_timer(self, pc: Any) -> None:
        self._cancel_disconnect_timer()
        grace = self._disconnected_grace_seconds
        if grace is None:
            return
        if grace == 0:
            self._closed.set()
            return
        delay = grace

        async def close_after_grace() -> None:
            try:
                await asyncio.sleep(delay)
                if pc is self._pc and pc.connectionState == "disconnected":
                    self._closed.set()
            except asyncio.CancelledError:
                raise

        self._disconnect_task = asyncio.create_task(close_after_grace())

    def _cancel_disconnect_timer(self) -> None:
        task, self._disconnect_task = self._disconnect_task, None
        if task is not None and not task.done():
            task.cancel()

    def _register_channel(self, channel: Any, primary: bool = False) -> None:
        @channel.on("message")
        def on_message(raw: Any) -> None:
            if isinstance(raw, bytes):
                try:
                    raw = raw.decode()
                except UnicodeDecodeError:
                    return
            try:
                message = json.loads(raw)
            except (TypeError, ValueError):
                return
            if isinstance(message, dict):
                self._session.receive(message)

        def make_current() -> None:
            if primary or self._channel is None:
                self._channel = channel
                self._session.channel_opened()

        if channel.readyState == "open":
            make_current()
        else:
            channel.on("open", make_current)

        @channel.on("close")
        def close_current_channel() -> None:
            if channel is self._channel:
                self._closed.set()

    def send(self, message: dict) -> bool:
        channel = self._channel
        if channel is None or channel.readyState != "open":
            return False
        _send_if_open(channel, json.dumps(message))
        return True

    async def wait_closed(self) -> None:
        await self._closed.wait()

    async def close(self) -> None:
        disconnect_task, self._disconnect_task = self._disconnect_task, None
        if disconnect_task is not None and not disconnect_task.done():
            disconnect_task.cancel()
            with suppress(asyncio.CancelledError):
                await disconnect_task
        pc, self._pc = self._pc, None
        self._channel = None
        self._closed.set()
        if pc is not None:
            await pc.close()


def __getattr__(name: str) -> Any:
    if name in {"GStreamerPeer", "PipelineSpec"}:
        from fal import wma_gstreamer

        return getattr(wma_gstreamer, name)
    if name in {
        "VideoProcessorBinding",
        "VideoProcessorPeer",
        "VideoProcessorPolicy",
        "VideoProcessorStats",
        "VideoProcessorTrack",
        "VideoSourcePeer",
        "VideoSourcePolicy",
        "VideoSourceStats",
        "VideoSourceTrack",
        "attach_video_processor",
    }:
        from fal import wma_models

        return getattr(wma_models, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
