"""Connection-oriented WMA application surface.

This is an experimental API: it may change in a minor release. See the
``fal.wma`` package docstring for the session protocol and lifecycle.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import logging
import math
import threading
from contextlib import suppress
from typing import (
    Any,
    Awaitable,
    Callable,
    ClassVar,
    Dict,
    List,
    Literal,
    Optional,
    Protocol,
)

from fastapi import Body, Header, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from starlette.background import BackgroundTask

import fal
from fal.compat import run_in_thread
from fal.wma._errors import InputValueError, InternalServerError
from fal.wma._raw import (
    INITIAL_CONNECT_TIMEOUT_SECONDS,
    ClientOfferError,
    close_peer_connection,
    negotiate_answer,
    sse_event,
)
from fal.wma._request_id import valid_fal_request_id
from fal.wma.contract import RealtimeContract, apply_contract, render_asyncapi
from fal.wma.telemetry import (
    CONNECTION_REPORT_VERSION,
    ConnectionReportObserver,
    observe_peer_connection,
)

SSE_KEEPALIVE_INTERVAL = 15
STREAM_START_TIMEOUT_SECONDS = 5
# fal-js creates this channel during offer construction. Keeping the label in
# one shared protocol constant makes the high-level peer and the published
# AsyncAPI describe the channel that browsers actually open.
DATA_CHANNEL_LABEL = "control"
START_SESSION_PATH = "/start-session"

FAL_BILLING_HEADER = "x-fal-billable-units"
FAL_BILLING_WEBHOOK_HEADER = "x-fal-billable-units-webhook"

_CLOSE_TASKS: set[asyncio.Task[Any]] = set()
_SESSION_TASKS: set[asyncio.Task[Any]] = set()
logger = logging.getLogger(__name__)

_BILLING_REST_CLIENT: Any = None


def _billing_rest_client() -> Any:
    """Process-wide fal REST client for deferred billing reports.

    Built lazily so importing this module never requires ``httpx`` (the
    module must stay importable on CPU hosts without the runner extras).
    """
    global _BILLING_REST_CLIENT  # noqa: PLW0603 - process-wide client cache
    if _BILLING_REST_CLIENT is None:
        from fal.wma._billing import make_fal_rest_client

        _BILLING_REST_CLIENT = make_fal_rest_client()
    return _BILLING_REST_CLIENT


# Pydantic evaluates model annotations at class-creation time, so these fields
# use ``typing`` generics rather than PEP 604/585 forms for the 3.8 floor.
class StartSessionRequest(BaseModel):
    """Offer forwarded by the WMA bridge."""

    sdp: str
    type: Literal["offer"] = "offer"
    session_id: Optional[str] = None
    ice_servers: List[Dict[str, Any]] = Field(default_factory=list)
    ice_status: Optional[str] = None
    credential_age_seconds: Optional[float] = Field(default=None, ge=0)


class SessionAnswer(BaseModel):
    """Negotiated answer returned by a :class:`PeerBackend`."""

    sdp: str
    type: str = "answer"
    metadata: Dict[str, Any] = Field(default_factory=dict)


class PeerBackend(Protocol):
    async def negotiate(self, offer: StartSessionRequest) -> SessionAnswer: ...

    async def wait_closed(self) -> None: ...

    async def close(self) -> None: ...


class SessionParams(Dict[str, Any]):
    """Session parameters synchronized over the WMA data channel."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._push: Callable[[dict[str, Any]], Any] | None = None

    def _bind(self, push: Callable[[dict[str, Any]], Any]) -> None:
        self._push = push

    def _sync(self) -> None:
        if self._push is not None:
            self._push(dict(self))

    def _merge_from_client(self, params: dict[str, Any]) -> None:
        super().update(params)

    def __setitem__(self, key: str, value: Any) -> None:
        dict.__setitem__(self, key, value)
        self._sync()

    def __delitem__(self, key: str) -> None:
        dict.__delitem__(self, key)
        self._sync()

    def update(self, *args: Any, **kwargs: Any) -> None:
        super().update(*args, **kwargs)
        self._sync()

    def pop(self, *args: Any) -> Any:
        result = dict.pop(self, *args)
        self._sync()
        return result

    def clear(self) -> None:
        super().clear()
        self._sync()

    def setdefault(self, key: str, default: Any = None) -> Any:
        result = super().setdefault(key, default)
        self._sync()
        return result

    def popitem(self) -> tuple[str, Any]:
        result = super().popitem()
        self._sync()
        return result

    def __ior__(self, other: Any) -> SessionParams:  # type: ignore[misc, override]
        super().update(other)
        self._sync()
        return self


def _send_if_open(channel: Any, payload: str) -> None:
    if channel.readyState == "open":
        channel.send(payload)


class Session:
    """Transport-neutral state and lifecycle for one WMA connection."""

    def __init__(
        self,
        request: StartSessionRequest,
        *,
        caller_user_id: str | None = None,
        request_id: str | None = None,
    ) -> None:
        self.id = request.session_id
        self.offer = request
        self.caller_user_id = caller_user_id
        # ``request_id`` is the caller-controlled ``x-fal-request-id`` header;
        # it is canonicalized (or dropped) here so a non-UUID value can never
        # reach the billing REST path (see fal.wma._request_id). The
        # isinstance guard covers direct ``start_session`` invocation, where
        # the FastAPI ``Header(None)`` sentinel arrives instead of a value.
        self.request_id = (
            valid_fal_request_id(request_id) if isinstance(request_id, str) else None
        )
        self._billable_units = 0.0
        self._billable_units_lock = threading.Lock()
        self._deferred_billing = False
        self.params = SessionParams()
        self.answer_metadata: dict[str, Any] = {}
        self.response_headers: dict[str, str] = {}
        self.state: dict[str, Any] = {}
        self._loop = asyncio.get_running_loop()
        self._handlers: dict[
            str, list[tuple[Callable[[dict[str, Any]], Any], bool]]
        ] = {}
        self._channel_open_handlers: list[Callable[[], Any]] = []
        self._channel_is_open = False
        self._sender: Callable[[dict[str, Any]], bool] | None = None
        self._sender_thread_safe = False
        self._backend: PeerBackend | None = None
        self._cleanup: list[Callable[[], Any]] = []
        self._tasks: set[asyncio.Task[Any]] = set()
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

    @property
    def billable_units(self) -> float:
        with self._billable_units_lock:
            return self._billable_units

    def add_billable_units(self, units: float = 1) -> None:
        """Accumulate usage for this session (e.g. one generated chunk).

        The unit is app-defined — chunks, seconds, tokens — and its price is
        platform pricing configuration, exactly as with the
        ``x-fal-billable-units`` header. The accumulated total is reported to
        fal billing once, automatically, when the session closes. Thread-safe:
        data-channel handlers may run off the session loop.
        """
        value = float(units)
        if not math.isfinite(value) or value < 0:
            raise ValueError("billable units must be a finite, non-negative number")
        with self._billable_units_lock:
            total = self._billable_units + value
            if not math.isfinite(total):
                # An overflowed total would fail the reporter's finite check
                # at close and void the WHOLE session's billing; refusing the
                # increment keeps everything accumulated so far billable.
                raise ValueError("billable units total overflowed")
            self._billable_units = total

    def _activate_deferred_billing(self) -> bool:
        """Switch this session's gateway request to report-at-close billing.

        Returns False (leaving billing on the immediate response headers)
        when there is no valid gateway request id to report against — e.g. a
        direct ``/start-session`` call that bypassed the fal gateway. Once
        activated, exactly one report is owed at close, even for zero units:
        the gateway parks the request as WAITING and only the report settles
        it.
        """
        if self.request_id is None:
            return False
        self._deferred_billing = True
        self.response_headers[FAL_BILLING_WEBHOOK_HEADER] = "1"
        return True

    async def _report_billable_units(self) -> None:
        if not self._deferred_billing or self.request_id is None:
            return
        self._deferred_billing = False
        try:
            from fal.wma._billing import report_stream_billing_units

            await report_stream_billing_units(
                _billing_rest_client(),
                self.request_id,
                self.billable_units,
                log_prefix="wma",
            )
        except Exception:
            # Billing must never break session teardown; a failed report
            # leaves the gateway request WAITING, which is the monitored
            # unbilled-session signal.
            logger.exception(
                "wma: billing report failed for request %s", self.request_id
            )

    def bind_backend(self, backend: PeerBackend) -> None:
        if self._backend is not None:
            raise RuntimeError("WMA session already has a peer backend")
        self._backend = backend

    def bind_sender(
        self,
        sender: Callable[[dict[str, Any]], bool],
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
        handler: Callable[[dict[str, Any]], Any],
        *,
        inline: bool = False,
    ) -> Callable[[dict[str, Any]], Any]:
        self._handlers.setdefault(kind, []).append((handler, inline))
        return handler

    def receive(self, message: dict[str, Any]) -> None:
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

    def _dispatch_inline(self, message: dict[str, Any]) -> bool:
        kind = message.get("type")
        if not isinstance(kind, str):
            return False
        handlers: list[tuple[Callable[[dict[str, Any]], Any], bool]] = (
            self._handlers.get(kind) or self._handlers.get("*") or []
        )
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

    def _dispatch(self, message: dict[str, Any], skip_inline: bool = False) -> None:
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
        handlers: list[tuple[Callable[[dict[str, Any]], Any], bool]] = (
            self._handlers.get(kind) or self._handlers.get("*") or []
        )
        for handler, inline in handlers:
            if not (skip_inline and inline):
                self._invoke_handler(handler, message)

    def _invoke_handler(
        self,
        handler: Callable[[dict[str, Any]], Any],
        message: dict[str, Any],
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

    def send(self, message: dict[str, Any]) -> bool:
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

    def create_task(self, awaitable: Awaitable[Any]) -> asyncio.Task[Any] | None:
        if self._is_closed:
            if inspect.iscoroutine(awaitable):
                awaitable.close()
            elif isinstance(awaitable, asyncio.Future):
                awaitable.cancel()
            return None

        async def run() -> Any:
            return await awaitable

        task = self._loop.create_task(run())
        self._tasks.add(task)
        task.add_done_callback(self._task_done)
        return task

    def _task_done(self, task: asyncio.Task[Any]) -> None:
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

            with self._inline_condition:
                inline_active = self._inline_active != 0
            if inline_active:
                await run_in_thread(self._wait_for_inline_handlers)

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
                    logger.warning("WMA deferred cleanup failed", exc_info=True)
            self._cleanup.clear()

            # Last step of the single close pass: the total is final once
            # handlers and tasks have stopped, and every teardown path
            # (client close, WebRTC drop, bridge abort, reaper, watchdog)
            # funnels through here exactly once.
            await self._report_billable_units()

            # Drop the session's object graph so peers, tracks, and handler
            # closures (which often hold GPU state) become refcount-collectable
            # at close instead of waiting for a cyclic GC pass: ``params`` is
            # bound to a lambda that closes over ``self``. Public ``state`` and
            # ``params`` contents are left readable.
            self._backend = None
            self._sender = None
            self._handlers.clear()
            self._channel_open_handlers.clear()
            self.params._push = None

    def _wait_for_inline_handlers(self) -> None:
        with self._inline_condition:
            while self._inline_active:
                self._inline_condition.wait()


def _schema_prefix(app_class_name: str) -> str:
    """Namespace for an app's published message schemas.

    ``components/schemas`` is flat and shared with the app's own request and
    response models, so the names generated here have to be unlikely to collide
    with them. The class name qualifies them; the ``App`` suffix is dropped
    because it says nothing about the message.
    """

    # str.removesuffix is 3.9+; the SDK supports 3.8.
    trimmed = (
        app_class_name[: -len("App")]
        if app_class_name.endswith("App")
        else app_class_name
    )
    return trimmed or app_class_name


class App(fal.App):
    """A WMA app whose one lifecycle endpoint owns a long-lived connection."""

    def __init_subclass__(cls, **kwargs: Any) -> None:
        # ``fal.App`` stores its resolved name on the class. Without clearing
        # the intermediate WMA base's inherited value ("app"), every concrete
        # subclass ignores both ``name=...`` and its own class-derived default.
        cls.app_name = None
        super().__init_subclass__(**kwargs)

    # ``secrets`` is deliberately NOT overridden here: an explicit ``[]`` is
    # an opt-in to *zero* secrets that every subclass would inherit, silently
    # stripping dashboard/pyproject-configured secrets from ported apps. WMA
    # itself needs no ambient credentials, so subclasses should declare the
    # narrowest ``secrets`` list for what they actually consume.

    # ``request_timeout`` is deliberately NOT set here: it is managed
    # dynamically per deployment by the platform, and a class default would
    # override that for every WMA app.

    #: What this app's live session needs and accepts. OpenAPI publishes the
    #: discovery link; :meth:`asyncapi` publishes the live-session contract.
    #: Left unset, the app's OpenAPI document is exactly what it was before
    #: contracts existed.
    realtime_contract: ClassVar[RealtimeContract | None] = None

    async def create_backend(self, session: Session) -> PeerBackend:
        raise NotImplementedError("WMA App subclasses must implement create_backend()")

    def openapi(self) -> dict[str, Any]:
        spec = super().openapi()
        contract = type(self).realtime_contract
        if contract is None:
            return spec
        return apply_contract(spec, path=START_SESSION_PATH)

    def asyncapi(self) -> dict[str, Any]:
        """Build the standalone realtime contract paired with ``openapi()``."""

        contract = type(self).realtime_contract
        if contract is None:
            raise ValueError("this WMA app does not declare a realtime contract")

        openapi_spec = self.openapi()
        session_path = openapi_spec.get("paths", {}).get(START_SESSION_PATH)
        if session_path is None or "post" not in session_path:
            raise ValueError(
                f"{type(self).__name__} declares a realtime_contract but does "
                f"not serve POST {START_SESSION_PATH}; the contract documents "
                "the session that endpoint negotiates, so it must exist"
            )
        operation_id = session_path["post"]["operationId"]
        return render_asyncapi(
            contract,
            title=f"{_schema_prefix(type(self).__name__)} realtime client API",
            schema_prefix=_schema_prefix(type(self).__name__),
            channel_address=DATA_CHANNEL_LABEL,
            openapi_operation_id=operation_id,
        )

    @classmethod
    def build_metadata(cls) -> dict[str, Any]:
        """Publish both documents through the deployment metadata channel."""

        app = cls(_allow_init=True)
        metadata = {"openapi": app.openapi()}
        if cls.realtime_contract is not None:
            metadata["asyncapi"] = app.asyncapi()
        return metadata

    @fal.endpoint(START_SESSION_PATH)
    async def start_session(
        self,
        request: StartSessionRequest = Body(...),
        # FastAPI resolves these annotations at runtime; ``typing`` forms keep
        # the 3.8 floor.
        x_fal_caller_user_id: Optional[str] = Header(None),
        x_fal_request_id: Optional[str] = Header(None),
    ) -> StreamingResponse:
        session = Session(
            request,
            caller_user_id=x_fal_caller_user_id,
            request_id=x_fal_request_id,
        )
        try:
            backend = await self.create_backend(session)
            session.bind_backend(backend)
            answer = await backend.negotiate(request)
        except ClientOfferError as exc:
            await session.close()
            # ClientOfferError is raised only when applying the request's SDP
            # (``type`` is already constrained to "offer" by validation), so
            # locate the 422 at the offending field.
            raise InputValueError.from_field_error(
                field="sdp",
                msg=f"WebRTC negotiation failed: {exc}",
            ) from exc
        except Exception as exc:
            await session.close()
            if isinstance(exc, HTTPException):
                # Platform-shaped errors already carry their own billing/retry
                # headers; merge in session headers (e.g. the app's
                # ``x-fal-billable-units: 0``) they did not set themselves.
                exc.headers = {**session.response_headers, **(exc.headers or {})}
                raise
            # A server-side setup/negotiation fault (aiortc createAnswer,
            # local-description, ICE gathering, ...) raised before the
            # streaming response carrying ``session.response_headers`` exists.
            # Translate it so the error response still answers with
            # ``x-fal-billable-units: 0`` and leaks no library internals.
            logger.exception("WMA session setup failed before streaming began")
            raise InternalServerError(input=None) from exc
        except BaseException:
            # Cancellation / shutdown: clean up but propagate unchanged.
            await session.close()
            raise

        # Success only: error paths above bill through their immediate
        # ``x-fal-billable-units`` headers and must never park the gateway
        # request in WAITING.
        session._activate_deferred_billing()

        stream_started = asyncio.Event()

        async def close_session() -> None:
            close_task = asyncio.ensure_future(session.close())
            _CLOSE_TASKS.add(close_task)
            close_task.add_done_callback(_CLOSE_TASKS.discard)
            with suppress(asyncio.CancelledError):
                await asyncio.shield(close_task)

        async def close_if_stream_never_starts() -> None:
            try:
                await asyncio.wait_for(
                    stream_started.wait(), timeout=STREAM_START_TIMEOUT_SECONDS
                )
            except asyncio.TimeoutError:
                await close_session()

        async def event_stream():
            stream_started.set()
            backend_closed = asyncio.ensure_future(backend.wait_closed())
            report_task: asyncio.Future[dict[str, str | int]] | None = None
            report_version = getattr(backend, "connection_report_version", None)
            wait_for_report = getattr(backend, "wait_connection_report", None)
            if report_version == CONNECTION_REPORT_VERSION and callable(
                wait_for_report
            ):
                try:
                    report_waiter = wait_for_report()
                    if inspect.isawaitable(report_waiter):
                        report_task = asyncio.ensure_future(report_waiter)
                    else:
                        logger.warning(
                            "WMA backend connection report waiter is not awaitable"
                        )
                except Exception:
                    logger.warning(
                        "WMA backend connection reporting could not start",
                        exc_info=True,
                    )
            try:
                payload = {
                    **session.answer_metadata,
                    **answer.metadata,
                    "sdp": answer.sdp,
                    "type": answer.type,
                    "session_id": request.session_id,
                }
                if report_task is not None:
                    payload["connection_report_version"] = report_version
                yield sse_event(payload)

                while not backend_closed.done():
                    waiters: set[asyncio.Future[Any]] = {backend_closed}
                    if report_task is not None:
                        waiters.add(report_task)
                    done, _ = await asyncio.wait(
                        waiters,
                        timeout=SSE_KEEPALIVE_INTERVAL,
                        return_when=asyncio.FIRST_COMPLETED,
                    )
                    if report_task is not None and report_task in done:
                        try:
                            report = report_task.result()
                        except asyncio.CancelledError:
                            pass
                        except Exception:
                            logger.warning(
                                "WMA backend connection reporting failed",
                                exc_info=True,
                            )
                        else:
                            try:
                                report_event = sse_event(
                                    report, event="connection_report"
                                )
                            except Exception:
                                logger.warning(
                                    "WMA backend connection report "
                                    "serialization failed",
                                    exc_info=True,
                                )
                            else:
                                yield report_event
                        report_task = None
                    if not done:
                        yield ": keepalive\n\n"
            finally:
                if report_task is not None:
                    if not report_task.done():
                        report_task.cancel()
                    with suppress(asyncio.CancelledError, Exception):
                        await report_task
                if not backend_closed.done():
                    backend_closed.cancel()
                try:
                    with suppress(asyncio.CancelledError):
                        await backend_closed
                finally:
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
    """aiortc implementation of the WMA peer-backend contract.

    ``initial_connect_timeout_seconds`` bounds how long a fresh peer may sit
    in a non-terminal ICE state before the session is abandoned; on a
    restrictive network ICE can stall in ``checking`` forever, which would
    otherwise hold the SSE response (and any deferred billing) until the
    platform's coarser request timeout. Pass ``None`` to disable the backstop.
    """

    def __init__(
        self,
        session: Session,
        on_connect: Callable[[Any], Any],
        *,
        create_default_channel: bool = True,
        rtc_configuration: Any = None,
        peer_connection_factory: Callable[[], Any] | None = None,
        disconnected_grace_seconds: float | None = 0,
        initial_connect_timeout_seconds: float | None = (
            INITIAL_CONNECT_TIMEOUT_SECONDS
        ),
    ) -> None:
        if rtc_configuration is not None and peer_connection_factory is not None:
            raise ValueError(
                "rtc_configuration and peer_connection_factory are mutually exclusive"
            )
        if disconnected_grace_seconds is not None and disconnected_grace_seconds < 0:
            raise ValueError("disconnected_grace_seconds cannot be negative")
        if (
            initial_connect_timeout_seconds is not None
            and initial_connect_timeout_seconds <= 0
        ):
            raise ValueError("initial_connect_timeout_seconds must be positive")
        self._session = session
        self._on_connect = on_connect
        self._create_default_channel = create_default_channel
        self._rtc_configuration = rtc_configuration
        self._peer_connection_factory = peer_connection_factory
        self._disconnected_grace_seconds = disconnected_grace_seconds
        self._initial_connect_timeout_seconds = initial_connect_timeout_seconds
        self._pc: Any = None
        self._channel: Any = None
        self._closed = asyncio.Event()
        self._connection_report: ConnectionReportObserver | None = None
        self._disconnect_task: asyncio.Task[Any] | None = None
        self._initial_connect_task: asyncio.Task[Any] | None = None

    async def negotiate(self, offer: StartSessionRequest) -> SessionAnswer:
        from aiortc import RTCPeerConnection

        if self._peer_connection_factory is not None:
            pc = self._peer_connection_factory()
            if inspect.isawaitable(pc):
                pc = await pc
        elif self._rtc_configuration is not None:
            pc = RTCPeerConnection(configuration=self._rtc_configuration)
        else:
            pc = RTCPeerConnection()
        self._pc = pc
        try:
            self._connection_report = observe_peer_connection(pc)
        except Exception:
            logger.warning(
                "WMA connection reporting could not observe the peer",
                exc_info=True,
            )
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
                self._cancel_initial_connect_timer()
                self._cancel_disconnect_timer()
            elif pc.connectionState == "disconnected":
                self._start_disconnect_timer(pc)
            elif pc.connectionState in ("closed", "failed"):
                self._cancel_initial_connect_timer()
                self._cancel_disconnect_timer()
                self._closed.set()
                if pc.connectionState == "failed":
                    task = asyncio.ensure_future(close_peer_connection(pc))
                    _CLOSE_TASKS.add(task)
                    task.add_done_callback(_CLOSE_TASKS.discard)

        try:
            result = self._on_connect(pc)
            if inspect.isawaitable(result):
                await result
            answer_sdp = await negotiate_answer(pc, offer.sdp, offer.type)
            self._start_initial_connect_timer(pc)
        except BaseException:
            await self.close()
            raise

        return SessionAnswer(sdp=answer_sdp)

    @property
    def connection_report_version(self) -> int | None:
        if self._connection_report is None:
            return None
        return self._connection_report.version

    async def wait_connection_report(self) -> dict[str, str | int]:
        observer = self._connection_report
        if observer is None:
            raise RuntimeError("WMA connection reporting is unavailable")
        return await observer.wait()

    def _start_initial_connect_timer(self, pc: Any) -> None:
        timeout = self._initial_connect_timeout_seconds
        if timeout is None or pc.connectionState == "connected":
            return
        timeout_seconds: float = timeout

        async def close_if_never_connected() -> None:
            await asyncio.sleep(timeout_seconds)
            if pc is self._pc and pc.connectionState != "connected":
                self._closed.set()

        self._initial_connect_task = asyncio.create_task(close_if_never_connected())

    def _cancel_initial_connect_timer(self) -> None:
        task, self._initial_connect_task = self._initial_connect_task, None
        if task is not None and not task.done():
            task.cancel()

    def _start_disconnect_timer(self, pc: Any) -> None:
        self._cancel_disconnect_timer()
        grace = self._disconnected_grace_seconds
        if grace is None:
            return
        if grace == 0:
            self._closed.set()
            return
        grace_seconds: float = grace

        async def close_after_grace() -> None:
            await asyncio.sleep(grace_seconds)
            if pc is self._pc and pc.connectionState == "disconnected":
                self._closed.set()

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

    def send(self, message: dict[str, Any]) -> bool:
        channel = self._channel
        if channel is None or channel.readyState != "open":
            return False
        _send_if_open(channel, json.dumps(message))
        return True

    async def wait_closed(self) -> None:
        await self._closed.wait()

    async def close(self) -> None:
        initial_connect_task, self._initial_connect_task = (
            self._initial_connect_task,
            None,
        )
        if initial_connect_task is not None and not initial_connect_task.done():
            initial_connect_task.cancel()
            with suppress(asyncio.CancelledError):
                await initial_connect_task
        disconnect_task, self._disconnect_task = self._disconnect_task, None
        if disconnect_task is not None and not disconnect_task.done():
            disconnect_task.cancel()
            with suppress(asyncio.CancelledError):
                await disconnect_task
        pc, self._pc = self._pc, None
        self._channel = None
        self._closed.set()
        if pc is not None:
            await close_peer_connection(pc)


__all__ = [
    "AiortcPeer",
    "App",
    "DATA_CHANNEL_LABEL",
    "PeerBackend",
    "Session",
    "SessionAnswer",
    "SessionParams",
    "StartSessionRequest",
]
