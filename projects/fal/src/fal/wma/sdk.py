"""Connection-oriented WebRTC sessions for fal apps."""

# Endpoint annotations must remain evaluated objects: fal ships this package
# by value with cloudpickle, which does not preserve globals referenced only
# by string annotations. FastAPI resolves them again on the runner.

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
    Protocol,
    Set,
    Tuple,
    Union,
)

from fastapi import Body, Header, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from starlette.background import BackgroundTask

import fal
from fal.compat import run_in_thread
from fal.exceptions import GPUException
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
    sanitize_connection_report,
)

SSE_KEEPALIVE_INTERVAL = 15
STREAM_START_TIMEOUT_SECONDS = 5
DATA_CHANNEL_LABEL = "control"
START_SESSION_PATH = "/start-session"

FAL_BILLING_HEADER = "x-fal-billable-units"
FAL_BILLING_WEBHOOK_HEADER = "x-fal-billable-units-webhook"

_CLOSE_TASKS: Set[asyncio.Task] = set()
_SESSION_TASKS: Set[asyncio.Task] = set()
logger = logging.getLogger(__name__)

_BILLING_REST_CLIENT: Any = None


def _billing_rest_client() -> Any:
    """Process-wide fal REST client for deferred billing reports.

    Built lazily so constructing the client only imports ``httpx`` when a
    deferred billing report is actually needed.
    """
    global _BILLING_REST_CLIENT  # noqa: PLW0603 - process-wide cache
    if _BILLING_REST_CLIENT is None:
        from fal.wma._billing import make_fal_rest_client

        _BILLING_REST_CLIENT = make_fal_rest_client()
    return _BILLING_REST_CLIENT


class StartSessionRequest(BaseModel):
    """Offer forwarded by the WMA bridge."""

    sdp: str
    type: Literal["offer"] = "offer"
    session_id: Union[str, None] = None
    ice_servers: List[Dict[str, Any]] = Field(default_factory=list)
    ice_status: Union[str, None] = None
    credential_age_seconds: Union[float, None] = Field(default=None, ge=0)


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
        self._push: Union[Callable[[Dict[str, Any]], Any], None] = None

    def _bind(self, push: Callable[[Dict[str, Any]], Any]) -> None:
        self._push = push

    def _sync(self) -> None:
        if self._push is not None:
            self._push(dict(self))

    def _merge_from_client(self, params: Dict[str, Any]) -> None:
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

    def popitem(self) -> Tuple[str, Any]:
        result = super().popitem()
        self._sync()
        return result

    def __ior__(self, other: Any) -> "SessionParams":  # type: ignore[misc, override]
        super().update(other)
        self._sync()
        return self


def _send_if_open(channel: Any, payload: str) -> None:
    if channel.readyState == "open":
        channel.send(payload)


def _task_waits_on(waiter: Any, target: Any) -> bool:
    """Read asyncio's wait links to identify an actual teardown dependency.

    Supported Python versions expose Task/gather links through _fut_waiter
    and _children. shield and wait_for wrap those links, so follow their
    standard-library closure/coroutine references as well. No application
    fields or cancellation history participate in this decision.
    """
    if waiter is None or target is None:
        return False
    pending = [waiter]
    seen: Set[int] = set()
    while pending:
        item = pending.pop()
        if item is target:
            return True
        if id(item) in seen:
            continue
        seen.add(id(item))
        if isinstance(item, asyncio.Future) and item.done():
            continue
        dependency = getattr(item, "_fut_waiter", None)
        if dependency is not None:
            pending.append(dependency)
        children = getattr(item, "_children", ())
        if isinstance(children, (list, tuple)):
            pending.extend(children)
        if isinstance(item, asyncio.Task):
            coro: Any = item.get_coro()
            while coro is not None:
                code = getattr(coro, "cr_code", None)
                frame = getattr(coro, "cr_frame", None)
                if frame is not None:
                    if code is asyncio.wait_for.__code__:
                        pending.append(frame.f_locals.get("fut"))
                    elif code is getattr(
                        getattr(asyncio.tasks, "_wait", None), "__code__", None
                    ):
                        pending.extend(frame.f_locals.get("fs", ()))
                coro = getattr(coro, "cr_await", None)
        # A shield Future has no _children link; its own completion callback
        # retains the inner Future. Restrict inspection to asyncio's callback.
        for callback, _context in getattr(item, "_callbacks", None) or ():
            if (
                getattr(callback, "__module__", None) == "asyncio.tasks"
                and getattr(callback, "__qualname__", None)
                == "shield.<locals>._outer_done_callback"
            ):
                for name, cell in zip(
                    callback.__code__.co_freevars, callback.__closure__ or ()
                ):
                    if name == "inner":
                        pending.append(cell.cell_contents)
    return False


class Session:
    """Transport-neutral state and lifecycle for one WMA connection."""

    def __init__(
        self,
        request: StartSessionRequest,
        *,
        http_request: Union[Request, None] = None,
        caller_user_id: Union[str, None] = None,
        request_id: Union[str, None] = None,
        billing_debug: bool = False,
        minimum_billable_units: float = 0,
    ) -> None:
        self.http_request = http_request
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
        #: Per-app opt-in for billing-event tracing, handed down from
        #: :attr:`App.billing_debug` by ``start_session``.
        self.billing_debug = billing_debug
        self._billable_units = 0.0
        self._billable_units_lock = threading.Lock()
        self._billing_finalized = False
        self.minimum_billable_units = minimum_billable_units
        self._deferred_billing = False
        self.params = SessionParams()
        self.answer_metadata: Dict[str, Any] = {}
        self.response_headers: Dict[str, str] = {}
        self.state: Dict[str, Any] = {}
        self._loop = asyncio.get_running_loop()
        self._handlers: Dict[
            str, List[Tuple[Callable[[Dict[str, Any]], Any], bool]]
        ] = {}
        self._channel_open_handlers: List[Callable[[], Any]] = []
        self._channel_is_open = False
        self._sender: Union[Callable[[Dict[str, Any]], bool], None] = None
        self._sender_thread_safe = False
        self._backend: Union[PeerBackend, None] = None
        self._cleanup: List[Callable[[], Any]] = []
        self._tasks: Set[asyncio.Task] = set()
        self._close_task: Union[asyncio.Task, None] = None
        self._close_waiters: Set[asyncio.Task] = set()
        # Resolve this runtime primitive on the destination Python version.
        from contextvars import ContextVar

        self._closing_context = ContextVar("wma_session_close", default=False)
        self._inline_condition = threading.Condition()
        self._inline_active = 0
        self._closed = asyncio.Event()
        self._is_closed = False
        self.params._bind(
            lambda params: self.send({"type": "session_params", "params": params})
        )
        # ``request_id_header_present`` distinguishes "no gateway header" from
        # "header dropped as non-canonical" — both leave deferred billing
        # unavailable, but only the latter is a caller/bridge bug.
        self.billing_debug_print(
            "session created",
            caller_user_id=self.caller_user_id,
            request_id_header_present=isinstance(request_id, str),
        )

    @property
    def minimum_billable_units(self) -> float:
        """Validated settlement floor, adjustable until billing is finalized."""
        return self._minimum_billable_units

    @minimum_billable_units.setter
    def minimum_billable_units(self, value: float) -> None:
        try:
            minimum = float(value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(
                "minimum billable units must be a finite, non-negative number"
            ) from exc
        if not math.isfinite(minimum) or minimum < 0:
            raise ValueError(
                "minimum billable units must be a finite, non-negative number"
            )
        with self._billable_units_lock:
            if self._billing_finalized:
                raise RuntimeError("billing has already been finalized")
            self._minimum_billable_units = minimum

    def billing_debug_print(self, event: str, **fields: Any) -> None:
        """Print one billing-event trace line when the app opted in.

        Gated by this session's ``billing_debug`` flag — a per-app class flag
        on :class:`App`, not an environment variable, so each app decides
        independently. Read per call, so a live worker patch can flip it.
        Every line carries the session's billing identity plus a fixed
        ``[wma-billing]`` tag, greppable alongside the deferred billing
        reporter's ``[wma]`` lines. Values are limited to ids, unit counts,
        and flags — never SDP, ICE credentials, or prompts.
        """
        if not self.billing_debug:
            return
        details = " ".join(
            f"{key}={value!r}"
            for key, value in {
                "session_id": self.id,
                "request_id": self.request_id,
                **fields,
            }.items()
        )
        logger.info("[wma-billing] %s (%s)", event, details)

    @property
    def closed(self) -> asyncio.Event:
        return self._closed

    @property
    def billable_units(self) -> float:
        with self._billable_units_lock:
            return self._billable_units

    def cap_billable_units(self, units: float) -> None:
        """Reconcile undelivered usage before deferred settlement and its floor.

        May only reduce accumulated units. Cleanup can repeat this operation;
        finalized billing cannot be changed.
        """
        value = float(units)
        if not math.isfinite(value) or value < 0:
            raise ValueError("billable units must be a finite, non-negative number")
        with self._billable_units_lock:
            if self._billing_finalized:
                raise RuntimeError(
                    "cannot cap billable units after billing is finalized"
                )
            self._billable_units = min(self._billable_units, value)

    def add_billable_units(self, units: float = 1) -> None:
        """Accumulate usage for this session (e.g. one generated chunk).

        The unit is app-defined — chunks, seconds, tokens — and its price is
        endpoint pricing configuration, exactly as with the
        ``x-fal-billable-units`` header. The accumulated total is reported to
        fal billing once, automatically, when the session closes. Thread-safe:
        data-channel handlers may run off the session loop.
        """
        value = float(units)
        if not math.isfinite(value) or value < 0:
            raise ValueError("billable units must be a finite, non-negative number")
        with self._billable_units_lock:
            if self._billing_finalized:
                self.billing_debug_print(
                    "billable units rejected: billing already finalized",
                    rejected_units=value,
                    total_units=self._billable_units,
                )
                raise RuntimeError(
                    "cannot add billable units after billing is finalized"
                )
            total = self._billable_units + value
            if not math.isfinite(total):
                # An overflowed total would fail the reporter's finite check
                # at close and void the WHOLE session's billing; refusing the
                # increment keeps everything accumulated so far billable.
                raise ValueError("billable units total overflowed")
            self._billable_units = total
        self.billing_debug_print(
            "billable units added",
            added_units=value,
            total_units=total,
        )

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
            self.billing_debug_print(
                "deferred billing not activated: no gateway request id",
            )
            return False
        self._deferred_billing = True
        self.response_headers[FAL_BILLING_WEBHOOK_HEADER] = "1"
        self.billing_debug_print(
            "deferred billing activated: one report owed at close",
        )
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
        sender: Callable[[Dict[str, Any]], bool],
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
        handler: Callable[[Dict[str, Any]], Any],
        *,
        inline: bool = False,
    ) -> Callable[[Dict[str, Any]], Any]:
        self._handlers.setdefault(kind, []).append((handler, inline))
        return handler

    def receive(self, message: Dict[str, Any]) -> None:
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

    def _dispatch_inline(self, message: Dict[str, Any]) -> bool:
        kind = message.get("type")
        if not isinstance(kind, str):
            return False
        handlers = self._handlers.get(kind) or self._handlers.get("*") or []
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

    def _dispatch(self, message: Dict[str, Any], skip_inline: bool = False) -> None:
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
        handlers = self._handlers.get(kind) or self._handlers.get("*") or []
        for handler, inline in handlers:
            if not (skip_inline and inline):
                self._invoke_handler(handler, message)

    def _invoke_handler(
        self,
        handler: Callable[[Dict[str, Any]], Any],
        message: Dict[str, Any],
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

    def send(self, message: Dict[str, Any]) -> bool:
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

    def create_task(self, awaitable: Awaitable[Any]) -> Union[asyncio.Task, None]:
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
        current = asyncio.current_task()
        # Only a real dependency cycle can bypass settlement. Cancellation
        # history and task ownership alone say nothing about who awaits whom.
        if self._closing_context.get() or _task_waits_on(self._close_task, current):
            return
        if current is not None:
            self._close_waiters.add(current)
        try:
            if self._close_task is None:
                self._close_task = asyncio.create_task(self._run_close())
            # One session-owned close pass survives cancellation of any caller;
            # other callers join that same pass through final billing settlement.
            while not self._close_task.done():
                if _task_waits_on(self._close_task, current):
                    return
                # A backend can yield before joining a worker that is already
                # awaiting close. Recheck that later dependency without
                # cancelling the shared teardown task when a caller exits.
                await asyncio.wait([self._close_task], timeout=0.05)
            await asyncio.shield(self._close_task)
        finally:
            if current is not None:
                self._close_waiters.discard(current)

    async def _run_close(self) -> None:
        token = self._closing_context.set(True)
        try:
            await self._close_once()
        finally:
            self._closing_context.reset(token)

    async def _close_once(self) -> None:
        if self._is_closed:
            return
        with self._inline_condition:
            self._is_closed = True
        self._closed.set()

        backend = self._backend
        if backend is not None:
            with suppress(Exception, asyncio.CancelledError):
                await backend.close()

        with self._inline_condition:
            inline_active = self._inline_active != 0
        if inline_active:
            await run_in_thread(self._wait_for_inline_handlers)

        current = asyncio.current_task()
        tasks = [
            task
            for task in self._tasks
            if task is not current and task not in self._close_waiters
        ]
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

        for cleanup in reversed(self._cleanup):
            try:
                result = cleanup()
                if inspect.isawaitable(result):
                    await result
            except (Exception, asyncio.CancelledError):
                logger.warning("WMA deferred cleanup failed", exc_info=True)
        self._cleanup.clear()

        # Last step of the single close pass: the total is final once
        # handlers and tasks have stopped, and every teardown path
        # (client close, WebRTC drop, bridge abort, reaper, watchdog)
        # funnels through here exactly once.
        accumulated_units: Union[float, None] = None
        with self._billable_units_lock:
            self._billing_finalized = True
            # The session floor applies only to the deferred close report
            # — a session that actually started. Sessions billing through
            # their immediate response headers (setup failures, direct
            # calls without a gateway request id) keep billing zero.
            if (
                self._deferred_billing
                and self._billable_units < self.minimum_billable_units
            ):
                accumulated_units = self._billable_units
                self._billable_units = self.minimum_billable_units
        if accumulated_units is not None:
            self.billing_debug_print(
                "billable units raised to the session minimum",
                accumulated_units=accumulated_units,
                minimum_billable_units=self.minimum_billable_units,
            )
        # ``deferred_report_pending=False`` means the session bills only
        # through the immediate response headers; no report follows.
        self.billing_debug_print(
            "billing finalized at session close",
            total_units=self.billable_units,
            deferred_report_pending=self._deferred_billing,
        )
        await self._report_billable_units()
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
        if "app_name" not in cls.__dict__:
            name_owner = next(
                base for base in cls.__mro__[1:] if "app_name" in base.__dict__
            )
            if name_owner is App:
                cls.app_name = None
        super().__init_subclass__(**kwargs)

    # Deferred billing reports authenticate to the fal REST API with the
    # app-scoped service credential. Subclasses that override this allow-list
    # must retain FAL_KEY alongside their app-specific secrets.
    secrets: ClassVar[List[str]] = ["FAL_KEY"]

    #: Per-app opt-in for billing-event tracing: subclasses set ``True`` to
    #: print one ``[wma-billing]`` line per billing event on every session
    #: (creation, unit accumulation, deferred activation, finalization,
    #: error-path headers). A class flag rather than an environment variable
    #: so each app decides independently.
    billing_debug: ClassVar[bool] = False

    #: Floor on a session's deferred billing report: a session that activated
    #: deferred billing bills at least this many units at close, even if it
    #: accumulated less (e.g. an early disconnect). Zero keeps usage-only
    #: billing. Sessions that never activated deferred billing — setup
    #: failures, direct calls without a gateway request id — are unaffected
    #: and still bill zero through their immediate response headers.
    minimum_billable_units: ClassVar[float] = 0

    # ``request_timeout`` is deliberately NOT set here: it is managed
    # dynamically per deployment by the platform, and a class default would
    # override that for every WMA app.

    #: What this app's live session needs and accepts. OpenAPI publishes the
    #: discovery link; :meth:`asyncapi` publishes the live-session contract.
    #: Left unset, the app's OpenAPI document is exactly what it was before
    #: contracts existed.
    realtime_contract: ClassVar[Union[RealtimeContract, None]] = None

    async def create_backend(self, session: Session) -> PeerBackend:
        raise NotImplementedError("WMA App subclasses must implement create_backend()")

    def openapi(self) -> Dict[str, Any]:
        spec = super().openapi()
        contract = type(self).realtime_contract
        if contract is None:
            return spec
        return apply_contract(spec, path=START_SESSION_PATH)

    def asyncapi(self) -> Dict[str, Any]:
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
    def build_metadata(cls) -> Dict[str, Any]:
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
        x_fal_caller_user_id: Union[str, None] = Header(None),
        x_fal_request_id: Union[str, None] = Header(None),
        http_request: Request = None,
    ) -> StreamingResponse:
        session = Session(
            request,
            http_request=http_request,
            caller_user_id=x_fal_caller_user_id,
            request_id=x_fal_request_id,
            billing_debug=self.billing_debug,
            minimum_billable_units=self.minimum_billable_units,
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
            if isinstance(exc, GPUException):
                # Preserve the platform signal to recycle a dead GPU runner.
                raise
            if isinstance(exc, HTTPException):
                # HTTP errors already carry their own billing/retry
                # headers; merge in session headers (e.g. the app's
                # ``x-fal-billable-units: 0``) they did not set themselves.
                exc.headers = {**session.response_headers, **(exc.headers or {})}
                session.billing_debug_print(
                    "session setup failed: billing rides the error response",
                    status_code=exc.status_code,
                    billable_units_header=exc.headers.get(FAL_BILLING_HEADER),
                )
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
            report_task: Union[asyncio.Future, None] = None
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
                    waiters: Set[asyncio.Future] = {backend_closed}
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
                            safe_report = sanitize_connection_report(report)
                            if safe_report is None:
                                logger.warning(
                                    "WMA backend connection report was invalid"
                                )
                            else:
                                try:
                                    report_event = sse_event(
                                        safe_report, event="connection_report"
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

    The WMA client creates the control channel when building its offer. Use
    that channel by default so replies reach the client's message listener.
    Set ``create_default_channel=True`` only for clients that explicitly
    consume a server-created channel.
    """

    def __init__(
        self,
        session: Session,
        on_connect: Callable[[Any], Any],
        *,
        create_default_channel: bool = False,
        rtc_configuration: Any = None,
        peer_connection_factory: Union[Callable[[], Any], None] = None,
        disconnected_grace_seconds: Union[float, None] = None,
        initial_connect_timeout_seconds: Union[
            float, None
        ] = INITIAL_CONNECT_TIMEOUT_SECONDS,
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
        self._connection_report: Union[ConnectionReportObserver, None] = None
        self._disconnect_task: Union[asyncio.Task, None] = None
        self._initial_connect_task: Union[asyncio.Task, None] = None

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
    def connection_report_version(self) -> Union[int, None]:
        if self._connection_report is None:
            return None
        return self._connection_report.version

    async def wait_connection_report(self) -> Dict[str, Union[str, int]]:
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
        # Other application channels must neither receive controls nor own the
        # session lifetime, regardless of which channel opens first.
        if not primary and channel.label != DATA_CHANNEL_LABEL:
            return

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

    def send(self, message: Dict[str, Any]) -> bool:
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
