from __future__ import annotations

import asyncio
import inspect
import json
import os
import queue
import re
import threading
import time
import typing
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass
from typing import Any, Callable, ClassVar, Literal, TypeVar

import fastapi
import grpc.aio as async_grpc
import httpx

from fal._serialization import include_modules_from
from fal.api import (
    SERVE_REQUIREMENTS,
    BaseServable,
    IsolatedFunction,
    RouteSignature,
)
from fal.api import (
    function as fal_function,
)
from fal.exceptions import FalServerlessException, RequestCancelledException
from fal.logging import get_logger
from fal.toolkit.file import request_lifecycle_preference
from fal.toolkit.file.providers.fal import LIFECYCLE_PREFERENCE

REALTIME_APP_REQUIREMENTS = ["websockets", "msgpack"]
REQUEST_ID_KEY = "x-fal-request-id"

EndpointT = TypeVar("EndpointT", bound=Callable[..., Any])
logger = get_logger(__name__)


async def _call_any_fn(fn, *args, **kwargs):
    if inspect.iscoroutinefunction(fn):
        return await fn(*args, **kwargs)
    else:
        return fn(*args, **kwargs)


async def open_isolate_channel(address: str) -> async_grpc.Channel | None:
    channel = async_grpc.insecure_channel(
        address,
        options=[
            ("grpc.max_send_message_length", -1),
            ("grpc.max_receive_message_length", -1),
            ("grpc.min_reconnect_backoff_ms", 0),
            ("grpc.max_reconnect_backoff_ms", 100),
            ("grpc.dns_min_time_between_resolutions_ms", 100),
        ],
    )

    try:
        channel_status = channel.channel_ready()

        await asyncio.wait_for(channel_status, timeout=1)
    except asyncio.TimeoutError:
        await channel.close(None)
        print("[DEBUG] Timed out trying to connect to local isolate")
        return None

    return channel


async def _set_logger_labels(
    logger_labels: dict[str, str], channel: async_grpc.Channel
):
    try:
        import sys

        from isolate.server import definitions

        # Flush any prints that were buffered before setting the logger labels
        sys.stderr.flush()
        sys.stdout.flush()

        isolate = definitions.IsolateStub(channel)
        isolate_request = definitions.SetMetadataRequest(
            # TODO: when submit is shipped, get task_id from an env var
            task_id="RUN",
            metadata=definitions.TaskMetadata(logger_labels=logger_labels),
        )
        res = isolate.SetMetadata(isolate_request)
        code = await res.code()
        assert str(code) == "StatusCode.OK", str(code)
    except BaseException:
        # NOTE hiding this for now to not print on every request
        # logger.debug("Failed to set logger labels", exc_info=True)
        pass


def wrap_app(cls: type[App], **kwargs) -> IsolatedFunction:
    include_modules_from(cls)

    def initialize_and_serve():
        app = cls()
        app.serve()

    metadata = {}
    app = cls(_allow_init=True)

    metadata["openapi"] = app.openapi()

    routes = app.collect_routes()
    realtime_app = any(route.is_websocket for route in routes)

    kind = cls.host_kwargs.pop("kind", "virtualenv")

    wrapper = fal_function(
        kind,
        requirements=cls.requirements,
        local_python_modules=cls.local_python_modules,
        machine_type=cls.machine_type,
        num_gpus=cls.num_gpus,
        **cls.host_kwargs,
        **kwargs,
        metadata=metadata,
        exposed_port=8080,
        serve=False,
    )
    fn = wrapper(initialize_and_serve)
    fn.options.add_requirements(SERVE_REQUIREMENTS)
    if realtime_app:
        fn.options.add_requirements(REALTIME_APP_REQUIREMENTS)

    return fn


@dataclass
class AppClientError(FalServerlessException):
    message: str
    status_code: int


class EndpointClient:
    def __init__(self, url, endpoint, signature, timeout: int | None = None):
        self.url = url
        self.endpoint = endpoint
        self.signature = signature
        self.timeout = timeout

        annotations = endpoint.__annotations__ or {}
        self.return_type = annotations.get("return") or None

    def __call__(self, data):
        with httpx.Client() as client:
            url = self.url + self.signature.path
            resp = client.post(
                self.url + self.signature.path,
                json=data.dict() if hasattr(data, "dict") else dict(data),
                timeout=self.timeout,
            )
            if not resp.is_success:
                # allow logs to be printed before raising the exception
                time.sleep(1)
                raise AppClientError(
                    f"Failed to POST {url}: {resp.status_code} {resp.text}",
                    status_code=resp.status_code,
                )
            resp_dict = resp.json()

        if not self.return_type:
            return resp_dict

        return self.return_type(**resp_dict)


class AppClient:
    def __init__(
        self,
        cls,
        url,
        timeout: int | None = None,
    ):
        self.url = url
        self.cls = cls

        for name, endpoint in inspect.getmembers(cls, inspect.isfunction):
            signature = getattr(endpoint, "route_signature", None)
            if signature is None:
                continue
            endpoint_client = EndpointClient(
                self.url,
                endpoint,
                signature,
                timeout=timeout,
            )
            setattr(self, name, endpoint_client)

    @classmethod
    @contextmanager
    def connect(
        cls,
        app_cls,
        *,
        health_request_timeout: int = 30,
        startup_timeout: int = 60,
        health_check_interval: float = 0.5,
    ):
        app = wrap_app(app_cls)
        info = app.spawn()
        _shutdown_event = threading.Event()

        def _print_logs():
            while not _shutdown_event.is_set():
                try:
                    log = info.logs.get(timeout=0.1)
                except queue.Empty:
                    continue
                print(log)

        _log_printer = threading.Thread(target=_print_logs, daemon=True)
        _log_printer.start()

        try:
            if info.url is None:
                raise AppClientError(
                    "App spawn failed: no URL returned",
                    status_code=500,
                )

            start_time = time.perf_counter()
            url = info.url + "/health"
            last_error = None
            attempt = 0

            with httpx.Client() as client:
                while time.perf_counter() - start_time < startup_timeout:
                    attempt += 1

                    try:
                        resp = client.get(url, timeout=health_request_timeout)
                    except httpx.TimeoutException:
                        last_error = (
                            f"Request timed out after {health_request_timeout} seconds"
                        )
                    except httpx.TransportError as e:
                        last_error = f"Network error: {e}"
                    else:
                        if resp.is_success:
                            break

                        if resp.status_code in (500, 404):
                            last_error = f"Server not ready (HTTP {resp.status_code})"
                        else:
                            raise AppClientError(
                                "Health check failed with non-retryable error: "
                                f"{resp.status_code} {resp.text}",
                                status_code=resp.status_code,
                            )

                    time.sleep(health_check_interval)
                else:
                    # retry loop completed without success
                    raise AppClientError(
                        f"Health check failed after {startup_timeout}s "
                        f"({attempt} attempts). Last error: {last_error}",
                        status_code=500,
                    )

            client = cls(app_cls, info.url)
            yield client
        finally:
            info.stream.cancel()
            _shutdown_event.set()
            _log_printer.join()

    def health(self):
        with httpx.Client() as client:
            resp = client.get(self.url + "/health")
            resp.raise_for_status()
            return resp.json()


PART_FINDER_RE = re.compile(r"[A-Z][a-z]*")


def _to_fal_app_name(name: str) -> str:
    # Convert MyGoodApp into my-good-app
    return "-".join(part.lower() for part in PART_FINDER_RE.findall(name))


def _print_python_packages() -> None:
    from importlib.metadata import distributions

    packages = [f"{dist.metadata['Name']}=={dist.version}" for dist in distributions()]

    print("[debug] Python packages installed:", ", ".join(packages))


class App(BaseServable):
    requirements: ClassVar[list[str]] = []
    local_python_modules: ClassVar[list[str]] = []
    machine_type: ClassVar[str] = "S"
    num_gpus: ClassVar[int | None] = None
    host_kwargs: ClassVar[dict[str, Any]] = {
        "_scheduler": "nomad",
        "_scheduler_options": {
            "storage_region": "us-east",
        },
        "resolver": "uv",
        "keep_alive": 60,
    }
    app_name: ClassVar[str]
    app_auth: ClassVar[Literal["private", "public", "shared", None]] = None
    request_timeout: ClassVar[int | None] = None
    startup_timeout: ClassVar[int | None] = None

    isolate_channel: async_grpc.Channel | None = None

    def __init_subclass__(cls, **kwargs):
        app_name = kwargs.pop("name", None) or _to_fal_app_name(cls.__name__)
        parent_settings = getattr(cls, "host_kwargs", {})
        cls.host_kwargs = {**parent_settings, **kwargs}

        if cls.request_timeout is not None:
            cls.host_kwargs["request_timeout"] = cls.request_timeout

        if cls.startup_timeout is not None:
            cls.host_kwargs["startup_timeout"] = cls.startup_timeout

        cls.app_name = getattr(cls, "app_name", app_name)

        if cls.__init__ is not App.__init__:
            raise ValueError(
                "App classes should not override __init__ directly. "
                "Use setup() instead."
            )

    def __init__(self, *, _allow_init: bool = False):
        if not _allow_init and not os.getenv("IS_ISOLATE_AGENT"):
            raise NotImplementedError(
                "Running apps through SDK is not implemented yet."
            )

    @classmethod
    def get_endpoints(cls) -> list[str]:
        return [
            signature.path
            for _, endpoint in inspect.getmembers(cls, inspect.isfunction)
            if (signature := getattr(endpoint, "route_signature", None))
        ]

    def collect_routes(self) -> dict[RouteSignature, Callable[..., Any]]:
        return {
            signature: endpoint
            for _, endpoint in inspect.getmembers(self, inspect.ismethod)
            if (signature := getattr(endpoint, "route_signature", None))
        }

    @asynccontextmanager
    async def lifespan(self, app: fastapi.FastAPI):
        _print_python_packages()
        await _call_any_fn(self.setup)
        try:
            yield
        finally:
            await _call_any_fn(self.teardown)

    def health(self):
        return {"version": self.version}

    def setup(self):
        """Setup the application before serving."""

    def teardown(self):
        """Teardown the application after serving."""

    def _add_extra_middlewares(self, app: fastapi.FastAPI):
        @app.middleware("http")
        async def provide_hints_headers(request, call_next):
            response = await call_next(request)
            try:
                # make sure the hints can be encoded in latin-1, so we don't crash
                # when serving.
                # https://github.com/encode/starlette/blob/a766a58d14007f07c0b5782fa78cdc370b892796/starlette/datastructures.py#L568
                hints = []
                for hint in self.provide_hints():
                    try:
                        _ = hint.encode("latin-1")
                        hints.append(hint)
                    except UnicodeEncodeError:
                        from fastapi.logger import logger

                        logger.warning(
                            "Ignoring hint %s for %s because it can't be encoded in "
                            "latin-1",
                            hint,
                            self.__class__.__name__,
                        )

                response.headers["X-Fal-Runner-Hints"] = ",".join(hints)
            except NotImplementedError:
                # This lets us differentiate between apps that don't provide hints
                # and apps that provide empty hints.
                pass
            except Exception:
                from fastapi.logger import logger

                logger.exception(
                    "Failed to provide hints for %s",
                    self.__class__.__name__,
                )
            return response

        @app.middleware("http")
        async def set_global_object_preference(request, call_next):
            try:
                preference_dict = request_lifecycle_preference(request)
                if preference_dict is not None:
                    # This will not work properly for apps with multiplexing enabled
                    # we may mix up the preferences between requests
                    LIFECYCLE_PREFERENCE.set(preference_dict)
            except Exception:
                from fastapi.logger import logger

                logger.exception(
                    "Failed set a global lifecycle preference %s",
                    self.__class__.__name__,
                )

            try:
                return await call_next(request)
            finally:
                # We may miss the global preference if there are operations
                # being done in the background that go beyond the request
                LIFECYCLE_PREFERENCE.set(None)

        @app.middleware("http")
        async def set_request_id(request, call_next):
            # NOTE: Setting request_id is not supported for websocket/realtime endpoints
            if not os.getenv("IS_ISOLATE_AGENT") or not os.environ.get(
                "NOMAD_ALLOC_PORT_grpc"
            ):
                # If not running in the expected environment, skip setting request_id
                return await call_next(request)

            if self.isolate_channel is None:
                grpc_port = os.environ.get("NOMAD_ALLOC_PORT_grpc")
                self.isolate_channel = await open_isolate_channel(
                    f"localhost:{grpc_port}"
                )

            if self.isolate_channel is None:
                return await call_next(request)

            request_id = request.headers.get(REQUEST_ID_KEY)
            if request_id is None:
                return await call_next(request)

            await _set_logger_labels(
                {"fal_request_id": request_id}, channel=self.isolate_channel
            )

            async def _unset_at_end():
                await _set_logger_labels({}, channel=self.isolate_channel)  # type: ignore

            try:
                response: fastapi.responses.Response = await call_next(request)
            except BaseException:
                await _unset_at_end()
                raise
            else:
                # We need to wait for the entire response to be sent before
                # we can set the logger labels back to the default.
                background_tasks = fastapi.BackgroundTasks()
                background_tasks.add_task(_unset_at_end)
                if response.background:
                    # We normally have no background tasks, but we should handle it
                    background_tasks.add_task(response.background)
                response.background = background_tasks

                return response

        @app.exception_handler(RequestCancelledException)
        async def value_error_exception_handler(
            request, exc: RequestCancelledException
        ):
            from fastapi.responses import JSONResponse

            # A 499 status code is not an officially recognized HTTP status code,
            # but it is sometimes used by servers to indicate that a client has closed
            # the connection without receiving a response
            return JSONResponse({"detail": str(exc)}, 499)

    def _add_extra_routes(self, app: fastapi.FastAPI):
        @app.get("/health")
        def health():
            return self.health()

    def provide_hints(self) -> list[str]:
        """Provide hints for routing the application."""
        raise NotImplementedError


def endpoint(
    path: str, *, is_websocket: bool = False
) -> Callable[[EndpointT], EndpointT]:
    """Designate the decorated function as an application endpoint."""

    def marker_fn(callable: EndpointT) -> EndpointT:
        if hasattr(callable, "route_signature"):
            raise ValueError(
                f"Can't set multiple routes for the same function: {callable.__name__}"
            )

        callable.route_signature = RouteSignature(path=path, is_websocket=is_websocket)  # type: ignore
        return callable

    return marker_fn


def _fal_websocket_template(
    func: EndpointT,
    route_signature: RouteSignature,
) -> EndpointT:
    # A template for fal's realtime websocket endpoints to basically
    # be a boilerplate for the user to fill in their inference function
    # and start using it.

    import asyncio
    from collections import deque
    from contextlib import suppress

    import msgpack
    from fastapi import WebSocket, WebSocketDisconnect

    async def mirror_input(queue: deque[Any], websocket: WebSocket) -> None:
        while True:
            try:
                raw_input = await asyncio.wait_for(
                    websocket.receive_bytes(),
                    timeout=route_signature.session_timeout,
                )
            except asyncio.TimeoutError:
                return

            input = msgpack.unpackb(raw_input, raw=False)
            if route_signature.input_modal:
                input = route_signature.input_modal(**input)

            queue.append(input)

    async def mirror_output(
        self,
        queue: deque[Any],
        websocket: WebSocket,
    ) -> None:
        loop = asyncio.get_event_loop()
        max_allowed_buffering = route_signature.buffering or 1
        outgoing_messages: asyncio.Queue[bytes] = asyncio.Queue(
            maxsize=max_allowed_buffering * 2  # x2 for outgoing timings
        )

        async def emit(message):
            if isinstance(message, bytes):
                await websocket.send_bytes(message)
            elif isinstance(message, str):
                await websocket.send_text(message)
            else:
                raise TypeError(f"Can't send message of type {type(message)}")

        async def background_emitter():
            while True:
                output = await outgoing_messages.get()
                await emit(output)
                outgoing_messages.task_done()

        emitter = asyncio.create_task(background_emitter())

        while True:
            if not queue:
                await asyncio.sleep(0.05)
                continue

            input = queue.popleft()
            if input is None or emitter.done():
                if not emitter.done():
                    await outgoing_messages.join()
                    emitter.cancel()

                with suppress(asyncio.CancelledError):
                    await emitter
                return None  # End of input

            batch = [input]
            while queue and len(batch) < route_signature.max_batch_size:
                next_input = queue.popleft()
                if hasattr(input, "can_batch") and not input.can_batch(
                    next_input, len(batch)
                ):
                    queue.appendleft(next_input)
                    break
                batch.append(next_input)

            t0 = loop.time()
            if inspect.iscoroutinefunction(func):
                output = await func(self, *batch)
            else:
                output = await loop.run_in_executor(None, func, self, *batch)  # type: ignore
            total_time = loop.time() - t0
            if not isinstance(output, dict):
                # Handle pydantic output modal
                if hasattr(output, "dict"):
                    output = output.dict()
                else:
                    raise TypeError(
                        "Expected a dict or pydantic model as output, got "
                        f"{type(output)}"
                    )

            messages = [
                msgpack.packb(output, use_bin_type=True),
            ]
            if route_signature.emit_timings:
                # We emit x-fal messages in JSON, no matter what the
                # input/output format is.
                timings = {
                    "type": "x-fal-message",
                    "action": "timings",
                    "timing": total_time,
                }
                messages.append(json.dumps(timings, separators=(",", ":")))

            for message in messages:
                try:
                    outgoing_messages.put_nowait(message)
                except asyncio.QueueFull:
                    await emit(message)

    async def websocket_template(self, websocket: WebSocket) -> None:
        import asyncio

        await websocket.accept()

        queue: deque[Any] = deque(maxlen=route_signature.buffering)
        input_task = asyncio.create_task(mirror_input(queue, websocket))
        input_task.add_done_callback(lambda _: queue.append(None))
        output_task = asyncio.create_task(mirror_output(self, queue, websocket))

        try:
            await asyncio.wait(
                {
                    input_task,
                    output_task,
                },
                return_when=asyncio.FIRST_COMPLETED,
            )
            if input_task.done():
                # User didn't send any input within the timeout
                # so we can just close the connection after the
                # processing of the last input is done.
                input_task.result()
                await asyncio.wait_for(
                    output_task, timeout=route_signature.session_timeout
                )
            else:
                assert output_task.done()

                # The execution of the inference function failed or exitted,
                # so just propagate the result.
                input_task.cancel()
                with suppress(asyncio.CancelledError):
                    await input_task

                output_task.result()
        except WebSocketDisconnect:
            input_task.cancel()
            output_task.cancel()
            with suppress(asyncio.CancelledError):
                await input_task

            with suppress(asyncio.CancelledError):
                await output_task
        except Exception as exc:
            import traceback

            traceback.print_exc()

            await websocket.send_json(
                {
                    "type": "x-fal-error",
                    "error": "INTERNAL_ERROR",
                    "reason": str(exc),
                }
            )
        else:
            await websocket.send_json(
                {
                    "type": "x-fal-error",
                    "error": "TIMEOUT",
                    "reason": "no inputs, reconnect when needed!",
                }
            )

        await websocket.close()

    # Seems like templating + stringified annotations don't play well,
    # so we have to set them manually.
    websocket_template.__annotations__ = {
        "websocket": WebSocket,
        "return": None,
    }
    websocket_template.route_signature = route_signature  # type: ignore
    websocket_template.original_func = func  # type: ignore
    return typing.cast(EndpointT, websocket_template)


_SENTINEL = object()


def realtime(
    path: str,
    *,
    buffering: int | None = None,
    session_timeout: float | None = None,
    input_modal: Any | None = _SENTINEL,
    max_batch_size: int = 1,
) -> Callable[[EndpointT], EndpointT]:
    """Designate the decorated function as a realtime application endpoint."""

    def marker_fn(original_func: EndpointT) -> EndpointT:
        nonlocal input_modal

        if hasattr(original_func, "route_signature"):
            raise ValueError(
                "Can't set multiple routes for the same function: "
                f"{original_func.__name__}"
            )

        if input_modal is _SENTINEL:
            type_hints = typing.get_type_hints(original_func)
            if len(type_hints) >= 1:
                input_modal = type_hints[list(type_hints.keys())[0]]
            else:
                input_modal = None

        route_signature = RouteSignature(
            path=path,
            is_websocket=True,
            input_modal=input_modal,
            buffering=buffering,
            session_timeout=session_timeout,
            max_batch_size=max_batch_size,
        )
        return _fal_websocket_template(
            original_func,
            route_signature,
        )

    return marker_fn
