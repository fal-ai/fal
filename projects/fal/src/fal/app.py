from __future__ import annotations

import inspect
import json
import os
import typing
from contextlib import asynccontextmanager
from typing import Any, Callable, ClassVar, TypeVar

from fastapi import FastAPI

import fal.api
from fal._serialization import add_serialization_listeners_for
from fal.api import RouteSignature
from fal.logging import get_logger
from fal.toolkit import mainify

REALTIME_APP_REQUIREMENTS = ["websockets", "msgpack"]

EndpointT = TypeVar("EndpointT", bound=Callable[..., Any])
logger = get_logger(__name__)


async def _call_any_fn(fn, *args, **kwargs):
    if inspect.iscoroutinefunction(fn):
        return await fn(*args, **kwargs)
    else:
        return fn(*args, **kwargs)


def wrap_app(cls: type[App], **kwargs) -> fal.api.IsolatedFunction:
    add_serialization_listeners_for(cls)

    def initialize_and_serve():
        app = cls()
        app.serve()

    metadata = {}
    try:
        app = cls(_allow_init=True)
        metadata["openapi"] = app.openapi()
    except Exception as exc:
        logger.warning("Failed to build OpenAPI specification for %s", cls.__name__)
        realtime_app = False
    else:
        routes = app.collect_routes()
        realtime_app = any(route.is_websocket for route in routes)

    wrapper = fal.api.function(
        "virtualenv",
        requirements=cls.requirements,
        machine_type=cls.machine_type,
        **cls.host_kwargs,
        **kwargs,
        metadata=metadata,
        exposed_port=8080,
        serve=False,
    )
    fn = wrapper(initialize_and_serve)
    fn.options.add_requirements(fal.api.SERVE_REQUIREMENTS)
    if realtime_app:
        fn.options.add_requirements(REALTIME_APP_REQUIREMENTS)

    return fn


@mainify
class App(fal.api.BaseServable):
    requirements: ClassVar[list[str]] = []
    machine_type: ClassVar[str] = "S"
    host_kwargs: ClassVar[dict[str, Any]] = {}

    def __init_subclass__(cls, **kwargs):
        parent_settings = getattr(cls, "host_kwargs", {})
        cls.host_kwargs = {**parent_settings, **kwargs}

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

    def collect_routes(self) -> dict[RouteSignature, Callable[..., Any]]:
        return {
            signature: endpoint
            for _, endpoint in inspect.getmembers(self, inspect.ismethod)
            if (signature := getattr(endpoint, "route_signature", None))
        }

    @asynccontextmanager
    async def lifespan(self, app: FastAPI):
        await _call_any_fn(self.setup)
        try:
            yield
        finally:
            await _call_any_fn(self.teardown)

    def setup(self):
        """Setup the application before serving."""

    def teardown(self):
        """Teardown the application after serving."""

    def _add_extra_middlewares(self, app: FastAPI):
        @app.middleware("http")
        async def provide_hints_headers(request, call_next):
            response = await call_next(request)
            try:
                response.headers["X-Fal-Runner-Hints"] = ",".join(self.provide_hints())
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

    def provide_hints(self) -> list[str]:
        """Provide hints for routing the application."""
        raise NotImplementedError


@mainify
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
            output = await loop.run_in_executor(None, func, self, *batch)  # type: ignore
            total_time = loop.time() - t0
            if not isinstance(output, dict):
                # Handle pydantic output modal
                if hasattr(output, "dict"):
                    output = output.dict()
                else:
                    raise TypeError(
                        f"Expected a dict or pydantic model as output, got {type(output)}"
                    )

            messages = [
                msgpack.packb(output, use_bin_type=True),
            ]
            if route_signature.emit_timings:
                # We emit x-fal messages in JSON, no matter what the input/output format is.
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


@mainify
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
                f"Can't set multiple routes for the same function: {original_func.__name__}"
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
