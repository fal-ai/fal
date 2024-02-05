from __future__ import annotations

import inspect
import os
import typing
from contextlib import asynccontextmanager
from typing import Any, Callable, ClassVar, NamedTuple, TypeVar

from fastapi import FastAPI

import fal.api
from fal._serialization import add_serialization_listeners_for
from fal.logging import get_logger
from fal.toolkit import mainify

REALTIME_APP_REQUIREMENTS = ["websockets", "msgpack"]

EndpointT = TypeVar("EndpointT", bound=Callable[..., Any])
logger = get_logger(__name__)


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
        serve=True,
    )
    fn = wrapper(initialize_and_serve)
    if realtime_app:
        fn.options.add_requirements(REALTIME_APP_REQUIREMENTS)
    return fn.on(
        serve=False,
        exposed_port=8080,
    )


@mainify
class RouteSignature(NamedTuple):
    path: str
    is_websocket: bool = False


@mainify
class App:
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

    def setup(self):
        """Setup the application before serving."""

    def provide_hints(self) -> list[str]:
        """Provide hints for routing the application."""
        raise NotImplementedError

    def serve(self) -> None:
        import uvicorn

        app = self._build_app()
        uvicorn.run(app, host="0.0.0.0", port=8080)

    def collect_routes(self) -> dict[RouteSignature, Callable[..., Any]]:
        return {
            signature: endpoint
            for _, endpoint in inspect.getmembers(self, inspect.ismethod)
            if (signature := getattr(endpoint, "route_signature", None))
        }

    def _build_app(self) -> FastAPI:
        from fastapi import FastAPI
        from fastapi.middleware.cors import CORSMiddleware

        @asynccontextmanager
        async def lifespan(app: FastAPI):
            self.setup()
            try:
                yield
            finally:
                self.teardown()

        _app = FastAPI(lifespan=lifespan)

        @_app.middleware("http")
        async def provide_hints(request, call_next):
            response = await call_next(request)
            try:
                response.headers["X-Fal-Runner-Hints"] = ",".join(self.provide_hints())
            except NotImplementedError:
                # This lets us differentiate between apps that don't provide hints
                # and apps that provide empty hints.
                pass
            except Exception as exc:
                from fastapi.logger import logger

                logger.exception(
                    "Failed to provide hints for %s",
                    self.__class__.__name__,
                    exc_info=exc,
                )
            return response

        _app.add_middleware(
            CORSMiddleware,
            allow_credentials=True,
            allow_headers=("*"),
            allow_methods=("*"),
            allow_origins=("*"),
        )

        routes = self.collect_routes()
        if not routes:
            raise ValueError("An application must have at least one route!")

        for signature, endpoint in routes.items():
            if signature.is_websocket:
                _app.add_api_websocket_route(
                    signature.path,
                    endpoint,
                    name=endpoint.__name__,
                )
            else:
                _app.add_api_route(
                    signature.path,
                    endpoint,
                    name=endpoint.__name__,
                    methods=["POST"],
                )

        return _app

    def openapi(self) -> dict[str, Any]:
        """
        Build the OpenAPI specification for the served function.
        Attach needed metadata for a better integration to fal.
        """
        app = self._build_app()
        spec = app.openapi()
        _mark_order_openapi(spec)
        return spec

    def teardown(self):
        """Teardown the application after serving."""


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
    buffering: int | None = None,
    session_timeout: float | None = None,
    input_modal: Any | None = None,
    max_batch_size: int = 1,
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
                    timeout=session_timeout,
                )
            except asyncio.TimeoutError:
                return

            input = msgpack.unpackb(raw_input, raw=False)
            if input_modal:
                input = input_modal(**input)

            queue.append(input)

    async def mirror_output(
        self,
        queue: deque[Any],
        websocket: WebSocket,
    ) -> None:
        loop = asyncio.get_event_loop()
        outgoing_messages: asyncio.Queue[bytes] = asyncio.Queue(maxsize=buffering or 1)

        async def emit(message):
            await websocket.send_bytes(message)

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
            while queue and len(batch) < max_batch_size:
                next_input = queue.popleft()
                if hasattr(input, "can_batch") and not input.can_batch(
                    next_input, len(batch)
                ):
                    queue.appendleft(next_input)
                    break
                batch.append(next_input)

            output = await loop.run_in_executor(None, func, self, *batch)  # type: ignore
            if not isinstance(output, dict):
                # Handle pydantic output modal
                if hasattr(output, "dict"):
                    output = output.dict()
                else:
                    raise TypeError(
                        f"Expected a dict or pydantic model as output, got {type(output)}"
                    )

            message = msgpack.packb(output, use_bin_type=True)
            try:
                outgoing_messages.put_nowait(message)
            except asyncio.QueueFull:
                await emit(message)

    async def websocket_template(self, websocket: WebSocket) -> None:
        import asyncio

        await websocket.accept()

        queue: deque[Any] = deque(maxlen=buffering)
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
                await asyncio.wait_for(output_task, timeout=session_timeout)
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

        callable = _fal_websocket_template(
            original_func,
            buffering=buffering,
            session_timeout=session_timeout,
            input_modal=input_modal,
            max_batch_size=max_batch_size,
        )
        callable.route_signature = RouteSignature(path=path, is_websocket=True)  # type: ignore
        callable.original_func = original_func  # type: ignore
        return callable

    return marker_fn


def _mark_order_openapi(spec: dict[str, Any]):
    """
    Add x-fal-order-* keys to the OpenAPI specification to help the rendering of UI.

    NOTE: We rely on the fact that fastapi and Python dicts keep the order of properties.
    """

    def mark_order(obj: dict[str, Any], key: str):
        obj[f"x-fal-order-{key}"] = list(obj[key].keys())

    mark_order(spec, "paths")

    def order_schema_object(schema: dict[str, Any]):
        """
        Mark the order of properties in the schema object.
        They can have 'allOf', 'properties' or '$ref' key.
        """
        if "allOf" in schema:
            for sub_schema in schema["allOf"]:
                order_schema_object(sub_schema)
        if "properties" in schema:
            mark_order(schema, "properties")

    for key in spec["components"].get("schemas") or {}:
        order_schema_object(spec["components"]["schemas"][key])

    return spec
