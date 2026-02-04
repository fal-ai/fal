from __future__ import annotations

import asyncio
import inspect
import json
import logging
import typing
from collections import deque
from contextlib import suppress
from typing import Any, Callable

from fastapi import WebSocket, WebSocketDisconnect

from fal._typing import EndpointT
from fal.api import RouteSignature

logger = logging.getLogger(__name__)


def msgpack_decode_message(message: bytes) -> Any:
    import msgpack

    return msgpack.unpackb(message, raw=False)


def msgpack_encode_message(message: Any) -> bytes:
    import msgpack

    return msgpack.packb(message, use_bin_type=True)


def _get_realtime_types(func: Callable[..., Any]) -> tuple[Any | None, Any | None]:
    signature = inspect.signature(func)
    params = [
        param
        for param in signature.parameters.values()
        if param.kind in (param.POSITIONAL_ONLY, param.POSITIONAL_OR_KEYWORD)
    ]
    if params and params[0].name in ("self", "cls"):
        params = params[1:]

    input_param = params[0].name if params else None
    type_hints = typing.get_type_hints(func)
    input_type = type_hints.get(input_param) if input_param else None
    output_type = type_hints.get("return")
    return input_type, output_type


def _unwrap_async_iterator(annotation: Any | None) -> Any | None:
    if annotation is None:
        return None

    import collections.abc

    if annotation in (
        typing.AsyncIterator,
        typing.AsyncGenerator,
        collections.abc.AsyncIterator,
        collections.abc.AsyncGenerator,
    ):
        return Any

    origin = typing.get_origin(annotation)
    if origin is None:
        return None

    if origin in (
        typing.AsyncIterator,
        typing.AsyncGenerator,
        collections.abc.AsyncIterator,
        collections.abc.AsyncGenerator,
    ):
        args = typing.get_args(annotation)
        return args[0] if args else Any

    return None


def _unwrap_sync_iterator(annotation: Any | None) -> Any | None:
    if annotation is None:
        return None

    import collections.abc

    if annotation in (
        typing.Iterator,
        typing.Iterable,
        typing.Generator,
        collections.abc.Iterator,
        collections.abc.Iterable,
        collections.abc.Generator,
    ):
        return Any

    origin = typing.get_origin(annotation)
    if origin is None:
        return None

    if origin in (
        typing.Iterator,
        typing.Iterable,
        typing.Generator,
        collections.abc.Iterator,
        collections.abc.Iterable,
        collections.abc.Generator,
    ):
        args = typing.get_args(annotation)
        return args[0] if args else Any

    return None


def _normalize_output(output: Any) -> dict[str, Any]:
    if not isinstance(output, dict):
        # Handle pydantic output modal
        if hasattr(output, "dict"):
            return output.dict()
        raise TypeError(
            "Expected a dict or pydantic model as output, got " f"{type(output)}"
        )
    return output


async def _mirror_input(
    queue: deque[Any],
    websocket: WebSocket,
    *,
    decode_message: Callable[[bytes], Any],
    input_modal: type | None,
    session_timeout: float | None,
) -> None:
    while True:
        try:
            raw_input = await asyncio.wait_for(
                websocket.receive_bytes(),
                timeout=session_timeout,
            )
        except asyncio.TimeoutError:
            return

        input = decode_message(raw_input)
        if input_modal:
            input = input_modal(**input)

        queue.append(input)


async def _receive_input(
    websocket: WebSocket,
    *,
    decode_message: Callable[[bytes], Any],
    input_modal: type | None,
    session_timeout: float | None,
) -> Any | None:
    try:
        raw_input = await asyncio.wait_for(
            websocket.receive_bytes(),
            timeout=session_timeout,
        )
    except (asyncio.TimeoutError, WebSocketDisconnect):
        return None

    decoded = decode_message(raw_input)
    if input_modal:
        decoded = input_modal(**decoded)
    return decoded


async def _iterate_inputs(
    websocket: WebSocket,
    *,
    decode_message: Callable[[bytes], Any],
    input_modal: type | None,
    session_timeout: float | None,
) -> typing.AsyncIterator[Any]:
    while True:
        item = await _receive_input(
            websocket,
            decode_message=decode_message,
            input_modal=input_modal,
            session_timeout=session_timeout,
        )
        if item is None:
            return
        yield item


def _next_sync_iterator_item(iterator: typing.Iterator[Any]) -> Any:
    try:
        return next(iterator)
    except StopIteration:
        return _SENTINEL


async def _iterate_sync_iterator(
    iterator: typing.Iterator[Any],
) -> typing.AsyncIterator[Any]:
    loop = asyncio.get_event_loop()
    while True:
        item = await loop.run_in_executor(None, _next_sync_iterator_item, iterator)
        if item is _SENTINEL:
            return
        yield item


async def _emit_message(websocket: WebSocket, message: bytes | str) -> None:
    if isinstance(message, bytes):
        await websocket.send_bytes(message)
    elif isinstance(message, str):
        await websocket.send_text(message)
    else:
        raise TypeError(f"Can't send message of type {type(message)}")


async def _background_emitter(
    outgoing_messages: asyncio.Queue[bytes | str],
    websocket: WebSocket,
) -> None:
    while True:
        output = await outgoing_messages.get()
        await _emit_message(websocket, output)
        outgoing_messages.task_done()


async def _mirror_output(
    self,
    queue: deque[Any],
    websocket: WebSocket,
    *,
    func: EndpointT,
    route_signature: RouteSignature,
    encode_message: Callable[[Any], bytes],
) -> None:
    loop = asyncio.get_event_loop()
    max_allowed_buffering = route_signature.buffering or 1
    outgoing_messages: asyncio.Queue[bytes | str] = asyncio.Queue(
        maxsize=max_allowed_buffering * 2  # x2 for outgoing timings
    )

    emitter = asyncio.create_task(_background_emitter(outgoing_messages, websocket))

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

        messages: list[bytes | str] = [encode_message(output)]
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
                await _emit_message(websocket, message)


async def _run_streaming_session(
    self,
    websocket: WebSocket,
    *,
    func: EndpointT,
    route_signature: RouteSignature,
    decode_message: Callable[[bytes], Any],
    encode_message: Callable[[Any], bytes],
    realtime_mode: str,
) -> None:
    loop = asyncio.get_event_loop()
    max_allowed_buffering = route_signature.buffering or 1
    outgoing_messages: asyncio.Queue[bytes | str] = asyncio.Queue(
        maxsize=max_allowed_buffering * 2  # x2 for outgoing timings
    )

    emitter = asyncio.create_task(_background_emitter(outgoing_messages, websocket))

    async def send_message(message):
        try:
            outgoing_messages.put_nowait(message)
        except asyncio.QueueFull:
            logger.warning(
                "Realtime outgoing queue full (maxsize=%s); sending message directly "
                "may reorder outputs.",
                outgoing_messages.maxsize,
            )
            await _emit_message(websocket, message)

    async def close_emitter():
        await outgoing_messages.join()
        emitter.cancel()
        with suppress(asyncio.CancelledError):
            await emitter

    if realtime_mode == "server_streaming":
        input_value = await _receive_input(
            websocket,
            decode_message=decode_message,
            input_modal=route_signature.input_modal,
            session_timeout=route_signature.session_timeout,
        )
        if input_value is None:
            await close_emitter()
            return

        t0 = loop.time()
        if inspect.iscoroutinefunction(func):
            result = await func(self, input_value)
        elif inspect.isasyncgenfunction(func):
            result = func(self, input_value)
        else:
            result = await loop.run_in_executor(None, func, self, input_value)  # type: ignore

        if not hasattr(result, "__aiter__") and not hasattr(result, "__iter__"):
            raise TypeError(
                "Expected an async iterator or iterator output for server-streaming "
                "realtime"
            )

        if hasattr(result, "__aiter__"):
            async_iterator = typing.cast(typing.AsyncIterator[Any], result)
            async for raw_output in async_iterator:
                output = _normalize_output(raw_output)
                await send_message(encode_message(output))
        else:
            sync_iterator = typing.cast(typing.Iterator[Any], result)
            async for raw_output in _iterate_sync_iterator(sync_iterator):
                output = _normalize_output(raw_output)
                await send_message(encode_message(output))

        total_time = loop.time() - t0
        if route_signature.emit_timings:
            timings = {
                "type": "x-fal-message",
                "action": "timings",
                "timing": total_time,
            }
            await send_message(json.dumps(timings, separators=(",", ":")))

        await close_emitter()
        return

    if realtime_mode == "client_streaming":
        if not inspect.iscoroutinefunction(func):
            raise TypeError(
                "Client-streaming realtime endpoints must be async functions"
            )
        result = await func(
            self,
            _iterate_inputs(
                websocket,
                decode_message=decode_message,
                input_modal=route_signature.input_modal,
                session_timeout=route_signature.session_timeout,
            ),
        )
        if hasattr(result, "__aiter__"):
            raise TypeError("Expected a single output for client-streaming realtime")
        output = _normalize_output(result)
        await send_message(encode_message(output))
        await close_emitter()
        return

    if realtime_mode == "bidi":
        if not inspect.iscoroutinefunction(func) and not inspect.isasyncgenfunction(
            func
        ):
            raise TypeError("Bidirectional realtime endpoints must be async functions")

        if inspect.isasyncgenfunction(func):
            result = func(
                self,
                _iterate_inputs(
                    websocket,
                    decode_message=decode_message,
                    input_modal=route_signature.input_modal,
                    session_timeout=route_signature.session_timeout,
                ),
            )
        else:
            result = await func(
                self,
                _iterate_inputs(
                    websocket,
                    decode_message=decode_message,
                    input_modal=route_signature.input_modal,
                    session_timeout=route_signature.session_timeout,
                ),
            )

        if not hasattr(result, "__aiter__"):
            raise TypeError(
                "Expected an async iterator output for bidirectional realtime"
            )

        async for raw_output in result:
            output = _normalize_output(raw_output)
            await send_message(encode_message(output))

        await close_emitter()
        return


async def _run_websocket_session(
    self,
    websocket: WebSocket,
    *,
    func: EndpointT,
    route_signature: RouteSignature,
    decode_message: Callable[[bytes], Any],
    encode_message: Callable[[Any], bytes],
    realtime_mode: str,
) -> None:
    await websocket.accept()

    if realtime_mode == "unary":
        queue: deque[Any] = deque(maxlen=route_signature.buffering)
        input_task = asyncio.create_task(
            _mirror_input(
                queue,
                websocket,
                decode_message=decode_message,
                input_modal=route_signature.input_modal,
                session_timeout=route_signature.session_timeout,
            )
        )
        input_task.add_done_callback(lambda _: queue.append(None))
        output_task = asyncio.create_task(
            _mirror_output(
                self,
                queue,
                websocket,
                func=func,
                route_signature=route_signature,
                encode_message=encode_message,
            )
        )
    else:
        input_task = None
        output_task = asyncio.create_task(
            _run_streaming_session(
                self,
                websocket,
                func=func,
                route_signature=route_signature,
                decode_message=decode_message,
                encode_message=encode_message,
                realtime_mode=realtime_mode,
            )
        )

    try:
        if input_task is not None:
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
        else:
            await output_task
    except WebSocketDisconnect:
        if input_task is not None:
            input_task.cancel()
            with suppress(asyncio.CancelledError, WebSocketDisconnect):
                await input_task
        output_task.cancel()

        with suppress(asyncio.CancelledError, WebSocketDisconnect):
            await output_task
    except Exception as exc:
        import traceback

        traceback.print_exc()

        with suppress(WebSocketDisconnect, RuntimeError):
            await websocket.send_json(
                {
                    "type": "x-fal-error",
                    "error": "INTERNAL_ERROR",
                    "reason": str(exc),
                }
            )
    else:
        with suppress(WebSocketDisconnect, RuntimeError):
            await websocket.send_json(
                {
                    "type": "x-fal-error",
                    "error": "TIMEOUT",
                    "reason": "no inputs, reconnect when needed!",
                }
            )

    with suppress(WebSocketDisconnect, RuntimeError):
        await websocket.close()


def _fal_websocket_template(
    func: EndpointT,
    route_signature: RouteSignature,
) -> EndpointT:
    # A template for fal's realtime websocket endpoints to basically
    # be a boilerplate for the user to fill in their inference function
    # and start using it.

    decode_message = route_signature.decode_message or msgpack_decode_message
    encode_message = route_signature.encode_message or msgpack_encode_message

    input_type, output_type = _get_realtime_types(func)
    input_stream_item = _unwrap_async_iterator(input_type)
    output_stream_item = _unwrap_async_iterator(output_type)
    output_sync_stream_item = _unwrap_sync_iterator(output_type)

    if input_stream_item is not None and output_stream_item is not None:
        realtime_mode = "bidi"
    elif input_stream_item is not None:
        realtime_mode = "client_streaming"
    elif output_stream_item is not None or output_sync_stream_item is not None:
        realtime_mode = "server_streaming"
    else:
        realtime_mode = "unary"

    async def websocket_template(self, websocket: WebSocket) -> None:
        await _run_websocket_session(
            self,
            websocket,
            func=func,
            route_signature=route_signature,
            decode_message=decode_message,
            encode_message=encode_message,
            realtime_mode=realtime_mode,
        )

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
    output_modal: Any | None = _SENTINEL,
    max_batch_size: int = 1,
    encode_message: Callable[[Any], bytes] | None = None,
    decode_message: Callable[[bytes], Any] | None = None,
) -> Callable[[EndpointT], EndpointT]:
    """Designate the decorated function as a realtime application endpoint."""

    def marker_fn(original_func: EndpointT) -> EndpointT:
        nonlocal input_modal, output_modal

        if hasattr(original_func, "route_signature"):
            raise ValueError(
                "Can't set multiple routes for the same function: "
                f"{original_func.__name__}"
            )

        input_type, output_type = _get_realtime_types(original_func)

        if input_modal is _SENTINEL:
            input_stream_item = _unwrap_async_iterator(input_type)
            if input_stream_item is not None:
                input_modal = None if input_stream_item is Any else input_stream_item
            else:
                input_modal = None if input_type is Any else input_type

        if output_modal is _SENTINEL:
            output_stream_item = _unwrap_async_iterator(output_type)
            output_sync_stream_item = _unwrap_sync_iterator(output_type)
            if output_stream_item is not None:
                output_modal = None if output_stream_item is Any else output_stream_item
            elif output_sync_stream_item is not None:
                output_modal = (
                    None if output_sync_stream_item is Any else output_sync_stream_item
                )
            else:
                output_modal = None if output_type is Any else output_type

        route_signature = RouteSignature(
            path=path,
            is_websocket=True,
            input_modal=input_modal,
            output_modal=output_modal,
            buffering=buffering,
            session_timeout=session_timeout,
            max_batch_size=max_batch_size,
            encode_message=encode_message,
            decode_message=decode_message,
        )
        return _fal_websocket_template(
            original_func,
            route_signature,
        )

    return marker_fn
