from __future__ import annotations

import asyncio
import inspect
import json
import typing
from collections import deque
from contextlib import suppress
from typing import Any, Callable

from fastapi import WebSocket, WebSocketDisconnect

from fal._typing import EndpointT
from fal.api import RouteSignature


def msgpack_decode_message(message: bytes) -> Any:
    import msgpack

    return msgpack.unpackb(message, raw=False)


def msgpack_encode_message(message: Any) -> bytes:
    import msgpack

    return msgpack.packb(message, use_bin_type=True)


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


async def _run_websocket_session(
    self,
    websocket: WebSocket,
    *,
    func: EndpointT,
    route_signature: RouteSignature,
    decode_message: Callable[[bytes], Any],
    encode_message: Callable[[Any], bytes],
) -> None:
    await websocket.accept()

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
            await asyncio.wait_for(output_task, timeout=route_signature.session_timeout)
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


def _fal_websocket_template(
    func: EndpointT,
    route_signature: RouteSignature,
) -> EndpointT:
    # A template for fal's realtime websocket endpoints to basically
    # be a boilerplate for the user to fill in their inference function
    # and start using it.

    decode_message = route_signature.decode_message or msgpack_decode_message
    encode_message = route_signature.encode_message or msgpack_encode_message

    async def websocket_template(self, websocket: WebSocket) -> None:
        await _run_websocket_session(
            self,
            websocket,
            func=func,
            route_signature=route_signature,
            decode_message=decode_message,
            encode_message=encode_message,
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
    max_batch_size: int = 1,
    encode_message: Callable[[Any], bytes] | None = None,
    decode_message: Callable[[bytes], Any] | None = None,
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
            encode_message=encode_message,
            decode_message=decode_message,
        )
        return _fal_websocket_template(
            original_func,
            route_signature,
        )

    return marker_fn
