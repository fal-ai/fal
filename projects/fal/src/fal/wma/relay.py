"""WebSocket signaling relay for partner-hosted WebRTC media (``@fal.realtime`` apps).

Some realtime apps do not terminate media on the runner:
the browser peers directly with the partner's media server, and the fal app is a
pure *signaling relay* riding a ``@fal.realtime`` WebSocket — it forwards SDP/ICE
to the partner's signaling endpoint and translates control updates (prompts,
reference images) both ways.

``WebSocketSignalingRelay`` owns the session machinery every such app needs:

- upstream WebSocket connect with a timeout and a structured connect error,
- bootstrap events pushed to the client before any signaling (``ready`` /
  ``iceServers`` in the Playground contract),
- inline forwarding of signaling messages (offer / ICE) so nothing can stall them,
- an ordered control worker gated on the upstream *answer*, so a slow control
  payload build (e.g. a hosted reference-image fetch) can neither reorder later
  controls nor reach the partner before the SDP handshake completes,
- error surfacing that keeps the session alive on per-message failures,
- full teardown when either side drops.

Apps plug in translation hooks and keep their own input/output models, so their
client-facing wire contract stays exactly as published:

    relay = WebSocketSignalingRelay(upstream_url=..., signaling_payload=...,
                                    control_payload=..., from_upstream=..., ...)

    @fal.realtime("/realtime")
    async def realtime(self, inputs):
        async for output in relay.run(inputs):
            yield output

A control-payload hook may raise ``ControlRejected`` to report a bad client
update (invalid reference image, oversized payload, ...); the relay surfaces it
via ``make_error`` and keeps the session going.
"""

import asyncio
from typing import Any, AsyncIterator, Awaitable, Callable, Sequence, Tuple, Type, Union


class ControlRejected(Exception):
    """A client control update was rejected by the app's ``control_payload`` hook.

    Surfaced to the client as an error output (never silently dropped, so e.g. a
    character-swap request never runs prompt-only without its reference image);
    the session stays alive.
    """


def _connection_closed_excs() -> Tuple[Type[BaseException], ...]:
    """Exception types that mean the upstream closed (normal end of session)."""
    try:
        from websockets.exceptions import ConnectionClosed
    except ImportError:  # pragma: no cover - websockets ships with relay users
        return ()
    return (ConnectionClosed,)


class WebSocketSignalingRelay:
    """Relay a ``@fal.realtime`` session to a partner signaling WebSocket.

    Hooks (all required unless noted):

    - ``signaling_payload(item) -> str | None``: upstream payload for a client
      signaling message (offer / ICE), or ``None`` to skip.
    - ``control_payload(item) -> Awaitable[str | None]``: upstream payload for a
      client control update; may raise ``ControlRejected``. Runs on the ordered
      control worker, so it may be slow (network fetches) without stalling
      signaling.
    - ``from_upstream(raw) -> output | None``: translate one upstream message to
      a client output; ``None`` drops it.
    - ``is_signaling(item) -> bool``: does this client message carry signaling?
    - ``has_control(item) -> bool``: does this client message carry a control
      update? (A message may carry both.)
    - ``is_session_ready(output) -> bool``: does this upstream-derived output
      mark the session as negotiated (typically the SDP answer)? Buffered
      controls are released once it fires.
    - ``make_error(message) -> output``: build a client-facing error output.
    - ``bootstrap``: outputs pushed to the client right after the upstream
      connects, before any signaling (e.g. ``ready`` + ``iceServers``).
    - ``connect_upstream`` (optional): override the upstream connection factory;
      must return an object with ``send``/``close`` and async iteration. The
      default connects ``upstream_url`` with the ``websockets`` package.
    """

    def __init__(
        self,
        *,
        upstream_url: str,
        signaling_payload: Callable[[Any], Union[str, None]],
        control_payload: Callable[[Any], Awaitable[Union[str, None]]],
        from_upstream: Callable[[Union[str, bytes]], Union[Any, None]],
        is_signaling: Callable[[Any], bool],
        has_control: Callable[[Any], bool],
        is_session_ready: Callable[[Any], bool],
        make_error: Callable[[str], Any],
        bootstrap: Sequence[Any] = (),
        connect_upstream: Union[Callable[[], Awaitable[Any]], None] = None,
        connect_timeout: float = 15.0,
        send_timeout: float = 30.0,
        ping_interval: Union[float, None] = 5,
        ping_timeout: Union[float, None] = 5,
        close_timeout: Union[float, None] = 10,
        connect_error: str = "Could not connect to the upstream service. Please retry.",
        init_error: str = "Could not start the session. Please retry.",
        forward_error: str = "Failed to forward the update.",
        control_error: str = "Failed to apply the update.",
        label: str = "wma-relay",
    ) -> None:
        self.upstream_url = upstream_url
        self.signaling_payload = signaling_payload
        self.control_payload = control_payload
        self.from_upstream = from_upstream
        self.is_signaling = is_signaling
        self.has_control = has_control
        self.is_session_ready = is_session_ready
        self.make_error = make_error
        self.bootstrap = list(bootstrap)
        self._connect_upstream = connect_upstream
        self.connect_timeout = connect_timeout
        self.send_timeout = send_timeout
        self.ping_interval = ping_interval
        self.ping_timeout = ping_timeout
        self.close_timeout = close_timeout
        self.connect_error = connect_error
        self.init_error = init_error
        self.forward_error = forward_error
        self.control_error = control_error
        self.label = label

    def _debug(self, msg: str) -> None:
        print(f"[{self.label}] {msg}", flush=True)

    async def _default_connect(self) -> Any:
        import websockets

        return await websockets.connect(
            self.upstream_url,
            ping_interval=self.ping_interval,
            ping_timeout=self.ping_timeout,
            close_timeout=self.close_timeout,
        )

    async def _connect(self) -> Any:
        connect = self._connect_upstream or self._default_connect
        return await asyncio.wait_for(connect(), timeout=self.connect_timeout)

    async def run(self, inputs: AsyncIterator[Any]) -> AsyncIterator[Any]:
        """Drive one relay session; yields client outputs until either side ends."""
        try:
            upstream = await self._connect()
        except Exception as e:
            # Timeout / handshake / unreachable: surface a structured error to
            # the client instead of letting the exception propagate out of the
            # generator (which would leave the client with no ready/error event).
            self._debug(f"upstream connect failed: {type(e).__name__}: {e}")
            yield self.make_error(self.connect_error)
            return

        closed_excs = _connection_closed_excs()
        output_queue: asyncio.Queue = asyncio.Queue()

        # Push the bootstrap events (e.g. ready / iceServers) before any client
        # signaling is answered. These run after the upstream is open but before
        # the main try/finally below, so guard them: close the upstream ourselves
        # if a put fails, otherwise the connection would leak (closed only at GC).
        try:
            for item in self.bootstrap:
                await output_queue.put(item)
        except Exception as e:
            self._debug(f"session init failed: {type(e).__name__}: {e}")
            try:
                await upstream.close()
            except Exception:
                pass
            yield self.make_error(self.init_error)
            return

        # Control messages are applied strictly in order by a single worker, off
        # the signaling path. Signaling (offer / ICE) is sent inline so a slow
        # control-payload build can neither stall it nor reorder later controls.
        control_queue: asyncio.Queue = asyncio.Queue()
        # Control updates are held until the session is negotiated (the upstream
        # answer arrives), so a control payload never reaches the partner before
        # the offer -> answer handshake. The answer is independent of any
        # control, so gating on it can't deadlock.
        session_ready = asyncio.Event()

        async def send_upstream(payload: Union[str, None]) -> None:
            if payload is None:
                return
            try:
                # ping_timeout only covers ping/pong frames; a live-but-slow
                # peer can still stall send() under TCP backpressure, so the
                # data send needs its own budget.
                await asyncio.wait_for(upstream.send(payload), self.send_timeout)
            except Exception as e:
                # Tell the client the update didn't reach the partner instead of
                # letting it assume the offer / ICE / control was applied.
                self._debug(f"upstream send failed: {e}")
                await output_queue.put(self.make_error(self.forward_error))

        async def send_control(item: Any) -> None:
            try:
                payload = await self.control_payload(item)
            except ControlRejected as e:
                # Report to the client and keep the session alive.
                await output_queue.put(self.make_error(str(e)))
                return
            await send_upstream(payload)

        async def control_worker() -> None:
            # Hold updates until the session is negotiated, then apply them
            # serially and in order (incl. slow payload builds), so a lagging
            # build never sends a stale control after a newer update.
            await session_ready.wait()
            while True:
                item = await control_queue.get()
                if item is None:
                    return
                try:
                    await send_control(item)
                except Exception as e:
                    # A single bad update must not kill the worker (which would
                    # leave all later updates queued forever); log, report, and
                    # keep going.
                    self._debug(f"control worker error: {e}")
                    await output_queue.put(self.make_error(self.control_error))

        async def client_pump() -> None:
            try:
                async for item in inputs:
                    # A message may carry signaling AND a control update; handle
                    # both. Signaling: inline and immediate — never blocked.
                    if self.is_signaling(item):
                        await send_upstream(self.signaling_payload(item))
                    # Control update: applied in order by the worker, off the
                    # signaling path. Present independently of signaling.
                    if self.has_control(item):
                        await control_queue.put(item)
            except Exception as e:
                self._debug(f"client->upstream error: {e}")
            finally:
                # Stop the control worker, then close the upstream so
                # upstream_pump reaches its sentinel and the session tears down
                # promptly (WebRTC signaling has no client-done message that
                # makes the partner hang up).
                await control_queue.put(None)
                try:
                    await upstream.close()
                except Exception:
                    pass

        async def upstream_pump() -> None:
            try:
                async for raw in upstream:
                    out = self.from_upstream(raw)
                    if out is not None:
                        await output_queue.put(out)
                        # Session negotiated — release any buffered controls.
                        if not session_ready.is_set() and self.is_session_ready(out):
                            session_ready.set()
            except closed_excs as e:
                # The partner hung up. Log it: a clean close right after the
                # handshake is otherwise indistinguishable from a client leave.
                self._debug(f"upstream closed: {e!r}")
            except Exception as e:
                self._debug(f"upstream->client error: {e}")
            finally:
                self._debug("upstream stream ended; ending session")
                await output_queue.put(None)

        recv_task = asyncio.create_task(upstream_pump())
        control_task = asyncio.create_task(control_worker())
        send_task = asyncio.create_task(client_pump())
        try:
            while True:
                item = await output_queue.get()
                if item is None:
                    break
                yield item
        finally:
            for task in (send_task, control_task):
                task.cancel()
                try:
                    await task
                except (asyncio.CancelledError, Exception):
                    pass
            try:
                await upstream.close()
            except Exception:
                pass
            try:
                await recv_task
            except (asyncio.CancelledError, Exception):
                pass
