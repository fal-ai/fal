"""Helpers for the *raw* WMA integration path (``POST /start-session`` + SSE).

This is the low-level half of ``fal.wma``, for apps that own their WebRTC
media pipeline instead of implementing a ``PeerBackend``:

- The app exposes ``POST /start-session`` receiving the client's SDP plus the
  bridge-assigned session id and short-lived runner ICE configuration. The WMA
  bridge at ``wma.fal.run`` forwards this request; direct callers omit the
  additive ICE fields and receive the runner's STUN-only fallback.
- The endpoint answers with an SSE stream whose *first* event is the SDP answer, and
  the HTTP response is then held open for the entire lifetime of the session. When the
  generator exits (client dropped, heartbeats stopped, peer closed) everything tears
  down together.
- Media flows peer-to-peer: the app pushes generated frames on an outbound video
  track, and receives control input (keys, prompts, resets) on a client-created
  data channel (app-defined message schema).

Only pure-Python pieces (SSE formatting, session slot, queue helper) live at module
level so they can be imported and unit-tested locally. Everything that needs
``aiortc``/``av`` (runner-only packages) is imported inside functions.
"""

from __future__ import annotations

import asyncio
import inspect
import ipaddress
import json
import logging
import threading
from typing import Any, AsyncIterator, Awaitable, Callable, Mapping

from fal.toolkit.utils.ssrf import is_globally_routable_ip
from fal.wma._aioice_teardown import install_orderly_ice_teardown
from fal.wma.telemetry import CONNECTION_REPORT_VERSION

# ---------------------------------------------------------------------------
# SSE formatting (the /start-session response is an SSE stream)
# ---------------------------------------------------------------------------

#: Comment line emitted periodically so intermediaries don't time the session out.
SSE_KEEPALIVE = ": keepalive\n\n"

logger = logging.getLogger(__name__)


def sse_event(payload: Mapping[str, Any], *, event: str | None = None) -> str:
    """Format a mapping as one optional named SSE event."""
    if event is not None and ("\n" in event or "\r" in event):
        raise ValueError("SSE event names cannot contain newlines")
    prefix = f"event: {event}\n" if event is not None else ""
    return f"{prefix}data: {json.dumps(dict(payload))}\n\n"


# ---------------------------------------------------------------------------
# Session slot (one interactive stream per runner)
# ---------------------------------------------------------------------------


class SessionSlot:
    """Guards the single interactive session a stateful runner can serve.

    World-model pipelines hold per-stream state (KV caches, VAE decode caches), so a
    runner can only serve one session at a time; a second concurrent offer must be
    rejected so the platform routes it to another runner.
    """

    def __init__(self) -> None:
        self._busy = False
        self._lock = threading.Lock()

    def try_acquire(self) -> bool:
        with self._lock:
            if self._busy:
                return False
            self._busy = True
            return True

    def release(self) -> None:
        with self._lock:
            self._busy = False


# ---------------------------------------------------------------------------
# asyncio queue helper
# ---------------------------------------------------------------------------


def queue_put_drop_oldest(queue: asyncio.Queue[Any], item: Any) -> None:
    """Put ``item`` on ``queue``, dropping the oldest entry when full.

    Frame queues must never exert unbounded backpressure on the generator loop; when
    the peer link is slower than generation, stale frames are the right thing to lose.
    """
    while True:
        try:
            queue.put_nowait(item)
            return
        except asyncio.QueueFull:
            try:
                queue.get_nowait()
            except asyncio.QueueEmpty:  # pragma: no cover - racy fallback
                continue


# ---------------------------------------------------------------------------
# WebRTC pieces (aiortc/av imported lazily: runner-only packages)
# ---------------------------------------------------------------------------

VIDEO_CLOCK_RATE = 90_000


def make_video_queue_track(
    frame_queue: asyncio.Queue[Any], fps: float, stats: dict | None = None
) -> Any:
    """Build an outbound video track fed by ``frame_queue`` of RGB uint8 ndarrays.

    Frames are paced at ``fps`` against a wall clock so a burst of generated frames
    (world models produce whole blocks at once) plays back smoothly on the client.

    Latency instrumentation: items may be ``(push_monotonic_ts, image)`` tuples; when
    ``stats`` is given, per-frame queue age (push->pop) and pacing sleep are appended
    to ``stats["queue_age_ms"]`` / ``stats["pace_sleep_ms"]`` (caller drains them).
    """
    import time
    from fractions import Fraction

    import av
    import numpy as np
    from aiortc import VideoStreamTrack

    class QueueVideoTrack(VideoStreamTrack):
        def __init__(self) -> None:
            super().__init__()
            self._pts = 0
            self._started_at: float | None = None

        async def recv(self) -> Any:
            item = await frame_queue.get()
            now = time.monotonic()
            if isinstance(item, tuple):
                push_ts, image = item
                if stats is not None:
                    stats.setdefault("queue_age_ms", []).append(
                        (now - push_ts) * 1000.0
                    )
            else:
                image = item
            if self._started_at is None:
                self._started_at = now
            else:
                self._pts += int(VIDEO_CLOCK_RATE / fps)
                target = self._started_at + self._pts / VIDEO_CLOCK_RATE
                delay = target - now
                if delay > 0:
                    if stats is not None:
                        stats.setdefault("pace_sleep_ms", []).append(delay * 1000.0)
                    await asyncio.sleep(delay)
                elif stats is not None:
                    stats.setdefault("pace_sleep_ms", []).append(0.0)
            # Decoded frames can be views of permuted tensors; av requires
            # C-contiguous input (no-op when already contiguous).
            frame = av.VideoFrame.from_ndarray(
                np.ascontiguousarray(image), format="rgb24"
            )
            frame.pts = self._pts
            frame.time_base = Fraction(1, VIDEO_CLOCK_RATE)
            return frame

    return QueueVideoTrack()


def _is_unsafe_ice_target(address: str) -> bool:
    """True if an ICE candidate address is an internal/link-local target.

    A client-supplied offer is otherwise free to list any UDP address as a host
    candidate; without this check the server's own ICE agent would happily fire
    STUN connectivity checks at it (e.g. cloud metadata IPs), which is an SSRF
    vector even though the response never reaches the client.

    Delegates to the shared :func:`fal.toolkit.utils.ssrf.is_globally_routable_ip`
    rather than re-deriving the same predicates, so this filter automatically
    picks up its coverage of CGNAT (``100.64.0.0/10``), IPv4-mapped IPv6
    (``::ffff:169.254.169.254``), and 6to4/NAT64 embeddings — ranges plain
    ``ipaddress`` predicates don't classify as private/reserved.

    Non-IP targets are hostnames. The only hostname form real ICE candidates use
    is mDNS (``*.local``, RFC 8445 6.2) for privacy-preserving host candidates —
    anything else is not a legitimate candidate shape and could be an
    attacker-controlled DNS name crafted to resolve to an internal address
    (DNS rebinding), so it's rejected rather than passed through.
    """
    try:
        ipaddress.ip_address(address)
    except ValueError:
        return not address.lower().endswith(".local")
    return not is_globally_routable_ip(address)


def filter_sdp_ice_candidates(sdp: str) -> str:
    """Strip ``a=candidate`` lines pointing at internal/reserved addresses.

    ``address`` is the 5th space-separated token of an ICE candidate line
    (https://www.rfc-editor.org/rfc/rfc5245#section-15.1). Lines that aren't
    literal IPs (rare mDNS/hostname candidates) are passed through unfiltered.
    """
    safe_lines = []
    for line in sdp.splitlines(keepends=True):
        if line.startswith("a=candidate:"):
            parts = line.split()
            if len(parts) >= 5 and _is_unsafe_ice_target(parts[4]):
                continue
        safe_lines.append(line)
    return "".join(safe_lines)


class ClientOfferError(Exception):
    """The remote offer was malformed or rejected by ``setRemoteDescription``.

    This is client input, unlike a failure in the later ``createAnswer`` /
    ``setLocalDescription`` steps, which run entirely on the server (parsing
    the offer already succeeded) and are a server-side/negotiation fault, not
    the client's.
    """


async def negotiate_answer(pc: Any, sdp: str, type_: str) -> str:
    """Run the server side of the SDP exchange and return the answer SDP.

    aiortc does not trickle ICE: ``setLocalDescription`` resolves only once candidate
    gathering finished, so the returned SDP is complete — exactly what the WMA bridge
    expects (it does not support trickle ICE either).

    Candidates aimed at internal/reserved addresses are stripped from the offer
    first (see :func:`filter_sdp_ice_candidates`) so a malicious offer can't use
    the server's ICE agent to probe internal network targets.

    A failure applying the remote offer raises :class:`ClientOfferError` (client
    input); a failure in the answer/local-ICE steps that follow propagates
    unchanged, since that's a server-side fault, not a bad request.

    Installs the orderly-ICE-teardown shim (see
    :mod:`fal.wma._aioice_teardown`) here — the first WebRTC step of a
    session, before any connection can be torn down — so every later
    ``pc.close()``, including aiortc-internal closes on ICE failure, benefits.
    """
    from aiortc import RTCSessionDescription

    install_orderly_ice_teardown()

    try:
        await pc.setRemoteDescription(
            RTCSessionDescription(sdp=filter_sdp_ice_candidates(sdp), type=type_)
        )
    except Exception as exc:
        raise ClientOfferError(str(exc)) from exc
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)
    return pc.localDescription.sdp


# Bound on how long a fresh peer connection may sit in a non-terminal ICE state
# (``new``/``connecting``/``checking``) before the session is abandoned and its
# runner slot freed. On a restrictive network (UDP blocked, no relay/TURN path)
# ICE may never reach ``connected`` *nor* ``failed`` — it just stalls in
# ``checking`` — so without this bound the slot would be held for the full
# ``MAX_SESSION_SECONDS`` (10 min), and every reconnect meanwhile gets a 503
# "already serving a session".
#
# This server-side backstop allows enough time for cold TURN/TCP/TLS relay
# allocation on a firewalled path, while still bounding how long a failed
# negotiation can retain application resources.
INITIAL_CONNECT_TIMEOUT_SECONDS = 35.0


def watch_connection_state(
    pc: Any,
    closed: asyncio.Event,
    connected: asyncio.Event | None = None,
    *,
    disconnected_grace: float | None = None,
) -> None:
    """Set ``connected`` when media can flow and ``closed`` when the peer
    connection ends, however it ends.

    ``disconnected`` is transient by default: it is an often self-recovering ICE
    blip (network jitter, a brief NAT rebind), not necessarily terminal, so with
    ``disconnected_grace=None`` (the default) only ``closed``/``failed`` set
    ``closed`` — treating ``disconnected`` as terminal would tear down a session
    that could have reconnected on its own, matching the client-side policy in
    ``wmaSession.ts``.

    When ``disconnected_grace`` (seconds) is given, a connection that enters
    ``disconnected`` and *stays* there that long is finally treated as terminal
    (``closed`` set). Recovering to ``connected`` before the grace elapses (or
    any transition to a terminal state) cancels the pending timer, so brief
    recoverable blips are still preserved while a session wedged in
    ``disconnected`` no longer holds the runner until ``MAX_SESSION_SECONDS``.
    The grace timer is scheduled on the running event loop, so this variant must
    be called from within a running loop (the app's request handler is).
    """
    grace: dict[str, asyncio.Task | None] = {"task": None}

    def _cancel_grace() -> None:
        task = grace["task"]
        if task is not None:
            task.cancel()
            grace["task"] = None

    async def _close_if_still_disconnected() -> None:
        try:
            await asyncio.sleep(disconnected_grace)  # type: ignore[arg-type]
        except asyncio.CancelledError:
            return
        finally:
            task = asyncio.current_task()
            if grace["task"] is task:
                grace["task"] = None
        # Only terminal if it never recovered during the grace window.
        if pc.connectionState == "disconnected":
            closed.set()

    @pc.on("connectionstatechange")
    def _on_state_change() -> None:
        state = pc.connectionState
        if state == "connected":
            _cancel_grace()
            if connected is not None:
                connected.set()
        elif state == "disconnected" and disconnected_grace is not None:
            if grace["task"] is None:
                grace["task"] = asyncio.ensure_future(_close_if_still_disconnected())
        if state in ("closed", "failed"):
            _cancel_grace()
            closed.set()


async def wait_for_initial_connect(
    connected: asyncio.Event,
    closed: asyncio.Event,
    timeout: float = INITIAL_CONNECT_TIMEOUT_SECONDS,
) -> bool:
    """Wait until the peer connection first reaches ``connected``.

    Returns ``True`` if it connected within ``timeout``; ``False`` if the timeout
    elapsed with no connection, or the connection reached a terminal state
    (``closed``/``failed``, which sets ``closed``) first. Racing on ``closed`` too
    is what makes an ICE *failure* free the caller immediately instead of blocking
    for the whole ``timeout`` — a plain ``connected.wait()`` ignores ``closed`` and
    would sit out the full window even though the connection is already dead.

    A ``False`` return is the producer's cue to abandon the session (which sets
    ``closed`` in its ``finally``, ending the SSE stream and releasing the slot).
    """
    conn_wait = asyncio.ensure_future(connected.wait())
    closed_wait = asyncio.ensure_future(closed.wait())
    try:
        await asyncio.wait(
            {conn_wait, closed_wait},
            timeout=timeout,
            return_when=asyncio.FIRST_COMPLETED,
        )
        return connected.is_set()
    finally:
        conn_wait.cancel()
        closed_wait.cancel()


async def close_peer_connection(pc: Any) -> None:
    """Idempotently close an aiortc ``RTCPeerConnection`` with orderly ICE teardown.

    This is the single teardown path shared by raw-path WMA apps.
    ``RTCPeerConnection.close()`` is itself idempotent (aiortc guards on an
    internal ``__isClosed`` future), so a double call — e.g. the negotiation-
    failure path *and* the session ``cleanup`` both firing — is safe. Any
    exception raised during teardown is swallowed: the connection is already
    being abandoned, and on the negotiation-failure path an unguarded raise here
    would otherwise mask the mapped, client-facing negotiation error we actually
    want to surface.

    The aioice STUN teardown race (fixed via orderly-teardown shim)
    ---------------------------------------------------------------
    ``await pc.close()`` is the orderly ICE teardown; it is what lets aioice cancel
    its in-flight STUN transactions. But *stock* aioice ``Connection.close()``
    (0.10.2, and unchanged on upstream ``main`` as of 2026-07-27) closes the
    underlying datagram transports while an in-flight ``stun.Transaction``'s
    retransmission timer — a ``loop.call_later(delay, Transaction.__retry)`` — is
    still scheduled and its owning check task not yet cancelled. It never cancels
    those check tasks (``connect()`` does, but misses *triggered* checks spawned
    after its cancel sweep, and never awaits the cancellation). When such a timer
    fires after its transport's socket is gone, ``__retry`` calls ``send_stun`` →
    ``transport.sendto`` on a ``_SelectorDatagramTransport`` whose ``_sock`` is
    already ``None`` (asyncio's ``_conn_lost`` fast-return guard is skipped because
    ICE host-candidate transports are created without a ``remote_addr``, so
    ``_address`` is unset), raising an ``AttributeError`` that asyncio logs as
    ``Exception in callback Transaction.__retry()``. On a session whose first
    attempt failed and was torn down, these orphaned timers keep firing for seconds
    *into the next (successful) attempt* — teardown log noise, never a functional
    fault, but real and avoidable.

    We fix it at the root: :func:`fal.wma.negotiate_answer` installs the
    :mod:`fal.wma._aioice_teardown` shim (idempotent, version-guarded,
    self-disabling on a future upstream fix), which makes ``Connection.close()``
    cancel *and await* its check tasks before closing the transports — so every
    retransmission timer is cancelled before its socket goes away. We deliberately
    do **not** install a global asyncio exception handler or blanket-swallow
    callback errors (that would risk masking unrelated failures). See
    ``tests/unit/wma/test_wma_helpers.py`` for a deterministic reproduction
    against the real aioice ``Connection`` that pins both the unpatched race and
    the patched fix.
    """
    try:
        await pc.close()
    except Exception:
        pass


async def wma_session_stream(
    answer_event: Mapping[str, Any],
    closed: asyncio.Event,
    cleanup: Callable[[], Awaitable[None]],
    keepalive_interval: float = 15.0,
    *,
    connection_report: Callable[[], Awaitable[Mapping[str, Any]]] | None = None,
) -> AsyncIterator[str]:
    """Yield the SSE body for a ``/start-session`` response.

    Implements the WMA raw-path contract: the first event is the SDP answer, then the
    stream stays open (emitting keepalive comments) until ``closed`` is set. The
    ``cleanup`` coroutine runs in ``finally`` so it fires whether the peer closed
    cleanly, the client dropped, or the bridge stopped receiving heartbeats.

    ``cleanup`` closes the peer connection via :func:`close_peer_connection` — the
    orderly ICE teardown, which (via the :mod:`fal.wma._aioice_teardown` shim
    installed in :func:`negotiate_answer`) cancels aioice's in-flight STUN check
    tasks before closing their transports; see :func:`close_peer_connection`.
    """
    report_task: asyncio.Future[Mapping[str, Any]] | None = None
    if connection_report is not None:
        try:
            report_waiter = connection_report()
            if inspect.isawaitable(report_waiter):
                report_task = asyncio.ensure_future(report_waiter)
            else:
                logger.warning("WMA connection report waiter is not awaitable")
        except Exception:
            logger.warning(
                "WMA connection reporting could not start",
                exc_info=True,
            )
    closed_task = asyncio.ensure_future(closed.wait())
    try:
        answer_payload = dict(answer_event)
        if report_task is not None:
            answer_payload["connection_report_version"] = CONNECTION_REPORT_VERSION
        yield sse_event(answer_payload)

        while not closed.is_set():
            waiters: set[asyncio.Future[Any]] = {closed_task}
            if report_task is not None:
                waiters.add(report_task)
            done, _ = await asyncio.wait(
                waiters,
                timeout=keepalive_interval,
                return_when=asyncio.FIRST_COMPLETED,
            )

            if report_task is not None and report_task in done:
                try:
                    report = report_task.result()
                except asyncio.CancelledError:
                    pass
                except Exception:
                    logger.warning(
                        "WMA connection report collection failed", exc_info=True
                    )
                else:
                    try:
                        report_event = sse_event(report, event="connection_report")
                    except Exception:
                        logger.warning(
                            "WMA connection report serialization failed",
                            exc_info=True,
                        )
                    else:
                        yield report_event
                report_task = None

            if not done:
                yield SSE_KEEPALIVE
    finally:
        tasks_to_await: list[asyncio.Future[Any]] = []
        for task in (report_task, closed_task):
            if task is None:
                continue
            if not task.done():
                task.cancel()
            tasks_to_await.append(task)
        if tasks_to_await:
            await asyncio.gather(*tasks_to_await, return_exceptions=True)
        closed.set()
        await cleanup()
