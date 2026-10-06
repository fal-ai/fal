"""Helpers for the *raw* WMA integration path (``POST /start-session`` + SSE).

These low-level ``fal.wma`` helpers support apps that manage their own
WebRTC media pipeline instead of implementing a ``PeerBackend``:

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
  data channel (message schema in ``fal.wma.protocol``).

Only pure-Python pieces (SSE formatting, session slot, queue helper) live at module
level so they can be imported and unit-tested locally. Everything that needs
``aiortc``/``av`` (runner-only packages) is imported inside functions.
"""

import asyncio
import inspect
import ipaddress
import json
import logging
import threading
from typing import (
    Any,
    AsyncIterator,
    Awaitable,
    Callable,
    Dict,
    List,
    Mapping,
    Set,
    Union,
)

from fal.toolkit.utils.ssrf import is_globally_routable_ip
from fal.wma._aioice_teardown import install_orderly_ice_teardown
from fal.wma.telemetry import (
    CONNECTION_REPORT_VERSION,
    sanitize_connection_report,
)

# ---------------------------------------------------------------------------
# SSE formatting (the /start-session response is an SSE stream)
# ---------------------------------------------------------------------------

#: Comment line emitted periodically so intermediaries don't time the session out.
SSE_KEEPALIVE = ": keepalive\n\n"

logger = logging.getLogger(__name__)


def sse_event(payload: Mapping[str, Any], *, event: Union[str, None] = None) -> str:
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


def queue_put_drop_oldest(queue: "asyncio.Queue[Any]", item: Any) -> None:
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
                queue.task_done()
            except asyncio.QueueEmpty:  # pragma: no cover - racy fallback
                continue


# ---------------------------------------------------------------------------
# WebRTC pieces (aiortc/av imported lazily: runner-only packages)
# ---------------------------------------------------------------------------

VIDEO_CLOCK_RATE = 90_000


def make_video_queue_track(
    frame_queue: "asyncio.Queue[Any]",
    fps: float,
    stats: Union[dict, None] = None,
    *,
    rebase_after_stall: bool = False,
    playout: Any = None,
    on_handoff: Any = None,
) -> Any:
    """Build an outbound video track fed by ``frame_queue`` of RGB uint8 ndarrays.

    Frames are paced at ``fps`` against a wall clock so a burst of generated frames
    (world models produce whole blocks at once) plays back smoothly on the client.

    ``rebase_after_stall`` preserves wall-clock gaps in RTP timestamps and
    acknowledges queue items only after track handoff (including reset invalidation).
    Leave it disabled for consumers using the legacy fixed RTP timeline.

    Latency instrumentation: items may be ``(push_monotonic_ts, image)`` tuples; when
    ``stats`` is given, per-frame queue age (push->pop) and pacing sleep are appended
    to ``stats["queue_age_ms"]`` / ``stats["pace_sleep_ms"]`` (caller drains them).
    """
    if playout is not None:
        return _make_playout_video_queue_track(
            frame_queue, fps, stats, playout, on_handoff, rebase_after_stall
        )

    import time
    from fractions import Fraction
    from typing import Any  # quirk with fal, we have to reimport this

    import av
    import numpy as np
    from aiortc import VideoStreamTrack

    class QueueVideoTrack(VideoStreamTrack):
        def __init__(self) -> None:
            super().__init__()
            self._pts = 0
            self._started_at: Union[float, None] = None

        async def recv(self) -> Any:
            while True:
                item = await frame_queue.get()
                generation = getattr(frame_queue, "generation", None)
                try:
                    frame = await self._render(item)
                    if not rebase_after_stall or generation == getattr(
                        frame_queue, "generation", None
                    ):
                        return frame
                finally:
                    if rebase_after_stall:
                        frame_queue.task_done()

        async def _render(self, item: Any) -> Any:
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
            if rebase_after_stall:
                now = time.monotonic()
                target = self._started_at + self._pts / VIDEO_CLOCK_RATE
                # Keep ordinary scheduling jitter on the fixed clock; a missed
                # frame deadline rebases RTP time as well as the playback clock.
                if now - target >= 0.5 / fps:
                    self._pts = max(
                        self._pts, round((now - self._started_at) * VIDEO_CLOCK_RATE)
                    )
            # Decoded frames can be views of permuted tensors; av requires
            # C-contiguous input (no-op when already contiguous).
            frame = av.VideoFrame.from_ndarray(
                np.ascontiguousarray(image),
                format="yuv420p" if image.ndim == 2 else "rgb24",
            )
            frame.pts = self._pts
            frame.time_base = Fraction(1, VIDEO_CLOCK_RATE)
            return frame

    return QueueVideoTrack()


def _is_unsafe_ice_target(address: str) -> bool:
    """True if an ICE candidate address is an internal/link-local target.

    Without this the server's ICE agent would fire STUN checks at any host
    candidate the offer lists (e.g. cloud metadata IPs) -- an SSRF vector.
    Delegates to :func:`fal.toolkit.utils.ssrf.is_globally_routable_ip` for
    CGNAT / IPv4-mapped / 6to4 coverage. Non-IP targets are hostnames; only
    mDNS (``*.local``, RFC 8445 6.2) is a legitimate candidate shape, anything
    else could be a DNS-rebinding name and is rejected.
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
    """The remote offer was malformed, rejected by ``setRemoteDescription``, or
    incompatible with the media this endpoint negotiates (client input, 422).

    See :func:`negotiate_answer` for which aiortc failures map here.
    """


async def negotiate_answer(pc: Any, sdp: str, type_: str) -> str:
    """Run the server side of the SDP exchange and return the complete answer SDP.

    aiortc does not trickle ICE (nor does the WMA bridge): ``setLocalDescription``
    resolves once gathering finished. Internal-address candidates are stripped
    from the offer first (:func:`filter_sdp_ice_candidates`). Malformed or
    incompatible offers -- including the ``ValueError`` aiortc raises in
    ``createAnswer``/``setLocalDescription`` when the offer omits an ``m=``
    section for a track the server streams -- raise :class:`ClientOfferError`;
    genuine ICE/DTLS faults propagate unchanged. Also installs the
    :mod:`fal.wma._aioice_teardown` shim before any connection can be torn down.
    """
    if not any(line.startswith("m=") for line in sdp.splitlines()):
        raise ClientOfferError("SDP offer has no media sections")

    from aiortc import RTCSessionDescription

    install_orderly_ice_teardown()

    try:
        await pc.setRemoteDescription(
            RTCSessionDescription(sdp=filter_sdp_ice_candidates(sdp), type=type_)
        )
    except Exception as exc:
        # Don't echo aiortc's internal parser wording to the caller (it shifts
        # across versions); ``from exc`` keeps it in the traceback for logs.
        raise ClientOfferError("SDP offer could not be applied") from exc
    try:
        answer = await pc.createAnswer()
        await pc.setLocalDescription(answer)
    except ValueError as exc:
        # A ValueError here means the offer's media sections don't line up with
        # the tracks the server added: a bad request (422), not a server fault.
        # ICE/DTLS faults raise other types and propagate. Don't echo aiortc's wording.
        raise ClientOfferError(
            "the offer is incompatible with the media this endpoint streams"
        ) from exc
    return pc.localDescription.sdp


# Bound on how long a fresh peer connection may sit in a non-terminal ICE state before
# the session is abandoned and its slot freed: on a UDP-blocked network ICE stalls in
# ``checking``, holding the slot for MAX_SESSION_SECONDS; 35s covers cold TURN/TLS.
INITIAL_CONNECT_TIMEOUT_SECONDS = 35.0


def watch_connection_state(
    pc: Any,
    closed: asyncio.Event,
    connected: Union[asyncio.Event, None] = None,
    *,
    disconnected_grace: Union[float, None] = None,
) -> None:
    """Set ``connected`` when media can flow and ``closed`` when the peer
    connection ends, however it ends.

    ``disconnected`` is a transient ICE blip by default (matches ``wmaSession.ts``):
    only ``closed``/``failed`` set ``closed`` unless ``disconnected_grace`` seconds
    is given, after which a connection still ``disconnected`` is treated as
    terminal. Recovery or a terminal transition cancels the pending timer. The
    timer needs a running event loop.
    """
    grace: Dict[str, Union[asyncio.Task, None]] = {"task": None}

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

    Returns ``True`` if it connected within ``timeout``; ``False`` on timeout or
    if ``closed`` fired first (an ICE failure frees the caller immediately
    instead of sitting out the window). ``False`` is the producer's cue to
    abandon the session.
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
    """Idempotently close an aiortc ``RTCPeerConnection``, swallowing teardown errors.

    ``RTCPeerConnection.close()`` is itself idempotent, so the negotiation-failure
    path and session ``cleanup`` may both call this. Exceptions are swallowed so a
    teardown fault cannot mask the client-facing negotiation error. Orderly ICE
    teardown (cancel STUN check tasks before closing transports) comes from the
    :mod:`fal.wma._aioice_teardown` shim installed in :func:`negotiate_answer`.
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
    connection_report: Union[Callable[[], Awaitable[Mapping[str, Any]]], None] = None,
) -> AsyncIterator[str]:
    """Yield the SSE body for a ``/start-session`` response.

    The first event is the SDP answer; the stream then stays open (emitting
    keepalive comments) until ``closed`` is set. ``cleanup`` runs in ``finally``
    so it fires however the session ends.
    """
    report_task: Union[asyncio.Future, None] = None
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
            waiters: Set[asyncio.Future] = {closed_task}
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
                    safe_report = sanitize_connection_report(report)
                    if safe_report is None:
                        logger.warning("WMA connection report was invalid")
                    else:
                        try:
                            report_event = sse_event(
                                safe_report,
                                event="connection_report",
                            )
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
        tasks_to_await: List[asyncio.Future] = []
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


def _make_playout_video_queue_track(
    frame_queue: "asyncio.Queue[Any]",
    fps: float,
    stats: Union[dict, None] = None,
    playout: Any = None,
    on_handoff: Any = None,
    acknowledge: bool = False,
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
    from typing import Any  # quirk with fal, we have to reimport this

    import av
    import numpy as np
    from aiortc import VideoStreamTrack

    class QueueVideoTrack(VideoStreamTrack):
        def __init__(self) -> None:
            super().__init__()
            self._pts = 0
            self._started_at: Union[float, None] = None

        async def recv(self) -> Any:
            next_pts = (
                self._pts
                if self._started_at is None
                else self._pts + int(VIDEO_CLOCK_RATE / fps)
            )
            while True:
                item = await frame_queue.get()
                try:
                    metadata = None
                    if playout is not None:
                        push_ts, image, metadata = item
                        item = (push_ts, image)
                    if metadata is not None and metadata["epoch"] != playout.epoch:
                        continue
                    now = time.monotonic()
                    if isinstance(item, tuple):
                        push_ts, image = item
                        if stats is not None:
                            stats.setdefault("queue_age_ms", []).append(
                                (now - push_ts) * 1000.0
                            )
                    else:
                        image = item
                    started_at = self._started_at
                    if started_at is None:
                        started_at = now
                    else:
                        target = started_at + next_pts / VIDEO_CLOCK_RATE
                        delay = target - now
                        if delay > 0:
                            if stats is not None:
                                stats.setdefault("pace_sleep_ms", []).append(
                                    delay * 1000.0
                                )
                            await asyncio.sleep(delay)
                        elif stats is not None:
                            stats.setdefault("pace_sleep_ms", []).append(0.0)
                        if playout is not None and delay < 0:
                            started_at = now - next_pts / VIDEO_CLOCK_RATE
                    if (
                        playout is not None
                        and metadata is not None
                        and metadata["epoch"] != playout.epoch
                    ):
                        continue
                    # A reset during the pacing sleep also invalidates this item.
                    # Commit clock changes only for frames that will be handed off.
                    self._started_at = started_at
                    self._pts = next_pts
                    # Decoded frames can be views of permuted tensors; av requires
                    # C-contiguous input (no-op when already contiguous).
                    # 2-D arrays are planar yuv420p (H*3/2, W); 3-D arrays are RGB.
                    frame = av.VideoFrame.from_ndarray(
                        np.ascontiguousarray(image),
                        format="yuv420p" if image.ndim == 2 else "rgb24",
                    )
                    frame.pts = self._pts
                    frame.time_base = Fraction(1, VIDEO_CLOCK_RATE)
                    if playout is not None and metadata is not None:
                        playout.handoff(metadata["epoch"])
                        if on_handoff is not None:
                            on_handoff(metadata, push_ts, time.monotonic())
                    return frame
                finally:
                    if acknowledge:
                        frame_queue.task_done()

    return QueueVideoTrack()
